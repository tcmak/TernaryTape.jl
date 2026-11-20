using Libdl
using Lux
using LuxCore
using Random
using TernaryTape
using Test

const REPO_ROOT = normpath(joinpath(@__DIR__, ".."))
const FIXTURES_DIR = joinpath(REPO_ROOT, "fixtures")
const C_HEADER_DIR = joinpath(REPO_ROOT, "c")

# --- Compile C99 ttape_reader.h shared library for fast in-process cross-validation ---
const C_SHLIB_DIR = mktempdir()
const C_WRAPPER_SRC = joinpath(C_SHLIB_DIR, "ttape_c_bridge.c")
const C_WRAPPER_SO = joinpath(C_SHLIB_DIR, "libttape_c_bridge." * Libdl.dlext)

write(
    C_WRAPPER_SRC,
    """
    #include <stddef.h>
    #include <stdint.h>
    #include "ttape_reader.h"

    int ttape_c_open_check(const uint8_t *buf, size_t len, int reader_profile) {
        struct ttape_reader r;
        return (int)ttape_reader_open(&r, buf, len, (ttape_v1_profile_t)reader_profile);
    }
    """,
)

const C_COMPILER = Sys.which("cc") !== nothing ? "cc" : "gcc"
run(
    `$C_COMPILER -std=c99 -Wall -Wextra -Wpedantic -Werror -O2 -fPIC -shared -I$C_HEADER_DIR $C_WRAPPER_SRC -o $C_WRAPPER_SO`,
)
const C_LIB_HANDLE = Libdl.dlopen(C_WRAPPER_SO)
const C_OPEN_SYM = Libdl.dlsym(C_LIB_HANDLE, :ttape_c_open_check)

const C_PROFILE_MAP = Dict(:core => Cint(0), :grouped => Cint(1), :extended => Cint(2))
const TTAPE_OK_CODE = Cint(0)
const TTAPE_ERR_UNSUPPORTED_PROFILE_CODE = Cint(7)

function c99_validate(blob::Vector{UInt8}, reader_profile::Symbol)::Cint
    return ccall(
        C_OPEN_SYM,
        Cint,
        (Ptr{UInt8}, Csize_t, Cint),
        blob,
        length(blob),
        C_PROFILE_MAP[reader_profile],
    )
end

const PYTHON_EXE = Sys.which("python3") !== nothing ? "python3" : "python"

function py_validate_all_profiles(blob::Vector{UInt8})::Dict{Symbol,String}
    mktempdir() do d
        p = joinpath(d, "probe.ttape")
        write(p, blob)
        script = """
import json, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import generate_golden as g
data = Path(sys.argv[2]).read_bytes()
out = {prof: g.validate_ttape(data, reader_profile=prof) for prof in ("core", "grouped", "extended")}
print(json.dumps(out))
"""
        raw = readchomp(`$PYTHON_EXE -c $script $FIXTURES_DIR $p`)
        # Parse simple 3-key JSON without external JSON package
        res = Dict{Symbol,String}()
        for prof in (:core, :grouped, :extended)
            m = match(Regex("\"$prof\"\\s*:\\s*\"([^\"]+)\""), raw)
            @assert m !== nothing "Failed to parse python validator output: $raw"
            res[prof] = String(m.captures[1])
        end
        return res
    end
end

"""
Cross-validate every emitted `.ttape` blob against BOTH:
1. `fixtures/generate_golden.py`'s `validate_ttape` (Python 3 stdlib reference)
2. `c/ttape_reader.h`'s `ttape_reader_open` (ISO C99 wire validator)
across all three reader conformance profiles (`core`, `grouped`, `extended`),
plus single-bit mutation rejection across every byte.
"""
function assert_cross_validated(blob::Vector{UInt8}; tape_profile::Symbol)
    py_res = py_validate_all_profiles(blob)

    expected = if tape_profile === :core
        Dict(
            :core => ("TTAPE_OK", TTAPE_OK_CODE),
            :grouped => ("TTAPE_OK", TTAPE_OK_CODE),
            :extended => ("TTAPE_OK", TTAPE_OK_CODE),
        )
    elseif tape_profile === :grouped
        Dict(
            :core => ("TTAPE_ERR_UNSUPPORTED_PROFILE", TTAPE_ERR_UNSUPPORTED_PROFILE_CODE),
            :grouped => ("TTAPE_OK", TTAPE_OK_CODE),
            :extended => ("TTAPE_OK", TTAPE_OK_CODE),
        )
    elseif tape_profile === :extended
        Dict(
            :core => ("TTAPE_ERR_UNSUPPORTED_PROFILE", TTAPE_ERR_UNSUPPORTED_PROFILE_CODE),
            :grouped => (
                "TTAPE_ERR_UNSUPPORTED_PROFILE",
                TTAPE_ERR_UNSUPPORTED_PROFILE_CODE,
            ),
            :extended => ("TTAPE_OK", TTAPE_OK_CODE),
        )
    else
        error("Unexpected tape_profile: $tape_profile")
    end

    for (rprof, (exp_py, exp_c)) in expected
        @test py_res[rprof] == exp_py
        @test c99_validate(blob, rprof) == exp_c
    end

    # Exhaustive single-bit mutation check (§4 invariant) in C99
    mut = copy(blob)
    for idx in eachindex(mut)
        mut[idx] = xor(mut[idx], 0x01)
        @test c99_validate(mut, :extended) != TTAPE_OK_CODE
        mut[idx] = blob[idx]
    end
end

function assert_goose_error(f::Function; expected_tag::String, expected_substr::String = "")
    caught = nothing
    try
        f()
    catch err
        caught = err
    end
    @test caught isa UnsupportedLayerError
    if caught isa UnsupportedLayerError
        msg = sprint(showerror, caught)
        @test occursin(expected_tag, msg)
        @test occursin("byte offset", msg)
        @test occursin("expected", msg)
        @test occursin("got", msg)
        if !isempty(expected_substr)
            @test occursin(expected_substr, msg)
        end
    end
end

struct CustomUnsupportedLuxLayer <: LuxCore.AbstractLuxLayer end

@testset "TernaryTape.jl — Task C1.2 Standalone Lux v1 Exporter" begin
    @testset "Project.toml compat & zero CRC32c dependency" begin
        proj_txt = read(joinpath(REPO_ROOT, "Project.toml"), String)
        @test !occursin("CRC32c", proj_txt)
        @test occursin("julia = \"1.10\"", proj_txt)
        @test occursin("Lux = \"1\"", proj_txt)
    end

    @testset "Pure-Julia FNV-1a-32 and FNV-1a-64 checksums" begin
        @test fnv1a_32(UInt8[]) === 0x811c9dc5
        @test fnv1a_64(UInt8[]) === 0xcbf29ce484222325
        @test fnv1a_32( collect(codeunits("foobar")) ) === 0xbf9cf968
        @test fnv1a_64( collect(codeunits("foobar")) ) === 0x85944171f73967e8

        for rel in (
            "accept/core_minimal_dot4.ttape",
            "accept/grouped_with_u16_scales.ttape",
            "accept/extended_with_fp32_aux.ttape",
        )
            raw = read(joinpath(FIXTURES_DIR, rel))
            stored_hdr_crc = ltoh(reinterpret(UInt32, raw[13:16])[1])
            stored_art_id = ltoh(reinterpret(UInt64, raw[17:24])[1])
            @test header_fnv1a_32(@view raw[1:32]) === stored_hdr_crc
            @test fnv1a_64(@view raw[33:end]) === stored_art_id
        end
    end

    @testset "Byte-exact parity with golden corpus (:core, :grouped, :extended)" begin
        # 1. core_minimal_dot4
        dense_dot4 = Dense(4 => 2; use_bias = false)
        ps_dot4 = (weight = Int8[1 -1 0 1; -1 1 1 0],)
        st_dot4 = (scale_exp = -2,)

        io = IOBuffer()
        nbytes = export_tape(io, dense_dot4, ps_dot4, st_dot4; profile = :core, context_cap = 16)
        blob_core_min = take!(io)
        @test nbytes == length(blob_core_min)
        @test blob_core_min == read(joinpath(FIXTURES_DIR, "accept/core_minimal_dot4.ttape"))
        assert_cross_validated(blob_core_min; tape_profile = :core)

        # 2. core_with_manifest_and_links
        manifest_nt = (
            producer_id = "ternarytape.jl/0.1.0",
            training_run_id = "run_20261118_bitnet_b158_seed042",
            quant_algorithm = "absmean-b158-pot",
            quant_params = "clip=2.0",
            content_hash = "sha256:7f83b1657ff1fc53b92dc18148a1d65d",
        )
        io = IOBuffer()
        export_tape(
            io,
            dense_dot4,
            ps_dot4,
            st_dot4;
            profile = :core,
            context_cap = 16,
            manifest = manifest_nt,
            link_table = Int32[4, -4],
        )
        blob_core_full = take!(io)
        @test blob_core_full ==
              read(joinpath(FIXTURES_DIR, "accept/core_with_manifest_and_links.ttape"))
        assert_cross_validated(blob_core_full; tape_profile = :core)

        # 3. grouped_with_u16_scales (also tests default profile = :grouped)
        dense_k6 = Dense(6 => 1; use_bias = false)
        ps_k6 = (weight = Int8[1 -1 0 1 -1 -1],)
        st_k6 = (group_size = 64, shift = 8, scales_u16 = reshape(UInt16[0x0180], 1, 1))
        io = IOBuffer()
        export_tape(io, dense_k6, ps_k6, st_k6; context_cap = 64)
        blob_grouped = take!(io)
        @test blob_grouped == read(joinpath(FIXTURES_DIR, "accept/grouped_with_u16_scales.ttape"))
        assert_cross_validated(blob_grouped; tape_profile = :grouped)

        # 4. extended_with_fp32_aux
        ext_model = Chain(
            Embedding(2 => 4),
            RMSNorm((4,)),
            Dense(4 => 2; use_bias = false),
        )
        ps_ext = (
            layer_1 = (
                weight = Float32[
                    0.25 -0.25;
                    -0.5 0.5;
                    0.75 -0.75;
                    1.0 0.125
                ],
            ),
            layer_2 = (scale = Float32[1.0, 1.0, 1.0, 1.0],),
            layer_3 = ps_dot4,
        )
        st_ext = (
            layer_1 = NamedTuple(),
            layer_2 = NamedTuple(),
            layer_3 = st_dot4,
        )
        io = IOBuffer()
        export_tape(
            io,
            ext_model,
            ps_ext,
            st_ext;
            profile = :extended,
            context_cap = 16,
            manifest = manifest_nt,
        )
        blob_extended = take!(io)
        @test blob_extended == read(joinpath(FIXTURES_DIR, "accept/extended_with_fp32_aux.ttape"))
        assert_cross_validated(blob_extended; tape_profile = :extended)

        # 5. lux_chain_grouped_mlp (2-layer deterministic Lux.Chain(Dense(8=>4), Dense(4=>2)))
        mlp_model = Chain(
            Dense(8 => 4; use_bias = false),
            Dense(4 => 2; use_bias = false),
        )
        w0 = Int8[
             1 -1  0  1 -1  0  1  1;
            -1  1  1  0  1 -1  0 -1;
             1  0 -1 -1  0  1 -1  1;
             0  1  1 -1  1  0  1 -1
        ]
        w1 = Int8[
             1 -1  0 -1;
            -1  1  1  0
        ]
        s0 = reshape(UInt16[0x0180, 0x0080, 0x0040, 0x0200], 4, 1)
        s1 = reshape(UInt16[0x0180, 0x0280], 2, 1)
        ps_mlp = (layer_1 = (weight = w0,), layer_2 = (weight = w1,))
        st_mlp = (
            layer_1 = (group_size = 64, shift = 8, scales_u16 = s0),
            layer_2 = (group_size = 64, shift = 8, scales_u16 = s1),
        )
        io = IOBuffer()
        export_tape(io, mlp_model, ps_mlp, st_mlp; profile = :grouped, context_cap = 64)
        blob_mlp = take!(io)
        @test blob_mlp == read(joinpath(FIXTURES_DIR, "accept/lux_chain_grouped_mlp.ttape"))
        assert_cross_validated(blob_mlp; tape_profile = :grouped)

        # Verify sidecar deterministic integer input_vector and expected_output_vector
        sidecar_txt = read(joinpath(FIXTURES_DIR, "accept/lux_chain_grouped_mlp.json"), String)
        @test occursin("\"input_vector\"", sidecar_txt)
        @test occursin("\"expected_output_vector\"", sidecar_txt)
        x_in = Int[4, -2, 6, -4, 2, 0, -6, 8]
        h_vec = [(Int(sum(Int.(w0[r, :]) .* x_in)) * Int(s0[r, 1])) >> 8 for r in 1:4]
        y_vec = [(Int(sum(Int.(w1[r, :]) .* h_vec)) * Int(s1[r, 1])) >> 8 for r in 1:2]
        @test h_vec == [3, -3, 4, -8]
        @test y_vec == [21, -5]
        # Verify Float32 Lux forward pass with effective group scales matches exact integer output
        ps_mlp_eff = (
            layer_1 = (weight = Float32.(w0) .* (Float32.(s0[:, 1]) .* 2.0f0^(-8)),),
            layer_2 = (weight = Float32.(w1) .* (Float32.(s1[:, 1]) .* 2.0f0^(-8)),),
        )
        st_mlp_lux = (layer_1 = NamedTuple(), layer_2 = NamedTuple())
        y_lux, _ = mlp_model(Float32.(x_in), ps_mlp_eff, st_mlp_lux)
        @test y_lux == Float32[21.0, -5.0]
    end

    @testset "Multi-group G in {64, 128, 256}, ragged dims, nested Chain & path export" begin
        rng = Random.MersenneTwister(20261120)
        for G in (64, 128, 256)
            # Ragged dim_in = 130 (not a multiple of 4, 64, 128, or 256)
            dim_in = 130
            dim_out = 3
            n_groups = cld(dim_in, G)
            w_mat = rand(rng, Int8[-1, 0, 1], dim_out, dim_in)
            scales_mat = rand(rng, UInt16(1):UInt16(1024), dim_out, n_groups)

            layer = Dense(dim_in => dim_out; use_bias = false)
            ps = (weight = w_mat,)
            st = (group_size = G, shift = 7, scales_u16 = scales_mat)

            mktempdir() do d
                out_path = joinpath(d, "ragged_g$(G).ttape")
                written = export_tape(out_path, layer, ps, st; profile = :grouped)
                disk_bytes = read(out_path)
                @test written == length(disk_bytes)
                assert_cross_validated(disk_bytes; tape_profile = :grouped)
            end
        end

        # Hybrid :extended nested Chain carrying BOTH 0x0006 group_scales_u16 and 0x0007 fp32_aux
        nested_ext = Chain(
            Embedding(5 => 8),
            RMSNorm((8,)),
            Chain(
                Dense(8 => 4; use_bias = false),
                Dense(4 => 3; use_bias = false),
            ),
        )
        ps_nested = (
            layer_1 = (weight = Float32.(reshape(collect(1:40), 8, 5) ./ 16),),
            layer_2 = (scale = fill(1.25f0, 8),),
            layer_3 = (
                layer_1 = (weight = rand(rng, Int8[-1, 0, 1], 4, 8),),
                layer_2 = (weight = rand(rng, Int8[-1, 0, 1], 3, 4),),
            ),
        )
        st_nested = (
            layer_1 = NamedTuple(),
            layer_2 = NamedTuple(),
            layer_3 = (
                layer_1 = (group_size = 64, shift = 6, scales_u16 = fill(UInt16(96), 4, 1)),
                layer_2 = (group_size = 64, shift = 5, scales_u16 = fill(UInt16(48), 3, 1)),
            ),
        )
        io = IOBuffer()
        export_tape(io, nested_ext, ps_nested, st_nested; profile = :extended)
        blob_nested_ext = take!(io)
        assert_cross_validated(blob_nested_ext; tape_profile = :extended)
    end

    @testset "Goose-style UnsupportedLayerError diagnostics" begin
        dense_ok = Dense(4 => 2; use_bias = false)
        ps_ok = (weight = Int8[1 0 -1 1; 0 1 -1 0],)
        st_ok = NamedTuple()

        # 1. Invalid profile symbol
        assert_goose_error(; expected_tag = "0x0000", expected_substr = "bogus") do
            export_tape(IOBuffer(), dense_ok, ps_ok, st_ok; profile = :bogus)
        end

        # 2. Custom / non-allowlisted LuxCore.AbstractLuxLayer (e.g. Dropout & custom struct)
        assert_goose_error(; expected_tag = "0x0003", expected_substr = "Dropout") do
            ch = Chain(dense_ok, Dropout(0.5f0))
            export_tape(IOBuffer(), ch, (layer_1 = ps_ok, layer_2 = NamedTuple()), (layer_1 = st_ok, layer_2 = NamedTuple()))
        end
        assert_goose_error(; expected_tag = "0x0003", expected_substr = "CustomUnsupportedLuxLayer") do
            export_tape(IOBuffer(), CustomUnsupportedLuxLayer(), NamedTuple(), NamedTuple())
        end

        # 3. Dense with use_bias = true
        assert_goose_error(; expected_tag = "0x0003", expected_substr = "use_bias") do
            d_bias = Dense(4 => 2)
            export_tape(IOBuffer(), d_bias, (weight = ps_ok.weight, bias = zeros(Float32, 2)), st_ok)
        end

        # 4. Dense with non-identity activation
        assert_goose_error(; expected_tag = "0x0003", expected_substr = "activation") do
            d_act = Dense(4 => 2, relu; use_bias = false)
            export_tape(IOBuffer(), d_act, ps_ok, st_ok)
        end

        # 5. Unquantized / non-ternary Dense weights
        assert_goose_error(; expected_tag = "0x0001", expected_substr = "0.25") do
            bad_w = Float32[1 0 -1 1; 0 0.25 -1 0]
            export_tape(IOBuffer(), dense_ok, (weight = bad_w,), st_ok; profile = :core)
        end
        assert_goose_error(; expected_tag = "0x0001", expected_substr = "2") do
            bad_w = Int8[1 0 2 1; 0 1 -1 0]
            export_tape(IOBuffer(), dense_ok, (weight = bad_w,), st_ok; profile = :core)
        end

        # 6. Profile/layer mismatches: RMSNorm and Embedding forbidden in :core and :grouped
        rms_chain = Chain(RMSNorm((4,)), dense_ok)
        ps_rms = (layer_1 = (scale = ones(Float32, 4),), layer_2 = ps_ok)
        st_rms = (layer_1 = NamedTuple(), layer_2 = st_ok)
        for bad_prof in (:core, :grouped)
            assert_goose_error(; expected_tag = "0x0007", expected_substr = "RMSNorm") do
                export_tape(IOBuffer(), rms_chain, ps_rms, st_rms; profile = bad_prof)
            end
        end

        emb_chain = Chain(Embedding(3 => 4), dense_ok)
        ps_emb = (layer_1 = (weight = ones(Float32, 4, 3),), layer_2 = ps_ok)
        st_emb = (layer_1 = NamedTuple(), layer_2 = st_ok)
        for bad_prof in (:core, :grouped)
            assert_goose_error(; expected_tag = "0x0007", expected_substr = "Embedding") do
                export_tape(IOBuffer(), emb_chain, ps_emb, st_emb; profile = bad_prof)
            end
        end

        # 7. :extended profile missing 0x0007 fp32_aux or missing ternary_dense
        assert_goose_error(; expected_tag = "0x0007") do
            export_tape(IOBuffer(), dense_ok, ps_ok, st_ok; profile = :extended)
        end
        assert_goose_error(; expected_tag = "0x0001") do
            export_tape(IOBuffer(), RMSNorm((4,)), (scale = ones(Float32, 4),), NamedTuple(); profile = :extended)
        end

        # 8. Non-finite NaN / Inf in 0x0007 fp32_aux
        assert_goose_error(; expected_tag = "0x0007", expected_substr = "NaN") do
            ps_nan = (layer_1 = (scale = Float32[1.0, NaN32, 1.0, 1.0],), layer_2 = ps_ok)
            export_tape(IOBuffer(), rms_chain, ps_nan, st_rms; profile = :extended)
        end
        assert_goose_error(; expected_tag = "0x0007", expected_substr = "Inf") do
            w_inf = ones(Float32, 4, 3)
            w_inf[2, 3] = Inf32
            export_tape(IOBuffer(), emb_chain, (layer_1 = (weight = w_inf,), layer_2 = ps_ok), st_emb; profile = :extended)
        end

        # 9. Invalid scale_exp, group_size, and shift
        assert_goose_error(; expected_tag = "0x0002", expected_substr = "-31") do
            export_tape(IOBuffer(), dense_ok, ps_ok, (scale_exp = -31,); profile = :core)
        end
        assert_goose_error(; expected_tag = "0x0006", expected_substr = "32") do
            export_tape(IOBuffer(), dense_ok, ps_ok, (group_size = 32,); profile = :grouped)
        end
        assert_goose_error(; expected_tag = "0x0006", expected_substr = "31") do
            export_tape(IOBuffer(), dense_ok, ps_ok, (shift = 31,); profile = :grouped)
        end
    end
end
