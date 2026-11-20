module TernaryTape

using Lux: Lux
using LuxCore: LuxCore

export UnsupportedLayerError, export_tape, fnv1a_32, fnv1a_64, header_fnv1a_32

# --- Wire Constants (spec/ttape-v1.0.md) ---
const TTAPE_MAGIC = UInt8['T', 'T', 'A', 'P']
const TTAPE_VERSION_MAJOR = UInt8(1)
const TTAPE_VERSION_MINOR = UInt8(0)

const FLAG_LINKS = UInt8(0x01)
const FLAG_MANIFEST = UInt8(0x02)
const FLAG_NON_CORE = UInt8(0x04)

const TAG_HEADER = UInt16(0x0000)
const TAG_WEIGHTS = UInt16(0x0001)
const TAG_SCALES = UInt16(0x0002)
const TAG_ARCH_DESCRIPTOR = UInt16(0x0003)
const TAG_MANIFEST = UInt16(0x0004)
const TAG_LINK_TABLE = UInt16(0x0005)
const TAG_GROUP_SCALES_U16 = UInt16(0x0006)
const TAG_FP32_AUX = UInt16(0x0007)

const LAYER_RMSNORM_SHIFT = UInt16(3)
const LAYER_TERNARY_DENSE = UInt16(4)
const LAYER_HOST_EMBEDDING = UInt16(6)

const AUX_ROLE_EMBEDDING = UInt8(1)
const AUX_ROLE_RMSNORM = UInt8(2)

const VALID_PROFILES = (:core, :grouped, :extended)
const VALID_GROUP_SIZES = (64, 128, 256)

"""
    UnsupportedLayerError <: Exception

Goose-style structured diagnostic thrown when `export_tape` encounters an unsupported
`LuxCore.AbstractLuxLayer`, profile/layer mismatch, unquantized weight, out-of-range
scale parameter, or malformed section input.
"""
struct UnsupportedLayerError <: Exception
    layer_path::String
    section_tag::UInt16
    byte_offset::Int
    profile::Symbol
    expected::String
    actual::String
end

function Base.showerror(io::IO, err::UnsupportedLayerError)
    print(
        io,
        "UnsupportedLayerError at layer `",
        err.layer_path,
        "` (section tag 0x",
        string(err.section_tag; base = 16, pad = 4),
        ", byte offset ",
        err.byte_offset,
        ", profile :",
        err.profile,
        "): expected ",
        err.expected,
        ", got ",
        err.actual,
    )
end

# --- Pure-Julia FNV-1a Checksums (spec/ttape-v1.0.md §2.3) ---

"""
    fnv1a_32(data::AbstractVector{UInt8}) -> UInt32

Compute 32-bit FNV-1a (`offset_basis = 0x811c9dc5`, `prime = 0x01000193`).
"""
function fnv1a_32(data::AbstractVector{UInt8})::UInt32
    h = 0x811c9dc5
    @inbounds for b in data
        h = xor(h, UInt32(b))
        h = h * 0x01000193
    end
    return h
end

"""
    fnv1a_64(data::AbstractVector{UInt8}) -> UInt64

Compute 64-bit FNV-1a (`offset_basis = 0xcbf29ce484222325`, `prime = 0x00000100000001b3`).
"""
function fnv1a_64(data::AbstractVector{UInt8})::UInt64
    h = 0xcbf29ce484222325
    @inbounds for b in data
        h = xor(h, UInt64(b))
        h = h * 0x00000100000001b3
    end
    return h
end

"""
    header_fnv1a_32(hdr::AbstractVector{UInt8}) -> UInt32

Compute `header_fnv1a` over the 28 non-checksum bytes `hdr[0x00..0x0B] || hdr[0x10..0x1F]`
of a 32-byte `ttape_v1_header`.
"""
function header_fnv1a_32(hdr::AbstractVector{UInt8})::UInt32
    length(hdr) == 32 || throw(
        ArgumentError("ttape_v1_header must be exactly 32 bytes, got $(length(hdr))"),
    )
    f = firstindex(hdr)
    h = 0x811c9dc5
    @inbounds for i in 0:11
        h = xor(h, UInt32(hdr[f + i]))
        h = h * 0x01000193
    end
    @inbounds for i in 16:31
        h = xor(h, UInt32(hdr[f + i]))
        h = h * 0x01000193
    end
    return h
end

# --- Little-Endian Binary Helpers ---

@inline write_u8(io::IO, v::UInt8) = write(io, v)
@inline write_u16_le(io::IO, v::UInt16) = write(io, htol(v))
@inline write_i16_le(io::IO, v::Int16) = write(io, htol(v))
@inline write_u32_le(io::IO, v::UInt32) = write(io, htol(v))
@inline write_i32_le(io::IO, v::Int32) = write(io, htol(v))
@inline write_u64_le(io::IO, v::UInt64) = write(io, htol(v))

@inline function clog2_u16(n::Integer)::UInt8
    stages = UInt8(0)
    span = UInt32(1)
    while span < UInt32(n)
        span <<= 1
        stages += UInt8(1)
    end
    return stages
end

# --- Model Walking & Parameter Extraction ---

struct FlatLayerEntry
    path::String
    layer::LuxCore.AbstractLuxLayer
    ps::Any
    st::Any
end

@inline function _lookup(container, key::Symbol, default = nothing)
    if container isa NamedTuple && haskey(container, key)
        return getfield(container, key)
    elseif container isa AbstractDict && haskey(container, key)
        return container[key]
    end
    return default
end

@inline function _layer_meta(ps, st, key::Symbol, default = nothing)
    val = _lookup(st, key, nothing)
    val !== nothing && return val
    return _lookup(ps, key, default)
end

function _collect_leaf_layers!(
    out::Vector{FlatLayerEntry},
    layer::LuxCore.AbstractLuxLayer,
    ps,
    st,
    path::String,
    profile::Symbol,
)
    if layer isa Lux.Chain
        sublayers = getfield(layer, :layers)
        if !(sublayers isa NamedTuple) || isempty(keys(sublayers))
            throw(
                UnsupportedLayerError(
                    path,
                    TAG_ARCH_DESCRIPTOR,
                    4 + 16 * length(out),
                    profile,
                    "non-empty Lux.Chain",
                    repr(typeof(layer)),
                ),
            )
        end
        for k in keys(sublayers)
            child_layer = getfield(sublayers, k)
            child_ps = _lookup(ps, k, NamedTuple())
            child_st = _lookup(st, k, NamedTuple())
            _collect_leaf_layers!(out, child_layer, child_ps, child_st, "$path.$k", profile)
        end
    else
        push!(out, FlatLayerEntry(path, layer, ps, st))
    end
    return out
end

function _has_explicit_group_meta(entry::FlatLayerEntry)::Bool
    return _layer_meta(entry.ps, entry.st, :scales_u16, nothing) !== nothing ||
           _layer_meta(entry.ps, entry.st, :group_size, nothing) !== nothing ||
           _layer_meta(entry.ps, entry.st, :shift, nothing) !== nothing
end

# --- Section Builders ---

struct ArchLayerSpec
    kind::UInt16
    dim_in::UInt16
    dim_out::UInt16
    reduce_stages::UInt8
    latency_cycles::UInt32
    context_cap::UInt32
end

struct GroupScaleSpec
    layer_index::UInt16
    group_size::UInt16
    shift::UInt8
    n_groups::UInt16
    scales_row_major::Vector{UInt16}
end

struct Fp32AuxSpec
    layer_index::UInt16
    aux_role::UInt8
    elem_count::UInt32
    words_le::Vector{UInt32}
end

function _validate_dim_u16(
    val,
    dim_name::String,
    path::String,
    arch_off::Int,
    profile::Symbol,
)::UInt16
    if !(val isa Integer) || (val isa Bool) || val < 1 || val > typemax(UInt16)
        throw(
            UnsupportedLayerError(
                path,
                TAG_ARCH_DESCRIPTOR,
                arch_off,
                profile,
                "$dim_name in 1..65535",
                repr(val),
            ),
        )
    end
    return UInt16(val)
end

function _pack_dense_weights!(
    weights_io::IOBuffer,
    W,
    dim_in::Int,
    dim_out::Int,
    path::String,
    profile::Symbol,
)
    if !(W isa AbstractMatrix) || size(W) != (dim_out, dim_in)
        actual_desc = W isa AbstractArray ? "array of size $(size(W))" : repr(typeof(W))
        throw(
            UnsupportedLayerError(
                path,
                TAG_WEIGHTS,
                position(weights_io),
                profile,
                "weight matrix of size ($dim_out, $dim_in)",
                actual_desc,
            ),
        )
    end

    base_off = position(weights_io)
    row_bytes = cld(dim_in, 4)

    @inbounds for r in 1:dim_out
        for b_idx in 0:(row_bytes - 1)
            packed = UInt8(0x00)
            for lane in 0:3
                c = 4 * b_idx + lane + 1
                if c <= dim_in
                    w = W[r, c]
                    if (w isa Bool) || !(w isa Real) || !isfinite(w) ||
                       !(w == -1 || w == 0 || w == 1)
                        byte_off = base_off + (r - 1) * row_bytes + b_idx
                        throw(
                            UnsupportedLayerError(
                                path,
                                TAG_WEIGHTS,
                                byte_off,
                                profile,
                                "strictly quantized ternary weight in {-1, 0, +1} at (row=$r, col=$c)",
                                repr(w),
                            ),
                        )
                    end
                    code = w == 0 ? UInt8(0x00) : (w == 1 ? UInt8(0x01) : UInt8(0x02))
                    packed |= code << (2 * lane)
                end
            end
            write_u8(weights_io, packed)
        end
    end
    return nothing
end

function _extract_group_scales(
    entry::FlatLayerEntry,
    layer_idx_zero_based::Int,
    dense_idx_zero_based::Int,
    dim_in::Int,
    dim_out::Int,
    cur_group_bytes::Int,
    default_group_size::Integer,
    default_shift::Integer,
    profile::Symbol,
)::Tuple{Int16,GroupScaleSpec}
    raw_g = _layer_meta(entry.ps, entry.st, :group_size, default_group_size)
    if !(raw_g isa Integer) || (raw_g isa Bool) || !(Int(raw_g) in VALID_GROUP_SIZES)
        throw(
            UnsupportedLayerError(
                entry.path,
                TAG_GROUP_SCALES_U16,
                cur_group_bytes + 2,
                profile,
                "group_size in {64, 128, 256}",
                repr(raw_g),
            ),
        )
    end
    G = Int(raw_g)
    n_groups = cld(dim_in, G)

    raw_shift = _layer_meta(entry.ps, entry.st, :shift, nothing)
    raw_scale_exp = _layer_meta(entry.ps, entry.st, :scale_exp, nothing)

    shift_val = if raw_shift !== nothing
        if !(raw_shift isa Integer) || (raw_shift isa Bool) || raw_shift < 0 || raw_shift > 30
            throw(
                UnsupportedLayerError(
                    entry.path,
                    TAG_GROUP_SCALES_U16,
                    cur_group_bytes + 4,
                    profile,
                    "group scale shift in 0..30",
                    repr(raw_shift),
                ),
            )
        end
        Int(raw_shift)
    elseif raw_scale_exp !== nothing
        if !(raw_scale_exp isa Integer) || (raw_scale_exp isa Bool) ||
           raw_scale_exp < -30 || raw_scale_exp > 0
            throw(
                UnsupportedLayerError(
                    entry.path,
                    TAG_GROUP_SCALES_U16,
                    cur_group_bytes + 4,
                    profile,
                    "non-positive scale_exp in -30..0 matching -shift",
                    repr(raw_scale_exp),
                ),
            )
        end
        -Int(raw_scale_exp)
    else
        Int(default_shift)
    end

    if shift_val < 0 || shift_val > 30
        throw(
            UnsupportedLayerError(
                entry.path,
                TAG_GROUP_SCALES_U16,
                cur_group_bytes + 4,
                profile,
                "group scale shift in 0..30",
                repr(shift_val),
            ),
        )
    end

    if raw_shift !== nothing && raw_scale_exp !== nothing &&
       Int(raw_scale_exp) != -shift_val
        throw(
            UnsupportedLayerError(
                entry.path,
                TAG_GROUP_SCALES_U16,
                2 * dense_idx_zero_based,
                profile,
                "scale_exp == -shift (-$shift_val)",
                "scale_exp=$(raw_scale_exp), shift=$shift_val",
            ),
        )
    end

    raw_scales = _layer_meta(entry.ps, entry.st, :scales_u16, nothing)
    scales_rm = Vector{UInt16}(undef, dim_out * n_groups)

    if raw_scales === nothing
        unit_mantissa = shift_val <= 15 ? (UInt16(1) << shift_val) : UInt16(1)
        fill!(scales_rm, unit_mantissa)
    else
        mat = if raw_scales isa AbstractVector && n_groups == 1 && length(raw_scales) == dim_out
            reshape(raw_scales, dim_out, 1)
        elseif raw_scales isa AbstractMatrix && size(raw_scales) == (dim_out, n_groups)
            raw_scales
        else
            actual_desc =
                raw_scales isa AbstractArray ? "array of size $(size(raw_scales))" :
                repr(typeof(raw_scales))
            throw(
                UnsupportedLayerError(
                    entry.path,
                    TAG_GROUP_SCALES_U16,
                    cur_group_bytes + 8,
                    profile,
                    "scales_u16 matrix of size ($dim_out, $n_groups)",
                    actual_desc,
                ),
            )
        end

        idx = 1
        @inbounds for r in 1:dim_out
            for g in 1:n_groups
                v = mat[r, g]
                if !(v isa Integer) || (v isa Bool) || v < 0 || v > typemax(UInt16)
                    byte_off = cur_group_bytes + 8 + 2 * (idx - 1)
                    throw(
                        UnsupportedLayerError(
                            entry.path,
                            TAG_GROUP_SCALES_U16,
                            byte_off,
                            profile,
                            "uint16 scale mantissa in 0..65535 at (row=$r, group=$g)",
                            repr(v),
                        ),
                    )
                end
                scales_rm[idx] = UInt16(v)
                idx += 1
            end
        end
    end

    spec = GroupScaleSpec(
        UInt16(layer_idx_zero_based),
        UInt16(G),
        UInt8(shift_val),
        UInt16(n_groups),
        scales_rm,
    )
    return (Int16(-shift_val), spec)
end

function _f32_checked_word(
    val,
    path::String,
    byte_off::Int,
    role_desc::String,
    profile::Symbol,
)::UInt32
    if (val isa Bool) || !(val isa Real)
        throw(
            UnsupportedLayerError(
                path,
                TAG_FP32_AUX,
                byte_off,
                profile,
                "finite IEEE-754 binary32 $role_desc",
                repr(val),
            ),
        )
    end
    f32 = Float32(val)
    word = reinterpret(UInt32, f32)
    if (word & 0x7f800000) == 0x7f800000
        throw(
            UnsupportedLayerError(
                path,
                TAG_FP32_AUX,
                byte_off,
                profile,
                "finite IEEE-754 binary32 $role_desc (NaN/±Inf forbidden)",
                repr(val),
            ),
        )
    end
    return word
end

function _build_manifest_payload(manifest, profile::Symbol)::Vector{UInt8}
    required_keys = (
        :producer_id,
        :training_run_id,
        :quant_algorithm,
        :quant_params,
        :content_hash,
    )
    io = IOBuffer()
    for (f_idx, k) in enumerate(required_keys)
        off = position(io)
        raw_val = _lookup(manifest, k, nothing)
        if raw_val === nothing
            throw(
                UnsupportedLayerError(
                    "manifest.$k",
                    TAG_MANIFEST,
                    off,
                    profile,
                    "non-empty manifest field `$k`",
                    "missing",
                ),
            )
        end
        bytes = if raw_val isa AbstractString
            collect(codeunits(raw_val))
        elseif raw_val isa AbstractVector{UInt8}
            collect(raw_val)
        else
            throw(
                UnsupportedLayerError(
                    "manifest.$k",
                    TAG_MANIFEST,
                    off,
                    profile,
                    "String or Vector{UInt8} for manifest field `$k`",
                    repr(typeof(raw_val)),
                ),
            )
        end
        blen = length(bytes)
        if blen == 0 || blen > typemax(UInt16)
            throw(
                UnsupportedLayerError(
                    "manifest.$k",
                    TAG_MANIFEST,
                    off,
                    profile,
                    "manifest field `$k` byte length in 1..65535",
                    string(blen),
                ),
            )
        end
        if f_idx == 2 && blen != 32
            throw(
                UnsupportedLayerError(
                    "manifest.training_run_id",
                    TAG_MANIFEST,
                    off,
                    profile,
                    "32-byte training_run_id",
                    "$blen bytes",
                ),
            )
        end
        if f_idx == 5 && blen < 8
            throw(
                UnsupportedLayerError(
                    "manifest.content_hash",
                    TAG_MANIFEST,
                    off,
                    profile,
                    "content_hash length >= 8 bytes",
                    "$blen bytes",
                ),
            )
        end
        write_u16_le(io, UInt16(blen))
        write(io, bytes)
    end
    return take!(io)
end

function _build_link_table_payload(
    links::AbstractVector{<:Integer},
    profile::Symbol,
)::Vector{UInt8}
    if isempty(links)
        throw(
            UnsupportedLayerError(
                "link_table",
                TAG_LINK_TABLE,
                0,
                profile,
                "non-empty 4-byte aligned link_table",
                "0 entries",
            ),
        )
    end
    total_bytes = 4 * length(links)
    io = IOBuffer()
    for (i, raw_delta) in enumerate(links)
        off = 4 * (i - 1)
        if (raw_delta isa Bool) || raw_delta < typemin(Int32) || raw_delta > typemax(Int32)
            throw(
                UnsupportedLayerError(
                    "link_table[$i]",
                    TAG_LINK_TABLE,
                    off,
                    profile,
                    "Int32 self-relative byte delta",
                    repr(raw_delta),
                ),
            )
        end
        delta = Int32(raw_delta)
        target = off + Int(delta)
        if delta == 0 || mod(delta, 4) != 0 || target < 0 || target >= total_bytes
            throw(
                UnsupportedLayerError(
                    "link_table[$i]",
                    TAG_LINK_TABLE,
                    off,
                    profile,
                    "non-zero 4-byte aligned in-bounds delta (0 <= off + delta < $total_bytes)",
                    string(delta),
                ),
            )
        end
        write_i32_le(io, delta)
    end
    return take!(io)
end

function _serialize_arch_payload(specs::Vector{ArchLayerSpec})::Vector{UInt8}
    io = IOBuffer()
    write_u16_le(io, UInt16(length(specs)))
    write_u16_le(io, UInt16(0))
    for s in specs
        write_u16_le(io, s.kind)
        write_u16_le(io, s.dim_in)
        write_u16_le(io, s.dim_out)
        write_u8(io, s.reduce_stages)
        write_u8(io, UInt8(0))
        write_u32_le(io, s.latency_cycles)
        write_u32_le(io, s.context_cap)
    end
    return take!(io)
end

function _serialize_group_scales_payload(specs::Vector{GroupScaleSpec})::Vector{UInt8}
    io = IOBuffer()
    write_u16_le(io, UInt16(length(specs)))
    write_u16_le(io, UInt16(0))
    for s in specs
        write_u16_le(io, s.layer_index)
        write_u16_le(io, s.group_size)
        write_u8(io, s.shift)
        write_u8(io, UInt8(0))
        write_u16_le(io, s.n_groups)
        for v in s.scales_row_major
            write_u16_le(io, v)
        end
    end
    return take!(io)
end

function _serialize_fp32_aux_payload(specs::Vector{Fp32AuxSpec})::Vector{UInt8}
    io = IOBuffer()
    write_u16_le(io, UInt16(length(specs)))
    write_u16_le(io, UInt16(0))
    for s in specs
        write_u16_le(io, s.layer_index)
        write_u8(io, s.aux_role)
        write_u8(io, UInt8(0))
        write_u32_le(io, s.elem_count)
        for w in s.words_le
            write_u32_le(io, w)
        end
    end
    return take!(io)
end

function _frame_ttape(
    sections::Vector{Tuple{UInt16,Vector{UInt8}}},
    flags::UInt8,
)::Vector{UInt8}
    sec_io = IOBuffer()
    for (tag, payload) in sections
        write_u16_le(sec_io, tag)
        write_u16_le(sec_io, UInt16(0))
        write_u32_le(sec_io, UInt32(length(payload)))
        write_u32_le(sec_io, fnv1a_32(payload))
        write(sec_io, payload)
    end
    sec_bytes = take!(sec_io)
    total_length = UInt32(length(sec_bytes))
    artifact_id = fnv1a_64(sec_bytes)

    hdr_io = IOBuffer()
    write(hdr_io, TTAPE_MAGIC)
    write_u8(hdr_io, TTAPE_VERSION_MAJOR)
    write_u8(hdr_io, TTAPE_VERSION_MINOR)
    write_u8(hdr_io, flags)
    write_u8(hdr_io, UInt8(length(sections)))
    write_u32_le(hdr_io, total_length)
    write_u32_le(hdr_io, UInt32(0))
    write_u64_le(hdr_io, artifact_id)
    for _ in 1:8
        write_u8(hdr_io, UInt8(0))
    end
    hdr_bytes = take!(hdr_io)
    hdr_crc = header_fnv1a_32(hdr_bytes)

    # Patch header_fnv1a at byte offset 0x0C (1-based indices 13..16)
    crc_le = htol(hdr_crc)
    hdr_bytes[13] = UInt8(crc_le & 0xff)
    hdr_bytes[14] = UInt8((crc_le >> 8) & 0xff)
    hdr_bytes[15] = UInt8((crc_le >> 16) & 0xff)
    hdr_bytes[16] = UInt8((crc_le >> 24) & 0xff)

    out = Vector{UInt8}(undef, 32 + length(sec_bytes))
    copyto!(out, 1, hdr_bytes, 1, 32)
    copyto!(out, 33, sec_bytes, 1, length(sec_bytes))
    return out
end

function _compile_tape_bytes(
    model::LuxCore.AbstractLuxLayer,
    ps,
    st;
    profile::Symbol = :grouped,
    group_size::Integer = 64,
    default_shift::Integer = 8,
    default_scale_exp::Integer = 0,
    context_cap::Integer = 64,
    include_group_scales::Union{Nothing,Bool} = nothing,
    manifest = nothing,
    link_table::Union{Nothing,AbstractVector{<:Integer}} = nothing,
)::Vector{UInt8}
    if !(profile in VALID_PROFILES)
        throw(
            UnsupportedLayerError(
                "model",
                TAG_HEADER,
                6,
                profile,
                "profile in (:core, :grouped, :extended)",
                repr(profile),
            ),
        )
    end

    if (context_cap isa Bool) || context_cap < 1 || context_cap > typemax(UInt32)
        throw(
            UnsupportedLayerError(
                "model",
                TAG_ARCH_DESCRIPTOR,
                16,
                profile,
                "context_cap in 1..$(typemax(UInt32))",
                repr(context_cap),
            ),
        )
    end
    cap_u32 = UInt32(context_cap)

    entries = FlatLayerEntry[]
    _collect_leaf_layers!(entries, model, ps, st, "model", profile)

    if isempty(entries) || length(entries) > typemax(UInt16)
        throw(
            UnsupportedLayerError(
                "model",
                TAG_ARCH_DESCRIPTOR,
                0,
                profile,
                "1..65535 leaf layers",
                "$(length(entries)) layers",
            ),
        )
    end

    emit_group_scales = if profile === :core
        if include_group_scales === true
            throw(
                UnsupportedLayerError(
                    "model",
                    TAG_GROUP_SCALES_U16,
                    0,
                    profile,
                    "include_group_scales == false in :core profile (tag 0x0006 forbidden)",
                    "true",
                ),
            )
        end
        false
    elseif profile === :grouped
        if include_group_scales === false
            throw(
                UnsupportedLayerError(
                    "model",
                    TAG_GROUP_SCALES_U16,
                    0,
                    profile,
                    "include_group_scales == true in :grouped profile (tag 0x0006 mandatory)",
                    "false",
                ),
            )
        end
        true
    else # :extended
        include_group_scales === nothing ? any(_has_explicit_group_meta, entries) :
        include_group_scales
    end

    weights_io = IOBuffer()
    scales_io = IOBuffer()
    arch_specs = ArchLayerSpec[]
    group_specs = GroupScaleSpec[]
    aux_specs = Fp32AuxSpec[]

    cur_group_bytes = 4
    cur_aux_bytes = 4
    dense_count = 0

    for (idx1, entry) in enumerate(entries)
        layer_idx0 = idx1 - 1
        arch_off = 4 + 16 * layer_idx0
        layer = entry.layer
        path = "$(entry.path) ($(nameof(typeof(layer))))"

        if layer isa Lux.Dense
            if Bool(layer.use_bias)
                throw(
                    UnsupportedLayerError(
                        path,
                        TAG_ARCH_DESCRIPTOR,
                        arch_off,
                        profile,
                        "quantized ternary Lux.Dense with use_bias=false (TTAPE ternary_dense has no bias section)",
                        "use_bias=true",
                    ),
                )
            end
            if layer.activation !== identity
                throw(
                    UnsupportedLayerError(
                        path,
                        TAG_ARCH_DESCRIPTOR,
                        arch_off,
                        profile,
                        "quantized ternary Lux.Dense with activation === identity",
                        "activation=$(repr(layer.activation))",
                    ),
                )
            end

            din_u16 = _validate_dim_u16(layer.in_dims, "in_dims", path, arch_off + 2, profile)
            dout_u16 =
                _validate_dim_u16(layer.out_dims, "out_dims", path, arch_off + 4, profile)
            din = Int(din_u16)
            dout = Int(dout_u16)

            W = _lookup(entry.ps, :weight, nothing)
            _pack_dense_weights!(weights_io, W, din, dout, path, profile)

            if emit_group_scales
                exp_s16, gspec = _extract_group_scales(
                    FlatLayerEntry(path, layer, entry.ps, entry.st),
                    layer_idx0,
                    dense_count,
                    din,
                    dout,
                    cur_group_bytes,
                    group_size,
                    default_shift,
                    profile,
                )
                write_i16_le(scales_io, exp_s16)
                push!(group_specs, gspec)
                cur_group_bytes += 8 + 2 * length(gspec.scales_row_major)
            else
                raw_exp = _layer_meta(entry.ps, entry.st, :scale_exp, default_scale_exp)
                if !(raw_exp isa Integer) || (raw_exp isa Bool) ||
                   raw_exp < -30 || raw_exp > 30
                    throw(
                        UnsupportedLayerError(
                            path,
                            TAG_SCALES,
                            2 * dense_count,
                            profile,
                            "s16 scale_exp in [-30, 30]",
                            repr(raw_exp),
                        ),
                    )
                end
                write_i16_le(scales_io, Int16(raw_exp))
            end

            stages = clog2_u16(din_u16)
            raw_lat = _layer_meta(entry.ps, entry.st, :latency_cycles, UInt32(stages))
            if !(raw_lat isa Integer) || (raw_lat isa Bool) ||
               raw_lat < Int(stages) || raw_lat > typemax(UInt32)
                throw(
                    UnsupportedLayerError(
                        path,
                        TAG_ARCH_DESCRIPTOR,
                        arch_off + 8,
                        profile,
                        "latency_cycles >= reduce_stages ($(Int(stages)))",
                        repr(raw_lat),
                    ),
                )
            end
            raw_cap = _layer_meta(entry.ps, entry.st, :context_cap, cap_u32)
            if !(raw_cap isa Integer) || (raw_cap isa Bool) ||
               raw_cap < 1 || raw_cap > typemax(UInt32)
                throw(
                    UnsupportedLayerError(
                        path,
                        TAG_ARCH_DESCRIPTOR,
                        arch_off + 12,
                        profile,
                        "context_cap >= 1",
                        repr(raw_cap),
                    ),
                )
            end

            push!(
                arch_specs,
                ArchLayerSpec(
                    LAYER_TERNARY_DENSE,
                    din_u16,
                    dout_u16,
                    stages,
                    UInt32(raw_lat),
                    UInt32(raw_cap),
                ),
            )
            dense_count += 1

        elseif layer isa Lux.RMSNorm
            if profile !== :extended
                throw(
                    UnsupportedLayerError(
                        path,
                        TAG_FP32_AUX,
                        cur_aux_bytes,
                        profile,
                        "profile :extended for Lux.RMSNorm (emits tag 0x0007 fp32_aux)",
                        "profile :$profile",
                    ),
                )
            end
            nshape = layer.normalized_shape
            if !(nshape isa Tuple) || length(nshape) != 1
                throw(
                    UnsupportedLayerError(
                        path,
                        TAG_ARCH_DESCRIPTOR,
                        arch_off + 2,
                        profile,
                        "1-D RMSNorm normalized_shape = (dim,)",
                        repr(nshape),
                    ),
                )
            end
            dim_u16 = _validate_dim_u16(nshape[1], "normalized_shape[1]", path, arch_off + 2, profile)
            dim = Int(dim_u16)
            if !Bool(layer.affine)
                throw(
                    UnsupportedLayerError(
                        path,
                        TAG_FP32_AUX,
                        cur_aux_bytes,
                        profile,
                        "Lux.RMSNorm with affine=true for 0x0007 rmsnorm_weight",
                        "affine=false",
                    ),
                )
            end
            scale_vec = _lookup(entry.ps, :scale, nothing)
            if !(scale_vec isa AbstractVector) || length(scale_vec) != dim
                actual_desc =
                    scale_vec isa AbstractArray ? "array of size $(size(scale_vec))" :
                    repr(typeof(scale_vec))
                throw(
                    UnsupportedLayerError(
                        path,
                        TAG_FP32_AUX,
                        cur_aux_bytes + 8,
                        profile,
                        "RMSNorm scale vector of length $dim",
                        actual_desc,
                    ),
                )
            end

            words = Vector{UInt32}(undef, dim)
            for k in 1:dim
                words[k] = _f32_checked_word(
                    scale_vec[k],
                    path,
                    cur_aux_bytes + 8 + 4 * (k - 1),
                    "rmsnorm_weight",
                    profile,
                )
            end

            push!(
                arch_specs,
                ArchLayerSpec(
                    LAYER_RMSNORM_SHIFT,
                    dim_u16,
                    dim_u16,
                    UInt8(0),
                    UInt32(1),
                    cap_u32,
                ),
            )
            push!(
                aux_specs,
                Fp32AuxSpec(UInt16(layer_idx0), AUX_ROLE_RMSNORM, UInt32(dim), words),
            )
            cur_aux_bytes += 8 + 4 * dim

        elseif layer isa Lux.Embedding
            if profile !== :extended
                throw(
                    UnsupportedLayerError(
                        path,
                        TAG_FP32_AUX,
                        cur_aux_bytes,
                        profile,
                        "profile :extended for Lux.Embedding (emits tag 0x0007 fp32_aux)",
                        "profile :$profile",
                    ),
                )
            end
            din_u16 = _validate_dim_u16(layer.in_dims, "in_dims", path, arch_off + 2, profile)
            dout_u16 =
                _validate_dim_u16(layer.out_dims, "out_dims", path, arch_off + 4, profile)
            din = Int(din_u16)
            dout = Int(dout_u16)

            emb_w = _lookup(entry.ps, :weight, nothing)
            if !(emb_w isa AbstractMatrix) || size(emb_w) != (dout, din)
                actual_desc =
                    emb_w isa AbstractArray ? "array of size $(size(emb_w))" :
                    repr(typeof(emb_w))
                throw(
                    UnsupportedLayerError(
                        path,
                        TAG_FP32_AUX,
                        cur_aux_bytes + 8,
                        profile,
                        "Embedding weight matrix of size ($dout, $din)",
                        actual_desc,
                    ),
                )
            end

            elem_count = din * dout
            words = Vector{UInt32}(undef, elem_count)
            idx = 1
            for tok in 1:din
                for d in 1:dout
                    words[idx] = _f32_checked_word(
                        emb_w[d, tok],
                        path,
                        cur_aux_bytes + 8 + 4 * (idx - 1),
                        "embedding_table",
                        profile,
                    )
                    idx += 1
                end
            end

            push!(
                arch_specs,
                ArchLayerSpec(
                    LAYER_HOST_EMBEDDING,
                    din_u16,
                    dout_u16,
                    UInt8(0),
                    UInt32(0),
                    cap_u32,
                ),
            )
            push!(
                aux_specs,
                Fp32AuxSpec(
                    UInt16(layer_idx0),
                    AUX_ROLE_EMBEDDING,
                    UInt32(elem_count),
                    words,
                ),
            )
            cur_aux_bytes += 8 + 4 * elem_count

        else
            throw(
                UnsupportedLayerError(
                    path,
                    TAG_ARCH_DESCRIPTOR,
                    arch_off,
                    profile,
                    "allowlisted quantized ternary Lux.Dense (or Lux.RMSNorm / Lux.Embedding in :extended profile)",
                    string(typeof(layer)),
                ),
            )
        end
    end

    if dense_count == 0
        throw(
            UnsupportedLayerError(
                "model",
                TAG_WEIGHTS,
                0,
                profile,
                "at least 1 quantized ternary Lux.Dense layer (tags 0x0001 and 0x0002 mandatory in all profiles)",
                "0 ternary_dense layers",
            ),
        )
    end

    if profile === :extended && isempty(aux_specs)
        throw(
            UnsupportedLayerError(
                "model",
                TAG_FP32_AUX,
                0,
                profile,
                "at least 1 host FP32 auxiliary layer (Lux.RMSNorm or Lux.Embedding) in :extended profile (tag 0x0007 mandatory)",
                "0 fp32_aux tensors",
            ),
        )
    end

    sections = Tuple{UInt16,Vector{UInt8}}[
        (TAG_WEIGHTS, take!(weights_io)),
        (TAG_SCALES, take!(scales_io)),
        (TAG_ARCH_DESCRIPTOR, _serialize_arch_payload(arch_specs)),
    ]

    flags = profile === :core ? UInt8(0x00) : FLAG_NON_CORE

    if manifest !== nothing
        push!(sections, (TAG_MANIFEST, _build_manifest_payload(manifest, profile)))
        flags |= FLAG_MANIFEST
    end

    if link_table !== nothing
        push!(sections, (TAG_LINK_TABLE, _build_link_table_payload(link_table, profile)))
        flags |= FLAG_LINKS
    end

    if emit_group_scales
        push!(sections, (TAG_GROUP_SCALES_U16, _serialize_group_scales_payload(group_specs)))
    end

    if !isempty(aux_specs)
        push!(sections, (TAG_FP32_AUX, _serialize_fp32_aux_payload(aux_specs)))
    end

    return _frame_ttape(sections, flags)
end

"""
    export_tape(io_or_path, model::LuxCore.AbstractLuxLayer, ps, st; profile::Symbol = :grouped, kwargs...) -> Int

Walk a `LuxCore.AbstractLuxLayer` model (`Lux.Dense`, `Lux.RMSNorm`, `Lux.Embedding`,
or nested `Lux.Chain` containers), validate all wire & quantization invariants fail-closed,
and write a deterministic `.ttape` v1 binary artifact to `io_or_path` (`IO` or file path).

Returns the number of bytes written. Rejects unsupported layers, profile mismatches,
non-ternary weights, non-finite FP32 auxiliary values, or invalid scale metadata with a
Goose-style `UnsupportedLayerError`.
"""
function export_tape(
    io::IO,
    model::LuxCore.AbstractLuxLayer,
    ps,
    st;
    profile::Symbol = :grouped,
    kwargs...,
)::Int
    blob = _compile_tape_bytes(model, ps, st; profile = profile, kwargs...)
    return write(io, blob)
end

function export_tape(
    path::AbstractString,
    model::LuxCore.AbstractLuxLayer,
    ps,
    st;
    profile::Symbol = :grouped,
    kwargs...,
)::Int
    blob = _compile_tape_bytes(model, ps, st; profile = profile, kwargs...)
    mkpath(dirname(abspath(path)))
    return open(path, "w") do io
        write(io, blob)
    end
end

end # module TernaryTape
