# TernaryTape.jl

**Canonical `TTAPE` (`.ttape`) binary container specification, zero-dependency golden fixture corpus, standalone ISO C99 wire parser (`c/ttape_reader.h`), and offline `Lux.jl` v1 model exporter.**

`TernaryTape.jl` has **zero external repository dependencies** and serves as the single upstream wire-format source of truth consumed via one-way HTTPS submodule by `setun`.

---

## 1. Repository Layout & Strict One-Way DAG

`spec/`, `fixtures/`, and `c/` sit at the repository root and have **zero Julia or PyPI dependencies**, allowing C and Python consumers (`setun`) to submodule this repository without installing Julia.

```text
TernaryTape.jl/
├── spec/
│   └── ttape-v1-draft.md         # Canonical TTAPE v1.0 (Draft) binary specification
├── fixtures/
│   ├── generate_golden.py        # Python 3 stdlib-only deterministic fixture generator & validator
│   ├── accept/                   # Golden valid .ttape + .json sidecars (core, grouped, extended)
│   └── reject/                   # Golden negative .ttape + .json sidecars for every §4 diagnostic
├── c/
│   ├── ttape_reader.h            # Zero-dependency ISO C99 header-only wire parser & validator
│   └── test_ttape_reader.c       # Standalone C99 corpus & exhaustive byte-mutation test runner
├── src/
│   └── TernaryTape.jl            # Julia 1.10+ / Lux v1 exporter (`export_tape`)
└── test/
    └── runtests.jl               # Cross-validated Julia, Python, and C99 test suite
```

| Repository | Upstream Repo Dependencies | Owns (Single Source of Truth) | Strictly Forbidden From Owning |
| :--- | :--- | :--- | :--- |
| **`TernaryTape.jl`** | **None** | `TTAPE` binary format spec (`spec/`), stdlib Python golden fixture generator (`fixtures/`), pure-C99 wire parser (`c/ttape_reader.h`), Julia/Lux v1 exporter (`src/`) | Any C/Python/FPGA inference runtime, hardware register maps, or co-sim interpreters |
| **`setun`** | `TernaryTape.jl` (`third_party/TernaryTape`) | Trit encoding, rounding rules, Migen RTL, LiteX SoC CSRs, synthesis reports, Python reference oracle (`setun/ref/`) | Wire format specification prose, host language compiler/runtime stacks |
| **`veriObjC`** | **None** | Objective-C subset Clang plugin (`compiler/`), freestanding runtime (`rt/`), `vobj_v1` `RelPtr` container, `vstd/` (with pure-CPU fallback `ternary.h`) | `TTAPE` code/submodules, hardware RTL models, Setun CSR structs, co-sim oracles |

---

## 2. `TTAPE` v1 Wire Overview (`spec/ttape-v1-draft.md`)

* **Magic & Extension:** 4-byte ASCII `"TTAP"` (`0x50415454` little-endian), file extension `.ttape`.
* **Framing & Integrity:**
  * 32-byte little-endian header (`struct ttape_v1_header`) protected by `header_fnv1a` (`FNV-1a-32`) and content-addressed by `artifact_id` (`FNV-1a-64` over all framed section bytes).
  * 12-byte inline section headers (`struct ttape_v1_section_header`) with per-section `payload_fnv1a` (`FNV-1a-32`).
* **Distinct Section Tags (No Overloading):**
  * `0x0001` `weights` — 2-bit LSB-first ternary trits (`0b00 = 0`, `0b01 = +1`, `0b10 = -1`, `0b11 = RESERVED/reject`).
  * `0x0002` `scales` — per-layer `s16` power-of-two shift exponents (`[-30, +30]`).
  * `0x0003` `arch_descriptor` — layer kinds, dimensions, reduction stages (`clog2(dim_in)`), and latency/context contracts.
  * `0x0004` `manifest` — optional 5-field length-prefixed provenance record.
  * `0x0005` `link_table` — optional 4-byte aligned `int32_t` self-relative offsets.
  * `0x0006` `group_scales_u16` — per-group `u16` scale mantissas, explicit `group_size G in {64, 128, 256}`, and shared `shift` for `TTAPE-grouped` (`setun`).
  * `0x0007` `fp32_aux` — host-only IEEE-754 `binary32` auxiliary tensors (embeddings, `RMSNorm`, softmax scales; `NaN`/`±Inf` rejected) for `TTAPE-extended`.

---

## 3. Verifying Golden Fixtures & Standalone C99 Parser

Both `fixtures/generate_golden.py` (Python 3 stdlib only) and `c/ttape_reader.h` (ISO C99 `<stdint.h>`, `<stddef.h>`, `<stdbool.h>` only — a wire parser/validator, **not** an inference runtime) validate the full 21-fixture corpus and exhaustive single-bit mutation sweep:

```bash
# 1. Verify on-disk Python golden fixtures are byte-identical and mutation-clean
python3 fixtures/generate_golden.py --check

# 2. Compile and run the standalone ISO C99 wire parser against fixtures/
cc -std=c99 -Wall -Wextra -Wpedantic -Werror c/test_ttape_reader.c -o /tmp/test_ttape_reader
/tmp/test_ttape_reader fixtures

# 3. Run the standalone Julia / Lux v1 exporter test suite
julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.test()'
```

---

## 4. Julia / `Lux.jl` v1 Exporter API (`TernaryTape.export_tape`)

`TernaryTape.jl` targets **`LuxCore.AbstractLuxLayer`** (Julia `1.10+`, `Lux = "1"`) and exposes `export_tape` (never the reserved keyword `export`):

```julia
using Lux, LuxCore, TernaryTape

# Walk a LuxCore.AbstractLuxLayer model and emit a verified .ttape artifact
export_tape("model_grouped.ttape", model, ps, st; profile = :grouped)
export_tape("model_extended.ttape", model, ps, st; profile = :extended)
```

### Supported Layer Allowlist
* Quantized ternary `Lux.Dense` → emits 2-bit packed trits (`0x0001`), per-layer shift exponent (`0x0002`), and per-group `u16` scale mantissas (`0x0006` in `:grouped` / `:extended` profiles).
* `Lux.RMSNorm` and embedding tables → emits host-consumed `fp32_aux` (`0x0007` in `:extended` profile).
* Any unsupported `AbstractLuxLayer` is rejected fail-closed with a typed `UnsupportedLayerError`.
