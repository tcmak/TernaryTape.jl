# TernaryTape.jl

**Canonical `TTAPE` (`.ttape`) binary container specification, zero-dependency golden fixture corpus, and offline `Lux.jl` v1 model exporter.**

`TernaryTape.jl` is the single source of truth for the wire contract connecting trained `Lux.jl` ternary models to downstream hardware and freestanding runtimes (`setun` and `veriObjC`).

---

## 1. Repository Layout & Ownership Boundaries

`spec/` and `fixtures/` sit at the repository root and have **zero Julia or PyPI dependencies**, allowing C, C++, and Python consumers (`setun`, `veriObjC`) to vendor or submodule this repository without installing Julia.

```text
TernaryTape.jl/
├── spec/
│   └── ttape-v1-draft.md         # Canonical TTAPE v1.0 (Draft) binary specification
├── fixtures/
│   ├── generate_golden.py        # Python 3 stdlib-only deterministic fixture generator & validator
│   ├── accept/                   # Golden valid .ttape + .json sidecars (core, grouped, extended)
│   └── reject/                   # Golden negative .ttape + .json sidecars for every §4 diagnostic
├── src/                          # (Phase C1.2) Julia 1.10+ / Lux v1 exporter (`export_tape`)
└── test/                         # (Phase C1.2) Julia unit & cross-repo M0 conformance tests
```

| Repository | Owns (Single Source of Truth) | Strictly Forbidden From Owning |
| :--- | :--- | :--- |
| **`TernaryTape.jl`** | `TTAPE` binary format spec (`spec/`), stdlib Python golden fixture generator (`fixtures/`), Julia/Lux v1 exporter (`src/`) | Any C/Python/FPGA runtime, hardware register maps, or co-sim interpreters |
| **`setun`** | Trit encoding, rounding rules, Migen RTL, LiteX SoC CSRs, synthesis reports, Python reference oracle (`setun/ref/`) | Wire format specification prose, host language compiler/runtime stacks |
| **`veriObjC`** | Objective-C subset Clang plugin (`compiler/`), freestanding runtime (`rt/`), `vobj_v1` `RelPtr` container, `ttape` reader (`vstd/`) | Hardware RTL models, hand-written Setun CSR structs, `TTAPE` spec definitions, co-sim oracles |

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

## 3. Verifying & Regenerating Golden Fixtures

`fixtures/generate_golden.py` uses **only the Python 3 standard library** (`argparse`, `json`, `math`, `pathlib`, `struct`, `sys`):

```bash
# Regenerate all accept/*.{ttape,json} and reject/*.{ttape,json} fixtures
python3 fixtures/generate_golden.py

# Verify on-disk fixtures are byte-identical and pass all validation + per-byte mutation checks
python3 fixtures/generate_golden.py --check
```

---

## 4. Planned Julia / `Lux.jl` v1 Exporter API (Phase C1.2)

`TernaryTape.jl` targets **`LuxCore.AbstractLuxLayer`** (Julia `1.10+`, `Lux = "1"`) and exposes `export_tape` (never the reserved keyword `export`):

```julia
using Lux, LuxCore, TernaryTape

# Walk a LuxCore.AbstractLuxLayer model and emit a verified .ttape artifact
export_tape("model_grouped.ttape", model, ps, st; profile = :grouped)
export_tape("model_extended.ttape", model, ps, st; profile = :extended)
```

### Supported Layer Allowlist (`setun/ref/m0.py` Parity)
* Quantized ternary `Lux.Dense` → emits 2-bit packed trits (`0x0001`), per-layer shift exponent (`0x0002`), and per-group `u16` scale mantissas (`0x0006` in `:grouped` / `:extended` profiles).
* `Lux.RMSNorm` and embedding tables → emits host-consumed `fp32_aux` (`0x0007` in `:extended` profile).
* Any unsupported `AbstractLuxLayer` is rejected fail-closed with a typed `UnsupportedLayerError`.
