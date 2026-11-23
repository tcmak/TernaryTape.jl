# TernaryTape.jl

**Canonical `TTAPE` (`.ttape`) binary container specification, zero-dependency golden fixture corpus, standalone ISO C99 wire parser (`c/ttape_reader.h`), and offline `Lux.jl` v1 model exporter.**

`TernaryTape.jl` has **zero external repository dependencies** and serves as the single upstream wire-format source of truth consumed via one-way HTTPS submodule by `setun`.

## 1. What Problem `TernaryTape.jl` Solves

Deploying $1.58$-bit ternary models ($\{-1, 0, +1\}$) to FPGAs, microcontrollers, or deterministic C/C++ runtimes usually encounters three core friction points:

* **Training-to-Hardware Coupling:** Formats like `GGUF` or `safetensors` are designed for host-side Python/C++ runtimes with dynamic metadata, floating-point dequantization, and heap allocation—forcing every FPGA or bare-metal project to write ad-hoc, unverified binary dumpers.

* **Silent Corruption & Undefined Codes:** Packing 4 trits per byte ($2\text{ bits/trit}$) leaves `0b11` as an illegal state, and raw memory dumps lack framing or checksums—meaning a single flipped bit, truncated transfer, or mismatched profile silently corrupts inference on hardware.

* **Cross-Language Drift:** Producers (Julia/Python) and consumers (C/RTL) frequently disagree on byte padding, group-scale alignment, or profile boundaries (`Core` vs. `Grouped` vs. `Extended` host-only `fp32_aux`).

**`TernaryTape.jl`** solves this with **TTAPE v1.0** ([`spec/ttape-v1.0.md`](spec/ttape-v1.0.md))—a frozen, little-endian, pointer-free, content-addressed binary container (`FNV-1a`) backed by:

1. A **single-header, freestanding C99 reader** (`c/ttape_reader.h`) with zero heap allocation and strict fail-closed validation.

2. A **Julia producer & reference package** (`src/TernaryTape.jl`) supporting `Lux.Chain` grouped exports.

3. A **21-fixture golden conformance corpus** (`fixtures/accept/` + `fixtures/reject/`) that every encoder and hardware decoder can test against.

## 2. Setup Requirements

Install **only** the toolchain required for your specific role:

| Role | Target Artifact / Directory | Toolchain Requirements | Dependencies | 
| ----- | ----- | ----- | ----- | 
| **C / Embedded / FPGA Consumer** | `c/ttape_reader.h` | Any **ISO C99** compiler (`gcc`, `clang`, MSVC) | Zero external libraries, zero `malloc` (`<stdint.h>`, `<stddef.h>`, `<stdbool.h>` only) | 
| ** Downstream `veriObjC` Firmware Consumer** | `c/ttape_reader.h` | `clang-23` with `libVeriObjCPlugin.so` | Zero Apple runtime / zero heap dependencies | 
| **Julia Producer / ML Engineer** | `src/TernaryTape.jl` | **Julia 1.10+** | Standard `Pkg` resolution (targets `LuxCore` / `Lux` v1) | 
| **Spec / Fixture Maintainer** | `fixtures/generate_golden.py` | **Python 3.10+** | Python standard library only (no `pip` packages required) | 

## 3. Repository Layout & Strict One-Way DAG

`spec/`, `fixtures/`, and `c/` sit at the repository root and have **zero Julia or PyPI dependencies**, allowing C and Python consumers (`setun`) to submodule this repository without installing Julia.

```
TernaryTape.jl/
├── spec/
│   └── ttape-v1.0.md             # Canonical TTAPE v1.0 (FROZEN v1.0) binary specification
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

### Upstream Dependency DAG

| Repository | Upstream Repo Dependencies | Owns (Single Source of Truth) | Strictly Forbidden From Owning | 
| ----- | ----- | ----- | ----- | 
| **`TernaryTape.jl`** | **None** | `TTAPE` binary format spec (`spec/`), stdlib Python golden fixture generator (`fixtures/`), pure-C99 wire parser (`c/ttape_reader.h`), Julia/Lux v1 exporter (`src/`) | Any C/Python/FPGA inference runtime, hardware register maps, or co-sim interpreters | 
| **`setun`** | `TernaryTape.jl` (`third_party/TernaryTape`) | Trit encoding, rounding rules, Migen RTL, LiteX SoC CSRs, synthesis reports, Python reference oracle (`setun/ref/`) | Wire format specification prose, host language compiler/runtime stacks | 
| **`veriObjC`** | **None** | Objective-C subset Clang plugin (`compiler/`), freestanding runtime (`rt/`), `vobj_v1` `RelPtr` container, `vstd/` (with pure-CPU fallback `ternary.h`) | `TTAPE` code/submodules, hardware RTL models, Setun CSR structs, co-sim oracles | 

## 4. `TTAPE` v1 Wire Overview ([`spec/ttape-v1.0.md`](spec/ttape-v1.0.md))

* **Magic & Extension:** 4-byte ASCII `"TTAP"` (`0x50415454` little-endian), file extension `.ttape`.

* **Framing & Integrity:**

  * 32-byte little-endian header (`struct ttape_v1_header`) protected by `header_fnv1a` (`FNV-1a-32`) and content-addressed by `artifact_id` (`FNV-1a-64` over all framed section bytes).

  * 12-byte inline section headers (`struct ttape_v1_section_header`) with per-section `payload_fnv1a` (`FNV-1a-32`).

* **Distinct Section Tags (No Overloading):**

  * `0x0001` `weights` — 2-bit LSB-first ternary trits (`0b00 = 0`, `0b01 = +1`, `0b10 = -1`, `0b11 = RESERVED/reject`).

  * `0x0002` `scales` — per-layer `s16` power-of-two shift exponents ($[-30, +30]$).

  * `0x0003` `arch_descriptor` — layer kinds, dimensions, reduction stages ($\lceil\log_2(\text{dim\_in})\rceil$), and latency/context contracts.

  * `0x0004` `manifest` — optional 5-field length-prefixed provenance record.

  * `0x0005` `link_table` — optional 4-byte aligned `int32_t` self-relative offsets.

  * `0x0006` `group_scales_u16` — per-group `u16` scale mantissas, explicit group size $G \in \{64, 128, 256\}$, and shared shift for `TTAPE-grouped` (`setun`).

  * `0x0007` `fp32_aux` — host-only IEEE-754 `binary32` auxiliary tensors (embeddings, `RMSNorm`, softmax scales; `NaN`/`±Inf` rejected) for `TTAPE-extended`.

## 5. How to Use & Verify

### A. In C / C++ / Bare-Metal Firmware (Reading a `.ttape`)

Include `c/ttape_reader.h` directly in your project:

```c
#include "ttape_reader.h"

struct ttape_reader reader;
ttape_v1_error_t st = ttape_reader_open(&reader, buffer, buffer_size, TTAPE_PROFILE_GROUPED);
if (st != TTAPE_OK) {
    // Handle fail-closed validation error (e.g., TTAPE_ERR_BAD_MAGIC)
}
```

To run the standalone C99 wire parser test suite against all 21 golden fixtures:

```bash
cc -std=c99 -Wall -Wextra -Wpedantic -Werror c/test_ttape_reader.c -o /tmp/test_ttape_reader
/tmp/test_ttape_reader fixtures
```

### B. In `veriObjC` (Freestanding Stack-Bounded Firmware & Drivers)

`c/ttape_reader.h` is fully compatible with `veriObjC` bare-metal firmware and drivers. Because `ttape_reader.h` performs zero heap allocations and uses standard ISO C types, `veriObjC` programs can include it directly using standard C syntax inside static-staging routines or `@nolatency` / `@pure` class methods.

```objc
#include <stdio.h>
#include <veriobjc_runtime.h>
#include <vstd/span.h>
#include <ttape_reader.h>

static uint8_t g_io_staging[65536];
static uint8_t g_arena_backing[65536];

veriobjc_cold
static int load_and_validate_tape(const char *path, ttape_v1_profile_t profile) {
    FILE *f = fopen(path, "rb");
    if (!f) return -100;
    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return -101; }
    long sz = ftell(f);
    if (fseek(f, 0, SEEK_SET) != 0) { fclose(f); return -101; }
    if (sz <= 0 || (size_t)sz > sizeof(g_io_staging)) { fclose(f); return -102; }

    /* Stage OS I/O into static memory; keep scoped arena refs inside VeriObjC */
    size_t rd = fread(g_io_staging, 1, (size_t)sz, f);
    fclose(f);
    if (rd != (size_t)sz) return -104;

    struct veriobjc_stack arena;
    veriobjc_stack_init(&arena, g_arena_backing, sizeof(g_arena_backing));

    int result = -105;
    {
        VERIOBJC_SCOPED_STACK(&arena);
        struct ttape_v1_reader *reader = (struct ttape_v1_reader *)veriobjc_stack_alloc(
            &arena, sizeof(struct ttape_v1_reader));
        if (!reader) return -106;

        result = (int)ttape_v1_reader_init(reader, g_io_staging, (size_t)sz, profile);
    }
    return result;
}
```

#### Compilation & Verification under `veriObjC`

Compile with `clang-23` using the `veriObjC` plugin and runtime headers:

```bash
PLUGIN_SO=$(find ~/veriObjC/build -name "libVeriObjCPlugin.so" | head -n 1)

clang-23 -std=gnu23 -O2 -Werror -fblocks \
  -Xclang -load -Xclang "$PLUGIN_SO" \
  -Xclang -add-plugin -Xclang veriobjc-check \
  -fpass-plugin="$PLUGIN_SO" \
  -I ~/veriObjC/rt/include -I ~/veriObjC/vstd/include \
  -isystem c \
  firmware_loader.m -o firmware_loader
```

Symbol auditing confirms zero dynamic ObjC dispatch (`objc_msgSend`) and zero heap allocations (`malloc`/`free`):

```bash
nm firmware_loader | grep -qE "objc_msgSend|objc_msg_lookup|__objc_exec_class|_OBJC_CLASS_|[[:space:]]U[[:space:]]+malloc|[[:space:]]U[[:space:]]+free" \
  && echo "AUDIT FAILED" || echo "AUDIT PASSED: Zero dynamic dispatch and zero heap symbols!"
```

### C. In Julia (Exporting Models)

Add the package to your Julia project:

```julia
using Pkg
Pkg.add(url="https://github.com/tcmak/TernaryTape.jl")
```

Run the complete Julia test suite (including `Lux.Chain` grouped MLP export and fixture parity):

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.test()'
```

### D. In Python / Fixture Maintenance

Verify that all on-disk golden `.ttape` files match their JSON sidecars and are mutation-clean:

```bash
python3 fixtures/generate_golden.py --check
```

## 6. Julia / `Lux.jl` v1 Exporter API (`TernaryTape.export_tape`)

`TernaryTape.jl` targets **`LuxCore.AbstractLuxLayer`** (Julia `1.10+`, `Lux = "1"`) and exposes `export_tape` (never the reserved keyword `export`):

```julia
using Lux, LuxCore, TernaryTape

# Walk a LuxCore.AbstractLuxLayer model and emit a verified .ttape artifact
export_tape("model_grouped.ttape", model, ps, st; profile = :grouped)
export_tape("model_extended.ttape", model, ps, st; profile = :extended)
```

### Supported Layer Allowlist

* Quantized ternary `Lux.Dense` $\rightarrow$ emits 2-bit packed trits (`0x0001`), per-layer shift exponent (`0x0002`), and per-group `u16` scale mantissas (`0x0006` in `:grouped` / `:extended` profiles).

* `Lux.RMSNorm` and embedding tables $\rightarrow$ emits host-consumed `fp32_aux` (`0x0007` in `:extended` profile).

* Any unsupported `AbstractLuxLayer` is rejected fail-closed with a typed `UnsupportedLayerError`.

