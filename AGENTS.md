# AGENTS.md — `TernaryTape.jl` Working Protocol & Standing Engineering Rules

`TernaryTape.jl` is the single home of the language-neutral `TTAPE` (`.ttape`) binary container specification, deterministic golden fixtures, standalone C99 wire parser/validator, and the Julia/Lux v1 model exporter.

## 1. Portfolio Ownership & Boundaries

* **`TernaryTape.jl` OWNS (zero external repo dependencies):**
  * `spec/` — the sole `TTAPE` wire specification.
  * `fixtures/` — stdlib-only Python golden generator + `.ttape`/`.json` corpus.
  * `c/ttape_reader.h` & `c/test_ttape_reader.c` — zero-dependency ISO C99 header-only wire parser/validator (strictly format validation and zero-copy section slicing, **not** an inference runtime).
  * `src/` — Julia `Lux` v1 exporter (`export_tape`).
* **`TernaryTape.jl` DOES NOT OWN:** Any C, Python, or FPGA inference runtime, co-simulation oracle (`setun/ref/` is the sole oracle), or hardware register map. `TernaryTape.jl` never imports, invokes, or submodules `setun` or `veriObjC`.

---

## 2. Session Workflow & Version Control

1. **Orient first:** Read `README.md` and `spec/` at the start of every session. Work on **one task at a time**, in the order specified by the cleanup/execution plan, unless told otherwise.
2. **Stop on open decisions:** Wire-format changes affect downstream consumers (`setun`). If a task requires changing header offsets, section tags, checksum algorithms, or profile rules, **stop and ask**—never mutate wire semantics silently.
3. **Atomic commits:** Commit after each satisfied acceptance criterion using the format `<task-id>: <what>`. Enforce LF line endings (`* text=auto eol=lf`).
4. **Spec versioning discipline (`FROZEN v1.0`):** `spec/ttape-v1.0.md` (`spec-v1.0`) is **FROZEN** across four verified implementations (`TernaryTape.jl`, `c/ttape_reader.h`, `fixtures/generate_golden.py`, and `setun`). Never mutate v1.0 wire layouts, tags, checksums, or error codes in place; any incompatible wire change requires a new `spec/ttape-v2.0.md` specification and tag (`spec-v*`), tagged separately from Julia package releases.

---

## 3. Standing Engineering Rules (Apply to Every Task)

1. **Red first, never weaken fixtures or tests.**
   * No spec rule, fixture generator update, C99 parser check, or Julia exporter feature lands without its failing check or `Pkg.test()` case written and observed red first.
   * **Never delete, weaken, or silently regenerate golden `.ttape` fixtures just to make a buggy exporter pass.** If a fixture conflicts with the spec, stop and confirm with the maintainer.

2. **Language-neutral `spec/`, `fixtures/`, and `c/` floor.**
   * `spec/`, `fixtures/`, and `c/` sit at the repository root so C and Python consumers never touch Julia files.
   * `fixtures/generate_golden.py` MUST use **Python 3 standard library only** (`struct`, `json`, `math`, `pathlib`, `argparse`, `sys`)—zero PyPI or Julia dependencies.
   * `c/ttape_reader.h` MUST use **standard ISO C99 headers only** (`<stdint.h>`, `<stddef.h>`, `<stdbool.h>`) with zero external dependencies.

3. **Pinned dependencies & current Lux/Julia idioms.**
   Pin explicit `[compat]` bounds in `Project.toml` (Julia `1.10+`, `Lux = "1"`). Target `LuxCore.AbstractLuxLayer` (never deprecated `AbstractExplicitLayer`), name the exporter function `export_tape` (never reserved keyword `export`), and allowlist strictly quantized ternary `Lux.Dense` and `Lux.RMSNorm` / embeddings—rejecting unsupported layers with a typed `UnsupportedLayerError`.

4. **Goose-style diagnostics.**
   Validator, generator, and exporter errors must name the layer or section tag (`0x0001`..`0x0007`), byte offset, expected profile/invariant, and actual value. Never throw a bare `ErrorException("invalid tape")`.

5. **Composition + positive/negative corpus parity.**
   * Every wire invariant in `spec/` (magic, version, unknown flags, section FNV-1a-32, FNV-1a-64 `artifact_id`, reserved trit `0b11`, non-zero padding, profile section restrictions, non-finite `fp32_aux`, `group_size` allowlist) must have both an `accept/*.ttape` fixture and a targeted `reject/*.ttape` fixture verified by both `fixtures/generate_golden.py --check` and `c/test_ttape_reader.c`.

---

## 4. Mandatory Pre-Completion Gate

A task is done **only** when every applicable check below exits `0`:
1. `python3 fixtures/generate_golden.py --check`
2. `cc -std=c99 -Wall -Wextra -Wpedantic -Werror c/test_ttape_reader.c -o /tmp/test_ttape_reader && /tmp/test_ttape_reader fixtures`
3. `julia --project=. -e 'using Pkg; Pkg.test()'` (once `Project.toml` and `src/` are initialized in Phase C1.2)
