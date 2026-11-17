# 🎯 TernaryTape.jl

**Offline export from trained Lux.jl models to VeriObjC ternary weight tapes.**

Julia stays in the training pipeline. VeriObjC runs on the hardware. The only thing that crosses between them is a **verified byte artifact**.

> [!WARNING]
> **Status: design stage — no code yet.** This document is the project brief, written to be self-contained so it can seed a fresh conversation without prior context.

---

## 🧭 Background & Context (self-contained)

**VeriObjC** is a hardened, verifiable dialect of Objective-C for deterministic low-latency systems (bare metal, FPGAs, HFT engines). A Clang 23 AST plugin bans dynamic dispatch, `id`, reflection, and exceptions with named diagnostics; a freestanding C micro-runtime replaces `libobjc` with compiler-assigned data stacks and `vpool` freelists. Its **veriStream** substrate runs matmul-free ternary inference (weights ∈ {-1, 0, +1}) on FPGAs with **zero DSP blocks**, enforced as a CI gate: CMUX ladders and popcount adder trees, never multipliers; weights packed as 2-bit codes, 4 per byte, pinned in BRAM.

**The interface decision that defines this project:** the boundary between the training tier and the inference binary is an *artifact, not an ABI*. Weight provenance is **training-source-agnostic** — any hosted `@cold` producer (Julia/Lux, PyTorch/BitNet-style, or a C++ tool) exports a content-addressed tape; `veriobjc_from_bytes` verifies framing, tags, checksums, and links *before* the image is hardware-visible. Julia is the provider of weights, never a runtime dependency.

**Why Lux.jl specifically:** Lux is pure-functional with explicit parameterization (`model`, `ps`, `st` are separate values) — the model is *data* you can walk, quantize, and emit, with no hidden state. That fits "model as artifact" exactly.

**Dependency direction (one-way):** TernaryTape.jl encodes against the **VeriObjC Ternary Tape Format Specification v1**. VeriObjC never imports, links, or requires Julia. See *Relationship to VeriObjC*.

---

## 🔄 What It Does

```
Lux model ──train──> Julia (normal Lux workflow, any device — never ships)
   │
   ▼  TernaryTape.jl
   ├─ Walk the layer tree (LuxCore traits: AbstractExplicitLayer, initialparameters/initialstates)
   ├─ Quantize → {-1, 0, +1}; scale factors rounded to POWERS OF TWO
   ├─ Encode → tape format v1: Bits(2) codes, 4/byte, FNV-1a framed, self-relative links
   └─ Emit tape + manifest (producer_id, training run ID, quant params, content hash)
   │
   ▼  content-addressed artifact
   │
   ▼  VeriObjC build
   ├─ veriobjc_from_bytes verifies BEFORE anything is hardware-visible
   └─ DSP=0 gate: kernels elaborate multipliers-free or the build fails
```

| Component | Role |
| :--- | :--- |
| **Layer walker** | Reads a trained Lux model via LuxCore traits; allowlist-only (see below); unsupported layers get a **clean rejection naming the layer and the nearest supported block** |
| **Quantizer** | BitNet b1.58-style per-tensor absmean → ternary codes; **scale factors forced to powers of two** so hardware normalization stays shifts-only (this rule is what keeps the DSP=0 gate intact) |
| **Tape encoder** | Conformant producer of Tape Format Spec v1; deterministic (same input → byte-identical output, required for content addressing) |
| **Reference interpreter** | Julia golden model mirroring veriStream semantics exactly (CMUX/popcount, shift norms) — the oracle for the VeriObjC project's C/RTL cosimulation equivalence gate |

---

## 🧪 Planned API

```julia
using Lux, TernaryTape

model, ps, st = train_my_model()                  # normal Lux training, unchanged
tape = TernaryTape.export(model, ps;
                          algorithm = :absmean_b158,
                          scales   = :power_of_two)
TernaryTape.verify(tape)                          # decode + spot-check vs reference interpreter
TernaryTape.write("ternary_glu.tape", tape)       # content-addressed artifact + manifest
```

---

## ✅ Supported Layer Set (v1 Scope — Ternary Allowlist)

| Lux layer kind | veriStream block |
| :--- | :--- |
| Dense (ternary-quantizable) | `ternary_dot` — CMUX + popcount, `log2(N)` reduce stages |
| Gated units (GLU-style) | `glu_mux` gates |
| Norm layers (RMSNorm family) | `rmsnorm_shift` — power-of-two scales, shifts only |
| Token shifting / delays | `token_shift` — register-delayed stream |
| KV storage | `kv_arena` — pre-banked BRAM against declared `ContextCap` |

Everything else — convolutions, float attention, anything with an unquantizable op — is **rejected with a diagnostic**, not approximated. Rejection is a feature here, exactly as it is in VeriObjC itself. The final allowlist must be reconciled against Lux's current layer set.

---

## 🚫 What This Project Is Not

- **Not a Lux `TrainingBackend`** — no training-loop integration; it reads a *trained* model. Think ONNX-export, not backend.
- **Not embedded** — no PackageCompiler, no `ccall` into Julia from any VeriObjC binary. Julia is build-pipeline-only by design.
- **Not general-purpose** — no conv nets, no float models, no "mostly ternary". Binary in, binary out, allowlist enforced.

---

## 🛡️ Verification Discipline

- **Golden model:** the reference interpreter is the equivalence oracle for VeriObjC's Task 5.4 (C/RTL cosimulation). "Zero DSPs" means nothing without something to be equivalent *to*.
- **Dual-direction corpus:** exported tapes must load clean via `veriobjc_from_bytes`, and every mutated byte must be rejected — mirroring VeriObjC's ban-corpus gate philosophy.
- **CI without Julia:** tapes enter VeriObjC CI as **committed fixtures**; the VeriObjC side runs its C loader against them. VeriObjC's CI never installs Julia.
- **No number, no claim:** quantization accuracy deltas (float vs ternary) are measured per model and published alongside losses, not promised.

---

## 🔗 Relationship to VeriObjC

| | |
| :--- | :--- |
| Repos | separate; this package depends on the Tape Format Spec v1 document, not on the VeriObjC toolchain |
| VeriObjC → Julia dependency | **none, ever** (producer-agnostic provenance is a VeriObjC design invariant) |
| Julia → VeriObjC dependency | spec conformance + committed golden tapes |
| VeriObjC plan hooks | Task 5.4 (cosim oracle), Task 5.6 (format spec + golden vector corpus), Task 6.1 (`vpm` content-addressed store hosts tapes) |

---

## 🗺️ Roadmap

1. **Format encoder + golden vectors** — gated on the VeriObjC Tape Format Spec being reconciled against `rt/` (spec is a proposal until then).
2. **Quantizer** — absmean b1.58 with power-of-two scales; refuse exports it cannot honor.
3. **Reference interpreter** — the cosim oracle.
4. **Layer walker** — LuxCore trait walk with the allowlist and clean rejections.
5. **Corpus generation** — deterministic fixtures committed into VeriObjC CI.