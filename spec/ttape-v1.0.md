# TernaryTape Binary Container Specification (`TTAPE` v1.0)

* **Status:** `FROZEN v1.0` (`spec-v1.0`) — Verified across independent producer and consumer implementations (`TernaryTape.jl` Lux v1 exporter, standalone ISO C99 `c/ttape_reader.h` wire validator, Python 3 stdlib reference validator, and `setun` Python loader & `M0` inference oracle) against the full 21-fixture golden corpus.
* **Canonical Owner:** `TernaryTape.jl` (`spec/`, `fixtures/`, and the standalone C99 wire parser `c/ttape_reader.h`). Downstream consumers (`setun`) submodule or pin this repository and MUST NOT duplicate wire specification prose.
* **File Extension:** `.ttape`
* **Endianness:** Little-endian (`<`) for all multi-byte integers and IEEE-754 floating-point values. Position-independent (zero raw pointers).

---

## 1. Scope, Boundaries & Conformance Profiles

`TTAPE` (`.ttape`) is a language- and hardware-neutral binary container for quantized ternary (`w in {-1, 0, +1}`) neural network weights, scaling metadata, architecture descriptors, provenance manifests, and optional host-consumed auxiliary FP32 tensors.

> [!NOTE]
> **Separation from `veriObjC` `VOBJ` v1:** The 24-byte `VOBJ` (`0x564F424A`) single-table `RelPtr` snapshot container (`veriobjc_to_bytes` / `veriobjc_from_bytes`) is an internal freestanding data-structure serialization format owned by `veriObjC`. `TTAPE` (`0x50415454`, `"TTAP"`) is the cross-portfolio multi-section ternary model tape format specified exclusively in this document.

### 1.1 Three Conformance Profiles

All three profiles share the exact same 32-byte file header, 12-byte inline section header framing, FNV-1a checksums, 2-bit LSB-first ternary packing, and precedence-ordered rejection contract:

| Profile | Header `flags` Bit 2 (`0x04`) | Permitted Section Tags | Target Use Case & Hardware Contract |
| :--- | :--- | :--- | :--- |
| **`TTAPE-core`** | `0` | `0x0001`..`0x0005` (`0x0006` and `0x0007` **forbidden**; `layer_kind in 1..5` only) | Strict zero-multiplier (`DSP = 0`) datapaths. Every `ternary_dense` layer uses a single per-layer power-of-two shift exponent (`0x0002` `scales`). |
| **`TTAPE-grouped`** | `1` | `0x0001`..`0x0006` (`0x0006` `group_scales_u16` **mandatory**; `0x0007` `fp32_aux` and `layer_kind in {6, 7}` **forbidden**) | Grouped ternary matmul engines (`setun` `ADR-003`/`ADR-004`/`ADR-011`) with parametric group size `G in {64, 128, 256}`, shared right-shift `shift`, and per-group `u16` scale mantissas. |
| **`TTAPE-extended`** | `1` | `0x0001`..`0x0007` (`0x0007` `fp32_aux` **mandatory**; `0x0006` `group_scales_u16` **optional**) | Hybrid inference models (e.g., `setun/ref/m0.py` and `Lux.Chain` exports) carrying host-only FP32 embeddings, `RMSNorm` weights, or softmax head scales in `0x0007` alongside ternary dense projections. |

### 1.2 Reader Conformance Classes (`reader_profile`)

A conforming decoder declares the maximum profile it supports (`core`, `grouped`, or `extended`):
1. **`core` reader (`TTAPE_PROFILE_CORE = 0`)**: Accepts valid `TTAPE-core` tapes (`(flags & 0x04) == 0`). Rejects any tape with `(flags & 0x04) != 0` with `TTAPE_ERR_UNSUPPORTED_PROFILE`.
2. **`grouped` reader (`TTAPE_PROFILE_GROUPED = 1`)**: Accepts valid `TTAPE-core` and `TTAPE-grouped` tapes. Rejects any tape containing tag `0x0007` (`fp32_aux`) or `layer_kind in {6, 7}` with `TTAPE_ERR_UNSUPPORTED_PROFILE`.
3. **`extended` reader (`TTAPE_PROFILE_EXTENDED = 2`)**: Accepts valid `TTAPE-core`, `TTAPE-grouped`, and `TTAPE-extended` tapes.

---

## 2. File Framing & Header Layout

A `.ttape` file consists of a **32-byte File Header** followed immediately by `section_count` contiguous **Framed Sections** (each a **12-byte Section Header** plus `length` payload bytes). Total buffer size MUST equal `32 + total_length`.

### 2.1 File Header (`struct ttape_v1_header`, 32 Bytes, Little-Endian `<4sBBBBIIQ8s`)

| Offset | Size | C / Python Type | Field | Normative Rule |
| :--- | :--- | :--- | :--- | :--- |
| `0x00` | `4` | `uint32_t` / `4s` | `magic` | ASCII `"TTAP"` (`0x54 0x54 0x41 0x50`, little-endian `u32` `0x50415454`). Mismatch → `TTAPE_ERR_BAD_MAGIC`. |
| `0x04` | `1` | `uint8_t` / `B` | `version_major` | Must equal `1`. Any other value → `TTAPE_ERR_UNSUPPORTED_VERSION`. |
| `0x05` | `1` | `uint8_t` / `B` | `version_minor` | `0` in v1.0. Readers accept `version_major == 1`. |
| `0x06` | `1` | `uint8_t` / `B` | `flags` | Bit 0 (`0x01`): `links_present` (`0x0005` present iff `1`).<br>Bit 1 (`0x02`): `manifest_present` (`0x0004` present iff `1`).<br>Bit 2 (`0x04`): `non_core_profile` (`0` = `TTAPE-core`; `1` = `TTAPE-grouped` or `TTAPE-extended`).<br>Bits 3–7 (`0xF8`): Reserved, MUST be `0` → else `TTAPE_ERR_UNKNOWN_FLAGS`. |
| `0x07` | `1` | `uint8_t` / `B` | `section_count` | Number of framed sections (`>= 3`). If `< 3` → `TTAPE_ERR_SECTION_COUNT_MISMATCH`. |
| `0x08` | `4` | `uint32_t` / `I` | `total_length` | Exact byte length of all framed sections after the 32-byte header. `32 + total_length` MUST equal total file byte length → else `TTAPE_ERR_LENGTH_MISMATCH`. |
| `0x0C` | `4` | `uint32_t` / `I` | `header_fnv1a` | 32-bit FNV-1a over the **28 non-checksum header bytes** (`0x00..0x0B` followed by `0x10..0x1F`) → mismatch rejects with `TTAPE_ERR_CHECKSUM_MISMATCH`. |
| `0x10` | `8` | `uint64_t` / `Q` | `artifact_id` | 64-bit FNV-1a over all `total_length` framed section bytes (`0x20 .. 0x20 + total_length - 1`) → mismatch rejects with `TTAPE_ERR_ARTIFACT_ID_MISMATCH`. |
| `0x18` | `8` | `uint8_t[8]` / `8s` | `reserved` | Must be all zero (`0x00 * 8`) → else `TTAPE_ERR_NONZERO_RESERVED`. |

### 2.2 Inline Section Header (`struct ttape_v1_section_header`, 12 Bytes, Little-Endian `<HHII`)

Repeated `section_count` times starting at byte offset `0x20`:

| Offset | Size | C / Python Type | Field | Normative Rule |
| :--- | :--- | :--- | :--- | :--- |
| `0x00` | `2` | `uint16_t` / `H` | `tag` | Section tag in `0x0001..0x0007`. Out of range → `TTAPE_ERR_UNKNOWN_TAG`; duplicate tag in same file → `TTAPE_ERR_DUPLICATE_TAG`. |
| `0x02` | `2` | `uint16_t` / `H` | `reserved` | Must equal `0` → else `TTAPE_ERR_NONZERO_RESERVED`. |
| `0x04` | `4` | `uint32_t` / `I` | `length` | Unpadded payload byte length. |
| `0x08` | `4` | `uint32_t` / `I` | `payload_fnv1a` | 32-bit FNV-1a over `payload[0 .. length - 1]` → mismatch rejects with `TTAPE_ERR_CHECKSUM_MISMATCH`. |
| `0x0C` | `length` | `uint8_t[length]` | `payload` | Section payload bytes. Sum of `(12 + length)` across all `section_count` sections MUST equal `total_length` → else `TTAPE_ERR_LENGTH_MISMATCH`. |

### 2.3 Checksum & Content-Address Algorithms

1. **FNV-1a-32 (`header_fnv1a` and `payload_fnv1a`):**
   * Offset basis: `0x811c9dc5`
   * Prime: `0x01000193`
   * For `header_fnv1a`, hash the 28-byte concatenation `header[0x00:0x0C] || header[0x10:0x20]`.
2. **FNV-1a-64 (`artifact_id`):**
   * Offset basis: `0xcbf29ce484222325`
   * Prime: `0x00000100000001b3`
   * Hash the entire section stream `file[0x20 : 0x20 + total_length]`.

---

## 3. Section Catalog (`0x0001` – `0x0007`)

| Tag (`u16`) | Name | Presence Rule | Summary |
| :--- | :--- | :--- | :--- |
| `0x0001` | `weights` | **Mandatory in all profiles** | 2-bit LSB-first packed ternary weights (`§3.1`). |
| `0x0002` | `scales` | **Mandatory in all profiles** | Per-layer `s16` power-of-two scale exponents (`§3.2`). |
| `0x0003` | `arch_descriptor` | **Mandatory in all profiles** | Layer list, dimensions, reduction stages, latency/context caps (`§3.3`). |
| `0x0004` | `manifest` | **Present iff `flags` bit 1 (`0x02`) == `1`** | 5 length-prefixed provenance fields (`§3.4`). |
| `0x0005` | `link_table` | **Present iff `flags` bit 0 (`0x01`) == `1`** | 4-byte aligned `int32_t` self-relative offsets (`§3.5`). |
| `0x0006` | `group_scales_u16` | **Mandatory in `TTAPE-grouped`; Optional in `TTAPE-extended`; Forbidden in `TTAPE-core`** | Per-group `u16` scale mantissas, `group_size G in {64, 128, 256}`, and shared `shift` (`§3.6`). |
| `0x0007` | `fp32_aux` | **Mandatory in `TTAPE-extended`; Forbidden in `TTAPE-core` and `TTAPE-grouped`** | Host-only IEEE-754 `binary32` auxiliary tensors (embeddings, `RMSNorm`, softmax scales) (`§3.7`). |

When `(flags & 0x04) != 0`, at least one of `0x0006` (`group_scales_u16`) or `0x0007` (`fp32_aux`) MUST be present → else `TTAPE_ERR_MISSING_REQUIRED_SECTION`.

### 3.1 Ternary Weights (`0x0001`)

* **2-Bit Trit Code Map (ADR-003):**
  * `0b00` (`0`) → `0`
  * `0b01` (`1`) → `+1`
  * `0b10` (`2`) → `-1`
  * `0b11` (`3`) → **RESERVED / INVALID.** Any active lane containing `0b11` MUST be rejected with `TTAPE_ERR_RESERVED_TERNARY_CODE`.
* **LSB-First Byte Packing:** Four consecutive trits along the input reduction dimension `dim_in` pack into one byte:
  $$\text{byte} = (c_0 \ll 0) \mid (c_1 \ll 2) \mid (c_2 \ll 4) \mid (c_3 \ll 6)$$
  where lane `k in 0..3` occupies bits `[2k + 1 : 2k]`.
* **Row Layout & Tail Padding:** Each `ternary_dense` layer (`layer_kind == 4`) with `dim_out` rows and `dim_in` columns occupies `dim_out * row_bytes` bytes in row-major order, where `row_bytes = (dim_in + 3) // 4`. If `dim_in % 4 != 0`, trailing lanes `c in [dim_in, row_bytes * 4)` MUST be `0b00` → else reject with `TTAPE_ERR_NONZERO_PADDING`.
  * *Consumer note for `TTAPE-grouped` (`setun`):* When `dim_in` is a multiple of `group_size G in {64, 128, 256}`, `row_bytes == (dim_in // G) * (G // 4)` and the row bytes map zero-copy to `(dim_out, n_groups, G // 4)`. For ragged rows (`dim_in % G != 0`), the loader unpacks the `dim_in` logical trits from `row_bytes` bytes and zero-pads the final group to `G` trits in memory.
* **Section Length Invariant:** `weights.length` MUST equal the sum of `dim_out * ((dim_in + 3) // 4)` across all `ternary_dense` layers in `arch_descriptor` (`0x0003`) → else `TTAPE_ERR_DIM_PAYLOAD_MISMATCH`.

### 3.2 Per-Layer Shift Scales (`0x0002`)

* Contains one little-endian signed 16-bit exponent (`int16_t` / `<h`) per `ternary_dense` layer in `arch_descriptor` order.
* `scales.length` MUST equal `2 * num_ternary_dense_layers` (with `num_ternary_dense_layers >= 1`) → else `TTAPE_ERR_DIM_PAYLOAD_MISMATCH`.
* Each `s16` exponent `exp_val` MUST lie in `[-30, +30]` → else `TTAPE_ERR_INVALID_SCALE`.
* In `TTAPE-core` (and `TTAPE-extended` layers without `0x0006`), the effective layer scale is $2^{\text{exp\_val}}$ (a hardware right-shift by $-\text{exp\_val}$ when $\text{exp\_val} \le 0$). When `0x0006` (`group_scales_u16`) is also present, `0x0002` stores `exp_val = -shift` for each `ternary_dense` layer.

### 3.3 Architecture Descriptor (`0x0003`)

* **Header (4 bytes, `<HH`):**
  * `u16 n_layers`: Must be `> 0`.
  * `u16 reserved`: Must be `0` → else `TTAPE_ERR_NONZERO_RESERVED`.
* **Layer Array (`n_layers * 16` bytes, each `<HHHBBII`):**
  * Total section length MUST equal `4 + 16 * n_layers` → else `TTAPE_ERR_MALFORMED_ARCH`.
  * Fields per 16-byte `ttape_v1_arch_layer`:
    1. `u16 layer_kind`:
       * `1 = token_shift` (`dim_in == dim_out > 0`, `reduce_stages == 0`)
       * `2 = glu_mux` (`dim_in == dim_out > 0`, `reduce_stages == 0`)
       * `3 = rmsnorm_shift` (`dim_in == dim_out > 0`, `reduce_stages == 0`)
       * `4 = ternary_dense` (`dim_in > 0`, `dim_out > 0`, `reduce_stages == clog2(dim_in)`, `latency_cycles >= reduce_stages`)
       * `5 = kv_arena` (`dim_in > 0`, `dim_out > 0`, `reduce_stages == 0`, `context_cap > 0`)
       * `6 = host_embedding` (**`TTAPE-extended` only**; `dim_in > 0`, `dim_out > 0`, `reduce_stages == 0`)
       * `7 = host_softmax_head` (**`TTAPE-extended` only**; `dim_in > 0`, `dim_out > 0`, `reduce_stages == 0`)
    2. `u16 dim_in` (`> 0`)
    3. `u16 dim_out` (`> 0`)
    4. `u8 reduce_stages` (`clog2(dim_in)` for `layer_kind == 4`, `0` otherwise, where `clog2(n) = ceil(log2(max(n, 1)))`)
    5. `u8 reserved` (`0` → else `TTAPE_ERR_NONZERO_RESERVED`)
    6. `u32 latency_cycles` (`>= reduce_stages` for `layer_kind == 4`)
    7. `u32 context_cap` (`> 0`)
* If `layer_kind in {6, 7}` appears when `(flags & 0x04) == 0`, reject with `TTAPE_ERR_PROFILE_SECTION_FORBIDDEN`. If `layer_kind in {6, 7}` appears on a `core` or `grouped` reader, reject with `TTAPE_ERR_UNSUPPORTED_PROFILE`.

### 3.4 Provenance Manifest (`0x0004`, Conditional on `flags` Bit 1)

Five sequential `u16`-length-prefixed fields (`<H` `byte_len > 0` followed by `byte_len` bytes):
1. `producer_id` (e.g., `"ternarytape.jl/0.1.0"`)
2. `training_run_id` (**must be exactly 32 bytes**)
3. `quant_algorithm` (e.g., `"absmean-b158-pot"` or `"absmean-b158-grouped-u16"`)
4. `quant_params` (algorithm parameter string/blob)
5. `content_hash` (producer digest, **`byte_len >= 8`**)

The sum of `(2 + byte_len)` across all 5 fields MUST equal `manifest.length` → else `TTAPE_ERR_MALFORMED_MANIFEST`.

### 3.5 Self-Relative Link Table (`0x0005`, Conditional on `flags` Bit 0)

An array of signed 32-bit self-relative byte offsets (`int32_t` / `<i`).
* `link_table.length` MUST satisfy `length > 0` and `length % 4 == 0`.
* For each entry at byte offset `off = 4 * i`:
  * `delta != 0` (self-link sentinel `0` is forbidden),
  * `(delta % 4) == 0` (4-byte aligned), and
  * `0 <= off + delta < link_table.length` (resolves within the section).
* Any violation rejects with `TTAPE_ERR_INVALID_LINK`.

### 3.6 Per-Group Unsigned 16-Bit Scales (`0x0006`, `group_scales_u16`)

Carries per-group `uint16` fixed-point scale mantissas and shared right-shift exponents for `setun` (`TTAPE-grouped` and `TTAPE-extended` profiles).
* **Profile Rule:** Forbidden when `(flags & 0x04) == 0` (`TTAPE_ERR_PROFILE_SECTION_FORBIDDEN`).
* **Header (4 bytes, `<HH`):**
  * `u16 tensor_count`: Must equal `num_ternary_dense_layers` (`> 0`) in `arch_descriptor` (`0x0003`).
  * `u16 reserved`: Must be `0` → else `TTAPE_ERR_NONZERO_RESERVED`.
* **Records (`tensor_count` records, in `ternary_dense` layer order):**
  Each record consists of an **8-byte record header (`<HHBBH`)** followed by `dim_out * n_groups` little-endian `uint16_t` scale mantissas (`dim_out * n_groups * 2` bytes):
  1. `u16 layer_index`: Must equal the layer index of the `i`-th `ternary_dense` (`layer_kind == 4`) layer in `arch_descriptor`.
  2. `u16 group_size`: Group size $G \in \{64, 128, 256\}$. Any value outside `{64, 128, 256}` MUST be rejected with `TTAPE_ERR_MALFORMED_GROUP_SCALES`.
  3. `u8 shift`: Shared right-shift exponent in `[0, 30]`. Must satisfy `scales_s16[i] == -shift` in section `0x0002`.
  4. `u8 reserved`: Must be `0` → else `TTAPE_ERR_NONZERO_RESERVED`.
  5. `u16 n_groups`: Number of groups per row. Must equal $\lceil \text{dim\_in} / G \rceil = (\text{dim\_in} + G - 1) // G$.
  6. `uint16_t scales[dim_out * n_groups]`: Row-major `(dim_out, n_groups)` little-endian `u16` scale mantissas. Effective real scale for row $n$, group $g$ is $\text{scales}(n, g) \cdot 2^{-\text{shift}}$.
* The exact byte sum `4 + sum(8 + 2 * dim_out * n_groups)` across all `tensor_count` records MUST equal `group_scales_u16.length` → else `TTAPE_ERR_MALFORMED_GROUP_SCALES`.

### 3.7 Host-Consumed FP32 Auxiliary Tensors (`0x0007`, `fp32_aux`)

Carries host-only IEEE-754 `binary32` tensors (embeddings, `RMSNorm` weights, softmax scales) in the `TTAPE-extended` profile. These tensors are verified by `payload_fnv1a` and `artifact_id` on load, but remain strictly host-consumed and are never mapped into hardware-visible BRAM/DMA regions.
* **Profile Rule:** Forbidden when `(flags & 0x04) == 0` (`TTAPE_ERR_PROFILE_SECTION_FORBIDDEN`). Rejected by `core` and `grouped` readers with `TTAPE_ERR_UNSUPPORTED_PROFILE`.
* **Header (4 bytes, `<HH`):**
  * `u16 tensor_count`: Must be `> 0`.
  * `u16 reserved`: Must be `0` → else `TTAPE_ERR_NONZERO_RESERVED`.
* **Records (`tensor_count` records):**
  Each record consists of an **8-byte record header (`<HBBI`)** followed by `elem_count` little-endian `float32` values (`elem_count * 4` bytes):
  1. `u16 layer_index`: Must be `< n_layers` in `arch_descriptor`.
  2. `u8 aux_role`:
     * `1 = embedding_table`: Target layer MUST have `layer_kind == 6` (`host_embedding`) and `elem_count == dim_in * dim_out`.
     * `2 = rmsnorm_weight`: Target layer MUST have `layer_kind == 3` (`rmsnorm_shift`) and `elem_count == dim_in`.
     * `3 = softmax_scale`: Target layer MUST have `layer_kind == 7` (`host_softmax_head`) and `elem_count == dim_out`.
  3. `u8 reserved`: Must be `0` → else `TTAPE_ERR_NONZERO_RESERVED`.
  4. `u32 elem_count`: Must be `> 0` and match the target layer dimension rule above.
  5. `float32 values[elem_count]`: `elem_count * 4` bytes of IEEE-754 `binary32` values. Any `NaN` or `±Inf` bit pattern (`(word & 0x7F800000) == 0x7F800000`) MUST be rejected with `TTAPE_ERR_MALFORMED_FP32_AUX`.
* The exact byte sum `4 + sum(8 + 4 * elem_count)` across all `tensor_count` records MUST equal `fp32_aux.length` → else `TTAPE_ERR_MALFORMED_FP32_AUX`.

---

## 4. Precedence-Ordered Rejection Contract & Diagnostics

Every conforming validator/reader MUST check invariants in the following precedence order and return the exact diagnostic symbol (`TTAPE_*`, with 1-to-1 transitional `VTPE_*` enum compatibility for C readers):

| Code | `TTAPE` Symbol | Transitional `VTPE` Alias | Trigger Condition |
| :--- | :--- | :--- | :--- |
| `0` | `TTAPE_OK` | `VTPE_OK` | Valid `.ttape` v1.0 artifact for the requested `reader_profile`. |
| `1` | `TTAPE_ERR_NULL_OR_TRUNCATED` | `VTPE_ERR_NULL_OR_TRUNCATED` | Input buffer is `NULL` or `in_len < 32`. |
| `2` | `TTAPE_ERR_BAD_MAGIC` | `VTPE_ERR_BAD_MAGIC` | Header bytes `0x00..0x03 != "TTAP"` (`0x50415454` LE). |
| `3` | `TTAPE_ERR_UNSUPPORTED_VERSION` | `VTPE_ERR_UNSUPPORTED_VERSION` | `version_major != 1`. |
| `4` | `TTAPE_ERR_NONZERO_RESERVED` | `VTPE_ERR_NONZERO_RESERVED` | Non-zero `reserved` field in file header, section header, arch descriptor, `group_scales_u16`, or `fp32_aux`. |
| `5` | `TTAPE_ERR_CHECKSUM_MISMATCH` | `VTPE_ERR_CHECKSUM_MISMATCH` | Header `header_fnv1a` or any section `payload_fnv1a` mismatch. |
| `6` | `TTAPE_ERR_UNKNOWN_FLAGS` | `VTPE_ERR_UNKNOWN_FLAGS` | Any reserved header `flags` bit 3–7 is set (`(flags & 0xF8) != 0`). |
| `7` | `TTAPE_ERR_UNSUPPORTED_PROFILE` | `VTPE_ERR_UNSUPPORTED_PROFILE` | `core` reader given `(flags & 0x04) != 0`, or `grouped` reader given tag `0x0007` (`fp32_aux`) / `layer_kind in {6, 7}`. |
| `8` | `TTAPE_ERR_SECTION_COUNT_MISMATCH` | `VTPE_ERR_SECTION_COUNT_MISMATCH` | `section_count < 3` or section walk count `!= section_count`. |
| `9` | `TTAPE_ERR_LENGTH_MISMATCH` | `VTPE_ERR_LENGTH_MISMATCH` | `32 + total_length != in_len` or sum of framed sections `!= total_length`. |
| `10` | `TTAPE_ERR_ARTIFACT_ID_MISMATCH` | `VTPE_ERR_ARTIFACT_ID_MISMATCH` | Header `artifact_id != fnv1a64(file[0x20 : 0x20 + total_length])`. |
| `11` | `TTAPE_ERR_UNKNOWN_TAG` | `VTPE_ERR_UNKNOWN_TAG` | Section tag outside `0x0001..0x0007`. |
| `12` | `TTAPE_ERR_DUPLICATE_TAG` | `VTPE_ERR_DUPLICATE_TAG` | Two sections in the same tape share the same tag. |
| `13` | `TTAPE_ERR_PROFILE_SECTION_FORBIDDEN` | `VTPE_ERR_PROFILE_SECTION_FORBIDDEN` | Tag `0x0006`, tag `0x0007`, or `layer_kind in {6, 7}` present in a Core tape (`(flags & 0x04) == 0`). |
| `14` | `TTAPE_ERR_MISSING_REQUIRED_SECTION` | `VTPE_ERR_MISSING_REQUIRED_SECTION` | Missing `0x0001`/`0x0002`/`0x0003`, `(flags & 0x04) != 0` with neither `0x0006` nor `0x0007` present, or `0x0004`/`0x0005` presence disagrees with `flags` bits 1/0. |
| `15` | `TTAPE_ERR_MALFORMED_ARCH` | `VTPE_ERR_MALFORMED_ARCH` | Invalid `arch_descriptor` (`0x0003`) length, `n_layers == 0`, `layer_kind`, dimensions, `reduce_stages`, or `latency_cycles`. |
| `16` | `TTAPE_ERR_DIM_PAYLOAD_MISMATCH` | `VTPE_ERR_DIM_PAYLOAD_MISMATCH` | `num_ternary_dense_layers == 0`, or `weights` (`0x0001`) / `scales` (`0x0002`) byte length disagrees with `arch_descriptor`. |
| `17` | `TTAPE_ERR_RESERVED_TERNARY_CODE` | `VTPE_ERR_RESERVED_TERNARY_CODE` | Active weight lane in `0x0001` contains reserved code `0b11`. |
| `18` | `TTAPE_ERR_NONZERO_PADDING` | `VTPE_ERR_NONZERO_PADDING` | Trailing row-padding lane (`c >= dim_in`) in `0x0001` is non-zero. |
| `19` | `TTAPE_ERR_INVALID_SCALE` | `VTPE_ERR_INVALID_SCALE` | Any `s16` exponent in `0x0002` is outside `[-30, +30]`. |
| `20` | `TTAPE_ERR_MALFORMED_MANIFEST` | `VTPE_ERR_MALFORMED_MANIFEST` | `manifest` (`0x0004`) framing, 32-byte `training_run_id`, or `>= 8`-byte `content_hash` rule violated. |
| `21` | `TTAPE_ERR_INVALID_LINK` | `VTPE_ERR_INVALID_LINK` | `link_table` (`0x0005`) offset is `0`, not 4-byte aligned, or resolves out of bounds. |
| `22` | `TTAPE_ERR_MALFORMED_FP32_AUX` | `VTPE_ERR_MALFORMED_FP32_AUX` | `fp32_aux` (`0x0007`) framing, layer/role/element binding, or `NaN`/`±Inf` check failed. |
| `23` | `TTAPE_ERR_MALFORMED_GROUP_SCALES` | `VTPE_ERR_MALFORMED_GROUP_SCALES` | `group_scales_u16` (`0x0006`) framing, `group_size not in {64, 128, 256}`, `n_groups != ceil(dim_in / G)`, `shift > 30`, or scale mismatch with `0x0002`. |

**Exhaustive Per-Byte Mutation Invariant:** Every byte index `0 .. len - 1` of any valid reference `.ttape` artifact MUST have a single-byte mutation (`byte ^= 0x01`) that transitions validation from `TTAPE_OK` to `TTAPE_ERR_*`.

---

## 5. Byte-Exact Worked Example (`accept/grouped_with_u16_scales.ttape`)

Consider the ragged 1-row tensor from `setun/docs/FORMAT.md` §6:
* 1 `ternary_dense` layer (`layer_index = 0`, `layer_kind = 4`), `dim_out = 1` ($N = 1$), `dim_in = 6` ($K = 6$), `reduce_stages = clog2(6) = 3`, `latency_cycles = 3`, `context_cap = 64`.
* Logical row weights: `w = [+1, -1, 0, +1, -1, -1]`, padded to `row_bytes = (6 + 3) // 4 = 2` bytes (`8` lanes, trailing 2 lanes = `0b00`):
  * Byte 0 (lanes 0..3: `+1, -1, 0, +1`): `(1 << 0) | (2 << 2) | (0 << 4) | (1 << 6) = 0x49`
  * Byte 1 (lanes 4..5: `-1, -1`; lanes 6..7: `0, 0`): `(2 << 0) | (2 << 2) = 0x0A`
* Group scale parameters (`0x0006` `group_scales_u16`): `group_size = 64`, `n_groups = ceil(6 / 64) = 1`, `shift = 8`, `scales_u16 = [[0x0180]]` (`384`, effective scale $384 \cdot 2^{-8} = 1.5$), with `0x0002` `scales` carrying `exp_val = -8` (`0xFFF8` LE).
* Section payloads:
  * `0x0001` (`weights`, 2 bytes): `49 0a`
  * `0x0002` (`scales`, 2 bytes): `f8 ff`
  * `0x0003` (`arch_descriptor`, 20 bytes): `01 00 00 00` (`n_layers=1`) + `04 00 06 00 01 00 03 00 03 00 00 00 40 00 00 00`
  * `0x0006` (`group_scales_u16`, 14 bytes): `01 00 00 00` (`tensor_count=1`) + `00 00 40 00 08 00 01 00` (`layer=0, G=64, shift=8, rsvd=0, n_groups=1`) + `80 01` (`scale_u16=384`)

---

## 6. Golden Fixture Corpus (`fixtures/`)

Generated deterministically by `python3 fixtures/generate_golden.py` (Python 3 standard library only) and verified by `python3 fixtures/generate_golden.py --check`:

### 6.1 Positive Corpus (`fixtures/accept/*.ttape` + `.json`)
1. `core_minimal_dot4`: `TTAPE-core` (`flags = 0x00`), 1 `ternary_dense` layer (`dim_in = 4, dim_out = 2`, `scale_exp = -2`).
2. `core_with_manifest_and_links`: `TTAPE-core` (`flags = 0x03`), adds `0x0004` (`manifest`) and `0x0005` (`link_table`).
3. `grouped_with_u16_scales`: `TTAPE-grouped` (`flags = 0x04`), ragged `dim_in = 6, dim_out = 1` layer with `0x0006` (`group_scales_u16`, `G = 64, shift = 8, scale_u16 = 0x0180`).
4. `lux_chain_grouped_mlp`: `TTAPE-grouped` (`flags = 0x04`), deterministic 2-layer `Lux.Chain(Dense(8 => 4), Dense(4 => 2))` export with `0x0006` (`group_scales_u16`, `G = 64, shift = 8`) and exact integer input/output vectors in its `.json` sidecar.
5. `extended_with_fp32_aux`: `TTAPE-extended` (`flags = 0x06`), 3-layer block (`host_embedding`, `rmsnorm_shift`, `ternary_dense`) with `0x0004` (`manifest`) and `0x0007` (`fp32_aux`).

### 6.2 Negative Corpus (`fixtures/reject/*.ttape` + `.json`)
1. `bad_magic` → `TTAPE_ERR_BAD_MAGIC`
2. `unsupported_version` → `TTAPE_ERR_UNSUPPORTED_VERSION`
3. `bad_section_checksum` → `TTAPE_ERR_CHECKSUM_MISMATCH`
4. `bad_artifact_id` → `TTAPE_ERR_ARTIFACT_ID_MISMATCH`
5. `reserved_ternary_code_0b11` → `TTAPE_ERR_RESERVED_TERNARY_CODE`
6. `nonzero_padding` → `TTAPE_ERR_NONZERO_PADDING`
7. `unknown_flags_bit3` → `TTAPE_ERR_UNKNOWN_FLAGS`
8. `unknown_section_tag` → `TTAPE_ERR_UNKNOWN_TAG`
9. `missing_required_section` → `TTAPE_ERR_MISSING_REQUIRED_SECTION`
10. `malformed_fp32_aux_nan` → `TTAPE_ERR_MALFORMED_FP32_AUX`
11. `core_reader_on_extended` → `TTAPE_ERR_UNSUPPORTED_PROFILE`
12. `core_reader_on_grouped` → `TTAPE_ERR_UNSUPPORTED_PROFILE`
13. `grouped_reader_on_extended` → `TTAPE_ERR_UNSUPPORTED_PROFILE`
14. `group_scales_in_core_profile` → `TTAPE_ERR_PROFILE_SECTION_FORBIDDEN`
15. `fp32_aux_in_core_profile` → `TTAPE_ERR_PROFILE_SECTION_FORBIDDEN`
16. `invalid_group_size` → `TTAPE_ERR_MALFORMED_GROUP_SCALES`
