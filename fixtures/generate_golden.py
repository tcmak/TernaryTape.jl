#!/usr/bin/env python3
"""Deterministic golden fixture generator & reference validator for TTAPE v1.0.

Uses ONLY Python 3 standard library modules (argparse, json, math, pathlib, struct, sys)
so C, C++, Python, and Julia consumers can regenerate or verify fixtures with zero
external dependencies.
"""

from __future__ import annotations

import argparse
import json
import math
import struct
import sys
from pathlib import Path

# --- Wire Constants (spec/ttape-v1.0.md) ---
MAGIC = b"TTAP"  # 0x50415454 little-endian
VERSION_MAJOR = 1
VERSION_MINOR = 0

FLAG_LINKS = 0x01
FLAG_MANIFEST = 0x02
FLAG_NON_CORE = 0x04
FLAG_RESERVED_MASK = 0xF8

TAG_WEIGHTS = 0x0001
TAG_SCALES = 0x0002
TAG_ARCH_DESCRIPTOR = 0x0003
TAG_MANIFEST = 0x0004
TAG_LINK_TABLE = 0x0005
TAG_GROUP_SCALES_U16 = 0x0006
TAG_FP32_AUX = 0x0007

LAYER_TOKEN_SHIFT = 1
LAYER_GLU_MUX = 2
LAYER_RMSNORM_SHIFT = 3
LAYER_TERNARY_DENSE = 4
LAYER_KV_ARENA = 5
LAYER_HOST_EMBEDDING = 6
LAYER_HOST_SOFTMAX_HEAD = 7

AUX_ROLE_EMBEDDING = 1
AUX_ROLE_RMSNORM = 2
AUX_ROLE_SOFTMAX = 3

VALID_GROUP_SIZES = (64, 128, 256)

# 32-byte File Header: magic(4s), ver_maj(B), ver_min(B), flags(B), sec_count(B),
#                      total_length(I), header_fnv1a(I), artifact_id(Q), reserved(8s)
HEADER_STRUCT = struct.Struct("<4sBBBBIIQ8s")
# 12-byte Section Header: tag(H), reserved(H), length(I), payload_fnv1a(I)
SECTION_HEADER_STRUCT = struct.Struct("<HHII")
# 16-byte Arch Layer Descriptor: kind(H), dim_in(H), dim_out(H), reduce_stages(B),
#                                reserved(B), latency_cycles(I), context_cap(I)
ARCH_LAYER_STRUCT = struct.Struct("<HHHBBII")
# 8-byte Group Scales Record Header: layer_index(H), group_size(H), shift(B),
#                                    reserved(B), n_groups(H)
GROUP_SCALE_REC_STRUCT = struct.Struct("<HHBBH")
# 8-byte FP32 Aux Record Header: layer_index(H), aux_role(B), reserved(B), elem_count(I)
FP32_AUX_REC_STRUCT = struct.Struct("<HBBI")


def fnv1a_32(data: bytes) -> int:
    h = 0x811C9DC5
    for b in data:
        h ^= b
        h = (h * 0x01000193) & 0xFFFFFFFF
    return h


def fnv1a_64(data: bytes) -> int:
    h = 0xCBF29CE484222325
    for b in data:
        h ^= b
        h = (h * 0x00000100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h


def header_fnv1a_32(hdr_bytes: bytes) -> int:
    assert len(hdr_bytes) == 32
    return fnv1a_32(hdr_bytes[:12] + hdr_bytes[16:32])


def clog2_u16(n: int) -> int:
    stages = 0
    span = 1
    while span < n:
        span <<= 1
        stages += 1
    return stages


def pack_trits_row(trits: list[int]) -> bytes:
    """Pack a single row of trits in {-1, 0, +1} LSB-first, zero-padded to a byte boundary."""
    code_map = {0: 0b00, 1: 0b01, -1: 0b10}
    row_bytes = (len(trits) + 3) // 4
    padded = list(trits) + [0] * (row_bytes * 4 - len(trits))
    out = bytearray()
    for i in range(0, len(padded), 4):
        b = (
            code_map[padded[i]]
            | (code_map[padded[i + 1]] << 2)
            | (code_map[padded[i + 2]] << 4)
            | (code_map[padded[i + 3]] << 6)
        )
        out.append(b)
    return bytes(out)


def make_arch_payload(
    layers: list[tuple[int, int, int, int, int, int, int]],
    hdr_reserved: int = 0,
) -> bytes:
    out = bytearray(struct.pack("<HH", len(layers), hdr_reserved))
    for kind, dim_in, dim_out, stages, res, lat, cap in layers:
        out.extend(ARCH_LAYER_STRUCT.pack(kind, dim_in, dim_out, stages, res, lat, cap))
    return bytes(out)


def make_manifest_payload(
    producer_id: bytes = b"ternarytape.jl/0.1.0",
    training_run_id: bytes = b"run_20261118_bitnet_b158_seed042",
    quant_algorithm: bytes = b"absmean-b158-pot",
    quant_params: bytes = b"clip=2.0",
    content_hash: bytes = b"sha256:7f83b1657ff1fc53b92dc18148a1d65d",
) -> bytes:
    fields = (producer_id, training_run_id, quant_algorithm, quant_params, content_hash)
    out = bytearray()
    for f in fields:
        out.extend(struct.pack("<H", len(f)))
        out.extend(f)
    return bytes(out)


def make_group_scales_payload(
    records: list[tuple[int, int, int, int, list[int]]],
    hdr_reserved: int = 0,
) -> bytes:
    """Build 0x0006 group_scales_u16 payload from (layer_idx, group_size, shift, n_groups, u16_scales)."""
    out = bytearray(struct.pack("<HH", len(records), hdr_reserved))
    for layer_idx, group_size, shift, n_groups, scales_u16 in records:
        out.extend(GROUP_SCALE_REC_STRUCT.pack(layer_idx, group_size, shift, 0, n_groups))
        for s in scales_u16:
            out.extend(struct.pack("<H", s))
    return bytes(out)


def make_fp32_aux_payload(
    records: list[tuple[int, int, list[int]]],
    hdr_reserved: int = 0,
) -> bytes:
    """Build 0x0007 fp32_aux payload from (layer_idx, aux_role, raw_u32_words)."""
    out = bytearray(struct.pack("<HH", len(records), hdr_reserved))
    for layer_idx, aux_role, words in records:
        out.extend(FP32_AUX_REC_STRUCT.pack(layer_idx, aux_role, 0, len(words)))
        for w in words:
            out.extend(struct.pack("<I", w))
    return bytes(out)


def f32_to_u32(val: float) -> int:
    return struct.unpack("<I", struct.pack("<f", val))[0]


def build_ttape(
    *,
    sections: list[tuple[int, bytes]],
    flags: int = 0x00,
    magic: bytes = MAGIC,
    version_major: int = VERSION_MAJOR,
    version_minor: int = VERSION_MINOR,
    section_count_override: int | None = None,
    corrupt_section_crc_idx: int | None = None,
    corrupt_artifact_id_xor: int = 0,
    reserved_bytes: bytes = b"\x00" * 8,
) -> bytes:
    sec_blobs = bytearray()
    for idx, (tag, payload) in enumerate(sections):
        crc = fnv1a_32(payload)
        if corrupt_section_crc_idx == idx:
            crc = 0xDEADBEEF
        sec_blobs.extend(SECTION_HEADER_STRUCT.pack(tag, 0, len(payload), crc))
        sec_blobs.extend(payload)

    sec_bytes = bytes(sec_blobs)
    sec_count = section_count_override if section_count_override is not None else len(sections)
    total_length = len(sec_bytes)
    artifact_id = fnv1a_64(sec_bytes) ^ corrupt_artifact_id_xor

    draft_hdr = HEADER_STRUCT.pack(
        magic,
        version_major,
        version_minor,
        flags,
        sec_count,
        total_length,
        0,
        artifact_id,
        reserved_bytes,
    )
    hdr_crc = header_fnv1a_32(draft_hdr)
    final_hdr = HEADER_STRUCT.pack(
        magic,
        version_major,
        version_minor,
        flags,
        sec_count,
        total_length,
        hdr_crc,
        artifact_id,
        reserved_bytes,
    )
    return final_hdr + sec_bytes


def validate_ttape(blob: bytes, reader_profile: str = "extended") -> str:
    """Reference validator implementing §4 precedence-ordered diagnostics of spec/ttape-v1.0.md."""
    if reader_profile not in ("core", "grouped", "extended"):
        raise ValueError(f"Unknown reader_profile: {reader_profile!r}")

    if blob is None or len(blob) < HEADER_STRUCT.size:
        return "TTAPE_ERR_NULL_OR_TRUNCATED"

    magic, ver_maj, _ver_min, flags, sec_count, total_len, hdr_crc, artifact_id, reserved = (
        HEADER_STRUCT.unpack_from(blob, 0)
    )
    if magic != MAGIC:
        return "TTAPE_ERR_BAD_MAGIC"
    if ver_maj != VERSION_MAJOR:
        return "TTAPE_ERR_UNSUPPORTED_VERSION"
    if any(b != 0 for b in reserved):
        return "TTAPE_ERR_NONZERO_RESERVED"
    if header_fnv1a_32(blob[:32]) != hdr_crc:
        return "TTAPE_ERR_CHECKSUM_MISMATCH"
    if (flags & FLAG_RESERVED_MASK) != 0:
        return "TTAPE_ERR_UNKNOWN_FLAGS"

    is_non_core_tape = (flags & FLAG_NON_CORE) != 0
    if reader_profile == "core" and is_non_core_tape:
        return "TTAPE_ERR_UNSUPPORTED_PROFILE"
    if sec_count < 3:
        return "TTAPE_ERR_SECTION_COUNT_MISMATCH"
    if len(blob) - HEADER_STRUCT.size != total_len:
        return "TTAPE_ERR_LENGTH_MISMATCH"

    sec_stream = blob[HEADER_STRUCT.size :]
    if fnv1a_64(sec_stream) != artifact_id:
        return "TTAPE_ERR_ARTIFACT_ID_MISMATCH"

    sec_map: dict[int, bytes] = {}
    off = 0
    for _ in range(sec_count):
        if total_len - off < SECTION_HEADER_STRUCT.size:
            return "TTAPE_ERR_LENGTH_MISMATCH"
        tag, sec_res, slen, payload_crc = SECTION_HEADER_STRUCT.unpack_from(sec_stream, off)
        if sec_res != 0:
            return "TTAPE_ERR_NONZERO_RESERVED"
        rem = total_len - off - SECTION_HEADER_STRUCT.size
        if slen > rem:
            return "TTAPE_ERR_LENGTH_MISMATCH"
        payload = sec_stream[off + SECTION_HEADER_STRUCT.size : off + SECTION_HEADER_STRUCT.size + slen]
        if fnv1a_32(payload) != payload_crc:
            return "TTAPE_ERR_CHECKSUM_MISMATCH"
        if tag < TAG_WEIGHTS or tag > TAG_FP32_AUX:
            return "TTAPE_ERR_UNKNOWN_TAG"
        if tag in sec_map:
            return "TTAPE_ERR_DUPLICATE_TAG"
        if tag in (TAG_GROUP_SCALES_U16, TAG_FP32_AUX) and not is_non_core_tape:
            return "TTAPE_ERR_PROFILE_SECTION_FORBIDDEN"
        if tag == TAG_FP32_AUX and reader_profile != "extended":
            return "TTAPE_ERR_UNSUPPORTED_PROFILE"
        sec_map[tag] = payload
        off += SECTION_HEADER_STRUCT.size + slen

    if off != total_len:
        return "TTAPE_ERR_LENGTH_MISMATCH"

    has_base = (
        TAG_WEIGHTS in sec_map
        and TAG_SCALES in sec_map
        and TAG_ARCH_DESCRIPTOR in sec_map
    )
    manifest_ok = ((flags & FLAG_MANIFEST) != 0) == (TAG_MANIFEST in sec_map)
    links_ok = ((flags & FLAG_LINKS) != 0) == (TAG_LINK_TABLE in sec_map)
    non_core_ok = (not is_non_core_tape) or (
        TAG_GROUP_SCALES_U16 in sec_map or TAG_FP32_AUX in sec_map
    )
    if not (has_base and manifest_ok and links_ok and non_core_ok):
        return "TTAPE_ERR_MISSING_REQUIRED_SECTION"

    # Validate 0x0003 arch_descriptor, 0x0001 weights, and 0x0002 scales
    arch_buf = sec_map[TAG_ARCH_DESCRIPTOR]
    weights_buf = sec_map[TAG_WEIGHTS]
    scales_buf = sec_map[TAG_SCALES]

    if len(arch_buf) < 4:
        return "TTAPE_ERR_MALFORMED_ARCH"
    n_layers, arch_res = struct.unpack_from("<HH", arch_buf, 0)
    if arch_res != 0:
        return "TTAPE_ERR_NONZERO_RESERVED"
    if n_layers == 0 or len(arch_buf) != 4 + 16 * n_layers:
        return "TTAPE_ERR_MALFORMED_ARCH"

    layers: list[tuple[int, int, int, int, int, int, int]] = []
    dense_layers: list[tuple[int, int, int]] = []  # (layer_idx, dim_in, dim_out)
    expected_weight_bytes = 0

    for i in range(n_layers):
        kind, dim_in, dim_out, stages, l_res, lat, cap = ARCH_LAYER_STRUCT.unpack_from(
            arch_buf, 4 + 16 * i
        )
        if l_res != 0:
            return "TTAPE_ERR_NONZERO_RESERVED"
        if kind < 1 or kind > 7 or dim_in == 0 or dim_out == 0 or cap == 0:
            return "TTAPE_ERR_MALFORMED_ARCH"
        if kind >= 6:
            if not is_non_core_tape:
                return "TTAPE_ERR_PROFILE_SECTION_FORBIDDEN"
            if reader_profile != "extended":
                return "TTAPE_ERR_UNSUPPORTED_PROFILE"
        if kind == LAYER_TERNARY_DENSE:
            exp_stages = clog2_u16(dim_in)
            if stages != exp_stages or lat < exp_stages:
                return "TTAPE_ERR_MALFORMED_ARCH"
            expected_weight_bytes += dim_out * ((dim_in + 3) // 4)
            dense_layers.append((i, dim_in, dim_out))
        else:
            if stages != 0:
                return "TTAPE_ERR_MALFORMED_ARCH"
            if kind in (LAYER_TOKEN_SHIFT, LAYER_GLU_MUX, LAYER_RMSNORM_SHIFT) and dim_in != dim_out:
                return "TTAPE_ERR_MALFORMED_ARCH"
        layers.append((kind, dim_in, dim_out, stages, l_res, lat, cap))

    if (
        len(dense_layers) == 0
        or len(weights_buf) != expected_weight_bytes
        or len(scales_buf) != 2 * len(dense_layers)
    ):
        return "TTAPE_ERR_DIM_PAYLOAD_MISMATCH"

    w_off = 0
    for _l_idx, dim_in, dim_out in dense_layers:
        row_bytes = (dim_in + 3) // 4
        total_lanes = row_bytes * 4
        for _r in range(dim_out):
            for c in range(total_lanes):
                b = weights_buf[w_off + (c >> 2)]
                code = (b >> ((c & 3) * 2)) & 0x03
                if c < dim_in:
                    if code == 0x03:
                        return "TTAPE_ERR_RESERVED_TERNARY_CODE"
                elif code != 0x00:
                    return "TTAPE_ERR_NONZERO_PADDING"
            w_off += row_bytes

    scale_exps: list[int] = []
    for s_idx in range(len(dense_layers)):
        (exp_val,) = struct.unpack_from("<h", scales_buf, 2 * s_idx)
        if exp_val < -30 or exp_val > 30:
            return "TTAPE_ERR_INVALID_SCALE"
        scale_exps.append(exp_val)

    # Validate 0x0004 manifest
    if TAG_MANIFEST in sec_map:
        m_buf = sec_map[TAG_MANIFEST]
        m_off = 0
        for f_idx in range(5):
            if len(m_buf) - m_off < 2:
                return "TTAPE_ERR_MALFORMED_MANIFEST"
            (flen,) = struct.unpack_from("<H", m_buf, m_off)
            if (
                flen == 0
                or flen > len(m_buf) - m_off - 2
                or (f_idx == 1 and flen != 32)
                or (f_idx == 4 and flen < 8)
            ):
                return "TTAPE_ERR_MALFORMED_MANIFEST"
            m_off += 2 + flen
        if m_off != len(m_buf):
            return "TTAPE_ERR_MALFORMED_MANIFEST"

    # Validate 0x0005 link_table
    if TAG_LINK_TABLE in sec_map:
        l_buf = sec_map[TAG_LINK_TABLE]
        if len(l_buf) == 0 or (len(l_buf) & 3) != 0:
            return "TTAPE_ERR_INVALID_LINK"
        for l_off in range(0, len(l_buf), 4):
            (delta,) = struct.unpack_from("<i", l_buf, l_off)
            target = l_off + delta
            if delta == 0 or (delta & 3) != 0 or target < 0 or target >= len(l_buf):
                return "TTAPE_ERR_INVALID_LINK"

    # Validate 0x0006 group_scales_u16
    if TAG_GROUP_SCALES_U16 in sec_map:
        g_buf = sec_map[TAG_GROUP_SCALES_U16]
        if len(g_buf) < 4:
            return "TTAPE_ERR_MALFORMED_GROUP_SCALES"
        tensor_count, g_res = struct.unpack_from("<HH", g_buf, 0)
        if g_res != 0:
            return "TTAPE_ERR_NONZERO_RESERVED"
        if tensor_count != len(dense_layers):
            return "TTAPE_ERR_MALFORMED_GROUP_SCALES"
        g_off = 4
        for d_idx, (exp_layer_idx, dim_in, dim_out) in enumerate(dense_layers):
            if len(g_buf) - g_off < GROUP_SCALE_REC_STRUCT.size:
                return "TTAPE_ERR_MALFORMED_GROUP_SCALES"
            layer_idx, group_size, shift, rec_res, n_groups = GROUP_SCALE_REC_STRUCT.unpack_from(
                g_buf, g_off
            )
            if rec_res != 0:
                return "TTAPE_ERR_NONZERO_RESERVED"
            if (
                layer_idx != exp_layer_idx
                or group_size not in VALID_GROUP_SIZES
                or shift > 30
                or scale_exps[d_idx] != -int(shift)
            ):
                return "TTAPE_ERR_MALFORMED_GROUP_SCALES"
            exp_groups = (dim_in + group_size - 1) // group_size
            if n_groups != exp_groups:
                return "TTAPE_ERR_MALFORMED_GROUP_SCALES"
            scales_bytes = dim_out * n_groups * 2
            if len(g_buf) - g_off - GROUP_SCALE_REC_STRUCT.size < scales_bytes:
                return "TTAPE_ERR_MALFORMED_GROUP_SCALES"
            g_off += GROUP_SCALE_REC_STRUCT.size + scales_bytes
        if g_off != len(g_buf):
            return "TTAPE_ERR_MALFORMED_GROUP_SCALES"

    # Validate 0x0007 fp32_aux
    if TAG_FP32_AUX in sec_map:
        a_buf = sec_map[TAG_FP32_AUX]
        if len(a_buf) < 4:
            return "TTAPE_ERR_MALFORMED_FP32_AUX"
        tensor_count, a_res = struct.unpack_from("<HH", a_buf, 0)
        if a_res != 0:
            return "TTAPE_ERR_NONZERO_RESERVED"
        if tensor_count == 0:
            return "TTAPE_ERR_MALFORMED_FP32_AUX"
        a_off = 4
        for _ in range(tensor_count):
            if len(a_buf) - a_off < FP32_AUX_REC_STRUCT.size:
                return "TTAPE_ERR_MALFORMED_FP32_AUX"
            layer_idx, aux_role, rec_res, elem_count = FP32_AUX_REC_STRUCT.unpack_from(
                a_buf, a_off
            )
            if rec_res != 0:
                return "TTAPE_ERR_NONZERO_RESERVED"
            if (
                layer_idx >= n_layers
                or aux_role < 1
                or aux_role > 3
                or elem_count == 0
                or elem_count > (len(a_buf) - a_off - FP32_AUX_REC_STRUCT.size) // 4
            ):
                return "TTAPE_ERR_MALFORMED_FP32_AUX"
            target_kind, t_din, t_dout, *_ = layers[layer_idx]
            exp_kind = (
                LAYER_HOST_EMBEDDING
                if aux_role == AUX_ROLE_EMBEDDING
                else (
                    LAYER_RMSNORM_SHIFT
                    if aux_role == AUX_ROLE_RMSNORM
                    else LAYER_HOST_SOFTMAX_HEAD
                )
            )
            exp_elems = (
                t_din * t_dout
                if aux_role == AUX_ROLE_EMBEDDING
                else (t_din if aux_role == AUX_ROLE_RMSNORM else t_dout)
            )
            if target_kind != exp_kind or elem_count != exp_elems:
                return "TTAPE_ERR_MALFORMED_FP32_AUX"
            vals_off = a_off + FP32_AUX_REC_STRUCT.size
            for k in range(elem_count):
                (word,) = struct.unpack_from("<I", a_buf, vals_off + 4 * k)
                if (word & 0x7F800000) == 0x7F800000:
                    return "TTAPE_ERR_MALFORMED_FP32_AUX"
            a_off += FP32_AUX_REC_STRUCT.size + elem_count * 4
        if a_off != len(a_buf):
            return "TTAPE_ERR_MALFORMED_FP32_AUX"

    return "TTAPE_OK"


def _header_hex_pair(blob: bytes) -> tuple[str, str]:
    hdr = HEADER_STRUCT.unpack_from(blob, 0)
    return f"0x{hdr[6]:08x}", f"0x{hdr[7]:016x}"


def build_fixture_catalog() -> dict[str, tuple[bytes, dict]]:
    catalog: dict[str, tuple[bytes, dict]] = {}

    # Common base payloads matching veriObjC/setun layouts
    # 2x4 ternary_dense layer: row0 = [+1, -1, 0, +1] -> 0x49; row1 = [-1, +1, +1, 0] -> 0x16
    w_dot4 = pack_trits_row([1, -1, 0, 1]) + pack_trits_row([-1, 1, 1, 0])  # b"\x49\x16"
    sc_neg2 = struct.pack("<h", -2)  # b"\xfe\xff"
    arch_dot4 = make_arch_payload([(LAYER_TERNARY_DENSE, 4, 2, 2, 0, 2, 16)])
    manifest_default = make_manifest_payload()
    links_default = struct.pack("<ii", 4, -4)

    # 1. accept/core_minimal_dot4
    bin_core_min = build_ttape(
        flags=0x00,
        sections=[
            (TAG_WEIGHTS, w_dot4),
            (TAG_SCALES, sc_neg2),
            (TAG_ARCH_DESCRIPTOR, arch_dot4),
        ],
    )
    hdr_crc_hex, art_id_hex = _header_hex_pair(bin_core_min)
    catalog["accept/core_minimal_dot4"] = (
        bin_core_min,
        {
            "artifact_id_hex": art_id_hex,
            "byte_length": len(bin_core_min),
            "error_code": "TTAPE_OK",
            "expected": "accept",
            "fixture": "accept/core_minimal_dot4.ttape",
            "flags": "0x00",
            "header_fnv1a_hex": hdr_crc_hex,
            "reader_profile": "core",
            "section_count": 3,
            "spot_checks": {
                "dim_in": 4,
                "dim_out": 2,
                "layer0_kind": "ternary_dense",
                "n_layers": 1,
                "reduce_stages": 2,
                "row0_weights": [1, -1, 0, 1],
                "row1_weights": [-1, 1, 1, 0],
                "scale_exp": -2,
            },
            "tape_profile": "core",
            "vtpe_error_code": "VTPE_OK",
        },
    )

    # 2. accept/core_with_manifest_and_links
    bin_core_full = build_ttape(
        flags=FLAG_LINKS | FLAG_MANIFEST,
        sections=[
            (TAG_WEIGHTS, w_dot4),
            (TAG_SCALES, sc_neg2),
            (TAG_ARCH_DESCRIPTOR, arch_dot4),
            (TAG_MANIFEST, manifest_default),
            (TAG_LINK_TABLE, links_default),
        ],
    )
    hdr_crc_hex, art_id_hex = _header_hex_pair(bin_core_full)
    catalog["accept/core_with_manifest_and_links"] = (
        bin_core_full,
        {
            "artifact_id_hex": art_id_hex,
            "byte_length": len(bin_core_full),
            "error_code": "TTAPE_OK",
            "expected": "accept",
            "fixture": "accept/core_with_manifest_and_links.ttape",
            "flags": "0x03",
            "header_fnv1a_hex": hdr_crc_hex,
            "reader_profile": "core",
            "section_count": 5,
            "spot_checks": {
                "link_deltas": [4, -4],
                "producer_id": "ternarytape.jl/0.1.0",
                "quant_algorithm": "absmean-b158-pot",
                "training_run_id": "run_20261118_bitnet_b158_seed042",
            },
            "tape_profile": "core",
            "vtpe_error_code": "VTPE_OK",
        },
    )

    # 3. accept/grouped_with_u16_scales (matches setun/docs/FORMAT.md §6 worked example:
    #    N=1, K=6, G=64, n_groups=1, shift=8, scale_u16=[[0x0180]], trits=[+1, -1, 0, +1, -1, -1])
    w_setun_k6 = pack_trits_row([1, -1, 0, 1, -1, -1])  # b"\x49\x0a"
    sc_neg8 = struct.pack("<h", -8)
    arch_setun_k6 = make_arch_payload([(LAYER_TERNARY_DENSE, 6, 1, 3, 0, 3, 64)])
    grp_scales_k6 = make_group_scales_payload([(0, 64, 8, 1, [0x0180])])
    bin_grouped = build_ttape(
        flags=FLAG_NON_CORE,
        sections=[
            (TAG_WEIGHTS, w_setun_k6),
            (TAG_SCALES, sc_neg8),
            (TAG_ARCH_DESCRIPTOR, arch_setun_k6),
            (TAG_GROUP_SCALES_U16, grp_scales_k6),
        ],
    )
    hdr_crc_hex, art_id_hex = _header_hex_pair(bin_grouped)
    catalog["accept/grouped_with_u16_scales"] = (
        bin_grouped,
        {
            "artifact_id_hex": art_id_hex,
            "byte_length": len(bin_grouped),
            "error_code": "TTAPE_OK",
            "expected": "accept",
            "fixture": "accept/grouped_with_u16_scales.ttape",
            "flags": "0x04",
            "header_fnv1a_hex": hdr_crc_hex,
            "reader_profile": "grouped",
            "section_count": 4,
            "spot_checks": {
                "dim_in": 6,
                "dim_out": 1,
                "group_size": 64,
                "n_groups": 1,
                "row0_weights": [1, -1, 0, 1, -1, -1],
                "scales_u16": [[384]],
                "shift": 8,
            },
            "tape_profile": "grouped",
            "vtpe_error_code": "VTPE_OK",
        },
    )

    # 3b. accept/lux_chain_grouped_mlp (2-layer Lux.Chain(Dense(8=>4), Dense(4=>2)), grouped G=64)
    w_mlp_l0 = (
        pack_trits_row([1, -1, 0, 1, -1, 0, 1, 1])
        + pack_trits_row([-1, 1, 1, 0, 1, -1, 0, -1])
        + pack_trits_row([1, 0, -1, -1, 0, 1, -1, 1])
        + pack_trits_row([0, 1, 1, -1, 1, 0, 1, -1])
    )
    w_mlp_l1 = pack_trits_row([1, -1, 0, -1]) + pack_trits_row([-1, 1, 1, 0])
    sc_mlp = struct.pack("<hh", -8, -8)
    arch_mlp = make_arch_payload(
        [
            (LAYER_TERNARY_DENSE, 8, 4, 3, 0, 3, 64),
            (LAYER_TERNARY_DENSE, 4, 2, 2, 0, 2, 64),
        ]
    )
    grp_scales_mlp = make_group_scales_payload(
        [
            (0, 64, 8, 1, [0x0180, 0x0080, 0x0040, 0x0200]),
            (1, 64, 8, 1, [0x0180, 0x0280]),
        ]
    )
    bin_mlp = build_ttape(
        flags=FLAG_NON_CORE,
        sections=[
            (TAG_WEIGHTS, w_mlp_l0 + w_mlp_l1),
            (TAG_SCALES, sc_mlp),
            (TAG_ARCH_DESCRIPTOR, arch_mlp),
            (TAG_GROUP_SCALES_U16, grp_scales_mlp),
        ],
    )
    hdr_crc_hex, art_id_hex = _header_hex_pair(bin_mlp)
    catalog["accept/lux_chain_grouped_mlp"] = (
        bin_mlp,
        {
            "artifact_id_hex": art_id_hex,
            "byte_length": len(bin_mlp),
            "error_code": "TTAPE_OK",
            "expected": "accept",
            "expected_output_vector": [21, -5],
            "fixture": "accept/lux_chain_grouped_mlp.ttape",
            "flags": "0x04",
            "header_fnv1a_hex": hdr_crc_hex,
            "input_vector": [4, -2, 6, -4, 2, 0, -6, 8],
            "reader_profile": "grouped",
            "section_count": 4,
            "spot_checks": {
                "expected_output_vector": [21, -5],
                "group_size": 64,
                "hidden_vector": [3, -3, 4, -8],
                "input_vector": [4, -2, 6, -4, 2, 0, -6, 8],
                "layer0_dim_in": 8,
                "layer0_dim_out": 4,
                "layer0_scales_u16": [[384], [128], [64], [512]],
                "layer0_shift": 8,
                "layer0_weights": [
                    [1, -1, 0, 1, -1, 0, 1, 1],
                    [-1, 1, 1, 0, 1, -1, 0, -1],
                    [1, 0, -1, -1, 0, 1, -1, 1],
                    [0, 1, 1, -1, 1, 0, 1, -1],
                ],
                "layer1_dim_in": 4,
                "layer1_dim_out": 2,
                "layer1_scales_u16": [[384], [640]],
                "layer1_shift": 8,
                "layer1_weights": [
                    [1, -1, 0, -1],
                    [-1, 1, 1, 0],
                ],
                "n_layers": 2,
            },
            "tape_profile": "grouped",
            "vtpe_error_code": "VTPE_OK",
        },
    )

    # 4. accept/extended_with_fp32_aux (3 layers: host_embedding 2x4, rmsnorm_shift 4x4, ternary_dense 4x2)
    arch_ext = make_arch_payload(
        [
            (LAYER_HOST_EMBEDDING, 2, 4, 0, 0, 0, 16),
            (LAYER_RMSNORM_SHIFT, 4, 4, 0, 0, 1, 16),
            (LAYER_TERNARY_DENSE, 4, 2, 2, 0, 2, 16),
        ]
    )
    fp32_aux_ext = make_fp32_aux_payload(
        [
            (
                0,
                AUX_ROLE_EMBEDDING,
                [
                    f32_to_u32(0.25),
                    f32_to_u32(-0.5),
                    f32_to_u32(0.75),
                    f32_to_u32(1.0),
                    f32_to_u32(-0.25),
                    f32_to_u32(0.5),
                    f32_to_u32(-0.75),
                    f32_to_u32(0.125),
                ],
            ),
            (
                1,
                AUX_ROLE_RMSNORM,
                [
                    f32_to_u32(1.0),
                    f32_to_u32(1.0),
                    f32_to_u32(1.0),
                    f32_to_u32(1.0),
                ],
            ),
        ]
    )
    bin_extended = build_ttape(
        flags=FLAG_MANIFEST | FLAG_NON_CORE,
        sections=[
            (TAG_WEIGHTS, w_dot4),
            (TAG_SCALES, sc_neg2),
            (TAG_ARCH_DESCRIPTOR, arch_ext),
            (TAG_MANIFEST, manifest_default),
            (TAG_FP32_AUX, fp32_aux_ext),
        ],
    )
    hdr_crc_hex, art_id_hex = _header_hex_pair(bin_extended)
    catalog["accept/extended_with_fp32_aux"] = (
        bin_extended,
        {
            "artifact_id_hex": art_id_hex,
            "byte_length": len(bin_extended),
            "error_code": "TTAPE_OK",
            "expected": "accept",
            "fixture": "accept/extended_with_fp32_aux.ttape",
            "flags": "0x06",
            "header_fnv1a_hex": hdr_crc_hex,
            "reader_profile": "extended",
            "section_count": 5,
            "spot_checks": {
                "fp32_aux_host_only": True,
                "fp32_aux_section_tag": "0x0007",
                "fp32_aux_tensor_count": 2,
                "n_layers": 3,
            },
            "tape_profile": "extended",
            "vtpe_error_code": "VTPE_OK",
        },
    )

    # --- REJECT FIXTURES ---
    # 1. reject/bad_magic
    catalog["reject/bad_magic"] = (
        build_ttape(
            magic=b"VOBJ",
            flags=0x00,
            sections=[
                (TAG_WEIGHTS, w_dot4),
                (TAG_SCALES, sc_neg2),
                (TAG_ARCH_DESCRIPTOR, arch_dot4),
            ],
        ),
        {
            "error_code": "TTAPE_ERR_BAD_MAGIC",
            "expected": "reject",
            "fixture": "reject/bad_magic.ttape",
            "reader_profile": "core",
            "reason": "Header magic is 'VOBJ' instead of 'TTAP' (0x50415454).",
            "vtpe_error_code": "VTPE_ERR_BAD_MAGIC",
        },
    )

    # 2. reject/unsupported_version
    catalog["reject/unsupported_version"] = (
        build_ttape(
            version_major=2,
            flags=0x00,
            sections=[
                (TAG_WEIGHTS, w_dot4),
                (TAG_SCALES, sc_neg2),
                (TAG_ARCH_DESCRIPTOR, arch_dot4),
            ],
        ),
        {
            "error_code": "TTAPE_ERR_UNSUPPORTED_VERSION",
            "expected": "reject",
            "fixture": "reject/unsupported_version.ttape",
            "reader_profile": "core",
            "reason": "Header version_major is 2 instead of supported v1.",
            "vtpe_error_code": "VTPE_ERR_UNSUPPORTED_VERSION",
        },
    )

    # 3. reject/bad_section_checksum
    catalog["reject/bad_section_checksum"] = (
        build_ttape(
            flags=0x00,
            corrupt_section_crc_idx=0,
            sections=[
                (TAG_WEIGHTS, w_dot4),
                (TAG_SCALES, sc_neg2),
                (TAG_ARCH_DESCRIPTOR, arch_dot4),
            ],
        ),
        {
            "error_code": "TTAPE_ERR_CHECKSUM_MISMATCH",
            "expected": "reject",
            "fixture": "reject/bad_section_checksum.ttape",
            "reader_profile": "core",
            "reason": "Section 0x0001 (weights) payload_fnv1a does not match its payload bytes.",
            "vtpe_error_code": "VTPE_ERR_CHECKSUM_MISMATCH",
        },
    )

    # 4. reject/bad_artifact_id
    catalog["reject/bad_artifact_id"] = (
        build_ttape(
            flags=0x00,
            corrupt_artifact_id_xor=0x00000000000000FF,
            sections=[
                (TAG_WEIGHTS, w_dot4),
                (TAG_SCALES, sc_neg2),
                (TAG_ARCH_DESCRIPTOR, arch_dot4),
            ],
        ),
        {
            "error_code": "TTAPE_ERR_ARTIFACT_ID_MISMATCH",
            "expected": "reject",
            "fixture": "reject/bad_artifact_id.ttape",
            "reader_profile": "core",
            "reason": "Header artifact_id FNV-1a-64 does not match framed section stream.",
            "vtpe_error_code": "VTPE_ERR_ARTIFACT_ID_MISMATCH",
        },
    )

    # 5. reject/reserved_ternary_code_0b11
    catalog["reject/reserved_ternary_code_0b11"] = (
        build_ttape(
            flags=0x00,
            sections=[
                (TAG_WEIGHTS, b"\x4b\x16"),  # lane 0 of byte 0 is 0b11
                (TAG_SCALES, sc_neg2),
                (TAG_ARCH_DESCRIPTOR, arch_dot4),
            ],
        ),
        {
            "error_code": "TTAPE_ERR_RESERVED_TERNARY_CODE",
            "expected": "reject",
            "fixture": "reject/reserved_ternary_code_0b11.ttape",
            "reader_profile": "core",
            "reason": "Weights section contains reserved 2-bit ternary code 0b11 in byte 0 lane 0.",
            "vtpe_error_code": "VTPE_ERR_RESERVED_TERNARY_CODE",
        },
    )

    # 6. reject/nonzero_padding
    arch_dim3 = make_arch_payload([(LAYER_TERNARY_DENSE, 3, 1, 2, 0, 2, 16)])
    catalog["reject/nonzero_padding"] = (
        build_ttape(
            flags=0x00,
            sections=[
                (TAG_WEIGHTS, b"\x41"),  # lane 3 (padding lane for dim_in=3) is 0b01 != 0b00
                (TAG_SCALES, sc_neg2),
                (TAG_ARCH_DESCRIPTOR, arch_dim3),
            ],
        ),
        {
            "error_code": "TTAPE_ERR_NONZERO_PADDING",
            "expected": "reject",
            "fixture": "reject/nonzero_padding.ttape",
            "reader_profile": "core",
            "reason": "Ragged row (dim_in=3) has non-zero trailing padding lane 3 in weights section.",
            "vtpe_error_code": "VTPE_ERR_NONZERO_PADDING",
        },
    )

    # 7. reject/unknown_flags_bit3
    catalog["reject/unknown_flags_bit3"] = (
        build_ttape(
            flags=0x08,  # bit 3 set
            sections=[
                (TAG_WEIGHTS, w_dot4),
                (TAG_SCALES, sc_neg2),
                (TAG_ARCH_DESCRIPTOR, arch_dot4),
            ],
        ),
        {
            "error_code": "TTAPE_ERR_UNKNOWN_FLAGS",
            "expected": "reject",
            "fixture": "reject/unknown_flags_bit3.ttape",
            "reader_profile": "core",
            "reason": "Reserved flag bit 3 (0x08) is set with a valid recomputed header checksum.",
            "vtpe_error_code": "VTPE_ERR_UNKNOWN_FLAGS",
        },
    )

    # 8. reject/unknown_section_tag
    catalog["reject/unknown_section_tag"] = (
        build_ttape(
            flags=0x00,
            sections=[
                (TAG_WEIGHTS, w_dot4),
                (TAG_SCALES, sc_neg2),
                (TAG_ARCH_DESCRIPTOR, arch_dot4),
                (0x0099, b"\x01\x02\x03\x04"),
            ],
        ),
        {
            "error_code": "TTAPE_ERR_UNKNOWN_TAG",
            "expected": "reject",
            "fixture": "reject/unknown_section_tag.ttape",
            "reader_profile": "core",
            "reason": "Section carries unrecognized tag 0x0099 outside 0x0001..0x0007.",
            "vtpe_error_code": "VTPE_ERR_UNKNOWN_TAG",
        },
    )

    # 9. reject/missing_required_section
    catalog["reject/missing_required_section"] = (
        build_ttape(
            flags=FLAG_NON_CORE,  # bit 2 = 1 but neither 0x0006 nor 0x0007 is present
            sections=[
                (TAG_WEIGHTS, w_dot4),
                (TAG_SCALES, sc_neg2),
                (TAG_ARCH_DESCRIPTOR, arch_dot4),
            ],
        ),
        {
            "error_code": "TTAPE_ERR_MISSING_REQUIRED_SECTION",
            "expected": "reject",
            "fixture": "reject/missing_required_section.ttape",
            "reader_profile": "extended",
            "reason": "Non-core profile flag (bit 2 = 1) is set without section 0x0006 or 0x0007.",
            "vtpe_error_code": "VTPE_ERR_MISSING_REQUIRED_SECTION",
        },
    )

    # 10. reject/malformed_fp32_aux_nan
    arch_norm_and_dense = make_arch_payload(
        [
            (LAYER_RMSNORM_SHIFT, 4, 4, 0, 0, 1, 16),
            (LAYER_TERNARY_DENSE, 4, 2, 2, 0, 2, 16),
        ]
    )
    fp32_aux_nan = make_fp32_aux_payload(
        [
            (
                0,
                AUX_ROLE_RMSNORM,
                [
                    f32_to_u32(1.0),
                    f32_to_u32(1.0),
                    f32_to_u32(1.0),
                    0x7FC00001,  # quiet NaN
                ],
            )
        ]
    )
    catalog["reject/malformed_fp32_aux_nan"] = (
        build_ttape(
            flags=FLAG_NON_CORE,
            sections=[
                (TAG_WEIGHTS, w_dot4),
                (TAG_SCALES, sc_neg2),
                (TAG_ARCH_DESCRIPTOR, arch_norm_and_dense),
                (TAG_FP32_AUX, fp32_aux_nan),
            ],
        ),
        {
            "error_code": "TTAPE_ERR_MALFORMED_FP32_AUX",
            "expected": "reject",
            "fixture": "reject/malformed_fp32_aux_nan.ttape",
            "reader_profile": "extended",
            "reason": "Extended tape fp32_aux (0x0007) contains a non-finite IEEE-754 NaN (0x7fc00001).",
            "vtpe_error_code": "VTPE_ERR_MALFORMED_FP32_AUX",
        },
    )

    # 11. reject/core_reader_on_extended
    catalog["reject/core_reader_on_extended"] = (
        bin_extended,
        {
            "error_code": "TTAPE_ERR_UNSUPPORTED_PROFILE",
            "expected": "reject",
            "fixture": "reject/core_reader_on_extended.ttape",
            "reader_profile": "core",
            "reason": "Valid Extended-profile tape (flags bit 2 = 1) rejected fail-closed by a Core-only reader.",
            "vtpe_error_code": "VTPE_ERR_UNSUPPORTED_PROFILE",
        },
    )

    # 12. reject/core_reader_on_grouped
    catalog["reject/core_reader_on_grouped"] = (
        bin_grouped,
        {
            "error_code": "TTAPE_ERR_UNSUPPORTED_PROFILE",
            "expected": "reject",
            "fixture": "reject/core_reader_on_grouped.ttape",
            "reader_profile": "core",
            "reason": "Valid Grouped-profile tape (flags bit 2 = 1, tag 0x0006) rejected fail-closed by a Core-only reader.",
            "vtpe_error_code": "VTPE_ERR_UNSUPPORTED_PROFILE",
        },
    )

    # 13. reject/grouped_reader_on_extended
    catalog["reject/grouped_reader_on_extended"] = (
        bin_extended,
        {
            "error_code": "TTAPE_ERR_UNSUPPORTED_PROFILE",
            "expected": "reject",
            "fixture": "reject/grouped_reader_on_extended.ttape",
            "reader_profile": "grouped",
            "reason": "Extended-profile tape carrying tag 0x0007 (fp32_aux) rejected fail-closed by a Grouped-only reader.",
            "vtpe_error_code": "VTPE_ERR_UNSUPPORTED_PROFILE",
        },
    )

    # 14. reject/group_scales_in_core_profile
    catalog["reject/group_scales_in_core_profile"] = (
        build_ttape(
            flags=0x00,  # Core profile (bit 2 = 0)
            sections=[
                (TAG_WEIGHTS, w_setun_k6),
                (TAG_SCALES, sc_neg8),
                (TAG_ARCH_DESCRIPTOR, arch_setun_k6),
                (TAG_GROUP_SCALES_U16, grp_scales_k6),
            ],
        ),
        {
            "error_code": "TTAPE_ERR_PROFILE_SECTION_FORBIDDEN",
            "expected": "reject",
            "fixture": "reject/group_scales_in_core_profile.ttape",
            "reader_profile": "grouped",
            "reason": "Core-profile tape (flags bit 2 = 0) carries Grouped-only section 0x0006 (group_scales_u16).",
            "vtpe_error_code": "VTPE_ERR_PROFILE_SECTION_FORBIDDEN",
        },
    )

    # 15. reject/fp32_aux_in_core_profile
    fp32_aux_valid_1 = make_fp32_aux_payload(
        [
            (
                0,
                AUX_ROLE_RMSNORM,
                [
                    f32_to_u32(1.0),
                    f32_to_u32(1.0),
                    f32_to_u32(1.0),
                    f32_to_u32(1.0),
                ],
            )
        ]
    )
    catalog["reject/fp32_aux_in_core_profile"] = (
        build_ttape(
            flags=0x00,  # Core profile (bit 2 = 0)
            sections=[
                (TAG_WEIGHTS, w_dot4),
                (TAG_SCALES, sc_neg2),
                (TAG_ARCH_DESCRIPTOR, arch_norm_and_dense),
                (TAG_FP32_AUX, fp32_aux_valid_1),
            ],
        ),
        {
            "error_code": "TTAPE_ERR_PROFILE_SECTION_FORBIDDEN",
            "expected": "reject",
            "fixture": "reject/fp32_aux_in_core_profile.ttape",
            "reader_profile": "extended",
            "reason": "Core-profile tape (flags bit 2 = 0) carries Extended-only section 0x0007 (fp32_aux).",
            "vtpe_error_code": "VTPE_ERR_PROFILE_SECTION_FORBIDDEN",
        },
    )

    # 16. reject/invalid_group_size
    grp_scales_bad_g = make_group_scales_payload([(0, 32, 8, 1, [0x0180])])  # G=32 not in {64, 128, 256}
    catalog["reject/invalid_group_size"] = (
        build_ttape(
            flags=FLAG_NON_CORE,
            sections=[
                (TAG_WEIGHTS, w_setun_k6),
                (TAG_SCALES, sc_neg8),
                (TAG_ARCH_DESCRIPTOR, arch_setun_k6),
                (TAG_GROUP_SCALES_U16, grp_scales_bad_g),
            ],
        ),
        {
            "error_code": "TTAPE_ERR_MALFORMED_GROUP_SCALES",
            "expected": "reject",
            "fixture": "reject/invalid_group_size.ttape",
            "reader_profile": "grouped",
            "reason": "Section 0x0006 (group_scales_u16) declares group_size=32 outside allowlist {64, 128, 256}.",
            "vtpe_error_code": "VTPE_ERR_MALFORMED_GROUP_SCALES",
        },
    )

    return catalog


def verify_catalog_semantics(catalog: dict[str, tuple[bytes, dict]]) -> list[str]:
    """Run reference validator and exhaustive per-byte mutation sweep over all fixtures."""
    errors: list[str] = []
    for rel_stem, (blob, meta) in sorted(catalog.items()):
        profile = str(meta["reader_profile"])
        expected_code = str(meta["error_code"])
        actual_code = validate_ttape(blob, reader_profile=profile)
        if actual_code != expected_code:
            errors.append(
                f"{rel_stem}: expected {expected_code} under profile={profile}, got {actual_code}"
            )
        if meta["expected"] == "accept":
            # Exhaustive per-byte single-bit mutation check (§4)
            for idx in range(len(blob)):
                mutated = bytearray(blob)
                mutated[idx] ^= 0x01
                m_code = validate_ttape(bytes(mutated), reader_profile="extended")
                if m_code == "TTAPE_OK":
                    errors.append(
                        f"{rel_stem}: byte offset {idx} single-bit flip did not flip accept -> reject"
                    )
                    break
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify on-disk fixtures match generated bytes and pass reference validation.",
    )
    args = parser.parse_args()

    fixtures_root = Path(__file__).resolve().parent
    catalog = build_fixture_catalog()

    semantic_errors = verify_catalog_semantics(catalog)
    if semantic_errors:
        print("Semantic validation errors in fixture catalog:", file=sys.stderr)
        for err in semantic_errors:
            print(f"  - {err}", file=sys.stderr)
        return 1

    mismatches: list[str] = []
    expected_files: set[Path] = set()

    for rel_stem, (blob, meta) in sorted(catalog.items()):
        bin_path = fixtures_root / f"{rel_stem}.ttape"
        json_path = fixtures_root / f"{rel_stem}.json"
        expected_files.add(bin_path)
        expected_files.add(json_path)
        json_bytes = (json.dumps(meta, indent=2, sort_keys=True) + "\n").encode("utf-8")

        if args.check:
            if not bin_path.exists() or bin_path.read_bytes() != blob:
                mismatches.append(str(bin_path))
            if not json_path.exists() or json_path.read_bytes() != json_bytes:
                mismatches.append(str(json_path))
        else:
            bin_path.parent.mkdir(parents=True, exist_ok=True)
            bin_path.write_bytes(blob)
            json_path.write_bytes(json_bytes)

    if not args.check:
        # Remove any stale files in accept/ or reject/ not in catalog
        for sub in ("accept", "reject"):
            subdir = fixtures_root / sub
            if subdir.exists():
                for p in subdir.iterdir():
                    if p.is_file() and p not in expected_files:
                        p.unlink()

    if args.check:
        spec_dir = fixtures_root.parent / "spec"
        spec_v1 = spec_dir / "ttape-v1.0.md"
        spec_draft = spec_dir / "ttape-v1-draft.md"
        if not spec_v1.exists():
            mismatches.append(f"missing frozen spec: {spec_v1}")
        elif "FROZEN v1.0" not in spec_v1.read_text(encoding="utf-8"):
            mismatches.append(f"spec header missing FROZEN v1.0 status: {spec_v1}")
        if spec_draft.exists():
            mismatches.append(f"stale draft spec still present: {spec_draft}")
        for sub in ("accept", "reject"):
            subdir = fixtures_root / sub
            if subdir.exists():
                for p in sorted(subdir.iterdir()):
                    if p.is_file() and p not in expected_files:
                        mismatches.append(f"stale file: {p}")
        if mismatches:
            print("Fixture drift detected in:", file=sys.stderr)
            for m in mismatches:
                print(f"  - {m}", file=sys.stderr)
            return 1
        print(
            f"OK: all {len(catalog)} TTAPE golden fixture pairs verified byte-identical and mutation-clean."
        )
        return 0

    print(f"Generated {len(catalog)} TTAPE golden fixture pairs in {fixtures_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
