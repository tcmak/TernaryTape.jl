#ifndef TTAPE_READER_H
#define TTAPE_READER_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* -------------------------------------------------------------------------
 * c/ttape_reader.h — Zero-Dependency ISO C99 TTAPE v1 Wire Parser/Validator
 * Canonical Spec: spec/ttape-v1-draft.md
 * Scope: Pure wire framing, checksum, and section layout parser/validator.
 *        Not an inference runtime.
 * ------------------------------------------------------------------------- */

#define TTAPE_V1_MAGIC         0x50415454u /* "TTAP" (0x54 0x54 0x41 0x50 LE) */
#define TTAPE_V1_VERSION_MAJOR 1u
#define TTAPE_V1_FLAG_LINKS    0x01u
#define TTAPE_V1_FLAG_MANIFEST 0x02u
#define TTAPE_V1_FLAG_NON_CORE 0x04u
#define TTAPE_V1_FLAG_RESERVED 0xF8u

#define TTAPE_V1_HEADER_BYTES           32u
#define TTAPE_V1_SECTION_HEADER_BYTES   12u
#define TTAPE_V1_ARCH_LAYER_BYTES       16u
#define TTAPE_V1_GROUP_SCALES_REC_BYTES 8u
#define TTAPE_V1_FP32_AUX_REC_BYTES     8u

#define TTAPE_V1_TAG_WEIGHTS          0x0001u
#define TTAPE_V1_TAG_SCALES           0x0002u
#define TTAPE_V1_TAG_ARCH_DESCRIPTOR  0x0003u
#define TTAPE_V1_TAG_MANIFEST         0x0004u
#define TTAPE_V1_TAG_LINK_TABLE       0x0005u
#define TTAPE_V1_TAG_GROUP_SCALES_U16 0x0006u
#define TTAPE_V1_TAG_FP32_AUX         0x0007u

#define TTAPE_LAYER_TOKEN_SHIFT       1u
#define TTAPE_LAYER_GLU_MUX           2u
#define TTAPE_LAYER_RMSNORM_SHIFT     3u
#define TTAPE_LAYER_TERNARY_DENSE     4u
#define TTAPE_LAYER_KV_ARENA          5u
#define TTAPE_LAYER_HOST_EMBEDDING    6u
#define TTAPE_LAYER_HOST_SOFTMAX_HEAD 7u

typedef enum ttape_v1_profile {
    TTAPE_PROFILE_CORE     = 0,
    TTAPE_PROFILE_GROUPED  = 1,
    TTAPE_PROFILE_EXTENDED = 2
} ttape_v1_profile_t;

typedef enum ttape_v1_error {
    TTAPE_OK = 0,
    TTAPE_ERR_NULL_OR_TRUNCATED,
    TTAPE_ERR_BAD_MAGIC,
    TTAPE_ERR_UNSUPPORTED_VERSION,
    TTAPE_ERR_NONZERO_RESERVED,
    TTAPE_ERR_CHECKSUM_MISMATCH,
    TTAPE_ERR_UNKNOWN_FLAGS,
    TTAPE_ERR_UNSUPPORTED_PROFILE,
    TTAPE_ERR_SECTION_COUNT_MISMATCH,
    TTAPE_ERR_LENGTH_MISMATCH,
    TTAPE_ERR_ARTIFACT_ID_MISMATCH,
    TTAPE_ERR_UNKNOWN_TAG,
    TTAPE_ERR_DUPLICATE_TAG,
    TTAPE_ERR_PROFILE_SECTION_FORBIDDEN,
    TTAPE_ERR_MISSING_REQUIRED_SECTION,
    TTAPE_ERR_MALFORMED_ARCH,
    TTAPE_ERR_DIM_PAYLOAD_MISMATCH,
    TTAPE_ERR_RESERVED_TERNARY_CODE,
    TTAPE_ERR_NONZERO_PADDING,
    TTAPE_ERR_INVALID_SCALE,
    TTAPE_ERR_MALFORMED_MANIFEST,
    TTAPE_ERR_INVALID_LINK,
    TTAPE_ERR_MALFORMED_FP32_AUX,
    TTAPE_ERR_MALFORMED_GROUP_SCALES
} ttape_v1_error_t;

struct ttape_v1_header {
    uint32_t magic;
    uint8_t  version_major;
    uint8_t  version_minor;
    uint8_t  flags;
    uint8_t  section_count;
    uint32_t total_length;
    uint32_t header_fnv1a;
    uint64_t artifact_id;
    uint8_t  reserved[8];
};

struct ttape_v1_section_header {
    uint16_t tag;
    uint16_t reserved;
    uint32_t length;
    uint32_t payload_fnv1a;
};

struct ttape_v1_arch_layer {
    uint16_t layer_kind;
    uint16_t dim_in;
    uint16_t dim_out;
    uint8_t  reduce_stages;
    uint8_t  reserved;
    uint32_t latency_cycles;
    uint32_t context_cap;
};

struct ttape_v1_group_scales_record {
    uint16_t layer_index;
    uint16_t group_size;
    uint8_t  shift;
    uint8_t  reserved;
    uint16_t n_groups;
};

struct ttape_v1_fp32_aux_record {
    uint16_t layer_index;
    uint8_t  aux_role;
    uint8_t  reserved;
    uint32_t elem_count;
};

struct ttape_section_span {
    const uint8_t *data;
    uint32_t length;
};

struct ttape_layer_desc {
    uint16_t layer_kind;
    uint16_t dim_in;
    uint16_t dim_out;
    uint8_t  reduce_stages;
    uint32_t latency_cycles;
    uint32_t context_cap;
    uint32_t weight_byte_offset;
    uint16_t scale_index;
};

struct ttape_group_scales_view {
    uint16_t layer_index;
    uint16_t group_size;
    uint8_t  shift;
    uint16_t n_groups;
    uint32_t elem_count;
    const uint8_t *raw_le_u16;
};

struct ttape_fp32_aux_tensor_view {
    uint16_t layer_index;
    uint8_t  aux_role;
    uint32_t elem_count;
    const uint8_t *raw_le_f32;
};

struct ttape_reader {
    const uint8_t *tape_base;
    size_t tape_size;
    uint8_t flags;
    ttape_v1_profile_t profile;
    uint16_t section_count;
    uint32_t total_length;
    uint32_t header_fnv1a;
    uint64_t artifact_id;
    struct ttape_section_span weights;
    struct ttape_section_span scales;
    struct ttape_section_span arch_descriptor;
    struct ttape_section_span manifest;
    struct ttape_section_span link_table;
    struct ttape_section_span group_scales_u16;
    struct ttape_section_span fp32_aux;
    uint16_t layer_count;
    uint16_t group_scales_count;
    uint16_t fp32_aux_count;
};

static inline uint16_t ttape_read_u16_le(const uint8_t *p) {
    return (uint16_t)((uint16_t)p[0] | ((uint16_t)p[1] << 8u));
}

static inline int16_t ttape_read_i16_le(const uint8_t *p) {
    uint16_t u = ttape_read_u16_le(p);
    return (u >= 0x8000u) ? (int16_t)((int32_t)u - 65536) : (int16_t)u;
}

static inline uint32_t ttape_read_u32_le(const uint8_t *p) {
    return ((uint32_t)p[0]) |
           ((uint32_t)p[1] << 8u) |
           ((uint32_t)p[2] << 16u) |
           ((uint32_t)p[3] << 24u);
}

static inline int32_t ttape_read_i32_le(const uint8_t *p) {
    uint32_t u = ttape_read_u32_le(p);
    return (u >= 0x80000000u) ? (int32_t)((int64_t)u - 4294967296LL) : (int32_t)u;
}

static inline uint64_t ttape_read_u64_le(const uint8_t *p) {
    uint64_t lo = (uint64_t)ttape_read_u32_le(p);
    uint64_t hi = (uint64_t)ttape_read_u32_le(p + 4u);
    return lo | (hi << 32u);
}

static inline void ttape_unpack_header(const uint8_t *b, struct ttape_v1_header *out) {
    out->magic = ttape_read_u32_le(b);
    out->version_major = b[4];
    out->version_minor = b[5];
    out->flags = b[6];
    out->section_count = b[7];
    out->total_length = ttape_read_u32_le(b + 8u);
    out->header_fnv1a = ttape_read_u32_le(b + 12u);
    out->artifact_id = ttape_read_u64_le(b + 16u);
    for (size_t i = 0u; i < 8u; ++i) {
        out->reserved[i] = b[24u + i];
    }
}

static inline void ttape_unpack_section_header(
    const uint8_t *b,
    struct ttape_v1_section_header *out
) {
    out->tag = ttape_read_u16_le(b);
    out->reserved = ttape_read_u16_le(b + 2u);
    out->length = ttape_read_u32_le(b + 4u);
    out->payload_fnv1a = ttape_read_u32_le(b + 8u);
}

static inline void ttape_unpack_arch_layer(const uint8_t *b, struct ttape_v1_arch_layer *out) {
    out->layer_kind = ttape_read_u16_le(b);
    out->dim_in = ttape_read_u16_le(b + 2u);
    out->dim_out = ttape_read_u16_le(b + 4u);
    out->reduce_stages = b[6];
    out->reserved = b[7];
    out->latency_cycles = ttape_read_u32_le(b + 8u);
    out->context_cap = ttape_read_u32_le(b + 12u);
}

static inline void ttape_unpack_group_scales_rec(
    const uint8_t *b,
    struct ttape_v1_group_scales_record *out
) {
    out->layer_index = ttape_read_u16_le(b);
    out->group_size = ttape_read_u16_le(b + 2u);
    out->shift = b[4];
    out->reserved = b[5];
    out->n_groups = ttape_read_u16_le(b + 6u);
}

static inline void ttape_unpack_fp32_aux_rec(
    const uint8_t *b,
    struct ttape_v1_fp32_aux_record *out
) {
    out->layer_index = ttape_read_u16_le(b);
    out->aux_role = b[2];
    out->reserved = b[3];
    out->elem_count = ttape_read_u32_le(b + 4u);
}

static inline uint32_t ttape_v1_fnv1a32(const uint8_t *data, size_t len) {
    uint32_t h = 0x811c9dc5u;
    for (size_t i = 0u; i < len; ++i) {
        h ^= (uint32_t)data[i];
        h *= 0x01000193u;
    }
    return h;
}

static inline uint64_t ttape_v1_fnv1a64(const uint8_t *data, size_t len) {
    uint64_t h = 0xcbf29ce484222325ULL;
    for (size_t i = 0u; i < len; ++i) {
        h ^= (uint64_t)data[i];
        h *= 0x100000001b3ULL;
    }
    return h;
}

static inline uint32_t ttape_v1_header_fnv1a32(const uint8_t *hdr_bytes) {
    uint8_t buf[28];
    for (size_t i = 0u; i < 12u; ++i) {
        buf[i] = hdr_bytes[i];
    }
    for (size_t i = 0u; i < 16u; ++i) {
        buf[12u + i] = hdr_bytes[16u + i];
    }
    return ttape_v1_fnv1a32(buf, sizeof(buf));
}

static inline uint8_t ttape_v1_clog2_u16(uint16_t n) {
    uint8_t stages = 0u;
    uint32_t span = 1u;
    while (span < (uint32_t)n) {
        span <<= 1u;
        stages++;
    }
    return stages;
}

static inline ttape_v1_error_t ttape_v1_validate_arch_weights_scales(
    bool is_non_core_tape,
    ttape_v1_profile_t reader_profile,
    const uint8_t *arch_ptr,
    uint32_t arch_len,
    const uint8_t *weights_ptr,
    uint32_t weights_len,
    const uint8_t *scales_ptr,
    uint32_t scales_len,
    uint16_t *out_n_layers,
    uint16_t *out_num_dense_layers
) {
    if (arch_len < 4u) {
        return TTAPE_ERR_MALFORMED_ARCH;
    }
    uint16_t n_layers = ttape_read_u16_le(arch_ptr);
    uint16_t arch_res = ttape_read_u16_le(arch_ptr + 2u);
    if (arch_res != 0u) {
        return TTAPE_ERR_NONZERO_RESERVED;
    }
    if (n_layers == 0u || arch_len != 4u + TTAPE_V1_ARCH_LAYER_BYTES * (uint32_t)n_layers) {
        return TTAPE_ERR_MALFORMED_ARCH;
    }

    uint32_t expected_weight_bytes = 0u;
    uint16_t num_dense_layers = 0u;
    for (uint16_t i = 0u; i < n_layers; ++i) {
        struct ttape_v1_arch_layer layer;
        ttape_unpack_arch_layer(arch_ptr + 4u + TTAPE_V1_ARCH_LAYER_BYTES * (uint32_t)i, &layer);
        if (layer.reserved != 0u) {
            return TTAPE_ERR_NONZERO_RESERVED;
        }
        if (layer.layer_kind < 1u || layer.layer_kind > 7u ||
            layer.dim_in == 0u || layer.dim_out == 0u || layer.context_cap == 0u) {
            return TTAPE_ERR_MALFORMED_ARCH;
        }
        if (layer.layer_kind >= 6u) {
            if (!is_non_core_tape) {
                return TTAPE_ERR_PROFILE_SECTION_FORBIDDEN;
            }
            if (reader_profile != TTAPE_PROFILE_EXTENDED) {
                return TTAPE_ERR_UNSUPPORTED_PROFILE;
            }
        }
        if (layer.layer_kind == TTAPE_LAYER_TERNARY_DENSE) {
            uint8_t exp_stages = ttape_v1_clog2_u16(layer.dim_in);
            if (layer.reduce_stages != exp_stages || layer.latency_cycles < (uint32_t)exp_stages) {
                return TTAPE_ERR_MALFORMED_ARCH;
            }
            expected_weight_bytes += (uint32_t)layer.dim_out * (((uint32_t)layer.dim_in + 3u) / 4u);
            num_dense_layers++;
        } else {
            if (layer.reduce_stages != 0u) {
                return TTAPE_ERR_MALFORMED_ARCH;
            }
            if ((layer.layer_kind == TTAPE_LAYER_TOKEN_SHIFT ||
                 layer.layer_kind == TTAPE_LAYER_GLU_MUX ||
                 layer.layer_kind == TTAPE_LAYER_RMSNORM_SHIFT) &&
                layer.dim_in != layer.dim_out) {
                return TTAPE_ERR_MALFORMED_ARCH;
            }
        }
    }

    if (num_dense_layers == 0u ||
        weights_len != expected_weight_bytes ||
        scales_len != 2u * (uint32_t)num_dense_layers) {
        return TTAPE_ERR_DIM_PAYLOAD_MISMATCH;
    }

    uint32_t w_off = 0u;
    for (uint16_t i = 0u; i < n_layers; ++i) {
        struct ttape_v1_arch_layer layer;
        ttape_unpack_arch_layer(arch_ptr + 4u + TTAPE_V1_ARCH_LAYER_BYTES * (uint32_t)i, &layer);
        if (layer.layer_kind != TTAPE_LAYER_TERNARY_DENSE) {
            continue;
        }
        uint32_t row_bytes = ((uint32_t)layer.dim_in + 3u) / 4u;
        uint32_t total_lanes = row_bytes * 4u;
        for (uint16_t r = 0u; r < layer.dim_out; ++r) {
            for (uint32_t c = 0u; c < total_lanes; ++c) {
                uint8_t b = weights_ptr[w_off + (c >> 2u)];
                uint8_t code = (uint8_t)((b >> ((c & 3u) * 2u)) & 0x03u);
                if (c < (uint32_t)layer.dim_in) {
                    if (code == 0x03u) {
                        return TTAPE_ERR_RESERVED_TERNARY_CODE;
                    }
                } else if (code != 0x00u) {
                    return TTAPE_ERR_NONZERO_PADDING;
                }
            }
            w_off += row_bytes;
        }
    }

    for (uint16_t s = 0u; s < num_dense_layers; ++s) {
        int16_t exp_val = ttape_read_i16_le(scales_ptr + 2u * (uint32_t)s);
        if (exp_val < -30 || exp_val > 30) {
            return TTAPE_ERR_INVALID_SCALE;
        }
    }

    *out_n_layers = n_layers;
    *out_num_dense_layers = num_dense_layers;
    return TTAPE_OK;
}

static inline ttape_v1_error_t ttape_v1_validate_manifest(const uint8_t *m_ptr, uint32_t m_len) {
    uint32_t m_off = 0u;
    for (uint32_t f = 0u; f < 5u; ++f) {
        if (m_len - m_off < 2u) {
            return TTAPE_ERR_MALFORMED_MANIFEST;
        }
        uint16_t flen = ttape_read_u16_le(m_ptr + m_off);
        if (flen == 0u || (uint32_t)flen > m_len - m_off - 2u ||
            (f == 1u && flen != 32u) || (f == 4u && flen < 8u)) {
            return TTAPE_ERR_MALFORMED_MANIFEST;
        }
        m_off += 2u + (uint32_t)flen;
    }
    return (m_off == m_len) ? TTAPE_OK : TTAPE_ERR_MALFORMED_MANIFEST;
}

static inline ttape_v1_error_t ttape_v1_validate_links(const uint8_t *l_ptr, uint32_t l_len) {
    if (l_len == 0u || (l_len & 3u) != 0u) {
        return TTAPE_ERR_INVALID_LINK;
    }
    for (uint32_t off = 0u; off < l_len; off += 4u) {
        int32_t delta = ttape_read_i32_le(l_ptr + off);
        int64_t target = (int64_t)off + (int64_t)delta;
        if (delta == 0 || ((uint32_t)delta & 3u) != 0u || target < 0 || target >= (int64_t)l_len) {
            return TTAPE_ERR_INVALID_LINK;
        }
    }
    return TTAPE_OK;
}

static inline ttape_v1_error_t ttape_v1_validate_group_scales(
    const uint8_t *g_ptr,
    uint32_t g_len,
    const uint8_t *arch_ptr,
    uint16_t n_layers,
    const uint8_t *scales_ptr
) {
    if (g_len < 4u) {
        return TTAPE_ERR_MALFORMED_GROUP_SCALES;
    }
    uint16_t tensor_count = ttape_read_u16_le(g_ptr);
    uint16_t g_res = ttape_read_u16_le(g_ptr + 2u);
    if (g_res != 0u) {
        return TTAPE_ERR_NONZERO_RESERVED;
    }

    uint16_t dense_seen = 0u;
    for (uint16_t i = 0u; i < n_layers; ++i) {
        struct ttape_v1_arch_layer layer;
        ttape_unpack_arch_layer(arch_ptr + 4u + TTAPE_V1_ARCH_LAYER_BYTES * (uint32_t)i, &layer);
        if (layer.layer_kind == TTAPE_LAYER_TERNARY_DENSE) {
            dense_seen++;
        }
    }
    if (tensor_count != dense_seen) {
        return TTAPE_ERR_MALFORMED_GROUP_SCALES;
    }

    uint32_t g_off = 4u;
    uint16_t dense_idx = 0u;
    for (uint16_t i = 0u; i < n_layers; ++i) {
        struct ttape_v1_arch_layer layer;
        ttape_unpack_arch_layer(arch_ptr + 4u + TTAPE_V1_ARCH_LAYER_BYTES * (uint32_t)i, &layer);
        if (layer.layer_kind != TTAPE_LAYER_TERNARY_DENSE) {
            continue;
        }
        if (g_len - g_off < TTAPE_V1_GROUP_SCALES_REC_BYTES) {
            return TTAPE_ERR_MALFORMED_GROUP_SCALES;
        }
        struct ttape_v1_group_scales_record rec;
        ttape_unpack_group_scales_rec(g_ptr + g_off, &rec);
        if (rec.reserved != 0u) {
            return TTAPE_ERR_NONZERO_RESERVED;
        }
        bool valid_g = (rec.group_size == 64u || rec.group_size == 128u || rec.group_size == 256u);
        int16_t exp_val = ttape_read_i16_le(scales_ptr + 2u * (uint32_t)dense_idx);
        if (rec.layer_index != i || !valid_g || exp_val != -(int16_t)rec.shift) {
            return TTAPE_ERR_MALFORMED_GROUP_SCALES;
        }
        uint16_t exp_groups = (uint16_t)(((uint32_t)layer.dim_in + (uint32_t)rec.group_size - 1u) /
                                         (uint32_t)rec.group_size);
        if (rec.n_groups != exp_groups) {
            return TTAPE_ERR_MALFORMED_GROUP_SCALES;
        }
        uint32_t scales_bytes = (uint32_t)layer.dim_out * (uint32_t)rec.n_groups * 2u;
        if (g_len - g_off - TTAPE_V1_GROUP_SCALES_REC_BYTES < scales_bytes) {
            return TTAPE_ERR_MALFORMED_GROUP_SCALES;
        }
        g_off += TTAPE_V1_GROUP_SCALES_REC_BYTES + scales_bytes;
        dense_idx++;
    }
    return (g_off == g_len) ? TTAPE_OK : TTAPE_ERR_MALFORMED_GROUP_SCALES;
}

static inline ttape_v1_error_t ttape_v1_validate_fp32_aux(
    const uint8_t *a_ptr,
    uint32_t a_len,
    const uint8_t *arch_ptr,
    uint16_t n_layers
) {
    if (a_len < 4u) {
        return TTAPE_ERR_MALFORMED_FP32_AUX;
    }
    uint16_t tensor_count = ttape_read_u16_le(a_ptr);
    uint16_t aux_res = ttape_read_u16_le(a_ptr + 2u);
    if (aux_res != 0u) {
        return TTAPE_ERR_NONZERO_RESERVED;
    }
    if (tensor_count == 0u) {
        return TTAPE_ERR_MALFORMED_FP32_AUX;
    }

    uint32_t a_off = 4u;
    for (uint16_t t = 0u; t < tensor_count; ++t) {
        if (a_len - a_off < TTAPE_V1_FP32_AUX_REC_BYTES) {
            return TTAPE_ERR_MALFORMED_FP32_AUX;
        }
        struct ttape_v1_fp32_aux_record rec;
        ttape_unpack_fp32_aux_rec(a_ptr + a_off, &rec);
        if (rec.reserved != 0u) {
            return TTAPE_ERR_NONZERO_RESERVED;
        }
        if (rec.layer_index >= n_layers || rec.aux_role < 1u || rec.aux_role > 3u ||
            rec.elem_count == 0u ||
            rec.elem_count > (a_len - a_off - TTAPE_V1_FP32_AUX_REC_BYTES) / 4u) {
            return TTAPE_ERR_MALFORMED_FP32_AUX;
        }

        struct ttape_v1_arch_layer layer;
        ttape_unpack_arch_layer(
            arch_ptr + 4u + TTAPE_V1_ARCH_LAYER_BYTES * (uint32_t)rec.layer_index, &layer
        );
        uint16_t exp_kind = (rec.aux_role == 1u) ? 6u : ((rec.aux_role == 2u) ? 3u : 7u);
        uint32_t exp_elems = (rec.aux_role == 1u)
            ? ((uint32_t)layer.dim_in * (uint32_t)layer.dim_out)
            : ((rec.aux_role == 2u) ? (uint32_t)layer.dim_in : (uint32_t)layer.dim_out);
        if (layer.layer_kind != exp_kind || rec.elem_count != exp_elems) {
            return TTAPE_ERR_MALFORMED_FP32_AUX;
        }

        const uint8_t *vals_ptr = a_ptr + a_off + TTAPE_V1_FP32_AUX_REC_BYTES;
        for (uint32_t k = 0u; k < rec.elem_count; ++k) {
            uint32_t word = ttape_read_u32_le(vals_ptr + 4u * k);
            if ((word & 0x7F800000u) == 0x7F800000u) {
                return TTAPE_ERR_MALFORMED_FP32_AUX;
            }
        }
        a_off += TTAPE_V1_FP32_AUX_REC_BYTES + rec.elem_count * 4u;
    }
    return (a_off == a_len) ? TTAPE_OK : TTAPE_ERR_MALFORMED_FP32_AUX;
}

static inline ttape_v1_error_t ttape_v1_validate_tape(
    const void *in_buf,
    size_t in_len,
    ttape_v1_profile_t reader_profile
) {
    if (!in_buf || in_len < TTAPE_V1_HEADER_BYTES) {
        return TTAPE_ERR_NULL_OR_TRUNCATED;
    }

    const uint8_t *base = (const uint8_t *)in_buf;
    struct ttape_v1_header hdr;
    ttape_unpack_header(base, &hdr);

    if (hdr.magic != TTAPE_V1_MAGIC) {
        return TTAPE_ERR_BAD_MAGIC;
    }
    if (hdr.version_major != TTAPE_V1_VERSION_MAJOR) {
        return TTAPE_ERR_UNSUPPORTED_VERSION;
    }

    uint8_t res_or = 0u;
    for (size_t i = 0u; i < 8u; ++i) {
        res_or |= hdr.reserved[i];
    }
    if (res_or != 0u) {
        return TTAPE_ERR_NONZERO_RESERVED;
    }

    if (ttape_v1_header_fnv1a32(base) != hdr.header_fnv1a) {
        return TTAPE_ERR_CHECKSUM_MISMATCH;
    }
    if ((hdr.flags & TTAPE_V1_FLAG_RESERVED) != 0u) {
        return TTAPE_ERR_UNKNOWN_FLAGS;
    }

    bool is_non_core_tape = ((hdr.flags & TTAPE_V1_FLAG_NON_CORE) != 0u);
    if (reader_profile == TTAPE_PROFILE_CORE && is_non_core_tape) {
        return TTAPE_ERR_UNSUPPORTED_PROFILE;
    }
    if (hdr.section_count < 3u) {
        return TTAPE_ERR_SECTION_COUNT_MISMATCH;
    }
    if (in_len - TTAPE_V1_HEADER_BYTES != (size_t)hdr.total_length) {
        return TTAPE_ERR_LENGTH_MISMATCH;
    }

    const uint8_t *sec_base = base + TTAPE_V1_HEADER_BYTES;
    if (ttape_v1_fnv1a64(sec_base, (size_t)hdr.total_length) != hdr.artifact_id) {
        return TTAPE_ERR_ARTIFACT_ID_MISMATCH;
    }

    const uint8_t *sec_ptr[8] = {NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL};
    uint32_t sec_len[8] = {0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u};
    size_t offset = 0u;

    for (uint8_t s = 0u; s < hdr.section_count; ++s) {
        if ((size_t)hdr.total_length - offset < TTAPE_V1_SECTION_HEADER_BYTES) {
            return TTAPE_ERR_LENGTH_MISMATCH;
        }
        struct ttape_v1_section_header sh;
        ttape_unpack_section_header(sec_base + offset, &sh);
        if (sh.reserved != 0u) {
            return TTAPE_ERR_NONZERO_RESERVED;
        }
        size_t rem = (size_t)hdr.total_length - offset - TTAPE_V1_SECTION_HEADER_BYTES;
        if ((size_t)sh.length > rem) {
            return TTAPE_ERR_LENGTH_MISMATCH;
        }
        const uint8_t *payload = sec_base + offset + TTAPE_V1_SECTION_HEADER_BYTES;
        if (ttape_v1_fnv1a32(payload, (size_t)sh.length) != sh.payload_fnv1a) {
            return TTAPE_ERR_CHECKSUM_MISMATCH;
        }
        if (sh.tag < TTAPE_V1_TAG_WEIGHTS || sh.tag > TTAPE_V1_TAG_FP32_AUX) {
            return TTAPE_ERR_UNKNOWN_TAG;
        }
        if (sec_ptr[sh.tag] != NULL) {
            return TTAPE_ERR_DUPLICATE_TAG;
        }
        if ((sh.tag == TTAPE_V1_TAG_GROUP_SCALES_U16 || sh.tag == TTAPE_V1_TAG_FP32_AUX) &&
            !is_non_core_tape) {
            return TTAPE_ERR_PROFILE_SECTION_FORBIDDEN;
        }
        if (sh.tag == TTAPE_V1_TAG_FP32_AUX && reader_profile != TTAPE_PROFILE_EXTENDED) {
            return TTAPE_ERR_UNSUPPORTED_PROFILE;
        }
        sec_ptr[sh.tag] = payload;
        sec_len[sh.tag] = sh.length;
        offset += TTAPE_V1_SECTION_HEADER_BYTES + (size_t)sh.length;
    }

    if (offset != (size_t)hdr.total_length) {
        return TTAPE_ERR_LENGTH_MISMATCH;
    }

    bool has_base = (sec_ptr[TTAPE_V1_TAG_WEIGHTS] != NULL) &&
                    (sec_ptr[TTAPE_V1_TAG_SCALES] != NULL) &&
                    (sec_ptr[TTAPE_V1_TAG_ARCH_DESCRIPTOR] != NULL);
    bool manifest_ok = (((hdr.flags & TTAPE_V1_FLAG_MANIFEST) != 0u) ==
                        (sec_ptr[TTAPE_V1_TAG_MANIFEST] != NULL));
    bool links_ok = (((hdr.flags & TTAPE_V1_FLAG_LINKS) != 0u) ==
                     (sec_ptr[TTAPE_V1_TAG_LINK_TABLE] != NULL));
    bool non_core_ok = (!is_non_core_tape ||
                        sec_ptr[TTAPE_V1_TAG_GROUP_SCALES_U16] != NULL ||
                        sec_ptr[TTAPE_V1_TAG_FP32_AUX] != NULL);
    if (!has_base || !manifest_ok || !links_ok || !non_core_ok) {
        return TTAPE_ERR_MISSING_REQUIRED_SECTION;
    }

    uint16_t n_layers = 0u;
    uint16_t num_dense = 0u;
    ttape_v1_error_t arch_err = ttape_v1_validate_arch_weights_scales(
        is_non_core_tape,
        reader_profile,
        sec_ptr[TTAPE_V1_TAG_ARCH_DESCRIPTOR], sec_len[TTAPE_V1_TAG_ARCH_DESCRIPTOR],
        sec_ptr[TTAPE_V1_TAG_WEIGHTS], sec_len[TTAPE_V1_TAG_WEIGHTS],
        sec_ptr[TTAPE_V1_TAG_SCALES], sec_len[TTAPE_V1_TAG_SCALES],
        &n_layers,
        &num_dense
    );
    if (arch_err != TTAPE_OK) {
        return arch_err;
    }

    if (sec_ptr[TTAPE_V1_TAG_MANIFEST] != NULL) {
        ttape_v1_error_t man_err = ttape_v1_validate_manifest(
            sec_ptr[TTAPE_V1_TAG_MANIFEST], sec_len[TTAPE_V1_TAG_MANIFEST]
        );
        if (man_err != TTAPE_OK) {
            return man_err;
        }
    }

    if (sec_ptr[TTAPE_V1_TAG_LINK_TABLE] != NULL) {
        ttape_v1_error_t link_err = ttape_v1_validate_links(
            sec_ptr[TTAPE_V1_TAG_LINK_TABLE], sec_len[TTAPE_V1_TAG_LINK_TABLE]
        );
        if (link_err != TTAPE_OK) {
            return link_err;
        }
    }

    if (sec_ptr[TTAPE_V1_TAG_GROUP_SCALES_U16] != NULL) {
        ttape_v1_error_t gs_err = ttape_v1_validate_group_scales(
            sec_ptr[TTAPE_V1_TAG_GROUP_SCALES_U16], sec_len[TTAPE_V1_TAG_GROUP_SCALES_U16],
            sec_ptr[TTAPE_V1_TAG_ARCH_DESCRIPTOR], n_layers,
            sec_ptr[TTAPE_V1_TAG_SCALES]
        );
        if (gs_err != TTAPE_OK) {
            return gs_err;
        }
    }

    if (sec_ptr[TTAPE_V1_TAG_FP32_AUX] != NULL) {
        ttape_v1_error_t aux_err = ttape_v1_validate_fp32_aux(
            sec_ptr[TTAPE_V1_TAG_FP32_AUX], sec_len[TTAPE_V1_TAG_FP32_AUX],
            sec_ptr[TTAPE_V1_TAG_ARCH_DESCRIPTOR], n_layers
        );
        if (aux_err != TTAPE_OK) {
            return aux_err;
        }
    }

    return TTAPE_OK;
}

static inline void ttape_zero_reader(struct ttape_reader *out) {
    out->tape_base = NULL;
    out->tape_size = 0u;
    out->flags = 0u;
    out->profile = TTAPE_PROFILE_CORE;
    out->section_count = 0u;
    out->total_length = 0u;
    out->header_fnv1a = 0u;
    out->artifact_id = 0u;
    out->weights.data = NULL;
    out->weights.length = 0u;
    out->scales.data = NULL;
    out->scales.length = 0u;
    out->arch_descriptor.data = NULL;
    out->arch_descriptor.length = 0u;
    out->manifest.data = NULL;
    out->manifest.length = 0u;
    out->link_table.data = NULL;
    out->link_table.length = 0u;
    out->group_scales_u16.data = NULL;
    out->group_scales_u16.length = 0u;
    out->fp32_aux.data = NULL;
    out->fp32_aux.length = 0u;
    out->layer_count = 0u;
    out->group_scales_count = 0u;
    out->fp32_aux_count = 0u;
}

static inline ttape_v1_error_t ttape_reader_open(
    struct ttape_reader *out,
    const uint8_t *buf,
    size_t len,
    ttape_v1_profile_t reader_profile
) {
    if (!out) {
        return TTAPE_ERR_NULL_OR_TRUNCATED;
    }
    ttape_v1_error_t st = ttape_v1_validate_tape(buf, len, reader_profile);
    if (st != TTAPE_OK) {
        return st;
    }

    ttape_zero_reader(out);
    struct ttape_v1_header hdr;
    ttape_unpack_header(buf, &hdr);

    out->tape_base = buf;
    out->tape_size = len;
    out->flags = hdr.flags;
    out->section_count = hdr.section_count;
    out->total_length = hdr.total_length;
    out->header_fnv1a = hdr.header_fnv1a;
    out->artifact_id = hdr.artifact_id;

    size_t off = TTAPE_V1_HEADER_BYTES;
    for (uint8_t i = 0u; i < hdr.section_count; ++i) {
        struct ttape_v1_section_header shdr;
        ttape_unpack_section_header(buf + off, &shdr);
        const uint8_t *p_ptr = buf + off + TTAPE_V1_SECTION_HEADER_BYTES;
        if (shdr.tag == TTAPE_V1_TAG_WEIGHTS) {
            out->weights.data = p_ptr;
            out->weights.length = shdr.length;
        } else if (shdr.tag == TTAPE_V1_TAG_SCALES) {
            out->scales.data = p_ptr;
            out->scales.length = shdr.length;
        } else if (shdr.tag == TTAPE_V1_TAG_ARCH_DESCRIPTOR) {
            out->arch_descriptor.data = p_ptr;
            out->arch_descriptor.length = shdr.length;
        } else if (shdr.tag == TTAPE_V1_TAG_MANIFEST) {
            out->manifest.data = p_ptr;
            out->manifest.length = shdr.length;
        } else if (shdr.tag == TTAPE_V1_TAG_LINK_TABLE) {
            out->link_table.data = p_ptr;
            out->link_table.length = shdr.length;
        } else if (shdr.tag == TTAPE_V1_TAG_GROUP_SCALES_U16) {
            out->group_scales_u16.data = p_ptr;
            out->group_scales_u16.length = shdr.length;
        } else {
            out->fp32_aux.data = p_ptr;
            out->fp32_aux.length = shdr.length;
        }
        off += TTAPE_V1_SECTION_HEADER_BYTES + (size_t)shdr.length;
    }

    if ((hdr.flags & TTAPE_V1_FLAG_NON_CORE) == 0u) {
        out->profile = TTAPE_PROFILE_CORE;
    } else if (out->fp32_aux.data != NULL) {
        out->profile = TTAPE_PROFILE_EXTENDED;
    } else {
        out->profile = TTAPE_PROFILE_GROUPED;
    }

    out->layer_count = ttape_read_u16_le(out->arch_descriptor.data);
    if (out->group_scales_u16.data) {
        out->group_scales_count = ttape_read_u16_le(out->group_scales_u16.data);
    }
    if (out->fp32_aux.data) {
        out->fp32_aux_count = ttape_read_u16_le(out->fp32_aux.data);
    }
    return TTAPE_OK;
}

static inline int ttape_reader_get_layer(
    const struct ttape_reader *r,
    uint16_t layer_idx,
    struct ttape_layer_desc *out_layer
) {
    if (!r || !out_layer || layer_idx >= r->layer_count) {
        return 0;
    }

    uint32_t w_off = 0u;
    uint16_t dense_idx = 0u;
    for (uint16_t i = 0u; i <= layer_idx; ++i) {
        struct ttape_v1_arch_layer al;
        ttape_unpack_arch_layer(
            r->arch_descriptor.data + 4u + TTAPE_V1_ARCH_LAYER_BYTES * (uint32_t)i, &al
        );
        if (i == layer_idx) {
            out_layer->layer_kind = al.layer_kind;
            out_layer->dim_in = al.dim_in;
            out_layer->dim_out = al.dim_out;
            out_layer->reduce_stages = al.reduce_stages;
            out_layer->latency_cycles = al.latency_cycles;
            out_layer->context_cap = al.context_cap;
            out_layer->weight_byte_offset = w_off;
            out_layer->scale_index = dense_idx;
        }
        uint32_t is_dense = (al.layer_kind == TTAPE_LAYER_TERNARY_DENSE) ? 1u : 0u;
        w_off += is_dense * (uint32_t)al.dim_out * (((uint32_t)al.dim_in + 3u) / 4u);
        dense_idx = (uint16_t)(dense_idx + is_dense);
    }
    return 1;
}

static inline int ttape_reader_get_scale_exp(
    const struct ttape_reader *r,
    uint16_t scale_idx,
    int16_t *out_exp
) {
    if (!r || !out_exp || (uint32_t)scale_idx * 2u + 2u > r->scales.length) {
        return 0;
    }
    *out_exp = ttape_read_i16_le(r->scales.data + (uint32_t)scale_idx * 2u);
    return 1;
}

static inline int ttape_reader_get_group_scales(
    const struct ttape_reader *r,
    uint16_t dense_idx,
    struct ttape_group_scales_view *out_view
) {
    if (!r || !out_view || dense_idx >= r->group_scales_count) {
        return 0;
    }
    uint32_t off = 4u;
    for (uint16_t i = 0u; i <= dense_idx; ++i) {
        struct ttape_v1_group_scales_record rec;
        ttape_unpack_group_scales_rec(r->group_scales_u16.data + off, &rec);
        struct ttape_v1_arch_layer layer;
        ttape_unpack_arch_layer(
            r->arch_descriptor.data + 4u + TTAPE_V1_ARCH_LAYER_BYTES * (uint32_t)rec.layer_index,
            &layer
        );
        uint32_t elems = (uint32_t)layer.dim_out * (uint32_t)rec.n_groups;
        if (i == dense_idx) {
            out_view->layer_index = rec.layer_index;
            out_view->group_size = rec.group_size;
            out_view->shift = rec.shift;
            out_view->n_groups = rec.n_groups;
            out_view->elem_count = elems;
            out_view->raw_le_u16 = r->group_scales_u16.data + off + TTAPE_V1_GROUP_SCALES_REC_BYTES;
        }
        off += TTAPE_V1_GROUP_SCALES_REC_BYTES + elems * 2u;
    }
    return 1;
}

static inline int ttape_reader_read_group_scale_u16(
    const struct ttape_group_scales_view *view,
    uint32_t elem_idx,
    uint16_t *out_val
) {
    if (!view || !out_val || elem_idx >= view->elem_count) {
        return 0;
    }
    *out_val = ttape_read_u16_le(view->raw_le_u16 + elem_idx * 2u);
    return 1;
}

static inline int ttape_reader_get_fp32_aux(
    const struct ttape_reader *r,
    uint16_t aux_idx,
    struct ttape_fp32_aux_tensor_view *out_view
) {
    if (!r || !out_view || aux_idx >= r->fp32_aux_count) {
        return 0;
    }
    uint32_t off = 4u;
    for (uint16_t i = 0u; i <= aux_idx; ++i) {
        struct ttape_v1_fp32_aux_record rec;
        ttape_unpack_fp32_aux_rec(r->fp32_aux.data + off, &rec);
        if (i == aux_idx) {
            out_view->layer_index = rec.layer_index;
            out_view->aux_role = rec.aux_role;
            out_view->elem_count = rec.elem_count;
            out_view->raw_le_f32 = r->fp32_aux.data + off + TTAPE_V1_FP32_AUX_REC_BYTES;
        }
        off += TTAPE_V1_FP32_AUX_REC_BYTES + rec.elem_count * 4u;
    }
    return 1;
}

static inline int ttape_reader_read_fp32_elem(
    const struct ttape_fp32_aux_tensor_view *view,
    uint32_t elem_idx,
    float *out_val
) {
    if (!view || !out_val || elem_idx >= view->elem_count) {
        return 0;
    }
    union {
        uint32_t u;
        float f;
    } conv;
    conv.u = ttape_read_u32_le(view->raw_le_f32 + elem_idx * 4u);
    *out_val = conv.f;
    return 1;
}

static inline int ttape_reader_is_host_only_ptr(
    const struct ttape_reader *r,
    const void *ptr,
    size_t byte_len
) {
    if (!r || !ptr || !r->fp32_aux.data) {
        return 0;
    }
    uintptr_t aux_lo = (uintptr_t)r->fp32_aux.data;
    uintptr_t aux_hi = aux_lo + (uintptr_t)r->fp32_aux.length;
    uintptr_t p_lo = (uintptr_t)ptr;
    uintptr_t p_hi = p_lo + (uintptr_t)byte_len;
    if (p_hi > aux_lo && p_lo < aux_hi) {
        return 1;
    }
    return 0;
}

#ifdef __cplusplus
}
#endif

#endif /* TTAPE_READER_H */
