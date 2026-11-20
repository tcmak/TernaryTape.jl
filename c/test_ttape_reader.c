#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include "ttape_reader.h"

#define MAX_TAPE_BYTES 65536u

static int g_failures = 0;

#define CHECK_TRUE(cond)                                                       \
    do {                                                                       \
        if (!(cond)) {                                                         \
            fprintf(stderr, "[FAIL] %s:%d: %s\n", __FILE__, __LINE__, #cond);  \
            g_failures++;                                                      \
        }                                                                      \
    } while (0)

#define CHECK_EQ_INT(expected, actual)                                         \
    do {                                                                       \
        long _exp = (long)(expected);                                          \
        long _act = (long)(actual);                                            \
        if (_exp != _act) {                                                    \
            fprintf(stderr, "[FAIL] %s:%d: expected %ld, got %ld\n",           \
                    __FILE__, __LINE__, _exp, _act);                           \
            g_failures++;                                                      \
        }                                                                      \
    } while (0)

static void write_u16_le(uint8_t *dst, uint16_t v) {
    dst[0] = (uint8_t)(v & 0xFFu);
    dst[1] = (uint8_t)((v >> 8u) & 0xFFu);
}

static void write_u32_le(uint8_t *dst, uint32_t v) {
    dst[0] = (uint8_t)(v & 0xFFu);
    dst[1] = (uint8_t)((v >> 8u) & 0xFFu);
    dst[2] = (uint8_t)((v >> 16u) & 0xFFu);
    dst[3] = (uint8_t)((v >> 24u) & 0xFFu);
}

static void write_u64_le(uint8_t *dst, uint64_t v) {
    write_u32_le(dst, (uint32_t)(v & 0xFFFFFFFFULL));
    write_u32_le(dst + 4u, (uint32_t)(v >> 32u));
}

static size_t load_fixture(const char *fixtures_dir, const char *rel_path, uint8_t *out_buf) {
    char path[1024];
    int n = snprintf(path, sizeof(path), "%s/%s", fixtures_dir, rel_path);
    if (n <= 0 || (size_t)n >= sizeof(path)) {
        fprintf(stderr, "[FAIL] fixture path too long: %s/%s\n", fixtures_dir, rel_path);
        g_failures++;
        return 0u;
    }
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        fprintf(stderr, "[FAIL] could not open fixture: %s\n", path);
        g_failures++;
        return 0u;
    }
    size_t read_n = fread(out_buf, 1u, MAX_TAPE_BYTES, fp);
    fclose(fp);
    CHECK_TRUE(read_n > 0u);
    return read_n;
}

static void recompute_tape_checksums(
    uint8_t *tape,
    size_t tape_len,
    bool recompute_artifact_id,
    bool recompute_hdr_crc
) {
    if (tape_len < TTAPE_V1_HEADER_BYTES) {
        return;
    }
    uint32_t total_len = ttape_read_u32_le(tape + 8u);
    if (recompute_artifact_id && tape_len >= TTAPE_V1_HEADER_BYTES + (size_t)total_len) {
        uint64_t art_id = ttape_v1_fnv1a64(tape + TTAPE_V1_HEADER_BYTES, (size_t)total_len);
        write_u64_le(tape + 16u, art_id);
    }
    if (recompute_hdr_crc) {
        uint32_t h_crc = ttape_v1_header_fnv1a32(tape);
        write_u32_le(tape + 12u, h_crc);
    }
}

static size_t append_section(
    uint8_t *secs,
    size_t cur_len,
    uint16_t tag,
    const uint8_t *payload,
    uint32_t payload_len,
    uint16_t reserved
) {
    write_u16_le(secs + cur_len, tag);
    write_u16_le(secs + cur_len + 2u, reserved);
    write_u32_le(secs + cur_len + 4u, payload_len);
    write_u32_le(secs + cur_len + 8u, ttape_v1_fnv1a32(payload, payload_len));
    for (uint32_t i = 0u; i < payload_len; ++i) {
        secs[cur_len + TTAPE_V1_SECTION_HEADER_BYTES + i] = payload[i];
    }
    return cur_len + TTAPE_V1_SECTION_HEADER_BYTES + (size_t)payload_len;
}

static size_t wrap_tape(
    const uint8_t *secs,
    size_t secs_len,
    uint8_t sec_count,
    uint8_t flags,
    uint8_t *out_tape
) {
    for (size_t i = 0u; i < TTAPE_V1_HEADER_BYTES; ++i) {
        out_tape[i] = 0u;
    }
    write_u32_le(out_tape, TTAPE_V1_MAGIC);
    out_tape[4] = 1u;
    out_tape[5] = 0u;
    out_tape[6] = flags;
    out_tape[7] = sec_count;
    write_u32_le(out_tape + 8u, (uint32_t)secs_len);
    for (size_t i = 0u; i < secs_len; ++i) {
        out_tape[TTAPE_V1_HEADER_BYTES + i] = secs[i];
    }
    size_t total = TTAPE_V1_HEADER_BYTES + secs_len;
    recompute_tape_checksums(out_tape, total, true, true);
    return total;
}

static size_t make_arch1(
    uint16_t kind,
    uint16_t din,
    uint16_t dout,
    uint8_t stages,
    uint8_t res,
    uint32_t lat,
    uint32_t cap,
    uint16_t n_layers,
    uint16_t hdr_res,
    uint8_t *out_ar
) {
    write_u16_le(out_ar, n_layers);
    write_u16_le(out_ar + 2u, hdr_res);
    write_u16_le(out_ar + 4u, kind);
    write_u16_le(out_ar + 6u, din);
    write_u16_le(out_ar + 8u, dout);
    out_ar[10] = stages;
    out_ar[11] = res;
    write_u32_le(out_ar + 12u, lat);
    write_u32_le(out_ar + 16u, cap);
    return 20u;
}

static size_t build_3sec_tape(
    const uint8_t *w,
    uint32_t w_len,
    const uint8_t *sc,
    uint32_t sc_len,
    const uint8_t *ar,
    uint32_t ar_len,
    uint8_t flags,
    uint8_t *out_tape
) {
    uint8_t secs[4096];
    size_t slen = 0u;
    slen = append_section(secs, slen, TTAPE_V1_TAG_WEIGHTS, w, w_len, 0u);
    slen = append_section(secs, slen, TTAPE_V1_TAG_SCALES, sc, sc_len, 0u);
    slen = append_section(secs, slen, TTAPE_V1_TAG_ARCH_DESCRIPTOR, ar, ar_len, 0u);
    return wrap_tape(secs, slen, 3u, flags, out_tape);
}

static void test_accept_and_reject_corpus(const char *fixtures_dir) {
    uint8_t buf[MAX_TAPE_BYTES];
    size_t len = 0u;

    /* 1. Accept corpus (4 fixtures) */
    len = load_fixture(fixtures_dir, "accept/core_minimal_dot4.ttape", buf);
    CHECK_EQ_INT(TTAPE_OK, ttape_v1_validate_tape(buf, len, TTAPE_PROFILE_CORE));
    CHECK_EQ_INT(TTAPE_OK, ttape_v1_validate_tape(buf, len, TTAPE_PROFILE_GROUPED));
    CHECK_EQ_INT(TTAPE_OK, ttape_v1_validate_tape(buf, len, TTAPE_PROFILE_EXTENDED));

    struct ttape_reader core_r;
    CHECK_EQ_INT(TTAPE_OK, ttape_reader_open(&core_r, buf, len, TTAPE_PROFILE_CORE));
    CHECK_EQ_INT(TTAPE_PROFILE_CORE, core_r.profile);
    CHECK_EQ_INT(1, core_r.layer_count);
    struct ttape_layer_desc l0;
    CHECK_EQ_INT(1, ttape_reader_get_layer(&core_r, 0u, &l0));
    CHECK_EQ_INT(TTAPE_LAYER_TERNARY_DENSE, l0.layer_kind);
    CHECK_EQ_INT(4, l0.dim_in);
    CHECK_EQ_INT(2, l0.dim_out);
    int16_t exp0 = 0;
    CHECK_EQ_INT(1, ttape_reader_get_scale_exp(&core_r, l0.scale_index, &exp0));
    CHECK_EQ_INT(-2, exp0);

    len = load_fixture(fixtures_dir, "accept/core_with_manifest_and_links.ttape", buf);
    CHECK_EQ_INT(TTAPE_OK, ttape_reader_open(&core_r, buf, len, TTAPE_PROFILE_CORE));
    CHECK_TRUE(core_r.manifest.data != NULL);
    CHECK_TRUE(core_r.link_table.data != NULL);

    len = load_fixture(fixtures_dir, "accept/grouped_with_u16_scales.ttape", buf);
    struct ttape_reader grp_r;
    CHECK_EQ_INT(TTAPE_OK, ttape_reader_open(&grp_r, buf, len, TTAPE_PROFILE_GROUPED));
    CHECK_EQ_INT(TTAPE_PROFILE_GROUPED, grp_r.profile);
    CHECK_EQ_INT(1, grp_r.group_scales_count);
    struct ttape_group_scales_view gview;
    CHECK_EQ_INT(1, ttape_reader_get_group_scales(&grp_r, 0u, &gview));
    CHECK_EQ_INT(64, gview.group_size);
    CHECK_EQ_INT(8, gview.shift);
    CHECK_EQ_INT(1, gview.n_groups);
    uint16_t scale_u16 = 0u;
    CHECK_EQ_INT(1, ttape_reader_read_group_scale_u16(&gview, 0u, &scale_u16));
    CHECK_EQ_INT(0x0180, scale_u16);

    len = load_fixture(fixtures_dir, "accept/extended_with_fp32_aux.ttape", buf);
    struct ttape_reader ext_r;
    CHECK_EQ_INT(TTAPE_OK, ttape_reader_open(&ext_r, buf, len, TTAPE_PROFILE_EXTENDED));
    CHECK_EQ_INT(TTAPE_PROFILE_EXTENDED, ext_r.profile);
    CHECK_EQ_INT(3, ext_r.layer_count);
    CHECK_EQ_INT(2, ext_r.fp32_aux_count);
    struct ttape_fp32_aux_tensor_view aview;
    CHECK_EQ_INT(1, ttape_reader_get_fp32_aux(&ext_r, 0u, &aview));
    float f0 = 0.0f;
    CHECK_EQ_INT(1, ttape_reader_read_fp32_elem(&aview, 0u, &f0));
    CHECK_TRUE(f0 == 0.25f);
    CHECK_EQ_INT(1, ttape_reader_is_host_only_ptr(&ext_r, aview.raw_le_f32, 4u));
    CHECK_EQ_INT(0, ttape_reader_is_host_only_ptr(&ext_r, ext_r.weights.data, 1u));

    /* 2. Reject corpus (16 fixtures) */
    struct {
        const char *rel_path;
        ttape_v1_profile_t profile;
        ttape_v1_error_t expected_err;
    } reject_cases[] = {
        {"reject/bad_magic.ttape", TTAPE_PROFILE_CORE, TTAPE_ERR_BAD_MAGIC},
        {"reject/unsupported_version.ttape", TTAPE_PROFILE_CORE, TTAPE_ERR_UNSUPPORTED_VERSION},
        {"reject/bad_section_checksum.ttape", TTAPE_PROFILE_CORE, TTAPE_ERR_CHECKSUM_MISMATCH},
        {"reject/bad_artifact_id.ttape", TTAPE_PROFILE_CORE, TTAPE_ERR_ARTIFACT_ID_MISMATCH},
        {"reject/reserved_ternary_code_0b11.ttape", TTAPE_PROFILE_CORE, TTAPE_ERR_RESERVED_TERNARY_CODE},
        {"reject/nonzero_padding.ttape", TTAPE_PROFILE_CORE, TTAPE_ERR_NONZERO_PADDING},
        {"reject/unknown_flags_bit3.ttape", TTAPE_PROFILE_CORE, TTAPE_ERR_UNKNOWN_FLAGS},
        {"reject/unknown_section_tag.ttape", TTAPE_PROFILE_CORE, TTAPE_ERR_UNKNOWN_TAG},
        {"reject/missing_required_section.ttape", TTAPE_PROFILE_EXTENDED, TTAPE_ERR_MISSING_REQUIRED_SECTION},
        {"reject/malformed_fp32_aux_nan.ttape", TTAPE_PROFILE_EXTENDED, TTAPE_ERR_MALFORMED_FP32_AUX},
        {"reject/core_reader_on_extended.ttape", TTAPE_PROFILE_CORE, TTAPE_ERR_UNSUPPORTED_PROFILE},
        {"reject/core_reader_on_grouped.ttape", TTAPE_PROFILE_CORE, TTAPE_ERR_UNSUPPORTED_PROFILE},
        {"reject/grouped_reader_on_extended.ttape", TTAPE_PROFILE_GROUPED, TTAPE_ERR_UNSUPPORTED_PROFILE},
        {"reject/group_scales_in_core_profile.ttape", TTAPE_PROFILE_GROUPED, TTAPE_ERR_PROFILE_SECTION_FORBIDDEN},
        {"reject/fp32_aux_in_core_profile.ttape", TTAPE_PROFILE_EXTENDED, TTAPE_ERR_PROFILE_SECTION_FORBIDDEN},
        {"reject/invalid_group_size.ttape", TTAPE_PROFILE_GROUPED, TTAPE_ERR_MALFORMED_GROUP_SCALES}
    };
    size_t num_reject = sizeof(reject_cases) / sizeof(reject_cases[0]);
    for (size_t i = 0u; i < num_reject; ++i) {
        size_t rlen = load_fixture(fixtures_dir, reject_cases[i].rel_path, buf);
        CHECK_EQ_INT(
            reject_cases[i].expected_err,
            ttape_v1_validate_tape(buf, rlen, reject_cases[i].profile)
        );
    }

    /* 3. Exhaustive per-byte single-bit mutation sweep across all 4 accept fixtures */
    const char *accept_paths[] = {
        "accept/core_minimal_dot4.ttape",
        "accept/core_with_manifest_and_links.ttape",
        "accept/grouped_with_u16_scales.ttape",
        "accept/extended_with_fp32_aux.ttape"
    };
    uint8_t mut[MAX_TAPE_BYTES];
    for (size_t f = 0u; f < 4u; ++f) {
        size_t alen = load_fixture(fixtures_dir, accept_paths[f], buf);
        CHECK_EQ_INT(TTAPE_OK, ttape_v1_validate_tape(buf, alen, TTAPE_PROFILE_EXTENDED));
        for (size_t idx = 0u; idx < alen; ++idx) {
            for (size_t k = 0u; k < alen; ++k) {
                mut[k] = buf[k];
            }
            mut[idx] ^= 0x01u;
            CHECK_TRUE(ttape_v1_validate_tape(mut, alen, TTAPE_PROFILE_EXTENDED) != TTAPE_OK);
        }
    }
}

static void test_defensive_edge_branches(const char *fixtures_dir) {
    uint8_t base[MAX_TAPE_BYTES];
    uint8_t t[MAX_TAPE_BYTES];
    size_t base_len = load_fixture(fixtures_dir, "accept/core_minimal_dot4.ttape", base);

    CHECK_EQ_INT(TTAPE_ERR_NULL_OR_TRUNCATED, ttape_v1_validate_tape(NULL, 64u, TTAPE_PROFILE_CORE));
    CHECK_EQ_INT(TTAPE_ERR_NULL_OR_TRUNCATED, ttape_v1_validate_tape(base, 16u, TTAPE_PROFILE_CORE));
    CHECK_EQ_INT(TTAPE_ERR_NULL_OR_TRUNCATED, ttape_reader_open(NULL, base, base_len, TTAPE_PROFILE_CORE));

    /* Nonzero header reserved */
    for (size_t i = 0u; i < base_len; ++i) {
        t[i] = base[i];
    }
    t[24] = 1u;
    CHECK_EQ_INT(TTAPE_ERR_NONZERO_RESERVED, ttape_v1_validate_tape(t, base_len, TTAPE_PROFILE_CORE));

    /* section_count < 3 */
    for (size_t i = 0u; i < base_len; ++i) {
        t[i] = base[i];
    }
    t[7] = 2u;
    recompute_tape_checksums(t, base_len, false, true);
    CHECK_EQ_INT(TTAPE_ERR_SECTION_COUNT_MISMATCH, ttape_v1_validate_tape(t, base_len, TTAPE_PROFILE_CORE));

    /* Duplicate tag */
    for (size_t i = 0u; i < base_len; ++i) {
        t[i] = base[i];
    }
    write_u16_le(t + 46u, TTAPE_V1_TAG_WEIGHTS);
    recompute_tape_checksums(t, base_len, true, true);
    CHECK_EQ_INT(TTAPE_ERR_DUPLICATE_TAG, ttape_v1_validate_tape(t, base_len, TTAPE_PROFILE_CORE));

    /* Invalid scale (< -30 and > +30), malformed manifest & links */
    uint8_t good_w[2] = {0x49u, 0x16u};
    uint8_t lo_sc[2] = {0xE1u, 0xFFu}; /* -31 */
    uint8_t hi_sc[2] = {31u, 0u};      /* +31 */
    uint8_t ar[20];
    make_arch1(4u, 4u, 2u, 2u, 0u, 2u, 16u, 1u, 0u, ar);
    size_t tlen = build_3sec_tape(good_w, 2u, lo_sc, 2u, ar, 20u, 0u, t);
    CHECK_EQ_INT(TTAPE_ERR_INVALID_SCALE, ttape_v1_validate_tape(t, tlen, TTAPE_PROFILE_CORE));
    tlen = build_3sec_tape(good_w, 2u, hi_sc, 2u, ar, 20u, 0u, t);
    CHECK_EQ_INT(TTAPE_ERR_INVALID_SCALE, ttape_v1_validate_tape(t, tlen, TTAPE_PROFILE_CORE));

    uint8_t bad_link[4] = {0u, 0u, 0u, 0u};
    CHECK_EQ_INT(TTAPE_ERR_INVALID_LINK, ttape_v1_validate_links(bad_link, 4u));
    uint8_t bad_man[2] = {0u, 0u};
    CHECK_EQ_INT(TTAPE_ERR_MALFORMED_MANIFEST, ttape_v1_validate_manifest(bad_man, 2u));
}

int main(int argc, char **argv) {
    const char *fixtures_dir = (argc > 1) ? argv[1] : "fixtures";
    test_accept_and_reject_corpus(fixtures_dir);
    test_defensive_edge_branches(fixtures_dir);

    if (g_failures != 0) {
        fprintf(stderr, "FAILED: %d check(s) failed in test_ttape_reader.\n", g_failures);
        return 1;
    }
    printf("OK: standalone C99 ttape_reader passed all 20 golden fixtures and mutation sweeps.\n");
    return 0;
}
