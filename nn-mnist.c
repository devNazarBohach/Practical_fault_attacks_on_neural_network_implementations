#include <stdint.h>
#include <string.h>

#include "hal.h"
#include "simpleserial.h"
#include "nn.h"

#ifndef EXPERIMENT
#  define EXPERIMENT 0
#endif
#ifndef EXP_RELU_WIN_START
#  define EXP_RELU_WIN_START 24
#endif
#ifndef EXP_RELU_WIN_STOP
#  define EXP_RELU_WIN_STOP 40
#endif

#define SS_VER SS_VER_2_1

#define INPUT_BYTES 784
#define UPLOAD_CHUNK_BYTES 48
#define UPLOAD_FRAME_BYTES 49
#define LOGITS_BYTES (NN_OUT * sizeof(float))
#define RESULT_PACKET_BYTES (8 + LOGITS_BYTES)

#define SNAP_CHUNK_BYTES 32u
#define SNAP_TOTAL_BYTES (NN_HID * sizeof(float))
#define SNAP_NUM_CHUNKS (SNAP_TOTAL_BYTES / SNAP_CHUNK_BYTES)

#define SNAP_ARRAY_LAST_Z1 0u
#define SNAP_ARRAY_LAST_A1 1u
#define SNAP_ARRAY_BASE_Z1 2u
#define SNAP_ARRAY_BASE_A1 3u

#define RELU_SUMMARY_BYTES 32u

#define PKT_MAGIC0 0xA5
#define PKT_MAGIC1 0x5A

#define RESULT_KIND_BASELINE 0u
#define RESULT_KIND_INFER 1u

#define RESULT_FLAG_INPUT_READY (1u << 0)
#define RESULT_FLAG_HAVE_BASELINE (1u << 1)
#define RESULT_FLAG_HAVE_LAST (1u << 2)
#define RESULT_FLAG_FAULT_ACTIVE (1u << 3)

typedef enum {
    TRIGMODE_ACTIVATION_FULL = 0,
    TRIGMODE_ACTIVATION_WINDOW = 1,
    TRIGMODE_FULL_NETWORK = 2,
} TriggerMode;

static uint8_t g_input_u8[INPUT_BYTES];
static uint16_t g_loaded = 0;
static uint8_t g_input_ready = 0;

static ActType g_act_type = ACT_RELU;
static FaultConfig g_fault = FAULT_CONFIG_NONE;
static TriggerMode g_trigger_mode = TRIGMODE_ACTIVATION_FULL;
static uint8_t g_trigger_start = EXP_RELU_WIN_START;
static uint8_t g_trigger_stop = EXP_RELU_WIN_STOP;

static MLPCache g_cache;
static float g_baseline_logits[NN_OUT];
static float g_baseline_a1[NN_HID];
static uint8_t g_have_baseline_a1 = 0;
static float g_last_a1[NN_HID];
static float g_last_z1[NN_HID];
static float g_last_logits[NN_OUT];
static uint8_t g_have_baseline = 0;
static uint8_t g_have_last = 0;
static uint8_t g_have_last_a1 = 0;
static uint8_t g_have_last_z1 = 0;
static float   g_baseline_z1[NN_HID];
static uint8_t g_have_baseline_z1 = 0;
static uint8_t g_baseline_pred = 0xFF;
static uint8_t g_last_pred = 0xFF;
static uint8_t g_result_seq = 0;

static void clear_input_state(void)
{
    memset(g_input_u8, 0, sizeof(g_input_u8));
    g_loaded = 0;
    g_input_ready = 0;
}

static void clear_runtime_state(void)
{
    memset(g_baseline_logits, 0, sizeof(g_baseline_logits));
    memset(g_last_logits, 0, sizeof(g_last_logits));
    memset(g_baseline_a1, 0, sizeof(g_baseline_a1));
    memset(g_last_a1, 0, sizeof(g_last_a1));
    memset(g_last_z1, 0, sizeof(g_last_z1));
    memset(g_baseline_z1, 0, sizeof(g_baseline_z1));
    g_have_baseline = 0;
    g_have_last = 0;
    g_have_baseline_a1 = 0;
    g_have_baseline_z1 = 0;
    g_have_last_a1 = 0;
    g_have_last_z1 = 0;
    g_baseline_pred = 0xFF;
    g_last_pred = 0xFF;
    g_result_seq = 0;
}

static void clamp_trigger_window(void)
{
    if (g_trigger_start > NN_HID) g_trigger_start = NN_HID;
    if (g_trigger_stop > NN_HID) g_trigger_stop = NN_HID;
    if (g_trigger_start > g_trigger_stop) g_trigger_start = g_trigger_stop;
}

static void u8_to_float(const uint8_t *src, float *dst)
{
    for (int i = 0; i < NN_IN; i++) {
        dst[i] = (float)src[i] / 255.0f;
    }
}

static void copy_logits(float *dst, const float *src)
{
    for (int i = 0; i < NN_OUT; i++) {
        dst[i] = src[i];
    }
}

static void cache_baseline_from_current_cache(uint8_t pred)
{
    g_baseline_pred = pred;
    copy_logits(g_baseline_logits, g_cache.logits);
    for (int i = 0; i < NN_HID; i++) {
        g_baseline_a1[i] = g_cache.a1[i];
        g_baseline_z1[i] = g_cache.z1[i];
    }
    g_have_baseline_a1 = 1;
    g_have_baseline_z1 = 1;
    g_have_baseline = 1;
}

static void cache_last_from_current_cache(uint8_t pred)
{
    g_last_pred = pred;
    copy_logits(g_last_logits, g_cache.logits);
    for (int i = 0; i < NN_HID; i++) {
        g_last_a1[i] = g_cache.a1[i];
        g_last_z1[i] = g_cache.z1[i];
    }
    g_have_last = 1;
    g_have_last_a1 = 1;
    g_have_last_z1 = 1;
}


static float f_abs32(float x)
{
    return (x < 0.0f) ? -x : x;
}

static void emit_result_packet(uint8_t kind, uint8_t seq, uint8_t pred, uint8_t fault_model, const float *logits)
{
    uint8_t resp[RESULT_PACKET_BYTES];
    uint8_t flags = 0;

    if (g_input_ready) flags |= RESULT_FLAG_INPUT_READY;
    if (g_have_baseline) flags |= RESULT_FLAG_HAVE_BASELINE;
    if (g_have_last) flags |= RESULT_FLAG_HAVE_LAST;
    if (fault_model != (uint8_t)FAULT_NONE) flags |= RESULT_FLAG_FAULT_ACTIVE;

    resp[0] = PKT_MAGIC0;
    resp[1] = PKT_MAGIC1;
    resp[2] = kind;
    resp[3] = seq;
    resp[4] = pred;
    resp[5] = (uint8_t)g_act_type;
    resp[6] = fault_model;
    resp[7] = flags;
    memcpy(&resp[8], logits, LOGITS_BYTES);
    simpleserial_put('r', RESULT_PACKET_BYTES, resp);
}

static void emit_zero_result_packet(uint8_t kind, uint8_t fault_model)
{
    float zeros[NN_OUT] = {0};
    emit_result_packet(kind, 0, 0xFF, fault_model, zeros);
}

static uint8_t run_baseline(void)
{
    float x[NN_IN];
    u8_to_float(g_input_u8, x);
    return (uint8_t)mlp_predict(x, &g_cache, g_act_type);
}

static uint8_t run_experiment(void)
{
    float x[NN_IN];

#if EXPERIMENT
    if (g_trigger_mode == TRIGMODE_FULL_NETWORK) {
        trigger_high();

        u8_to_float(g_input_u8, x);
        mlp_hidden(x, &g_cache);
        mlp_activate(&g_cache, g_act_type, &g_fault);
        mlp_output(&g_cache);

        uint8_t pred = (uint8_t)mlp_argmax(g_cache.logits);

        trigger_low();
        return pred;
    }
#endif

    u8_to_float(g_input_u8, x);
    mlp_hidden(x, &g_cache);

#if EXPERIMENT
    if (g_trigger_mode == TRIGMODE_ACTIVATION_WINDOW) {
        clamp_trigger_window();

        mlp_activate_range(&g_cache, g_act_type, &g_fault, 0, (int)g_trigger_start);

        trigger_high();
        mlp_activate_range(&g_cache, g_act_type, &g_fault,
                           (int)g_trigger_start, (int)g_trigger_stop);
        trigger_low();

        mlp_activate_range(&g_cache, g_act_type, &g_fault,
                           (int)g_trigger_stop, NN_HID);
    } else {
        trigger_high();
        mlp_activate(&g_cache, g_act_type, &g_fault);
        trigger_low();
    }
#else
    mlp_activate(&g_cache, g_act_type, &g_fault);
#endif

    mlp_output(&g_cache);
    return (uint8_t)mlp_argmax(g_cache.logits);
}

static uint8_t cmd_ping(uint8_t cmd, uint8_t scmd, uint8_t len, uint8_t *buf)
{
    (void)cmd; (void)scmd; (void)len; (void)buf;
    uint8_t resp[4] = { 0xDE, 0xAD, 0xBE, 0xEF };
    simpleserial_put('r', 4, resp);
    return 0;
}

static uint8_t cmd_clear(uint8_t cmd, uint8_t scmd, uint8_t len, uint8_t *buf)
{
    (void)cmd; (void)scmd; (void)len; (void)buf;
    clear_input_state();
    clear_runtime_state();
    uint8_t resp[2] = { 0xAA, 0x55 };
    simpleserial_put('r', 2, resp);
    return 0;
}

static uint8_t cmd_set_act(uint8_t cmd, uint8_t scmd, uint8_t len, uint8_t *buf)
{
    (void)cmd; (void)scmd;
    if (len >= 1) {
        uint8_t requested = buf[0];
        if (requested <= (uint8_t)ACT_RELU_EXT) {
            g_act_type = (ActType)requested;
        }
    }
    uint8_t resp[1] = { (uint8_t)g_act_type };
    simpleserial_put('r', 1, resp);
    return 0;
}

static uint8_t cmd_set_fault(uint8_t cmd, uint8_t scmd, uint8_t len, uint8_t *buf)
{
    (void)cmd; (void)scmd;
    if (len >= 7) {
        uint8_t model = buf[0];
        if (model <= (uint8_t)FAULT_PERTURB) {
            g_fault.model = (FaultModel)model;
            g_fault.target_neuron = buf[1];
            g_fault.window_stop = buf[2];

            union { uint32_t u; float f; } delta;
            delta.u = (uint32_t)buf[3]
                    | ((uint32_t)buf[4] <<  8)
                    | ((uint32_t)buf[5] << 16)
                    | ((uint32_t)buf[6] << 24);
            g_fault.perturb_delta = delta.f;
        }
    }
    uint8_t resp[1] = { (uint8_t)g_fault.model };
    simpleserial_put('r', 1, resp);
    return 0;
}

static uint8_t cmd_set_trigger_window(uint8_t cmd, uint8_t scmd, uint8_t len, uint8_t *buf)
{
    (void)cmd; (void)scmd;
    if (len >= 3) {
        uint8_t requested_mode = buf[0];
        if (requested_mode <= (uint8_t)TRIGMODE_FULL_NETWORK) {
            g_trigger_mode = (TriggerMode)requested_mode;
        }
        g_trigger_start = buf[1];
        g_trigger_stop  = buf[2];
        clamp_trigger_window();
    }
    uint8_t resp[3] = { (uint8_t)g_trigger_mode, g_trigger_start, g_trigger_stop };
    simpleserial_put('r', 3, resp);
    return 0;
}

static uint8_t cmd_upload(uint8_t cmd, uint8_t scmd, uint8_t len, uint8_t *buf)
{
    (void)cmd; (void)scmd;
    uint8_t resp[2];
    if (len != UPLOAD_FRAME_BYTES) {
        resp[0] = (uint8_t)(g_loaded & 0xFF);
        resp[1] = (uint8_t)((g_loaded >> 8) & 0xFF);
        simpleserial_put('r', 2, resp);
        return 0;
    }
    uint8_t chunk_idx = buf[0];
    uint16_t start = (uint16_t)chunk_idx * UPLOAD_CHUNK_BYTES;
    if (start < INPUT_BYTES) {
        uint16_t remaining = INPUT_BYTES - start;
        uint16_t copy_len = (remaining >= UPLOAD_CHUNK_BYTES) ? UPLOAD_CHUNK_BYTES : remaining;
        memcpy(&g_input_u8[start], &buf[1], copy_len);
        uint16_t end = start + copy_len;
        if (end > g_loaded) g_loaded = end;
        if (g_loaded >= INPUT_BYTES) g_input_ready = 1;
    }
    resp[0] = (uint8_t)(g_loaded & 0xFF);
    resp[1] = (uint8_t)((g_loaded >> 8) & 0xFF);
    simpleserial_put('r', 2, resp);
    return 0;
}

static uint8_t cmd_baseline(uint8_t cmd, uint8_t scmd, uint8_t len, uint8_t *buf)
{
    (void)cmd; (void)scmd; (void)len; (void)buf;
    if (!g_input_ready) {
        uint8_t zero[LOGITS_BYTES] = { 0 };
        simpleserial_put('r', LOGITS_BYTES, zero);
        return 0;
    }
    cache_baseline_from_current_cache(run_baseline());
    simpleserial_put('r', LOGITS_BYTES, (uint8_t *)g_baseline_logits);
    return 0;
}

static uint8_t cmd_baseline_packet(uint8_t cmd, uint8_t scmd, uint8_t len, uint8_t *buf)
{
    (void)cmd; (void)scmd; (void)len; (void)buf;
    if (!g_input_ready) {
        emit_zero_result_packet(RESULT_KIND_BASELINE, (uint8_t)FAULT_NONE);
        return 0;
    }
    cache_baseline_from_current_cache(run_baseline());
    g_result_seq = (uint8_t)(g_result_seq + 1u);
    emit_result_packet(RESULT_KIND_BASELINE, g_result_seq, g_baseline_pred, (uint8_t)FAULT_NONE, g_baseline_logits);
    return 0;
}

static uint8_t cmd_infer(uint8_t cmd, uint8_t scmd, uint8_t len, uint8_t *buf)
{
    (void)cmd; (void)scmd; (void)len; (void)buf;
    uint8_t resp[1] = { 0xFF };
    if (!g_input_ready) {
        simpleserial_put('r', 1, resp);
        return 0;
    }
    cache_last_from_current_cache(run_experiment());
    resp[0] = g_last_pred;
    simpleserial_put('r', 1, resp);
    return 0;
}

static uint8_t cmd_infer_packet(uint8_t cmd, uint8_t scmd, uint8_t len, uint8_t *buf)
{
    (void)cmd; (void)scmd; (void)len; (void)buf;
    if (!g_input_ready) {
        emit_zero_result_packet(RESULT_KIND_INFER, (uint8_t)g_fault.model);
        return 0;
    }
    cache_last_from_current_cache(run_experiment());
    g_result_seq = (uint8_t)(g_result_seq + 1u);
    emit_result_packet(RESULT_KIND_INFER, g_result_seq, g_last_pred, (uint8_t)g_fault.model, g_last_logits);
    return 0;
}

static uint8_t cmd_last_logits(uint8_t cmd, uint8_t scmd, uint8_t len, uint8_t *buf)
{
    (void)cmd; (void)scmd; (void)len; (void)buf;
    if (!g_have_last) {
        uint8_t zero[LOGITS_BYTES] = { 0 };
        simpleserial_put('r', LOGITS_BYTES, zero);
        return 0;
    }
    simpleserial_put('r', LOGITS_BYTES, (uint8_t *)g_last_logits);
    return 0;
}

static uint8_t cmd_last_act_lo(uint8_t cmd, uint8_t scmd, uint8_t len, uint8_t *buf)
{
    (void)cmd; (void)scmd; (void)len; (void)buf;
    if (!g_have_last) {
        uint8_t zero[128] = { 0 };
        simpleserial_put('r', 128, zero);
        return 0;
    }
    simpleserial_put('r', 128, (uint8_t *)g_last_a1);
    return 0;
}

static uint8_t cmd_last_z_lo(uint8_t cmd, uint8_t scmd, uint8_t len, uint8_t *buf)
{
    (void)cmd; (void)scmd; (void)len; (void)buf;
    if (!g_have_last) {
        uint8_t zero[128] = { 0 };
        simpleserial_put('r', 128, zero);
        return 0;
    }
    simpleserial_put('r', 128, (uint8_t *)g_last_z1);
    return 0;
}

static uint8_t cmd_last_z_hi(uint8_t cmd, uint8_t scmd, uint8_t len, uint8_t *buf)
{
    (void)cmd; (void)scmd; (void)len; (void)buf;
    if (!g_have_last) {
        uint8_t zero[128] = { 0 };
        simpleserial_put('r', 128, zero);
        return 0;
    }
    simpleserial_put('r', 128, (uint8_t *)g_last_z1 + 128);
    return 0;
}

static uint8_t cmd_baseline_z_lo(uint8_t cmd, uint8_t scmd, uint8_t len, uint8_t *buf)
{
    (void)cmd; (void)scmd; (void)len; (void)buf;
    if (!g_have_baseline_z1) {
        uint8_t zero[128] = { 0 };
        simpleserial_put('r', 128, zero);
        return 0;
    }
    simpleserial_put('r', 128, (uint8_t *)g_baseline_z1);
    return 0;
}

static uint8_t cmd_baseline_z_hi(uint8_t cmd, uint8_t scmd, uint8_t len, uint8_t *buf)
{
    (void)cmd; (void)scmd; (void)len; (void)buf;
    if (!g_have_baseline_z1) {
        uint8_t zero[128] = { 0 };
        simpleserial_put('r', 128, zero);
        return 0;
    }
    simpleserial_put('r', 128, (uint8_t *)g_baseline_z1 + 128);
    return 0;
}

static uint8_t cmd_baseline_logits_read(uint8_t cmd, uint8_t scmd, uint8_t len, uint8_t *buf)
{
    (void)cmd; (void)scmd; (void)len; (void)buf;
    if (!g_have_baseline) {
        uint8_t zero[LOGITS_BYTES] = { 0 };
        simpleserial_put('r', LOGITS_BYTES, zero);
        return 0;
    }
    simpleserial_put('r', LOGITS_BYTES, (uint8_t *)g_baseline_logits);
    return 0;
}

static uint8_t cmd_last_act_hi(uint8_t cmd, uint8_t scmd, uint8_t len, uint8_t *buf)
{
    (void)cmd; (void)scmd; (void)len; (void)buf;
    if (!g_have_last) {
        uint8_t zero[128] = { 0 };
        simpleserial_put('r', 128, zero);
        return 0;
    }
    simpleserial_put('r', 128, (uint8_t *)g_last_a1 + 128);
    return 0;
}

static uint8_t cmd_baseline_act_lo(uint8_t cmd, uint8_t scmd, uint8_t len, uint8_t *buf)
{
    (void)cmd; (void)scmd; (void)len; (void)buf;
    if (!g_have_baseline_a1) {
        uint8_t zero[128] = { 0 };
        simpleserial_put('r', 128, zero);
        return 0;
    }
    simpleserial_put('r', 128, (uint8_t *)g_baseline_a1);
    return 0;
}

static uint8_t cmd_baseline_act_hi(uint8_t cmd, uint8_t scmd, uint8_t len, uint8_t *buf)
{
    (void)cmd; (void)scmd; (void)len; (void)buf;
    if (!g_have_baseline_a1) {
        uint8_t zero[128] = { 0 };
        simpleserial_put('r', 128, zero);
        return 0;
    }
    simpleserial_put('r', 128, (uint8_t *)g_baseline_a1 + 128);
    return 0;
}


static const uint8_t *snapshot_array_ptr(uint8_t array_id, uint8_t *available)
{
    *available = 0;

    switch (array_id) {
    case SNAP_ARRAY_LAST_Z1:
        *available = g_have_last_z1;
        return (const uint8_t *)g_last_z1;
    case SNAP_ARRAY_LAST_A1:
        *available = g_have_last_a1;
        return (const uint8_t *)g_last_a1;
    case SNAP_ARRAY_BASE_Z1:
        *available = g_have_baseline_z1;
        return (const uint8_t *)g_baseline_z1;
    case SNAP_ARRAY_BASE_A1:
        *available = g_have_baseline_a1;
        return (const uint8_t *)g_baseline_a1;
    default:
        *available = 0;
        return 0;
    }
}

static uint8_t cmd_snapshot_chunk32(uint8_t cmd, uint8_t scmd, uint8_t len, uint8_t *buf)
{
    (void)cmd; (void)scmd;
    uint8_t zero[SNAP_CHUNK_BYTES] = { 0 };

    if (len < 2 || buf == 0) {
        simpleserial_put('r', SNAP_CHUNK_BYTES, zero);
        return 0;
    }

    uint8_t array_id = buf[0];
    uint8_t chunk_id = buf[1];

    if (chunk_id >= SNAP_NUM_CHUNKS) {
        simpleserial_put('r', SNAP_CHUNK_BYTES, zero);
        return 0;
    }

    uint8_t available = 0;
    const uint8_t *base = snapshot_array_ptr(array_id, &available);
    if (!available || base == 0) {
        simpleserial_put('r', SNAP_CHUNK_BYTES, zero);
        return 0;
    }

    simpleserial_put('r', SNAP_CHUNK_BYTES, base + ((uint16_t)chunk_id * SNAP_CHUNK_BYTES));
    return 0;
}

static uint8_t cmd_relu_summary(uint8_t cmd, uint8_t scmd, uint8_t len, uint8_t *buf)
{
    (void)cmd; (void)scmd; (void)len; (void)buf;

    uint8_t resp[RELU_SUMMARY_BYTES];
    memset(resp, 0, sizeof(resp));

    if (!g_have_last_z1 || !g_have_last_a1) {
        resp[0] = 0u; /* invalid */
        simpleserial_put('r', RELU_SUMMARY_BYTES, resp);
        return 0;
    }

    const float eps = 1.0e-6f;
    const float err_eps = 1.0e-5f;

    uint8_t n_viol = 0;
    uint8_t n_viol_win = 0;
    uint8_t first_bad = 0xFFu;
    uint8_t n_neg_not_zeroed = 0;
    uint8_t n_pos_zeroed = 0;
    uint8_t n_wrong_value = 0;

    float max_err = 0.0f;
    float first_z = 0.0f;
    float first_a = 0.0f;
    float first_expected = 0.0f;
    float max_z1_delta_to_baseline = 0.0f;

    for (uint8_t i = 0; i < NN_HID; i++) {
        float z = g_last_z1[i];
        float a = g_last_a1[i];
        float expected = (z > 0.0f) ? z : 0.0f;
        float err = f_abs32(a - expected);

        uint8_t neg_not_zeroed = (z < -eps && f_abs32(a) > eps);
        uint8_t pos_zeroed = (z > eps && f_abs32(a) < eps);
        uint8_t wrong_value = (!neg_not_zeroed && !pos_zeroed && err > err_eps);
        uint8_t violated = (neg_not_zeroed || pos_zeroed || wrong_value);

        if (g_have_baseline_z1) {
            float dz = f_abs32(g_last_z1[i] - g_baseline_z1[i]);
            if (dz > max_z1_delta_to_baseline) {
                max_z1_delta_to_baseline = dz;
            }
        }

        if (violated) {
            if (n_viol < 255u) n_viol++;
            if (i >= g_trigger_start && i < g_trigger_stop) {
                if (n_viol_win < 255u) n_viol_win++;
            }
            if (neg_not_zeroed && n_neg_not_zeroed < 255u) n_neg_not_zeroed++;
            if (pos_zeroed && n_pos_zeroed < 255u) n_pos_zeroed++;
            if (wrong_value && n_wrong_value < 255u) n_wrong_value++;

            if (err > max_err) {
                max_err = err;
            }

            if (first_bad == 0xFFu) {
                first_bad = i;
                first_z = z;
                first_a = a;
                first_expected = expected;
            }
        }
    }

    resp[0] = 1u; /* valid */
    resp[1] = n_viol;
    resp[2] = n_viol_win;
    resp[3] = first_bad;
    memcpy(resp + 4,  &max_err, sizeof(float));
    memcpy(resp + 8,  &first_z, sizeof(float));
    memcpy(resp + 12, &first_a, sizeof(float));
    memcpy(resp + 16, &first_expected, sizeof(float));
    memcpy(resp + 20, &max_z1_delta_to_baseline, sizeof(float));
    resp[24] = n_neg_not_zeroed;
    resp[25] = n_pos_zeroed;
    resp[26] = n_wrong_value;
    resp[27] = g_trigger_start;
    resp[28] = g_trigger_stop;
    resp[29] = g_last_pred;
    resp[30] = g_baseline_pred;
    resp[31] = g_result_seq;

    simpleserial_put('r', RELU_SUMMARY_BYTES, resp);
    return 0;
}

static uint8_t cmd_status(uint8_t cmd, uint8_t scmd, uint8_t len, uint8_t *buf)
{
    (void)cmd; (void)scmd; (void)len; (void)buf;
    uint8_t resp[16];
    resp[0] = (uint8_t)g_act_type;
    resp[1] = (uint8_t)g_fault.model;
    resp[2] = g_fault.target_neuron;
    resp[3] = g_fault.window_stop;
    resp[4] = g_input_ready;
    resp[5] = g_have_baseline;
    resp[6] = g_have_last;
    resp[7] = g_baseline_pred;
    resp[8] = g_last_pred;
    resp[9] = (uint8_t)(g_loaded & 0xFF);
    resp[10] = (uint8_t)((g_loaded >> 8) & 0xFF);
    resp[11] = g_result_seq;
    resp[12] = (uint8_t)g_trigger_mode;
    resp[13] = g_trigger_start;
    resp[14] = g_trigger_stop;
    resp[15] = 0;
    if (g_have_baseline_a1) resp[15] |= 0x01;
    if (g_have_baseline_z1) resp[15] |= 0x02;
    if (g_have_last_a1)     resp[15] |= 0x04;
    if (g_have_last_z1)     resp[15] |= 0x08;
    simpleserial_put('r', 16, resp);
    return 0;
}

int main(void)
{
    platform_init();
    init_uart();
    trigger_setup();

    clear_input_state();
    clear_runtime_state();
    clamp_trigger_window();

    simpleserial_init();

    simpleserial_addcmd('z', 0, cmd_ping);
    simpleserial_addcmd('c', 0, cmd_clear);
    simpleserial_addcmd('a', 1, cmd_set_act);
    simpleserial_addcmd('f', 7, cmd_set_fault);
    simpleserial_addcmd('t', 3, cmd_set_trigger_window);
    simpleserial_addcmd('x', UPLOAD_FRAME_BYTES, cmd_upload);
    simpleserial_addcmd('b', 0, cmd_baseline);
    simpleserial_addcmd('n', 0, cmd_baseline_packet);
    simpleserial_addcmd('i', 0, cmd_infer);
    simpleserial_addcmd('g', 0, cmd_infer_packet);
    simpleserial_addcmd('o', 0, cmd_last_logits);
    simpleserial_addcmd('s', 0, cmd_status);
    simpleserial_addcmd('h', 0, cmd_last_act_lo);
    simpleserial_addcmd('j', 0, cmd_last_act_hi);
    simpleserial_addcmd('q', 0, cmd_baseline_act_lo);
    simpleserial_addcmd('u', 0, cmd_baseline_act_hi);
    simpleserial_addcmd('k', 0, cmd_last_z_lo);
    simpleserial_addcmd('l', 0, cmd_last_z_hi);
    simpleserial_addcmd('m', 0, cmd_baseline_z_lo);
    simpleserial_addcmd('Y', 0, cmd_baseline_z_hi);
    simpleserial_addcmd('d', 0, cmd_baseline_logits_read);

    // Robust post-glitch evidence commands:
    //  'w' reads 32-byte chunks of z1/a1 snapshots.
    //  'V' returns compact on-device ReLU violation summary.
    simpleserial_addcmd('W', 2, cmd_snapshot_chunk32);
    simpleserial_addcmd('V', 0, cmd_relu_summary);

    while (1) {
        simpleserial_get();
    }
}
