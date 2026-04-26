// #include "nn.h"
// #include "model_data.h"
//
// #include <math.h>
// #include <stdint.h>
// #include <string.h>
//
// static float act_relu(float v)
// {
//     return (v > 0.0f) ? v : 0.0f;
// }
//
// static float act_sigmoid(float v)
// {
//     return 1.0f / (1.0f + expf(-v));
// }
//
// static float act_tanh_f(float v)
// {
//     return tanhf(v);
// }
//
// static float act_relu_ext(float v)
// {
//     uint32_t bits;
//     memcpy(&bits, &v, sizeof(bits));
//
//     uint32_t sign = bits >> 31;
//     uint32_t mask = (uint32_t)(sign - 1u);
//     bits &= mask;
//
//     volatile uint32_t acc = bits;
// #if EXP_RELU_REPEAT > 0
//     for (volatile int d = 0; d < EXP_RELU_REPEAT; d++) {
//         acc ^= (uint32_t)(d * 0x6B8B4567u);
//         __asm__ volatile("nop" ::: "memory");
//     }
// #endif
//     (void)acc;
//
//     float out;
//     memcpy(&out, &bits, sizeof(out));
//     return out;
// }
//
// static float apply_act(float v, ActType act)
// {
//     switch (act) {
//         case ACT_SIGMOID:  return act_sigmoid(v);
//         case ACT_TANH:     return act_tanh_f(v);
//         case ACT_RELU_EXT: return act_relu_ext(v);
//         case ACT_RELU:
//         default:           return act_relu(v);
//     }
// }
//
// NN_NOINLINE void mlp_hidden(const float *x, MLPCache *c)
// {
//     for (int h = 0; h < NN_HID; h++) {
//         float acc = model_b1[h];
//         for (int i = 0; i < NN_IN; i++) {
//             acc += model_W1[h][i] * x[i];
//         }
//         c->z1[h] = acc;
//     }
// }
//
// NN_NOINLINE void mlp_activate_range(MLPCache *c, ActType act, const FaultConfig *fc, int start, int stop)
// {
//     if (start < 0) start = 0;
//     if (stop > NN_HID) stop = NN_HID;
//     if (start >= stop) return;
//
//     float prev_a = (start > 0) ? c->a1[start - 1] : 0.0f;
//
//     for (int h = start; h < stop; h++) {
//         float v = c->z1[h];
//         float a;
//
//         switch (fc->model) {
//             case FAULT_SKIP_NEURON:
//                 a = (h == (int)fc->target_neuron) ? v : apply_act(v, act);
//                 break;
//             case FAULT_ZERO_NEURON:
//                 a = (h == (int)fc->target_neuron) ? 0.0f : apply_act(v, act);
//                 break;
//             case FAULT_KEEP_PREV:
//                 a = (h == (int)fc->target_neuron) ? prev_a : apply_act(v, act);
//                 break;
//             case FAULT_PARTIAL_WIN:
//                 a = (h < (int)fc->window_stop) ? apply_act(v, act) : 0.0f;
//                 break;
//             case FAULT_EARLY_TERM:
//                 a = (h < (int)fc->window_stop) ? apply_act(v, act) : v;
//                 break;
//             case FAULT_PERTURB:
//                 a = apply_act(v, act);
//                 if (h == (int)fc->target_neuron) {
//                     a += fc->perturb_delta;
//                 }
//                 break;
//             case FAULT_NONE:
//             default:
//                 a = apply_act(v, act);
//                 break;
//         }
//
//         prev_a = a;
//         c->a1[h] = a;
//     }
// }
//
// NN_NOINLINE void mlp_activate(MLPCache *c, ActType act, const FaultConfig *fc)
// {
//     mlp_activate_range(c, act, fc, 0, NN_HID);
// }
//
// NN_NOINLINE void mlp_output(MLPCache *c)
// {
//     for (int o = 0; o < NN_OUT; o++) {
//         float acc = model_b2[o];
//         for (int h = 0; h < NN_HID; h++) {
//             acc += model_W2[o][h] * c->a1[h];
//         }
//         c->logits[o] = acc;
//     }
// }
//
// NN_NOINLINE int mlp_argmax(const float *logits)
// {
//     int best = 0;
//     float bestv = logits[0];
//     for (int i = 1; i < NN_OUT; i++) {
//         if (logits[i] > bestv) {
//             bestv = logits[i];
//             best = i;
//         }
//     }
//     return best;
// }
//
// NN_NOINLINE int mlp_predict(const float *x, MLPCache *c, ActType act)
// {
//     static const FaultConfig no_fault = FAULT_CONFIG_NONE;
//     mlp_hidden(x, c);
//     mlp_activate(c, act, &no_fault);
//     mlp_output(c);
//     return mlp_argmax(c->logits);
// }
//
// NN_NOINLINE int mlp_predict_faulty(const float *x, MLPCache *c, ActType act, const FaultConfig *fc)
// {
//     mlp_hidden(x, c);
//     mlp_activate(c, act, fc);
//     mlp_output(c);
//     return mlp_argmax(c->logits);
// }
//
// NN_NOINLINE int mlp_predict_embedded(const float *x, MLPCache *c)
// {
//     return mlp_predict(x, c, ACT_RELU);
// }


#include "nn.h"
#include "model_data.h"

#include <math.h>
#include <stdint.h>
#include <string.h>

static float act_relu(float v)
{
    volatile float vv = v;

#if EXP_RELU_REPEAT > 0
    for (volatile int d = 0; d < EXP_RELU_REPEAT; d++) {
        __asm__ volatile("nop" ::: "memory");
    }
#endif

    return (vv > 0.0f) ? vv : 0.0f;
}

static float act_sigmoid(float v)
{
    return 1.0f / (1.0f + expf(-v));
}

static float act_tanh_f(float v)
{
    return tanhf(v);
}

static float act_relu_ext(float v)
{
    uint32_t bits;
    memcpy(&bits, &v, sizeof(bits));

    uint32_t sign = bits >> 31;
    uint32_t mask = (uint32_t)(sign - 1u);
    bits &= mask;

    volatile uint32_t acc = bits;
#if EXP_RELU_REPEAT > 0
    for (volatile int d = 0; d < EXP_RELU_REPEAT; d++) {
        acc ^= (uint32_t)(d * 0x6B8B4567u);
        __asm__ volatile("nop" ::: "memory");
    }
#endif
    (void)acc;

    float out;
    memcpy(&out, &bits, sizeof(out));
    return out;
}

static float apply_act(float v, ActType act)
{
    switch (act) {
        case ACT_SIGMOID:  return act_sigmoid(v);
        case ACT_TANH:     return act_tanh_f(v);
        case ACT_RELU_EXT: return act_relu_ext(v);
        case ACT_RELU:
        default:           return act_relu(v);
    }
}

NN_NOINLINE void mlp_hidden(const float *x, MLPCache *c)
{
    for (int h = 0; h < NN_HID; h++) {
        float acc = model_b1[h];
        for (int i = 0; i < NN_IN; i++) {
            acc += model_W1[h][i] * x[i];
        }
        c->z1[h] = acc;
    }
}

NN_NOINLINE void mlp_activate_range(MLPCache *c, ActType act, const FaultConfig *fc, int start, int stop)
{
    if (start < 0) start = 0;
    if (stop > NN_HID) stop = NN_HID;
    if (start >= stop) return;

    float prev_a = (start > 0) ? c->a1[start - 1] : 0.0f;

    for (int h = start; h < stop; h++) {
        float v = c->z1[h];
        float a;

        switch (fc->model) {
            case FAULT_SKIP_NEURON:
                a = (h == (int)fc->target_neuron) ? v : apply_act(v, act);
                break;
            case FAULT_ZERO_NEURON:
                a = (h == (int)fc->target_neuron) ? 0.0f : apply_act(v, act);
                break;
            case FAULT_KEEP_PREV:
                a = (h == (int)fc->target_neuron) ? prev_a : apply_act(v, act);
                break;
            case FAULT_PARTIAL_WIN:
                a = (h < (int)fc->window_stop) ? apply_act(v, act) : 0.0f;
                break;
            case FAULT_EARLY_TERM:
                a = (h < (int)fc->window_stop) ? apply_act(v, act) : v;
                break;
            case FAULT_PERTURB:
                a = apply_act(v, act);
                if (h == (int)fc->target_neuron) {
                    a += fc->perturb_delta;
                }
                break;
            case FAULT_NONE:
            default:
                a = apply_act(v, act);
                break;
        }

        prev_a = a;
        c->a1[h] = a;
    }
}

NN_NOINLINE void mlp_activate(MLPCache *c, ActType act, const FaultConfig *fc)
{
    mlp_activate_range(c, act, fc, 0, NN_HID);
}

NN_NOINLINE void mlp_output(MLPCache *c)
{
    for (int o = 0; o < NN_OUT; o++) {
        float acc = model_b2[o];
        for (int h = 0; h < NN_HID; h++) {
            acc += model_W2[o][h] * c->a1[h];
        }
        c->logits[o] = acc;
    }
}

NN_NOINLINE int mlp_argmax(const float *logits)
{
    int best = 0;
    float bestv = logits[0];
    for (int i = 1; i < NN_OUT; i++) {
        if (logits[i] > bestv) {
            bestv = logits[i];
            best = i;
        }
    }
    return best;
}

NN_NOINLINE int mlp_predict(const float *x, MLPCache *c, ActType act)
{
    static const FaultConfig no_fault = FAULT_CONFIG_NONE;
    mlp_hidden(x, c);
    mlp_activate(c, act, &no_fault);
    mlp_output(c);
    return mlp_argmax(c->logits);
}

NN_NOINLINE int mlp_predict_faulty(const float *x, MLPCache *c, ActType act, const FaultConfig *fc)
{
    mlp_hidden(x, c);
    mlp_activate(c, act, fc);
    mlp_output(c);
    return mlp_argmax(c->logits);
}

NN_NOINLINE int mlp_predict_embedded(const float *x, MLPCache *c)
{
    return mlp_predict(x, c, ACT_RELU);
}
