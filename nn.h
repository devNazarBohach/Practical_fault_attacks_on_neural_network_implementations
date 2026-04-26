/*
 * nn.h — Math-only MLP header
 */
#ifndef NN_H
#define NN_H

#include <stdint.h>

#define IMG_W 28
#define IMG_H 28
#define NN_IN (IMG_W * IMG_H)
#define NN_HID 64
#define NN_OUT 10

#ifndef EXP_RELU_REPEAT
#  define EXP_RELU_REPEAT 8
#endif

typedef enum {
    ACT_RELU = 0,
    ACT_SIGMOID = 1,
    ACT_TANH = 2,
    ACT_RELU_EXT = 3,
} ActType;

typedef enum {
    FAULT_NONE = 0,
    FAULT_SKIP_NEURON = 1,
    FAULT_ZERO_NEURON = 2,
    FAULT_KEEP_PREV = 3,
    FAULT_PARTIAL_WIN = 4,
    FAULT_EARLY_TERM = 5,
    FAULT_PERTURB = 6,
} FaultModel;

typedef struct {
    FaultModel model;
    uint8_t target_neuron;
    uint8_t window_stop;
    float perturb_delta;
} FaultConfig;

#define FAULT_CONFIG_NONE  { FAULT_NONE, 0, 0, 0.0f }

typedef struct {
    float z1[NN_HID];
    float a1[NN_HID];
    float logits[NN_OUT];
} MLPCache;

#if defined(__GNUC__)
#  define NN_NOINLINE __attribute__((noinline))
#else
#  define NN_NOINLINE
#endif

NN_NOINLINE void mlp_hidden(const float *x, MLPCache *c);
NN_NOINLINE void mlp_activate(MLPCache *c, ActType act, const FaultConfig *fc);
NN_NOINLINE void mlp_activate_range(MLPCache *c, ActType act, const FaultConfig *fc, int start, int stop);
NN_NOINLINE void mlp_output(MLPCache *c);
NN_NOINLINE int mlp_argmax(const float *logits);
NN_NOINLINE int mlp_predict(const float *x, MLPCache *c, ActType act);
NN_NOINLINE int mlp_predict_faulty(const float *x, MLPCache *c, ActType act, const FaultConfig *fc);
NN_NOINLINE int mlp_predict_embedded(const float *x, MLPCache *c);

#endif
