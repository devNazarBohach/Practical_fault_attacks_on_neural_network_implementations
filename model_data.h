#ifndef MODEL_DATA_H
#define MODEL_DATA_H

#include "nn.h"

extern const float model_W1[NN_HID][NN_IN];
extern const float model_b1[NN_HID];
extern const float model_W2[NN_OUT][NN_HID];
extern const float model_b2[NN_OUT];

#endif