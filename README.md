# Practical Fault Attacks on Neural Network Implementations

This repository contains experiments for my bachelor thesis. The focus is on practical clock glitch attacks on a neural network running on an STM32 microcontroller using ChipWhisperer.

## What this project is about

The goal was to check how hardware faults (clock glitches) can affect neural network inference.

I tested several activation functions (ReLU, sigmoid, tanh), but the main focus is on ReLU, where I was able to observe and explain a specific fault using internal network values.

## Structure


experiment/
ReluExperiment.ipynb # main experiment focused on ReLU
allActivationsExperiment.ipynb # comparison between different activations

firmware/
nn-mnist.c # SimpleSerial interface for ChipWhisperer
nn.c / nn.h # neural network implementation
model_data_*.c # weights for each activation
Makefile # build configuration

results/
reluResults/ # results for ReLU experiment
allActivationsResults/ # comparison results


## Main result (ReLU)

In normal execution (no glitch), ReLU behaves as expected:


a1 = max(z1, 0)


But under a specific clock glitch, I observed a violation of this rule for one neuron:


z1[16] = -3.916523
expected a1[16] = 0
actual a1[16] = -3.916523


So the negative value was not zeroed. This error then propagated through the network and changed the final prediction.

## Important notes

- Final ReLU experiment uses `FAULT_NONE` (no software fault)
- All effects are caused by hardware clock glitching
- ReLU execution was slightly extended (`EXP_RELU_REPEAT=32`) to make timing easier to target
- Firmware is intended to be used inside the ChipWhisperer environment

## Build example


make PLATFORM=CWLITEARM CRYPTO_TARGET=NONE SS_VER=SS_VER_2_1 ACTIVATION=relu EXPERIMENT=1 EXP_RELU_REPEAT=32


## Hardware

- ChipWhisperer-Lite
- STM32F3 target
- SimpleSerial2 communication
