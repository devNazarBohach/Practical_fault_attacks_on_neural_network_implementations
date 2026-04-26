# Practical Fault Attacks on Neural Network Implementations

This repository contains experiments for a bachelor thesis focused on practical clock-glitch fault attacks against neural network implementations on an embedded STM32 target using ChipWhisperer.

## Project Overview

The main goal is to study how hardware clock glitches can affect neural network inference, especially the ReLU activation function. The experiments compare several activation functions and then analyze one ReLU-specific fault case in detail using internal network snapshots.

## Repository Structure

```text
experiment/
  ReluExperiment.ipynb              # Detailed ReLU fault experiment
  allActivationsExperiment.ipynb    # Comparison across activation functions

firmware/
  nn-mnist.c                        # SimpleSerial firmware interface
  nn.c / nn.h                       # Neural network implementation
  model_data_*.c                    # Weights for different activation functions
  Makefile                          # Build configuration for ChipWhisperer firmware

results/
  reluResults/                      # ReLU experiment outputs
  allActivationsResults/            # Cross-activation comparison outputs
Main ReLU Result

The final ReLU experiment demonstrates a snapshot-backed ReLU fault. In the no-glitch control, the ReLU output follows:

a1 = max(z1, 0)

Under the targeted clock glitch, one neuron inside the selected ReLU trigger window violates this rule:

z1[16] = -3.916523
expected a1[16] = 0
glitched a1[16] = -3.916523

This changes the network logits and flips the prediction from class 9 to class 4.

Important Notes
Final ReLU attack uses FAULT_NONE.
Software fault models are included only for debugging and comparison.
The ReLU operation is temporally extended with EXP_RELU_REPEAT=32 to make the hardware fault window observable.
The firmware is intended to be built inside the ChipWhisperer firmware environment.
Example Build Command
make PLATFORM=CWLITEARM CRYPTO_TARGET=NONE SS_VER=SS_VER_2_1 ACTIVATION=relu EXPERIMENT=1 EXP_RELU_REPEAT=32
Hardware Setup
ChipWhisperer-Lite
STM32F3 target
SimpleSerial2 protocol
Clock glitching through ChipWhisperer
