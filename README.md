# Practical Fault Attacks on Neural Network Implementations

Bachelor thesis project — clock glitch attacks on a neural network running on STM32F3
via ChipWhisperer-Lite. 

---

## Repository structure

```
firmware/
  nn.h / nn.c          — math-only MLP (hidden layer + activation + output)
  nn-mnist.c           — SimpleSerial2 command interface, glitch hooks
  model_data.h         — extern declarations (shared by all activations)
  model_data_relu.c    — weights for ReLU (auto-generated)
  model_data_sigmoid.c — weights for Sigmoid (auto-generated)
  model_data_tanh.c    — weights for Tanh (auto-generated)
  model_data_relu_ext.c— weights for Extended ReLU (auto-generated, same training as relu)
  Makefile             — build config, activation selected via ACTIVATION=

training/
  train_mlp.py         — trains all four MLP variants, saves weights_*.npz
  gen_model_data_c.py  — converts weights_*.npz → model_data_*.c (C float arrays)

experiment/
  ReluExperiment.ipynb        — deep-dive ReLU fault analysis with z1/a1 snapshots
  allActivationsExperiment.ipynb — comparative sweep across all four activations

results/
  reluResults/         — plots, CSVs, and report-ready artifacts from ReLU experiment
  allActivasionsResults/ — comparison plots across all activations
```

---

## How to reproduce from scratch

### Prerequisites

| Tool | Version used | Notes |
|---|---|---|
| Python | 3.10 | |
| TensorFlow / Keras | 2.x | for training only |
| NumPy | any recent | |
| ChipWhisperer | 5.7.x | `pip install chipwhisperer` |
| Jupyter | any | to run notebooks |
| ARM GCC toolchain | arm-none-eabi-gcc | part of ChipWhisperer install |
| ChipWhisperer hardware | CWLITEARM | STM32F3 target board |

---

### Step 0 — Clone ChipWhisperer repository

The firmware `Makefile` depends on the ChipWhisperer SDK — specifically
`../Makefile.inc`, the HAL (hardware abstraction layer for STM32), and the
SimpleSerial 2 library. These are **not** included in this repo and must come
from the official ChipWhisperer source:

```bash
git clone https://github.com/newaetech/chipwhisperer.git
```

After cloning, place the `firmware/` folder from this repo inside the
ChipWhisperer hardware victims directory:

```
chipwhisperer/
  firmware/
    mcu/
      nn-all/         ← put this repo's firmware/ contents here
      nn.c
      nn.h
      nn-mnist.c
      model_data.h
      model_data_relu.c
      ...
      Makefile
```

The path matters because `Makefile` does `include ../Makefile.inc`.
---

### Step 2 — Train models

Run from the `training/` directory:

```bash
cd training
python train_mlp.py
```

This trains four models (relu, sigmoid, tanh, relu_ext) on MNIST and saves:

```
weights_relu.npz
weights_sigmoid.npz
weights_tanh.npz
weights_relu_ext.npz
```

**Note on relu_ext:** Extended ReLU uses the same Keras training as plain ReLU
(`activation="relu"`). The difference is only in the C firmware implementation —
`act_relu_ext` uses a bitwise mask instead of a branch. Both share the same
trained weights by design.

---

### Step 3 — Convert weights to C arrays

Still in `training/`:

```bash
python gen_model_data_c.py
```

This reads the four `.npz` files and writes:

```
model_data_relu.c
model_data_sigmoid.c
model_data_tanh.c
model_data_relu_ext.c
```

---

### Step 4 — Build firmware

```bash
# ReLU — with extended repeat loop for easier timing (used in the main experiment)
make PLATFORM=CWLITEARM CRYPTO_TARGET=NONE SS_VER=SS_VER_2_1 \
     ACTIVATION=relu EXPERIMENT=1 EXP_RELU_REPEAT=32

# Sigmoid
make PLATFORM=CWLITEARM CRYPTO_TARGET=NONE SS_VER=SS_VER_2_1 \
     ACTIVATION=sigmoid EXPERIMENT=1

# Tanh
make PLATFORM=CWLITEARM CRYPTO_TARGET=NONE SS_VER=SS_VER_2_1 \
     ACTIVATION=tanh EXPERIMENT=1

# Extended ReLU
make PLATFORM=CWLITEARM CRYPTO_TARGET=NONE SS_VER=SS_VER_2_1 \
     ACTIVATION=relu_ext EXPERIMENT=1 EXP_RELU_REPEAT=32
```

Each build produces `nn-mnist-<ACTIVATION>-CWLITEARM.hex`.

**Build flags explained:**

| Flag | Meaning |
|---|---|
| `EXPERIMENT=1` | enables fault injection hooks (`mlp_activate_range`, snapshot buffers) |
| `EXP_RELU_REPEAT=32` | adds a NOP loop inside `act_relu` to stretch timing for easier glitch targeting |
| `EXP_RELU_WIN_START/STOP` | default trigger window [24, 40), can be changed at runtime via notebook |

Without `EXPERIMENT=1` the firmware runs a plain MLP with no SimpleSerial hooks —
useful to verify clean inference only.

---

### Step 5 — Set up notebooks

In both notebooks, update the paths at the top of the configuration cell:

**`allActivationsExperiment.ipynb`:**
```python
FW_DIR = r'C:\path\to\chipwhisperer\firmware\nn-all'
MNIST_CACHE = r'D:\mnist_cache'   # or any directory
```

**`ReluExperiment.ipynb`:**
```python
FW_RELU_HEX = r'C:\path\to\nn-mnist-relu-CWLITEARM.hex'
MNIST_CACHE = r'D:\mnist_cache'
```

MNIST will be downloaded automatically on first run into `MNIST_CACHE`.

---

### Step 6 — Run ReLU deep-dive (ReluExperiment.ipynb)

Run cells top to bottom. The recommended order:

1. **Sections 1–8** — imports, config, scope setup, transport helpers, firmware flash, sanity check
2. **Section 9** — MNIST load
3. **Section 10** — clean baseline accuracy (should be ~97–98% for ReLU; if lower, check weight conversion)
4. **Section 13** — fast coarse sweep; finds candidate `(width, offset, ext_offset)` tuples
5. **Section 14** — identifies hot configurations with most glitch effect
6. **Section 15** — fine sweep around the best coarse candidate
7. **Section 16** — deep-dive: uploads one image, glitches it, reads back `z1`, `a1`, logits, ReLU violation summary
8. **Section 17** — verdict
9. **Section 19** — validation: 50× no-glitch control + 50× targeted glitch repeats

The deep-dive in section 16 is the core result: it shows `z1` unchanged while
`a1` violates `a1 = max(z1, 0)` for one neuron inside the trigger window.

---

### Step 7 — Run cross-activation comparison (allActivationsExperiment.ipynb)

This notebook requires all four firmware `.hex` files and `FAIR_MODEL_MODE = True`
(each activation uses its own trained weights).

Run cells top to bottom. The main entry point is `run_all_activations_main()` at the
bottom — it runs FULL and WINDOW mode sweeps for all four activations and produces
comparison CSVs and plots in `results_all_activations/`.

```

---

## Key result (ReLU)

Under a clock glitch at `width=3.0, offset=10.0, ext_offset=61` targeting
neurons [16, 24):

```
z1[16] = -3.916523   (matrix multiply result — unchanged by glitch)
expected a1[16] = max(-3.916523, 0) = 0
actual   a1[16] = -3.916523         (ReLU did not zero the negative value)
```

`max_abs_z1_delta = 0.0` — the fault is isolated to the activation step,
not the matrix multiply. Reproduced 50/50 times under identical parameters.

## Hardware setup

- **Platform:** ChipWhisperer-Lite (CWLITEARM)
- **Target:** STM32F303
- **Clock:** 7.37 MHz (CLKGEN), glitch output = `clock_xor`
- **Communication:** SimpleSerial 2.1 over UART
- **Trigger:** `ext_single`, fired from firmware GPIO at start of inference (FULL mode)
  or at start of activation window (WINDOW mode)
