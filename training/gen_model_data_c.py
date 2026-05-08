from pathlib import Path
import numpy as np

ACTS = ["relu", "sigmoid", "tanh", "relu_ext"]


def fmt_float(x: float) -> str:
    return repr(float(x)) + "f"


def dump(npz_path: Path, out_c_path: Path) -> None:
    d = np.load(npz_path)
    w1 = d["W1"]
    b1 = d["b1"]
    w2 = d["W2"]
    b2 = d["b2"]

    with out_c_path.open("w", encoding="utf-8") as f:
        f.write(f"/* Auto-generated from {npz_path.name} */\n")
        f.write('#include "model_data.h"\n\n')

        f.write("const float model_W1[NN_HID][NN_IN] = {\n")
        for h in range(w1.shape[0]):
            row = ", ".join(fmt_float(v) for v in w1[h])
            f.write(f"  {{ {row} }},\n")
        f.write("};\n\n")

        f.write("const float model_b1[NN_HID] = {\n  ")
        f.write(", ".join(fmt_float(v) for v in b1))
        f.write("\n};\n\n")

        f.write("const float model_W2[NN_OUT][NN_HID] = {\n")
        for o in range(w2.shape[0]):
            row = ", ".join(fmt_float(v) for v in w2[o])
            f.write(f"  {{ {row} }},\n")
        f.write("};\n\n")

        f.write("const float model_b2[NN_OUT] = {\n  ")
        f.write(", ".join(fmt_float(v) for v in b2))
        f.write("\n};\n")

    print(f"wrote {out_c_path}")


def main() -> None:
    here = Path(".")
    for act in ACTS:
        dump(here / f"weights_{act}.npz", here / f"model_data_{act}.c")


if __name__ == "__main__":
    main()
