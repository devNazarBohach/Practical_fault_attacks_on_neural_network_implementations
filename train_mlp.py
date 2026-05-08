import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

SEED = 42
EPOCHS = 20
BATCH_SIZE = 128
LEARNING_RATE = 1e-3


def build_model(activation_name: str) -> keras.Model:
    model = keras.Sequential([
        layers.Input(shape=(784,)),
        layers.Dense(64, activation=activation_name, name="hidden"),
        layers.Dense(10, activation=None, name="output"),
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    return model


def main() -> None:
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = x_train.reshape(-1, 784).astype(np.float32) / 255.0
    x_test = x_test.reshape(-1, 784).astype(np.float32) / 255.0

    activations = {
        "relu": "relu",
        "sigmoid": "sigmoid",
        "tanh": "tanh",
        "relu_ext": "relu",
    }

    for act_key, act_fn in activations.items():
        tf.random.set_seed(SEED)
        np.random.seed(SEED)
        print(f"\n=== training {act_key} ===")
        model = build_model(act_fn)
        model.fit(
            x_train,
            y_train,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            validation_split=0.1,
            verbose=2,
        )
        test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
        print(f"{act_key}: test_acc = {test_acc:.4f}")

        w1 = model.get_layer("hidden").kernel.numpy().T
        b1 = model.get_layer("hidden").bias.numpy()
        w2 = model.get_layer("output").kernel.numpy().T
        b2 = model.get_layer("output").bias.numpy()

        assert w1.shape == (64, 784)
        assert b1.shape == (64,)
        assert w2.shape == (10, 64)
        assert b2.shape == (10,)

        np.savez(
            f"weights_{act_key}.npz",
            W1=w1,
            b1=b1,
            W2=w2,
            b2=b2,
            test_acc=np.array([test_acc], dtype=np.float32),
        )


if __name__ == "__main__":
    main()
