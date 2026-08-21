import os
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

from .common import set_seed


BASE_DIR = "models"


def _ensure_dir(filepath):
    """Ensures the directory for the given filepath exists."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)


def generate_target_model_cpu(q, model_shape=(64, 20, 20, 10), save_path=None, target_seed=0):
    if save_path:
        full_save_path = os.path.join(BASE_DIR, save_path)
        if os.path.exists(full_save_path):
            print(f"Loading cached target model from: {full_save_path}")
            return load_target_model(save_path)

    print("Initiating generation of a structurally sound target model...")
    digits = load_digits()

    set_seed(target_seed)
    noise = np.random.normal(0, 0.01, digits.data.shape)
    X_data = (digits.data + noise).astype(np.float32)
    y_data = digits.target

    while True:
        X_train, X_test, y_train, y_test = train_test_split(
            X_data, y_data, test_size=0.2, random_state=target_seed
        )

        mlp = MLPClassifier(
            hidden_layer_sizes=model_shape[1:-1],
            activation="relu",
            max_iter=1000,
            random_state=target_seed,
        )
        mlp.fit(X_train, y_train)

        W = [np.round(mlp.coefs_[i] * q).astype(np.int64) for i in range(len(model_shape) - 1)]
        B = [np.round(mlp.intercepts_[i] * q).astype(np.int64) for i in range(len(model_shape) - 1)]

        is_invalid_model = False

        for i in range(len(W)):
            invalid_col = np.any(np.all(W[i] >= 0, axis=0)) or np.any(np.all(W[i] <= 0, axis=0))
            invalid_row = np.any(np.all(W[i] >= 0, axis=1)) or np.any(np.all(W[i] <= 0, axis=1))

            if invalid_col or invalid_row:
                is_invalid_model = True
                break

        if is_invalid_model:
            # print(f"Seed {target_seed} rejected: Monotonic or disconnected (dead) neurons detected. Regenerating...")
            target_seed += 1
            set_seed(target_seed)
            continue

        print(f"Target model successfully generated (Seed: {target_seed}).")
        break

    if save_path:
        _ensure_dir(full_save_path)
        save_dict = {"q": q, "num_layers": len(W)}
        for i, (w, b) in enumerate(zip(W, B)):
            save_dict[f"W_{i}"] = w
            save_dict[f"B_{i}"] = b

        np.savez(full_save_path, **save_dict)
        print(f"Target model saved successfully to: {full_save_path}")

    return W, B


def load_target_model(load_path):
    full_load_path = os.path.join(BASE_DIR, load_path)
    data = np.load(full_load_path)
    num_layers = int(data["num_layers"])

    W = [data[f"W_{i}"] for i in range(num_layers)]
    B = [data[f"B_{i}"] for i in range(num_layers)]

    saved_q = int(data["q"])
    print(f"Model loaded successfully (Quantization scale q={saved_q}, Layers={num_layers}).")

    return W, B


class CPUOracleCounter:
    def __init__(self, W_list, B_list, q, model_shape):
        self.W = W_list
        self.B = B_list
        self.q = q
        self.model_shape = model_shape
        self.total_queries = 0

    def prob_oracle(self, X_batch, sf_leak=False):
        if len(X_batch.shape) == 1:
            self.total_queries += 1
            Y = X_batch.reshape(1, -1).copy()
        else:
            self.total_queries += X_batch.shape[0]
            Y = X_batch.copy()

        leaked_sf = []
        for i in range(len(self.model_shape) - 1):
            M = np.dot(Y, self.W[i])
            sf = np.random.randint(0, self.q, size=M.shape)
            if sf_leak:
                leaked_sf.append(sf)
            Y = np.floor(M / self.q).astype(np.int64) + (M % self.q > sf).astype(np.int64) + self.B[i]
            if i < len(self.model_shape) - 2:
                Y = np.maximum(0, Y)

        if sf_leak:
            return Y, leaked_sf
        return Y