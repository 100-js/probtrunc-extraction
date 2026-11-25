"""Empirical E_max and spectral bound simulations (Appendix: Empirical E_max Estimation)."""

from __future__ import annotations

import os
from pathlib import Path
import random
from typing import Sequence

import numpy as np

from .io import export_to_dat


def random_matrices_scenario(
    layer_dims: Sequence[int] = (64, 20, 20, 10),
    q: int = 512,
    X_anchor: float = 1e13,
    Delta_X: float = 1e10,
    num_models: int = 500,
    seed: int = 42,
    large_samples: int = 5000,
    output_dir: str | Path = "runs/emax/random",
) -> dict:
    """Measure projected truncation noise |e_proj| and E_max on He-initialized integer MLPs."""
    rng = np.random.default_rng(seed)
    L = len(layer_dims) - 1

    print("Initiating spectral bound and |e_proj| distribution simulation (Random Matrices).")
    print(f"Parameters: q={q}, L={L}, X_anchor={X_anchor:.0e}, Delta_X={Delta_X:.0e}, Seed={seed}\n")

    emp_E_maxs: list[float] = []
    all_e_projs_abs: list[float] = []
    sigma_max_records: dict[int, list[float]] = {i: [] for i in range(1, L + 1)}

    X_base = np.ones(layer_dims[0], dtype=np.int64) * int(X_anchor)
    X_step = X_base.copy()
    X_step[0] += int(Delta_X)

    for _ in range(num_models):
        weights = []
        for i in range(L):
            d_in, d_out = layer_dims[i], layer_dims[i + 1]
            w = rng.standard_normal((d_in, d_out)) * np.sqrt(2.0 / d_in)
            W_quantized = np.round(w * q).astype(np.int64)
            weights.append(W_quantized)
            sigma_max_records[i + 1].append(float(np.linalg.norm(W_quantized / float(q), ord=2)))

        p_vec = rng.integers(1, 3, size=layer_dims[-1]) * rng.choice([-1, 1], size=layer_dims[-1])

        def md_ml_oracle(X_batch: np.ndarray) -> np.ndarray:
            H = X_batch.copy()
            for i in range(L):
                M = np.dot(H, weights[i])
                prob = (M % q) / float(q)
                noise = (rng.random(M.shape) < prob).astype(np.int64)
                H = (M // q) + noise
                if i < L - 1:
                    H = np.maximum(0, H)
            return H

        Y_base_all = md_ml_oracle(np.tile(X_base, (large_samples, 1)))
        Y_step_all = md_ml_oracle(np.tile(X_step, (large_samples, 1)))

        Y_base_proj = np.dot(Y_base_all, p_vec)
        Y_step_proj = np.dot(Y_step_all, p_vec)

        mean_base_proj = np.mean(Y_base_proj)
        mean_step_proj = np.mean(Y_step_proj)

        e_proj_abs_base = np.abs(Y_base_proj - mean_base_proj)
        e_proj_abs_step = np.abs(Y_step_proj - mean_step_proj)

        all_e_projs_abs.extend(e_proj_abs_base.tolist())
        emp_E_maxs.append(float(max(np.max(e_proj_abs_base), np.max(e_proj_abs_step))))

    all_e_projs_arr = np.array(all_e_projs_abs)
    emp_E_maxs_arr = np.array(emp_E_maxs)

    sigma_summary = {}
    for i in range(1, L + 1):
        max_sig = float(np.max(sigma_max_records[i]))
        sigma_summary[f"W{i}_max_sigma"] = max_sig
        print(f"- W{i} maximum sigma_max: {max_sig:.4f}")

    print(f"- |e_proj| global mean ({num_models * large_samples:,} samples): {np.mean(all_e_projs_arr):.4f}")
    print(f"- E_max absolute maximum (Global Max): {np.max(emp_E_maxs_arr):.4f}")

    exported = [
        str(export_to_dat(all_e_projs_arr, "eproj_abs_hist.dat", output_dir=output_dir, bins=60)),
        str(export_to_dat(emp_E_maxs_arr, "emax_hist.dat", output_dir=output_dir, bins=40)),
    ]
    for i in range(1, L + 1):
        exported.append(
            str(
                export_to_dat(
                    np.array(sigma_max_records[i]),
                    f"sigma_max_W{i}.dat",
                    output_dir=output_dir,
                    bins=30,
                )
            )
        )

    return {
        "scenario": "random",
        "layer_dims": list(layer_dims),
        "q": q,
        "num_models": num_models,
        "large_samples": large_samples,
        "seed": seed,
        "e_proj_abs_mean": float(np.mean(all_e_projs_arr)),
        "e_max_global_max": float(np.max(emp_E_maxs_arr)),
        "sigma_summary": sigma_summary,
        "files": exported,
    }


def trained_models_scenario(
    layer_dims: Sequence[int] = (64, 20, 20, 10),
    q: int = 512,
    X_anchor: float = 1e13,
    Delta_X: float = 1e10,
    num_models: int = 500,
    batch_size: int = 100,
    seed: int = 42,
    use_gpu: bool = True,
    large_samples: int = 5000,
    output_dir: str | Path = "runs/emax/trained",
) -> dict:
    """Measure projected truncation noise |e_proj| and E_max on empirically trained Digits MLPs."""
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split

    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
    except ImportError as error:
        raise ImportError(
            "PyTorch is required for the 'trained' E_max scenario. "
            "Install optional dependencies with: pip install -e '.[emax]'"
        ) from error

    torch.set_num_threads(1)
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    random.seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    torch.use_deterministic_algorithms(True, warn_only=True)

    L = len(layer_dims) - 1
    q_L = q**L

    print("Initiating spectral bound & |e_proj| simulation (Empirically Trained Models).")
    print(f"Topology: {tuple(layer_dims)}, Models: {num_models}, Device: {device.type.upper()}")
    print(f"Parameters: q={q}, L={L}, X_anchor={X_anchor:.0e}, Delta_X={Delta_X:.0e}, Seed={seed}\n")

    digits = load_digits()
    X_data = digits.data.astype(np.float32)
    y_data = digits.target

    emp_E_maxs: list[float] = []
    grad_errors_N1: list[float] = []
    all_e_projs_abs: list[float] = []

    W_tensors_list: list[list[np.ndarray]] = [[] for _ in range(L)]
    sigma_max_records: dict[int, list[float]] = {i: [] for i in range(1, L + 1)}
    weight_dist_records: dict[int, list[float]] = {i: [] for i in range(1, L + 1)}
    bias_dist_records: dict[int, list[float]] = {i: [] for i in range(1, L + 1)}

    for m in range(num_models):
        X_train, _, y_train, _ = train_test_split(X_data, y_data, test_size=0.2, random_state=seed + m)
        X_t = torch.tensor(X_train, device=device)
        y_t = torch.tensor(y_train, device=device, dtype=torch.long)

        layers = []
        for i in range(L):
            layers.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))
            if i < L - 1:
                layers.append(nn.ReLU())

        model = nn.Sequential(*layers).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)

        model.train()
        for _ in range(200):
            optimizer.zero_grad()
            loss = nn.CrossEntropyLoss()(model(X_t), y_t)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            linear_idx = 0
            for layer in model:
                if isinstance(layer, nn.Linear):
                    w = layer.weight.t().cpu().numpy()
                    W_quantized = np.round(w * q).astype(np.float64)
                    W_tensors_list[linear_idx].append(W_quantized)

                    b = layer.bias.cpu().numpy()
                    b_quantized = np.round(b * q).astype(np.float64)
                    bias_dist_records[linear_idx + 1].extend((b_quantized / float(q)).flatten().tolist())

                    effective_w = W_quantized / float(q)
                    weight_dist_records[linear_idx + 1].extend(effective_w.flatten().tolist())
                    sigma_max_records[linear_idx + 1].append(float(np.linalg.norm(effective_w, ord=2)))
                    linear_idx += 1

    W_batch_all = [
        torch.tensor(np.array(W_tensors_list[i]), device=device, dtype=torch.int64)
        for i in range(L)
    ]

    p_vec_np = rng.integers(1, 3, size=layer_dims[-1]) * rng.choice([-1, 1], size=layer_dims[-1])
    p_vec_t = torch.tensor(p_vec_np, device=device, dtype=torch.float64).unsqueeze(1)

    X_base_val = np.ones(layer_dims[0], dtype=np.float64) * X_anchor
    X_step_val = X_base_val.copy()
    X_step_val[0] += Delta_X

    X_base_t = torch.tensor(X_base_val, device=device, dtype=torch.int64).unsqueeze(0).unsqueeze(0)
    X_step_t = torch.tensor(X_step_val, device=device, dtype=torch.int64).unsqueeze(0).unsqueeze(0)

    cpu_generator = torch.Generator(device="cpu")
    cpu_generator.manual_seed(seed)

    processed = 0
    with torch.no_grad():
        while processed < num_models:
            current_batch = min(batch_size, num_models - processed)
            W_batch = [W_batch_all[i][processed : processed + current_batch] for i in range(L)]

            def batched_md_ml_oracle(X_tensor):
                H_long = X_tensor.expand(current_batch, large_samples, -1).clone()
                for i in range(L):
                    M_long = torch.bmm(H_long, W_batch[i])
                    prob = (M_long % q).double() / q
                    noise_cpu = torch.rand(
                        prob.size(), generator=cpu_generator, dtype=torch.float64, device="cpu"
                    )
                    noise_int = (noise_cpu.to(M_long.device) < prob).to(torch.int64)
                    H_long = torch.div(M_long, q, rounding_mode="floor") + noise_int
                    if i < L - 1:
                        H_long = torch.clamp(H_long, min=0)
                return H_long.double()

            Y_base_all = batched_md_ml_oracle(X_base_t)
            Y_step_all = batched_md_ml_oracle(X_step_t)

            Y_base_proj = torch.matmul(Y_base_all, p_vec_t).squeeze(-1)
            Y_step_proj = torch.matmul(Y_step_all, p_vec_t).squeeze(-1)

            mean_base_proj = torch.mean(Y_base_proj, dim=1, keepdim=True)
            mean_step_proj = torch.mean(Y_step_proj, dim=1, keepdim=True)

            e_proj_abs_base = torch.abs(Y_base_proj - mean_base_proj)
            e_proj_abs_step = torch.abs(Y_step_proj - mean_step_proj)

            E_max_base = torch.max(e_proj_abs_base, dim=1).values
            E_max_step = torch.max(e_proj_abs_step, dim=1).values
            emp_E_max = torch.maximum(E_max_base, E_max_step)

            true_diff = mean_step_proj.squeeze(1) - mean_base_proj.squeeze(1)
            obs_diff = Y_step_proj[:, 0] - Y_base_proj[:, 0]
            grad_err = torch.abs(obs_diff - true_diff) / Delta_X * q_L

            all_e_projs_abs.extend(e_proj_abs_base.flatten().cpu().numpy().tolist())
            emp_E_maxs.extend(emp_E_max.cpu().numpy().tolist())
            grad_errors_N1.extend(grad_err.cpu().numpy().tolist())
            processed += current_batch

    all_e_projs_arr = np.array(all_e_projs_abs)
    emp_E_maxs_arr = np.array(emp_E_maxs)
    grad_errors_arr = np.array(grad_errors_N1)

    sigma_summary = {}
    for i in range(1, L + 1):
        mean_sig = float(np.mean(sigma_max_records[i]))
        sigma_summary[f"W{i}_mean_sigma"] = mean_sig
        print(f"- W{i} mean sigma_max: {mean_sig:.4f}")

    print(f"- |e_proj| global mean (all samples): {np.mean(all_e_projs_arr):.4f}")
    print(f"- E_max mean (worst-case cluster): {np.mean(emp_E_maxs_arr):.4f}")
    print(f"- E_max absolute maximum (Global Max): {np.max(emp_E_maxs_arr):.4f}")
    print(f"- Maximum N=1 gradient error: {np.max(grad_errors_arr):.8f}")

    exported = [
        str(export_to_dat(all_e_projs_arr, "eproj_abs_hist.dat", output_dir=output_dir, bins=60)),
        str(export_to_dat(emp_E_maxs_arr, "emax_hist.dat", output_dir=output_dir, bins=40)),
    ]
    for i in range(1, L + 1):
        exported.append(
            str(
                export_to_dat(
                    np.array(sigma_max_records[i]),
                    f"sigma_max_W{i}.dat",
                    output_dir=output_dir,
                    bins=30,
                )
            )
        )

    return {
        "scenario": "trained",
        "layer_dims": list(layer_dims),
        "q": q,
        "num_models": num_models,
        "large_samples": large_samples,
        "seed": seed,
        "device": device.type,
        "e_proj_abs_mean": float(np.mean(all_e_projs_arr)),
        "e_max_mean": float(np.mean(emp_E_maxs_arr)),
        "e_max_global_max": float(np.max(emp_E_maxs_arr)),
        "max_n1_grad_error": float(np.max(grad_errors_arr)),
        "sigma_summary": sigma_summary,
        "files": exported,
    }
