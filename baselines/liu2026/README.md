# Liu et al. (2026) First-Layer Baseline Comparison

This directory reproduces the comparative evaluation against Liu et al.'s cryptanalytic first-layer extraction attack under our fixed-point probabilistic truncation (PT) oracle on the shared `64-20-20-10` ($S=9$, $q=512$) target (`checkpoints/digits/(64, 20, 20, 10)-01.npz`).

## Environment Setup (Separate Virtual Environment)

Liu et al.'s released upstream implementation (`third_party/liu_extraction`, MIT License, upstream commit `5206db797c123136aad21f69e7e01aeb15c2d9e0`) requires TensorFlow 2.16 and NumPy 1.26, which conflict with the NumPy 2.x environment used by the main `probtrunc_extraction` package. Create an isolated Python 3.12 virtual environment:

```bash
python3.12 -m venv baselines/liu2026/.venv
# Windows PowerShell:
.\baselines\liu2026\.venv\Scripts\pip.exe install -r baselines/liu2026/requirements-liu.txt
# Linux / macOS:
./baselines/liu2026/.venv/bin/pip install -r baselines/liu2026/requirements-liu.txt
```

## Upstream Compatibility Edits

Four documented compatibility edits are applied inside `third_party/liu_extraction/` so Liu's released attack can run cleanly against 20-neuron multi-output targets without accessing ground-truth weights:
1. `signature_recovery/precision_improvement.py`: defaults `tol=1e-10` for hidden widths other than 8/16 (width 20 otherwise raises `UnboundLocalError`).
2. `signature_recovery/compute_critical_point_values.py`: guards empty diagnostic imprecision lists when oracle-only mode disables true-layer labels.
3. `signature_recovery/cluster_critical_points/merge_good_critical_points.py`: raises an explicit error when clustering finds zero components.
4. `signature_recovery/signature_recovery.py`: squeezes the singleton batch axis and selects the requested output logit in the multi-output sign solver.

Target weights remain strictly isolated inside `oracle_bridge.py` and are never exposed to Liu's attack code, which receives an independent decoy Keras model and runtime guards from `first_layer_adapter.py`.

## Reproduction Commands

Run all commands from the `probtrunc_extraction/` repository root using the Liu virtual environment Python (`$LIU_PY`).

### 1. Connection Sanity Check (1 Query)

```bash
$LIU_PY baselines/liu2026/check_connection.py
```

### 2. Default-Step Liu Baseline (`float`, `float-grid`, and `pt`)

```bash
# Continuous float control (succeeds in ~713k queries)
$LIU_PY baselines/liu2026/run_first_layer.py --mode float --repeats 1 --seed 42 --max-queries 3000000 --max-seconds 600 --output-path runs/liu/default_float --execute

# Input-grid deterministic control (stalls in first sweep up to cap)
$LIU_PY baselines/liu2026/run_first_layer.py --mode float-grid --repeats 1 --seed 42 --max-queries 3000000 --max-seconds 600 --output-path runs/liu/default_float_grid --execute

# Single-response and 4-response PT (stall in first sweep up to cap)
$LIU_PY baselines/liu2026/run_first_layer.py --mode pt --repeats 1 --seed 42 --max-queries 750000 --max-seconds 900 --output-path runs/liu/default_pt_r1 --execute
$LIU_PY baselines/liu2026/run_first_layer.py --mode pt --repeats 4 --seed 42 --max-queries 3000000 --max-seconds 300 --output-path runs/liu/default_pt_r4 --execute
```

### 3. Grid-Adapted Micro-Probe Ratio Diagnostic (Up to 256-Response Averaging)

Evaluates Liu's exact first-layer ratio estimator (`get_ratios_lstsq`) at true critical points with $\varepsilon \in \{10^{-4}, k/q\}$ under `float`, `float-grid`, and `pt` with up to 256 repetitions:

```bash
$LIU_PY baselines/liu2026/probe_liu_ratio_grid_epsilon.py --points 5 --pt-repeats 1,4,16,64,256 --execute --output runs/liu/grid_epsilon_r256.json
```

### 4. Macro-Adapted Liu First-Layer Recovery (Scale-Matched to Our Attack)

Runs Liu's full first-layer pipeline scaled to macro input radius ($\sim 10^{13}$ codes) and macro probe step ($10^{10}$ codes):

```bash
$LIU_PY baselines/liu2026/run_macro_first_layer.py --mode float --repeats 1 --seed 42 --output-path runs/liu/macro_float --execute
$LIU_PY baselines/liu2026/run_macro_first_layer.py --mode float-grid --repeats 1 --seed 42 --output-path runs/liu/macro_float_grid --execute
$LIU_PY baselines/liu2026/run_macro_first_layer.py --mode pt --repeats 1 --seed 42 --output-path runs/liu/macro_pt --execute
```

### 5. Independent Post-Hoc Scoring

Score any completed Liu output directory with the main package CLI:

```bash
python -m probtrunc_extraction score-liu --run-dir runs/liu/macro_pt
```

### 6. Unit Tests

```bash
$LIU_PY -m unittest discover -s baselines/liu2026 -p "test_*.py" -v
```
