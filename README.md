# probtrunc_extraction

Reference implementation for **Exact Model Extraction Attacks against Fixed-Point Neural Networks with Probabilistic Truncation**.

This repository reproduces all experiments reported in the paper:
- **Original Digits Benchmark (`64-20-20-10`, $S=9$)**: 5 target models $\times$ 10 trials = 50 runs, including Layer-2 SIR diagnostics and per-layer signed bit-width tracking.
- **Narrow MNIST Benchmark (`784-20-20-10`, $S \in \{6, 9, 12\}$)**: 3 target models $\times$ 5 trials = 15 runs per precision level.
- **Wide MNIST Benchmark (`784-128-64-10`, $S=6$)**: 3 target models $\times$ 5 trials = 15 runs.
- **Appendix Empirical $E_{\max}$ Simulation**: Spectral-norm and projected truncation-noise distributions for both random and trained models.
- **Comparative Baseline (Liu et al., 2026)**: Default-step, grid-adapted micro-probe, and macro-adapted first-layer comparisons under the identical PT oracle (`baselines/liu2026/`).

## Installation

Python 3.12 is required.

```bash
# Core attack, target training, and evaluation (no C++ build dependencies)
pip install -e .

# Optional: fast native LLL via fpylll (Linux/macOS; falls back to SymPy LLL automatically when absent)
pip install -e ".[fpylll]"

# Optional: Appendix E_max trained-model simulation and plotting
pip install -e ".[emax]"
```

For the Liu et al. (2026) baseline comparison (which requires TensorFlow 2.16 and NumPy 1.26), follow the separate virtual-environment setup in [baselines/liu2026/README.md](baselines/liu2026/README.md).

## Target Checkpoints & Datasets

- The 17 quantized target checkpoints (`checkpoints/digits/`, `checkpoints/mnist-narrow/`, `checkpoints/mnist-wide/`) are bundled with SHA-256 checksums (`checkpoints/checkpoints.sha256`).
- `mnist.npz` is downloaded automatically to `data/mnist.npz` on first use and verified by SHA-256.
- Every target model can also be retrained from scratch using `train-targets`.

Verify bundled checksums and fixed-point arithmetic invariants:

```bash
python -m probtrunc_extraction verify-inputs
python -m probtrunc_extraction verify-arithmetic
```

Evaluate target model accuracies (Digits holdout and official MNIST test split):

```bash
python -m probtrunc_extraction score-targets --config configs/digits_64_20_20_10_s9.toml
python -m probtrunc_extraction score-targets --config configs/mnist_784_20_20_10_s6.toml
python -m probtrunc_extraction score-targets --config configs/mnist_784_20_20_10_s9.toml
python -m probtrunc_extraction score-targets --config configs/mnist_784_20_20_10_s12.toml
python -m probtrunc_extraction score-targets --config configs/mnist_784_128_64_10_s6.toml
```

To retrain targets from scratch into a fresh directory:

```bash
python -m probtrunc_extraction train-targets --config configs/mnist_784_20_20_10_s6.toml --output runs/targets/mnist-s6
```

## Running Extraction Benchmarks

Every benchmark command runs trials in parallel, saves per-trial JSON records and logs, and resumes cleanly if interrupted. Pass `--model` and `--trial-seed` for a single-trial smoke run, or `--checkpoint-dir` to attack newly trained targets.

### 1. Original Digits (`64-20-20-10`, $S=9$, 50 trials)

```bash
python -m probtrunc_extraction run-benchmark --config configs/digits_64_20_20_10_s9.toml --output runs/benchmarks/digits-s9 --workers 4
```

### 2. Narrow MNIST (`784-20-20-10`, $S \in \{6, 9, 12\}$, 15 trials each)

```bash
python -m probtrunc_extraction run-benchmark --config configs/mnist_784_20_20_10_s6.toml --output runs/benchmarks/mnist-narrow-s6 --workers 4
python -m probtrunc_extraction run-benchmark --config configs/mnist_784_20_20_10_s9.toml --output runs/benchmarks/mnist-narrow-s9 --workers 4
python -m probtrunc_extraction run-benchmark --config configs/mnist_784_20_20_10_s12.toml --output runs/benchmarks/mnist-narrow-s12 --workers 4
```

### 3. Wide MNIST (`784-128-64-10`, $S=6$, 15 trials)

```bash
python -m probtrunc_extraction run-benchmark --config configs/mnist_784_128_64_10_s6.toml --output runs/benchmarks/mnist-wide-s6 --workers 4
```

### 4. Aggregating Live Results into Tables

Generate CSV and Markdown benchmark, SIR, and bit-width tables from completed run directories:

```bash
python -m probtrunc_extraction summarize \
  --runs runs/benchmarks/digits-s9 runs/benchmarks/mnist-narrow-s6 runs/benchmarks/mnist-narrow-s9 runs/benchmarks/mnist-narrow-s12 runs/benchmarks/mnist-wide-s6 \
  --output runs/summary
```

## Appendix Empirical $E_{\max}$ Simulation

```bash
python -m probtrunc_extraction run-emax random --output runs/emax/random
python -m probtrunc_extraction run-emax trained --output runs/emax/trained
```

Pass `--plot` (requires `.[emax]`) to visualize the exported `.dat` histograms.

## Liu et al. (2026) Comparison

See [baselines/liu2026/README.md](baselines/liu2026/README.md) to run the default-step, grid-adapted micro-probe, and macro-adapted first-layer experiments. Completed runs can be scored independently via:

```bash
python -m probtrunc_extraction score-liu --run-dir runs/liu/macro_pt
```

## Running Tests

```bash
python -m unittest discover -s tests -v
```
