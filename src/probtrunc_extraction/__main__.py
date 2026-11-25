"""Unified CLI for probtrunc_extraction."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_info

from .evaluation.accuracy import score_checkpoint
from .evaluation.liu_metrics import score_liu_run
from .evaluation.summarize import summarize_runs
from .fixedpoint import check_arithmetic
from .runners.benchmark import run_benchmark
from .targets import digits, mnist
from .targets.checkpoints import load_checkpoint, save_checkpoint
from .targets.config import load_benchmark_config, load_target_config


def _verify_file(path: Path, expected: str) -> None:
    with path.open("rb") as source:
        actual = hashlib.file_digest(source, "sha256").hexdigest()
    if actual != expected:
        raise ValueError(f"SHA-256 mismatch for {path}: expected {expected}, found {actual}")


def _model_indices(config, index: int | None):
    if index is None:
        return range(1, len(config.model_seeds) + 1)
    if not 1 <= index <= len(config.model_seeds):
        raise ValueError(f"Model index must be between 1 and {len(config.model_seeds)}")
    return [index]


def _default_checkpoint_dir(config, checkpoints_root: Path) -> Path | None:
    if config.checkpoint_group is not None:
        return checkpoints_root / config.checkpoint_group
    return None


def cmd_verify_inputs(args: argparse.Namespace) -> dict:
    root = args.checkpoints.resolve()
    manifest = root / "checkpoints.sha256"
    verified: set[Path] = set()
    for line in manifest.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line:
            continue
        expected, name = line.split(maxsplit=1)
        path = (root / name).resolve()
        if not path.is_relative_to(root) or path in verified:
            raise ValueError(f"Invalid or duplicate checkpoint path: {name}")
        _verify_file(path, expected)
        verified.add(path)
    if not verified:
        raise ValueError(f"No checkpoint hashes in {manifest}")

    result: dict = {"verified_checkpoints": len(verified)}
    if args.mnist is not None and args.mnist.is_file():
        verified_mnist = mnist.ensure_mnist(args.mnist, allow_download=False)
        result["verified_mnist"] = str(verified_mnist)
    return result


def cmd_score_targets(args: argparse.Namespace) -> dict:
    config = load_target_config(args.config)
    source = args.checkpoint_dir or _default_checkpoint_dir(config, args.checkpoints)
    if source is None:
        raise ValueError("Set --checkpoint-dir for a configuration without checkpoint_group")
    data_path = args.mnist if config.dataset == "mnist" else None

    effective_seeds: dict[int, int] = {}
    training_manifest = source / "training.json"
    if training_manifest.is_file():
        training = json.loads(training_manifest.read_text(encoding="utf-8"))
        if training.get("configuration") == config.name:
            effective_seeds = {row["model"]: row["effective_seed"] for row in training["models"]}

    scores = []
    for index in _model_indices(config, args.model):
        ckpt_path = config.checkpoint_path(source, index)
        score = score_checkpoint(
            config,
            ckpt_path,
            data_path,
            config.model_seeds[index - 1],
            effective_seed=effective_seeds.get(index),
            allow_download=not args.no_download,
        )
        scores.append({"model": index, **score})

    return {
        "configuration": config.name,
        "models": scores,
        "mean_quantized_accuracy_percent": sum(r["quantized_accuracy_percent"] for r in scores) / len(scores),
        "mean_float_accuracy_percent": sum(r["float_accuracy_percent"] for r in scores) / len(scores),
    }


def _compare_checkpoints(checkpoint, reference) -> bool:
    if checkpoint.q != reference.q or len(checkpoint.weights) != len(reference.weights):
        return False
    arrays = zip(checkpoint.weights + checkpoint.biases, reference.weights + reference.biases)
    return all(np.array_equal(current, saved) for current, saved in arrays)


def cmd_train_targets(args: argparse.Namespace) -> dict:
    config = load_target_config(args.config)
    reference = _default_checkpoint_dir(config, args.checkpoints)
    data_path = args.mnist if config.dataset == "mnist" else None
    output = args.output or Path("runs/targets") / args.config.stem
    indices = _model_indices(config, args.model)
    paths = {index: config.checkpoint_path(output, index) for index in indices}
    manifest = output / "training.json"

    for path in [*paths.values(), manifest]:
        if path.exists():
            raise ValueError(f"Output already exists: {path}. Choose a fresh --output directory.")

    libraries = [
        {key: entry[key] for key in ("internal_api", "version", "num_threads")}
        for entry in threadpool_info()
    ]
    models = []
    for index, path in paths.items():
        seed = config.model_seeds[index - 1]
        if config.dataset == "digits":
            checkpoint, effective_seed = digits.train(config, seed)
        else:
            checkpoint, effective_seed = mnist.train(config, seed, data_path)

        matched = None
        if reference is not None and config.checkpoint_path(reference, index).is_file():
            saved = load_checkpoint(config.checkpoint_path(reference, index))
            matched = _compare_checkpoints(checkpoint, saved)

        score = score_checkpoint(config, checkpoint, data_path, seed, effective_seed=effective_seed)
        save_checkpoint(path, checkpoint)
        models.append({
            "model": index,
            "nominal_seed": seed,
            "effective_seed": effective_seed,
            "matches_bundled_checkpoint": matched,
            "checkpoint": str(path),
            "score": score,
        })

    result = {"configuration": config.name, "threadpools": libraries, "models": models}
    manifest.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def cmd_run_benchmark(args: argparse.Namespace) -> dict:
    config = load_benchmark_config(args.config)
    output = args.output or Path("runs/benchmarks") / config.target.name
    return run_benchmark(
        args.config,
        output,
        model_index=args.model,
        trial_seed=args.trial_seed,
        workers=args.workers,
        checkpoint_dir=args.checkpoint_dir,
        save_extracted=args.save_extracted,
    )


def cmd_summarize(args: argparse.Namespace) -> dict:
    return summarize_runs(args.runs, args.output)


def cmd_score_liu(args: argparse.Namespace) -> dict:
    return score_liu_run(
        args.target,
        run_dir=args.run_dir,
        signature_path=args.signature,
        recovered_path=args.recovered,
        manifest_path=args.manifest,
        direction_tolerance=args.direction_tolerance,
    )


def cmd_run_emax(args: argparse.Namespace) -> dict:
    from .emax.simulation import random_matrices_scenario, trained_models_scenario

    output_dir = args.output or Path("runs/emax") / args.scenario
    if args.scenario == "random":
        result = random_matrices_scenario(
            layer_dims=(64, 20, 20, 10),
            q=args.q,
            X_anchor=args.x_anchor,
            Delta_X=args.delta_x,
            num_models=args.num_models,
            seed=args.seed,
            large_samples=args.samples,
            output_dir=output_dir,
        )
    else:
        result = trained_models_scenario(
            layer_dims=(64, 20, 20, 10),
            q=args.q,
            X_anchor=args.x_anchor,
            Delta_X=args.delta_x,
            num_models=args.num_models,
            batch_size=args.batch_size,
            seed=args.seed,
            use_gpu=not args.cpu,
            large_samples=args.samples,
            output_dir=output_dir,
        )
    if args.plot:
        from .emax.visualization import plot_all_dat_results

        plot_all_dat_results(output_dir=output_dir)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m probtrunc_extraction")
    commands = parser.add_subparsers(required=True)

    p_verify = commands.add_parser("verify-inputs", help="Verify bundled checkpoint SHA-256 digests")
    p_verify.add_argument("--checkpoints", type=Path, default=Path("checkpoints"))
    p_verify.add_argument("--mnist", type=Path, default=Path("data/mnist.npz"))
    p_verify.set_defaults(run=cmd_verify_inputs)

    p_arith = commands.add_parser("verify-arithmetic", help="Verify exact fixed-point arithmetic invariants")
    p_arith.add_argument("--seed", type=int, default=42)
    p_arith.set_defaults(run=lambda a: check_arithmetic(seed=a.seed))

    p_score = commands.add_parser("score-targets", help="Evaluate target checkpoints on canonical test splits")
    p_score.add_argument("--config", type=Path, required=True)
    p_score.add_argument("--checkpoints", type=Path, default=Path("checkpoints"))
    p_score.add_argument("--checkpoint-dir", type=Path)
    p_score.add_argument("--mnist", type=Path, default=Path("data/mnist.npz"))
    p_score.add_argument("--model", type=int, help="Select one model by 1-based index")
    p_score.add_argument("--no-download", action="store_true")
    p_score.set_defaults(run=cmd_score_targets)

    p_train = commands.add_parser("train-targets", help="Retrain and score target checkpoints from scratch")
    p_train.add_argument("--config", type=Path, required=True)
    p_train.add_argument("--checkpoints", type=Path, default=Path("checkpoints"))
    p_train.add_argument("--mnist", type=Path, default=Path("data/mnist.npz"))
    p_train.add_argument("--output", type=Path)
    p_train.add_argument("--model", type=int, help="Select one model by 1-based index")
    p_train.set_defaults(run=cmd_train_targets)

    p_bench = commands.add_parser("run-benchmark", help="Execute live model extraction benchmark trials")
    p_bench.add_argument("--config", type=Path, required=True)
    p_bench.add_argument("--output", type=Path)
    p_bench.add_argument("--model", type=int, help="Select one model by 1-based index")
    p_bench.add_argument("--trial-seed", type=int, help="Run a single trial seed")
    p_bench.add_argument("--workers", type=int, default=4)
    p_bench.add_argument("--checkpoint-dir", type=Path, help="Override bundled checkpoints with custom targets")
    p_bench.add_argument("--save-extracted", action="store_true", help="Save recovered NPZ weights for successful trials")
    p_bench.set_defaults(run=cmd_run_benchmark)

    p_sum = commands.add_parser("summarize", help="Generate CSV & Markdown tables from completed benchmark runs")
    p_sum.add_argument("--runs", type=Path, nargs="+", required=True)
    p_sum.add_argument("--output", type=Path, default=Path("runs/summary"))
    p_sum.set_defaults(run=cmd_summarize)

    p_liu = commands.add_parser("score-liu", help="Score live Liu et al. first-layer recovery outputs")
    p_liu.add_argument(
        "--target",
        type=Path,
        default=Path("checkpoints/digits/(64, 20, 20, 10)-01.npz"),
    )
    p_liu.add_argument("--run-dir", type=Path, help="Completed Liu run directory containing manifest and NPZ outputs")
    p_liu.add_argument("--signature", type=Path, help="Optional explicit signature_checkpoint_1.npz path")
    p_liu.add_argument("--recovered", type=Path, help="Optional explicit recovered_1.npz path")
    p_liu.add_argument("--manifest", type=Path, help="Optional explicit run_manifest.json path")
    p_liu.add_argument("--direction-tolerance", type=float, default=0.05)
    p_liu.set_defaults(run=cmd_score_liu)

    p_emax = commands.add_parser("run-emax", help="Run Appendix empirical E_max and spectral-norm simulations")
    p_emax.add_argument("scenario", choices=("random", "trained"))
    p_emax.add_argument("--num-models", type=int, default=500)
    p_emax.add_argument("--samples", type=int, default=5000)
    p_emax.add_argument("--seed", type=int, default=42)
    p_emax.add_argument("--q", type=int, default=512)
    p_emax.add_argument("--x-anchor", type=float, default=1e13)
    p_emax.add_argument("--delta-x", type=float, default=1e10)
    p_emax.add_argument("--batch-size", type=int, default=100)
    p_emax.add_argument("--cpu", action="store_true", help="Force CPU execution for trained scenario")
    p_emax.add_argument("--output", type=Path)
    p_emax.add_argument("--plot", action="store_true")
    p_emax.set_defaults(run=cmd_run_emax)

    args = parser.parse_args(argv)
    try:
        result = args.run(args)
    except (OSError, ValueError, ImportError) as error:
        parser.exit(1, f"{error}\n")
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
