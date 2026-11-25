"""Parallel, resumable benchmark runner for Original Digits and Unified MNIST extraction."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import nullcontext, redirect_stdout
from dataclasses import asdict
from hashlib import file_digest
import json
from pathlib import Path
from statistics import mean, pstdev
from tempfile import NamedTemporaryFile
from time import perf_counter

import numpy as np
from threadpoolctl import threadpool_limits

from probtrunc_extraction.extractors import mnist as mnist_extractor
from probtrunc_extraction.extractors import original_digits as digits_extractor
from probtrunc_extraction.extractors.common import set_seed
from probtrunc_extraction.extractors.verification import (
    unshuffle_and_verify_B2,
    unshuffle_and_verify_L1,
    unshuffle_and_verify_L_last,
    unshuffle_and_verify_W2,
)
from probtrunc_extraction.fixedpoint import signed_bit_width
from probtrunc_extraction.oracle import CPUOracleCounter
from probtrunc_extraction.targets.checkpoints import Checkpoint, load_checkpoint, save_checkpoint
from probtrunc_extraction.targets.config import BenchmarkConfig, ExtractionConfig, load_benchmark_config


def _file_hash(path: Path) -> str:
    with path.open("rb") as source:
        return file_digest(source, "sha256").hexdigest()


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        "w", encoding="utf-8", newline="\n", dir=path.parent, suffix=".tmp", delete=False
    ) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _trial_path(output_dir: Path, model_index: int, trial_seed: int) -> Path:
    return output_dir / "trials" / f"M{model_index}-seed{trial_seed}.json"


def run_trial(
    checkpoint_path: str | Path,
    architecture: tuple[int, ...],
    q: int,
    model_index: int,
    trial_seed: int,
    ext: ExtractionConfig,
    *,
    save_extracted_path: str | Path | None = None,
) -> dict:
    """Execute one full end-to-end extraction trial and return structured telemetry."""
    set_seed(trial_seed)
    checkpoint = load_checkpoint(checkpoint_path)
    if checkpoint.q != q or checkpoint.architecture != architecture:
        raise ValueError(f"Checkpoint does not match configuration: {checkpoint_path}")

    W_true, B_true = checkpoint.weights, checkpoint.biases
    oracle = CPUOracleCounter(
        W_true,
        B_true,
        q,
        architecture,
        backend=ext.oracle_backend,
        track_bitwidth=ext.track_bitwidth,
    )

    bias_tols = ext.bias_tolerances if ext.bias_tolerances is not None else (None, None, None)
    tol_b1, tol_b2, tol_b3 = bias_tols
    total_layers = len(architecture) - 1

    result: dict = {
        "architecture": list(architecture),
        "S": q.bit_length() - 1,
        "q": q,
        "extractor": ext.extractor,
        "oracle_backend": ext.oracle_backend,
        "model_idx": model_index,
        "extraction_seed": trial_seed,
        "L1_success": False,
        "W2_success": False,
        "B2_success": False,
        "L_last_success": False,
        "end_to_end_success": False,
        "L1_queries": 0,
        "W2_queries": 0,
        "B2_queries": 0,
        "L_last_queries": 0,
        "total_queries": 0,
        "elapsed_time": 0.0,
        "L1_max_weight_err": None,
        "W2_max_weight_err": None,
        "L_last_max_weight_err": None,
        "L1_max_bias_err": None,
        "B2_max_bias_err": None,
        "L_last_max_bias_err": None,
        "l2_avg_sir": None,
        "bitwidth": None,
    }

    start = perf_counter()
    try:
        # 1. Layer 1 (W1, B1)
        if ext.extractor == "original-digits":
            dec_W1, dec_B1 = digits_extractor.decrypt_L1(
                oracle.prob_oracle,
                architecture[0],
                architecture[1],
                architecture[-1],
                q,
                total_layers,
                W1_anchors=ext.w1_anchors,
                W1_steps_per_anchor=ext.w1_steps_per_anchor,
                W1_queries=ext.w1_queries,
                B1_anchors=ext.b1_anchors,
                B1_queries=ext.b1_queries,
            )
        else:
            dec_W1, dec_B1 = mnist_extractor.decrypt_L1(
                oracle.prob_oracle,
                architecture[0],
                architecture[1],
                architecture[-1],
                q,
                total_layers,
                W1_anchors=ext.w1_anchors,
                W1_steps_per_anchor=ext.w1_steps_per_anchor,
                W1_queries=ext.w1_queries,
                B1_anchors=ext.b1_anchors,
                B1_queries=ext.b1_queries,
                rank1_consensus=ext.rank1_consensus,
                b1_high_snr_filter=ext.b1_high_snr_filter,
            )

        if dec_W1 is None or dec_B1 is None:
            return result

        W1_recon, B1_recon, pi_1 = unshuffle_and_verify_L1(
            dec_W1,
            dec_B1,
            W_true[0],
            B_true[0],
            W_true[1],
            tolerance=tol_b1,
            weight_tolerance=0,
        )
        if W1_recon is None or B1_recon is None or pi_1 is None:
            return result

        result["L1_success"] = True
        q_l1 = oracle.total_queries
        result["L1_queries"] = q_l1
        result["L1_max_weight_err"] = int(np.max(np.abs(W1_recon - W_true[0])))
        result["L1_max_bias_err"] = int(np.max(np.abs(B1_recon - B_true[0])))

        # 2. Layer 2 Weights (W2)
        if ext.extractor == "original-digits":
            dec_W2, sir_details = digits_extractor.decrypt_W2(
                oracle.prob_oracle,
                dec_W1,
                architecture[0],
                architecture[1],
                architecture[2],
                architecture[-1],
                q,
                total_layers,
                max_anchors=ext.w2_max_anchors,
                steps_per_anchor=ext.w2_steps_per_anchor,
                p1_queries=ext.w2_p1_queries,
                p2_queries=ext.w2_p2_queries,
                p3_queries=ext.w2_p3_queries,
                return_sir=True,
            )
        else:
            dec_W2, sir_details = mnist_extractor.decrypt_W2(
                oracle.prob_oracle,
                dec_W1,
                architecture[0],
                architecture[1],
                architecture[2],
                architecture[-1],
                q,
                total_layers,
                max_anchors=ext.w2_max_anchors,
                steps_per_anchor=ext.w2_steps_per_anchor,
                p1_queries=ext.w2_p1_queries,
                p2_queries=ext.w2_p2_queries,
                p3_queries=ext.w2_p3_queries,
                walk_mode=ext.w2_walk_mode,
                w2_inv_backend=ext.w2_inv_backend,
                return_sir=True,
            )

        if sir_details:
            valid_sirs = [d["sir"] for d in sir_details if d.get("linprog_success")]
            if valid_sirs:
                result["l2_avg_sir"] = float(mean(valid_sirs))

        if dec_W2 is None:
            return result

        W2_recon, pi_2 = unshuffle_and_verify_W2(dec_W2, W_true[1], pi_1)
        if W2_recon is None or pi_2 is None:
            return result

        result["W2_success"] = True
        result["W2_max_weight_err"] = int(np.max(np.abs(W2_recon - W_true[1])))
        q_w2 = oracle.total_queries - q_l1
        result["W2_queries"] = q_w2

        # 3. Layer 2 Biases (B2)
        if ext.extractor == "original-digits":
            dec_B2 = digits_extractor.decrypt_B2(
                oracle.prob_oracle,
                dec_W1,
                dec_B1,
                dec_W2,
                architecture[0],
                architecture[1],
                architecture[2],
                q,
                num_bases=ext.b2_num_bases,
                test_queries=ext.b2_test_queries,
            )
        else:
            dec_B2 = mnist_extractor.decrypt_B2(
                oracle.prob_oracle,
                dec_W1,
                dec_B1,
                dec_W2,
                architecture[0],
                architecture[1],
                architecture[2],
                q,
                num_bases=ext.b2_num_bases,
                test_queries=ext.b2_test_queries,
            )

        B2_recon = unshuffle_and_verify_B2(dec_B2, B_true[1], pi_2, tolerance=tol_b2)
        if B2_recon is None:
            return result

        result["B2_success"] = True
        q_b2 = oracle.total_queries - (q_l1 + q_w2)
        result["B2_queries"] = q_b2
        result["B2_max_bias_err"] = int(np.max(np.abs(B2_recon - B_true[1])))

        # 4. Final Layer (W_last, B_last)
        if ext.extractor == "original-digits":
            dec_W_last, dec_B_last = digits_extractor.decrypt_L_last(
                oracle.prob_oracle,
                [dec_W1, dec_W2],
                [dec_B1, dec_B2],
                architecture[-1],
                q,
                last_layer_repetitions=ext.last_layer_repetitions,
            )
        else:
            dec_W_last, dec_B_last = mnist_extractor.decrypt_L_last(
                oracle.prob_oracle,
                [dec_W1, dec_W2],
                [dec_B1, dec_B2],
                architecture[-1],
                q,
                solver=ext.last_layer_solver,
                last_layer_repetitions=ext.last_layer_repetitions,
            )

        is_match, W_last_recon, B_last_recon = unshuffle_and_verify_L_last(
            dec_W_last,
            dec_B_last,
            W_true[-1],
            B_true[-1],
            pi_2,
            tolerance=tol_b3,
        )
        if not is_match:
            return result

        result["L_last_success"] = True
        result["end_to_end_success"] = True
        q_last = oracle.total_queries - (q_l1 + q_w2 + q_b2)
        result["L_last_queries"] = q_last
        result["L_last_max_weight_err"] = int(np.max(np.abs(W_last_recon - W_true[-1])))
        result["L_last_max_bias_err"] = int(np.max(np.abs(B_last_recon - B_true[-1])))

        if save_extracted_path is not None:
            extracted_ckpt = Checkpoint(
                q=q,
                weights=(W1_recon, W2_recon, W_last_recon),
                biases=(B1_recon, B2_recon, B_last_recon),
            )
            save_checkpoint(save_extracted_path, extracted_ckpt)
    finally:
        result["total_queries"] = oracle.total_queries
        result["elapsed_time"] = perf_counter() - start
        if ext.track_bitwidth:
            result["bitwidth"] = oracle.bitwidth_report()

    return result


def _execute_trial(task: dict) -> dict:
    output_dir = Path(task["output_dir"])
    log_path = output_dir / "logs" / f"M{task['model_index']}-seed{task['trial_seed']}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    ext = ExtractionConfig(
        **{
            key: tuple(value) if key in ("trial_seeds", "bias_tolerances") and value is not None else value
            for key, value in task["extraction"].items()
        }
    )
    save_extracted = None
    if task.get("save_extracted"):
        save_extracted = output_dir / "extracted" / f"M{task['model_index']}-seed{task['trial_seed']}.npz"

    blas_ctx = threadpool_limits(limits=1) if task.get("limit_blas_threads") else nullcontext()
    with blas_ctx, log_path.open("w", encoding="utf-8", buffering=1) as log, redirect_stdout(log):
        result = run_trial(
            task["checkpoint_path"],
            tuple(task["architecture"]),
            task["q"],
            task["model_index"],
            task["trial_seed"],
            ext,
            save_extracted_path=save_extracted,
        )
    _write_json(_trial_path(output_dir, task["model_index"], task["trial_seed"]), result)
    return result


def summarize_trials(results: list[dict], expected_count: int, config: BenchmarkConfig) -> dict:
    """Aggregate per-trial JSON records into paper-ready benchmark, SIR, and bit-width summaries."""
    l1_ok = sum(1 for row in results if row["L1_success"])
    w2_ok = sum(1 for row in results if row["W2_success"])
    b2_ok = sum(1 for row in results if row["B2_success"])
    w3_ok = sum(1 for row in results if row["L_last_success"])
    complete = [row for row in results if row["end_to_end_success"]]

    summary: dict = {
        "configuration": config.target.name,
        "dataset": config.target.dataset,
        "architecture": list(config.target.architecture),
        "fraction_bits": config.target.precision,
        "q": config.target.q,
        "extractor": config.extraction.extractor,
        "oracle_backend": config.extraction.oracle_backend,
        "completed_trials": len(results),
        "expected_trials": expected_count,
        "end_to_end_successes": len(complete),
        "exact_weight_counts": [
            [l1_ok, len(results)],
            [w2_ok, l1_ok],
            [w3_ok, b2_ok],
        ],
        "W1": f"{l1_ok}/{len(results)}",
        "W2": f"{w2_ok}/{l1_ok}",
        "W3": f"{w3_ok}/{b2_ok}",
    }

    if complete:
        summary["bias_max_lsb"] = [
            max(row[key] for row in complete)
            for key in ("L1_max_bias_err", "B2_max_bias_err", "L_last_max_bias_err")
        ]
        queries = [row["total_queries"] for row in complete]
        summary["queries_mean"] = float(mean(queries))
        summary["queries_population_sd"] = float(pstdev(queries))
        summary["stage_query_means"] = {
            stage: float(mean(row[stage] for row in complete))
            for stage in ("L1_queries", "W2_queries", "B2_queries", "L_last_queries")
        }

    # SIR summary grouped by W2 recovery outcome (Table 2 in rebuttal).
    sir_rows = []
    for outcome_flag, label in ((True, "Exact W2"), (False, "Incorrect W2")):
        vals = [
            row["l2_avg_sir"]
            for row in results
            if row.get("l2_avg_sir") is not None and bool(row["W2_success"]) is outcome_flag
        ]
        if vals:
            sir_rows.append({
                "outcome": label,
                "trials": len(vals),
                "mean": float(mean(vals)),
                "population_sd": float(pstdev(vals)),
                "minimum": float(min(vals)),
                "maximum": float(max(vals)),
            })
    if sir_rows:
        summary["sir_summary"] = sir_rows

    # Signed bit-width summary across completed trials (Table 3 in rebuttal).
    bw_reports = [row["bitwidth"] for row in complete if row.get("bitwidth") is not None]
    if bw_reports:
        num_layers = len(bw_reports[0]["layers"])
        bw_layers = []
        for idx in range(num_layers):
            pre_max = max(rep["layers"][idx]["pre_trunc_max_abs"] for rep in bw_reports)
            post_max = max(rep["layers"][idx]["post_trunc_max_abs"] for rep in bw_reports)
            bw_layers.append({
                "layer": idx + 1,
                "reports": len(bw_reports),
                "pre_trunc_max_abs": pre_max,
                "pre_trunc_signed_bits": signed_bit_width(pre_max),
                "post_trunc_max_abs": post_max,
                "post_trunc_signed_bits": signed_bit_width(post_max),
            })
        summary["bitwidth_summary"] = {
            "reports": len(bw_reports),
            "input_max_abs": max(rep["input_max_abs"] for rep in bw_reports),
            "layers": bw_layers,
        }

    summary["trial_results"] = results
    return summary


def run_benchmark(
    config_path: str | Path,
    output_dir: str | Path,
    *,
    model_index: int | None = None,
    trial_seed: int | None = None,
    workers: int = 4,
    checkpoint_dir: str | Path | None = None,
    save_extracted: bool = False,
) -> dict:
    """Run (or resume) a configured extraction benchmark and write structured JSON outputs."""
    config_file = Path(config_path).resolve()
    config = load_benchmark_config(config_file)
    target = config.target
    ext = config.extraction

    if len(target.architecture) != 4:
        raise ValueError("The benchmark runner currently supports two-hidden-layer MLPs")
    if type(workers) is not int or workers < 1:
        raise ValueError("workers must be a positive integer")
    if model_index is not None and model_index not in range(1, len(target.model_seeds) + 1):
        raise ValueError("model_index is outside the configured target models")
    if trial_seed is not None and (type(trial_seed) is not int or trial_seed < 0):
        raise ValueError("trial_seed must be a nonnegative integer")

    indices = range(1, len(target.model_seeds) + 1) if model_index is None else (model_index,)
    seeds = ext.trial_seeds if trial_seed is None else (trial_seed,)

    if checkpoint_dir is None and target.checkpoint_group is None:
        raise ValueError("A custom configuration without checkpoint_group requires --checkpoint-dir")
    target_dir = (
        Path(checkpoint_dir).resolve()
        if checkpoint_dir is not None
        else (config_file.parent.parent / "checkpoints" / target.checkpoint_group).resolve()
    )

    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)

    ext_payload = asdict(ext)
    tasks = []
    for index in indices:
        ckpt_path = target.checkpoint_path(target_dir, index).resolve()
        ckpt = load_checkpoint(ckpt_path)
        if ckpt.q != target.q or ckpt.architecture != target.architecture:
            raise ValueError(f"Checkpoint mismatch: {ckpt_path}")
        ckpt_hash = _file_hash(ckpt_path)
        for seed in seeds:
            tasks.append({
                "checkpoint_path": str(ckpt_path),
                "checkpoint_sha256": ckpt_hash,
                "architecture": list(target.architecture),
                "q": target.q,
                "model_index": index,
                "trial_seed": seed,
                "output_dir": str(destination),
                "extraction": ext_payload,
                "save_extracted": bool(save_extracted),
                "limit_blas_threads": workers > 1,
            })

    extractors_dir = Path(__file__).resolve().parent.parent / "extractors"
    extractor_file = (
        extractors_dir / "original_digits.py"
        if ext.extractor == "original-digits"
        else extractors_dir / "mnist.py"
    )
    metadata = {
        "configuration": target.name,
        "configuration_sha256": _file_hash(config_file),
        "extractor_sha256": _file_hash(extractor_file),
        "common_sha256": _file_hash(extractors_dir / "common.py"),
        "oracle_sha256": _file_hash(Path(__file__).resolve().parent.parent / "oracle.py"),
        "architecture": list(target.architecture),
        "q": target.q,
        "extraction": ext_payload,
        "tasks": [
            {key: task[key] for key in ("model_index", "trial_seed", "checkpoint_sha256")}
            for task in tasks
        ],
    }

    manifest = destination / "run-manifest.json"
    if manifest.exists():
        existing = json.loads(manifest.read_text(encoding="utf-8"))
        if existing != metadata:
            raise ValueError(f"Output directory has a conflicting run manifest: {destination}")
    elif any(destination.iterdir()):
        raise ValueError(f"Output directory is not empty: {destination}")
    else:
        _write_json(manifest, metadata)

    pending = [
        task for task in tasks
        if not _trial_path(destination, task["model_index"], task["trial_seed"]).exists()
    ]

    if pending:
        if workers == 1 or len(pending) == 1:
            for task in pending:
                result = _execute_trial(task)
                _print_progress(result)
        else:
            with ProcessPoolExecutor(max_workers=min(workers, len(pending))) as pool:
                futures = {pool.submit(_execute_trial, task): task for task in pending}
                for future in as_completed(futures):
                    _print_progress(future.result())

    results = [
        json.loads(
            _trial_path(destination, task["model_index"], task["trial_seed"]).read_text(encoding="utf-8")
        )
        for task in tasks
    ]
    summary = summarize_trials(results, len(tasks), config)
    _write_json(destination / "summary.json", summary)
    return summary


def _print_progress(result: dict) -> None:
    status = "SUCCESS" if result["end_to_end_success"] else "FAIL"
    print(
        f"M{result['model_idx']} seed {result['extraction_seed']}: {status} | "
        f"{result['total_queries']:,} queries | "
        f"B_err=({result['L1_max_bias_err']}, {result['B2_max_bias_err']}, {result['L_last_max_bias_err']}) | "
        f"{result['elapsed_time']:.1f}s",
        flush=True,
    )
