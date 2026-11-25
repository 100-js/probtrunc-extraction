"""Run Liu et al.'s first-layer extraction under the macro-scaled attack policy.

This driver executes the first-layer pipeline (``layerID == 1, real_attack == 1``):
  1. Macro critical-point line search (``offset_codes ~ floor(N(0,1)*1e13)``,
     ``direction_codes ~ floor(N(0,1)*1e11)``, ``t in [0, 10]``, no-wrap input
     preflight, 3-query midpoint curvature screen ``> 10.0``, and ``all_grid``
     recursive bisection tolerances with side step ``10^10 / ||d_code||_inf``).
  2. Macro ratio estimation (``gather_ratios`` with ``GRAD_EPS = 10^10 / q``
     and macro retries ``[GRAD_EPS, GRAD_EPS/2, GRAD_EPS/4]``, plus fixing the
     upstream Python loop-scoping bug when all retries raise ``AcceptableFailure``).
  3. Graph clustering and post-cluster coordinate verification
     (``cluster_critical_points`` with macro ``eps = 10^10 / q``).
  4. Initial signature checkpoint save (``signature_checkpoint_1.npz``).
  5. Witness refill to ``target_length`` per component using scale-matched
     hyperplane-membership tolerance ``1e-6 * ||point||_2`` and macro second-gradient
     verification ``eps = 10^10 / q``, followed by SVD hyperplane precision
     refinement (``improve_precision``).
  6. Macro first-layer sign recovery (``continuous_input_point ~ floor(N(0,1)*1e13)/q``,
     ``eps = 10^9 / q``) and final signed parameter save (``recovered_1.npz``).

Target weights and biases are held exclusively by the oracle bridge and are never
exposed to the attack code, which receives an independent decoy Keras model and
runtime guards via ``first_layer_adapter``.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import datetime as dt
import hashlib
import json
import math
import os
import random
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
from scipy.linalg import null_space

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SRC_ROOT = ROOT / "src"
LIU_ROOT = HERE / "third_party" / "liu_extraction"
for source_root in (SRC_ROOT, LIU_ROOT, HERE):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

from first_layer_adapter import (  # noqa: E402
    ContinuousOracleBridge,
    build_decoy_model,
    install_first_layer_guards,
)
from oracle_bridge import (  # noqa: E402
    AttackBudgetExceeded,
    ProbTruncBridge,
    install_bridge,
    load_integer_target,
)


class StopAfterSignature(RuntimeError):
    """ Internal control exception used by --stop-after-signature."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _default_output_path(model_path: Path, mode: str, repeats: int, seed: int) -> Path:
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
    tag = f"macro_{mode}_r{repeats}_s{seed}_{stamp}"
    return HERE / "runs_first_layer" / model_path.stem / tag


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _queries(bridge: ProbTruncBridge | ContinuousOracleBridge) -> int:
    if isinstance(bridge, ProbTruncBridge):
        return int(bridge.oracle.total_queries)
    return int(bridge.total_queries)


def _audited_constants_table(q: int, options: argparse.Namespace) -> list[dict[str, Any]]:
    grad_eps = float(options.ratio_eps_codes) / float(q)
    sign_eps = float(options.sign_eps_codes) / float(q)
    offset_decoded = float(options.offset_scale_codes) / float(q)
    direction_decoded = float(options.direction_scale_codes) / float(q)
    return [
        {
            "stage": "1_critical_point_search_line",
            "location": "signature_recovery/critical_point_search.py:sweep_for_critical_points",
            "original_value": "offset ~ N(0,1), direction ~ N(0,1), t in [-1e5, 1e5]",
            "physical_units": "decoded input units and scalar line parameter t",
            "macro_value": {
                "offset_scale_codes": options.offset_scale_codes,
                "offset_scale_decoded": offset_decoded,
                "direction_scale_codes": options.direction_scale_codes,
                "direction_scale_decoded": direction_decoded,
                "t_interval": [options.t_low, options.t_high],
                "curvature_screen_threshold_decoded": options.curvature_threshold,
                "max_input_code_preflight_limit": options.max_input_code_limit,
            },
        },
        {
            "stage": "1_critical_point_search_bisection",
            "location": "signature_recovery/critical_point_search.py:do_better_sweep",
            "original_value": {
                "SKIP_LINEAR_TOL": 1e-8,
                "linear_floor": 0.0,
                "min_interval_width": 1e-8,
                "center_match_tol": 1e-6,
                "side_diff_step_t": 1e-4,
                "grad_match_tol": 1e-7,
            },
            "physical_units": "decoded output units (linearity/center), scalar t (width/step), decoded output per unit t (slope)",
            "macro_value": {
                "skip_linear_tol": options.skip_linear_tol,
                "linear_floor": options.linear_floor,
                "min_interval_width": options.min_interval_width,
                "center_match_tol": options.center_match_tol,
                "sweep_delta_x_codes": options.sweep_delta_x_codes,
                "side_diff_step_t_formula": "sweep_delta_x_codes / max(abs(direction_codes))",
                "grad_match_tol": options.grad_match_tol,
            },
        },
        {
            "stage": "2_ratio_estimation",
            "location": "signature_recovery/compute_critical_point_values.py:gather_ratios",
            "original_value": {
                "GRAD_EPS": 1e-4,
                "retry_schedule": [1e-4, 1e-5, 1e-6],
                "loop_scoping_bug": "ratio/h_matrix not reset before retry loop",
            },
            "physical_units": "decoded input coordinate step per +/-1 coordinate",
            "macro_value": {
                "ratio_eps_codes": options.ratio_eps_codes,
                "GRAD_EPS_decoded": grad_eps,
                "retry_schedule_decoded": [grad_eps, grad_eps / 2.0, grad_eps / 4.0],
                "loop_scoping_fix": "reset ratio=None, h_matrix=None before retry loop and drop point if all retries fail",
            },
        },
        {
            "stage": "3_clustering_and_component_filter",
            "location": "signature_recovery/cluster_critical_points/cluster_critical_points.py",
            "original_value": {
                "BLOCK_ERROR_TOL": 1e-3,
                "post_cluster_coord_eps": 1e-4,
                "post_cluster_flat_tol": 1e-5,
            },
            "physical_units": "dimensionless normalized ratio L1 diff; decoded input step along axis e_k; dimensionless slope jump",
            "macro_value": {
                "BLOCK_ERROR_TOL": 1e-3,
                "post_cluster_coord_eps_decoded": grad_eps,
                "post_cluster_flat_tol": 1e-5,
                "empty_component_guard": True,
            },
        },
        {
            "stage": "4_witness_refill_and_precision_improvement",
            "location": "signature_recovery/signature_recovery.py:1719-1796 & precision_improvement.py",
            "original_value": {
                "target_length": "int(dimOfInput * 1.5) = 96",
                "point_membership_tol": 1e-5,
                "quality_check_eps": 1e-4,
                "quality_check_flat_threshold": 1e-5,
                "svd_nullspace_tol": 1e-10,
            },
            "physical_units": "decoded preactivation residual; decoded input step; dimensionless slope jump; SVD singular values",
            "macro_value": {
                "target_length": options.target_length,
                "point_membership_tol_formula": f"{options.membership_rel_tol} * max(1.0, ||point||_2)",
                "quality_check_eps_decoded": grad_eps,
                "quality_check_directions": options.refill_check_directions,
                "quality_check_flat_threshold": 1e-5,
                "svd_nullspace_policy": "outlier-trimmed centered TLS right singular vector Vh[-1, :] of (X - x_bar) with b = -x_bar @ w, verified by Liu diff < 1e-2 against extracted_weight",
            },
        },
        {
            "stage": "5_sign_recovery",
            "location": "signature_recovery/signature_recovery.py:1820-1904",
            "original_value": {
                "continuous_input_point": "Uniform(-10, 10)",
                "eps": 1e-3,
                "active_soln_threshold": 1e-5,
            },
            "physical_units": "decoded input coordinates; decoded input step; dimensionless active-neuron gain",
            "macro_value": {
                "continuous_input_point_codes": "floor(N(0,1) * offset_scale_codes)",
                "sign_eps_codes": options.sign_eps_codes,
                "sign_eps_decoded": sign_eps,
                "relative_step_ratio": float(options.sign_eps_codes) / float(options.offset_scale_codes),
                "active_soln_threshold": 1e-5,
                "lstsq_relative_residual_guard": 1e-4,
            },
        },
    ]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        default=ROOT / "checkpoints" / "digits" / "(64, 20, 20, 10)-01.npz",
        help="Path to saved integer target model (.npz).",
    )
    parser.add_argument(
        "--mode",
        choices=("pt", "float", "float-grid"),
        default="pt",
        help="Oracle mode: stochastic PT, continuous float, or deterministic input-grid float.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Oracle responses averaged per logical query in PT mode.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Base attack seed.")
    parser.add_argument("--noise-seed", type=int, default=1337, help="Independent PT oracle RNG seed.")
    parser.add_argument("--decoy-seed", dest="decoy_seed", type=int, default=271828, help="Decoy model seed.")
    parser.add_argument("--count", type=int, default=500, help="Initial critical points to collect before clustering.")
    parser.add_argument("--output-index", type=int, default=0, help="Logit coordinate queried by Liu.")
    parser.add_argument("--max-queries", type=int, default=1_000_000, help="Hard physical oracle query cap.")
    parser.add_argument("--max-seconds", type=float, default=600.0, help="Hard wall-time cap in seconds.")
    parser.add_argument("--output-path", type=Path, default=None, help="Directory for logs, checkpoints, and manifest.")
    parser.add_argument(
        "--stop-after-signature",
        action="store_true",
        help="Stop right after writing signature_checkpoint_1.npz (unsigned checkpoint).",
    )
    # Macro coordinate & step policy parameters (fixed defaults from verified macro diagnostics)
    parser.add_argument("--offset-scale-codes", type=float, default=1e13)
    parser.add_argument("--direction-scale-codes", type=float, default=1e11)
    parser.add_argument("--t-low", type=float, default=0.0)
    parser.add_argument("--t-high", type=float, default=10.0)
    parser.add_argument("--curvature-threshold", type=float, default=10.0)
    parser.add_argument("--max-input-code-limit", type=int, default=200_000_000_000_000)
    parser.add_argument("--sweep-delta-x-codes", type=int, default=10_000_000_000)
    parser.add_argument("--skip-linear-tol", type=float, default=1e-8)
    parser.add_argument("--linear-floor", type=float, default=1.0)
    parser.add_argument("--min-interval-width", type=float, default=1e-8)
    parser.add_argument("--center-match-tol", type=float, default=5.0)
    parser.add_argument("--grad-match-tol", type=float, default=25.0)
    parser.add_argument("--ratio-eps-codes", type=int, default=10_000_000_000)
    parser.add_argument("--sign-eps-codes", type=int, default=1_000_000_000)
    parser.add_argument(
        "--target-length",
        type=int,
        default=96,
        help="Target witness critical points per neuron for improve_precision (default 96 = 1.5 * 64).",
    )
    parser.add_argument(
        "--membership-rel-tol",
        type=float,
        default=1e-6,
        help="Relative hyperplane membership tolerance (multiplied by ||point||_2) during witness refill.",
    )
    parser.add_argument(
        "--refill-check-directions",
        type=int,
        default=50,
        help="Number of random directions checked per point during witness refill (upstream default: 50).",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually run the extraction; without this flag, prints the 0-query dry-run manifest.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    options = parse_args(argv)
    if options.mode != "pt" and options.repeats != 1:
        raise ValueError("--repeats != 1 is only meaningful in --mode pt")
    if options.count < 1:
        raise ValueError("--count must be positive")
    if options.max_queries is not None and options.max_queries < 1:
        raise ValueError("--max-queries must be positive")
    if options.max_seconds is not None and options.max_seconds <= 0:
        raise ValueError("--max-seconds must be positive")

    model_path = options.model.resolve(strict=True)
    weights, biases, q, shape = load_integer_target(model_path)
    if not 0 <= options.output_index < shape[-1]:
        raise ValueError(f"--output-index must be in [0, {shape[-1] - 1}]")

    if options.mode == "pt":
        bridge: ProbTruncBridge | ContinuousOracleBridge = ProbTruncBridge(
            weights,
            biases,
            q,
            shape,
            noise_seed=options.noise_seed,
            repeats=options.repeats,
            max_queries=options.max_queries,
            max_seconds=options.max_seconds,
        )
    else:
        bridge = ContinuousOracleBridge(
            weights,
            biases,
            q,
            shape,
            grid_input=(options.mode == "float-grid"),
            max_queries=options.max_queries,
            max_seconds=options.max_seconds,
        )

    # Import Liu modules and install oracle-only bridge + truth-leak guards.
    import model_extraction  # noqa: F401
    from global_vars import GlobalConfig
    from signature_recovery import (
        compute_critical_point_values as ccpv_module,
        critical_point_search as cps_module,
        precision_improvement as pi_module,
        signature_recovery as recovery_module,
        utils as utils_module,
    )
    from signature_recovery.cluster_critical_points import (
        cluster_critical_points as ccp_module,
    )
    from whitebox import getWhiteboxSignatures

    query_patches = install_bridge(bridge)
    guard_patches = install_first_layer_guards()
    model = build_decoy_model(shape, q, options.decoy_seed)

    GlobalConfig.set_save_queries = False
    GlobalConfig.output_index = options.output_index
    if shape[-1] > 1:
        GlobalConfig.dataset = "cifar10"

    output_path = (
        options.output_path.resolve()
        if options.output_path is not None
        else _default_output_path(model_path, options.mode, options.repeats, options.seed)
    )
    recovered_path = output_path / "recovered_1.npz"
    pre_precision_recovered_path = output_path / "recovered_pre_precision_1.npz"
    signature_path = output_path / "signature_checkpoint_1.npz"

    grad_eps_decoded = float(options.ratio_eps_codes) / float(q)
    sign_eps_decoded = float(options.sign_eps_codes) / float(q)

    telemetry: dict[str, Any] = {
        "search_stage": {
            "lines_attempted": 0,
            "lines_skipped_preflight": 0,
            "lines_screened_linear": 0,
            "lines_swept": 0,
            "sweeps_completed": 0,
            "unfiltered_candidates_yielded": 0,
            "initial_candidates_collected": 0,
            "physical_queries": 0,
            "logical_queries": 0,
            "elapsed_seconds": 0.0,
        },
        "ratio_stage": {
            "candidates_attempted": 0,
            "candidates_succeeded": 0,
            "acceptable_failures": 0,
            "retry_level_successes": {"0": 0, "1": 0, "2": 0},
            "physical_queries": 0,
            "logical_queries": 0,
            "elapsed_seconds": 0.0,
        },
        "clustering_stage": {
            "good_critical_points": 0,
            "graph_components_recovered": 0,
            "witness_counts_before_coord_filter": [],
            "witness_counts_after_coord_filter": [],
            "empty_component_guards_triggered": 0,
            "physical_queries": 0,
            "logical_queries": 0,
            "elapsed_seconds": 0.0,
        },
        "precision_stage": {
            "target_length": int(options.target_length),
            "initial_min_witness_count": 0,
            "final_min_witness_count": 0,
            "final_witness_counts": [],
            "refill_lines_attempted": 0,
            "refill_lines_screened_linear": 0,
            "refill_lines_swept": 0,
            "refill_candidates_pulled": 0,
            "refill_membership_hits": 0,
            "refill_quality_passed": 0,
            "refill_quality_rejected": 0,
            "svd_components_improved": 0,
            "svd_components_kept_original": 0,
            "svd_per_component": [],
            "physical_queries": 0,
            "logical_queries": 0,
            "elapsed_seconds": 0.0,
        },
        "sign_stage": {
            "attempts": 0,
            "lstsq_relative_residual": None,
            "active_neuron_count_at_probe": 0,
            "signs_recovered_count": 0,
            "recovered_signs": [],
            "physical_queries": 0,
            "logical_queries": 0,
            "elapsed_seconds": 0.0,
        },
    }

    manifest: dict[str, Any] = {
        "baseline_variant": "modified_liu_macro_first_layer_v1",
        "executed": bool(options.execute),
        "status": "dry_run" if not options.execute else "running",
        "stage": "first_layer_only",
        "last_reached_stage": "init",
        "stop_after_signature": bool(options.stop_after_signature),
        "model_path": str(model_path),
        "model_sha256": _sha256(model_path),
        "target_Access": "oracle_only_decoy_model",
        "decoy_seed": options.decoy_seed,
        "q": q,
        "shape": list(shape),
        "mode": options.mode,
        "repeats": options.repeats,
        "seed": options.seed,
        "noise_seed": options.noise_seed if options.mode == "pt" else None,
        "count": options.count,
        "output_index": options.output_index,
        "liu_Multi_output_branch": "cifar10" if shape[-1] > 1 else "mnist",
        "max_queries": options.max_queries,
        "max_seconds": options.max_seconds,
        "physical_oracle_queries": 0,
        "logical_liu_queries": 0,
        "output_path": str(output_path),
        "recovered_path": str(recovered_path),
        "pre_precision_recovered_path": str(pre_precision_recovered_path),
        "signature_checkpoint_path": str(signature_path),
        "query_patches": query_patches,
        "guard_patches": guard_patches,
        "macro_policy": {
            "offset_scale_codes": options.offset_scale_codes,
            "direction_scale_codes": options.direction_scale_codes,
            "t_low": options.t_low,
            "t_high": options.t_high,
            "curvature_threshold": options.curvature_threshold,
            "max_input_code_limit": options.max_input_code_limit,
            "sweep_delta_x_codes": options.sweep_delta_x_codes,
            "skip_linear_tol": options.skip_linear_tol,
            "linear_floor": options.linear_floor,
            "min_interval_width": options.min_interval_width,
            "center_match_tol": options.center_match_tol,
            "grad_match_tol": options.grad_match_tol,
            "ratio_eps_codes": options.ratio_eps_codes,
            "ratio_eps_decoded": grad_eps_decoded,
            "sign_eps_codes": options.sign_eps_codes,
            "sign_eps_decoded": sign_eps_decoded,
            "target_length": options.target_length,
            "membership_rel_tol": options.membership_rel_tol,
            "refill_check_directions": options.refill_check_directions,
        },
        "audited_constants": _audited_constants_table(q, options),
        "telemetry": telemetry,
    }

    if not options.execute:
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return

    # ------------------------------------------------------------------
    # Install in-memory macro stage patches
    # ------------------------------------------------------------------
    input_width = shape[0]
    line_counter = {"index": 0}

    def _run_macro_single_line_sweep(
        offset: np.ndarray,
        direction: np.ndarray,
        diff_step: float,
        weights_all: list[np.ndarray],
        biases_all: list[np.ndarray],
    ) -> list[np.ndarray]:
        memo: dict[float, float] = {}
        relus: list[np.ndarray] = []

        def memo_forward_pass(t_val: float) -> float:
            if t_val not in memo:
                sample = (offset + direction * t_val)[np.newaxis, :]
                out = utils_module.predict_manual_fast(sample, weights_all, biases_all)
                GlobalConfig.crit_query_count += 1
                memo[t_val] = float(out.flatten()[options.output_index])
            return memo[t_val]

        def search(cur_low: float, cur_high: float) -> None:
            mid = (cur_low + cur_high) * 0.5
            y1 = f_low = memo_forward_pass(cur_low)
            f_mid = memo_forward_pass(mid)
            y2 = f_high = memo_forward_pass(cur_high)

            interval_width = cur_high - cur_low
            bound = max(options.skip_linear_tol * (interval_width ** 0.5), options.linear_floor)
            if abs(f_mid - (f_high + f_low) * 0.5) < bound:
                return
            if interval_width < options.min_interval_width:
                return

            q1 = (cur_low + mid) * 0.5
            q3 = (cur_high + mid) * 0.5
            f_q1 = memo_forward_pass(q1)
            f_q3 = memo_forward_pass(q3)

            m1 = (f_q1 - f_low) / (q1 - cur_low)
            m2 = (f_q3 - f_high) / (q3 - cur_high)

            if m1 != m2:
                d = cur_high - cur_low
                denom = m1 - m2
                alpha = (y2 - y1 - d * m2) / (d * denom)
                x_should_be = cur_low + (y2 - y1 - d * m2) / denom
                height_should_be = y1 + m1 * (y2 - y1 - d * m2) / denom

                if 0.25 + 1e-5 < alpha < 0.75 - 1e-5:
                    real_h_at_x = memo_forward_pass(x_should_be)
                    if abs(real_h_at_x - height_should_be) < options.center_match_tol:
                        eighth_left = x_should_be - diff_step
                        eighth_right = x_should_be + diff_step
                        grad_left = (memo_forward_pass(eighth_left) - real_h_at_x) / (eighth_left - x_should_be)
                        grad_right = (memo_forward_pass(eighth_right) - real_h_at_x) / (eighth_right - x_should_be)
                        if (
                            abs(grad_left - m1) <= options.grad_match_tol
                            and abs(grad_right - m2) <= options.grad_match_tol
                        ):
                            relus.append(offset + direction * x_should_be)
                            return

            search(cur_low, mid)
            search(mid, cur_high)

        search(float(options.t_low), float(options.t_high))
        return relus

    def macro_sweep_for_critical_points(args_obj, model_obj, std=1, dataset="mnist"):
        weights_all, biases_all = utils_module.getAllWeightsAndBiases(model_obj)
        while True:
            is_refill = manifest.get("last_reached_stage") == "precision_improvement"
            line_seed = int(options.seed) + int(line_counter["index"])
            line_counter["index"] += 1
            if is_refill:
                telemetry["precision_stage"]["refill_lines_attempted"] += 1
            else:
                telemetry["search_stage"]["lines_attempted"] += 1

            rng = np.random.RandomState(line_seed)
            offset_codes = np.floor(rng.randn(input_width) * options.offset_scale_codes).astype(np.int64)
            direction_codes = np.floor(rng.randn(input_width) * options.direction_scale_codes).astype(np.int64)
            direction_inf_codes = int(np.max(np.abs(direction_codes)))
            if direction_inf_codes == 0:
                if not is_refill:
                    telemetry["search_stage"]["lines_skipped_preflight"] += 1
                continue

            max_endpoint_code = int(
                max(
                    np.max(np.abs(offset_codes + np.rint(options.t_low * direction_codes).astype(np.int64))),
                    np.max(np.abs(offset_codes + np.rint(options.t_high * direction_codes).astype(np.int64))),
                )
            )
            max_code_with_probe = max_endpoint_code + max(
                int(options.sweep_delta_x_codes), int(options.ratio_eps_codes) * 2
            )
            if max_code_with_probe > int(options.max_input_code_limit):
                if not is_refill:
                    telemetry["search_stage"]["lines_skipped_preflight"] += 1
                continue

            offset = offset_codes.astype(np.float64) / float(q)
            direction = direction_codes.astype(np.float64) / float(q)
            diff_step = float(options.sweep_delta_x_codes) / float(direction_inf_codes)

            t_mid = 0.5 * (float(options.t_low) + float(options.t_high))
            screen_pts = np.stack(
                [
                    offset + direction * float(options.t_low),
                    offset + direction * t_mid,
                    offset + direction * float(options.t_high),
                ],
                axis=0,
            )
            screen_vals = utils_module.predict_manual_fast(screen_pts, weights_all, biases_all)[
                :, options.output_index
            ]
            GlobalConfig.crit_query_count += 3
            midpoint_residual = abs(float(screen_vals[1] - 0.5 * (screen_vals[0] + screen_vals[2])))
            if midpoint_residual <= float(options.curvature_threshold):
                if is_refill:
                    telemetry["precision_stage"]["refill_lines_screened_linear"] += 1
                else:
                    telemetry["search_stage"]["lines_screened_linear"] += 1
                continue

            if is_refill:
                telemetry["precision_stage"]["refill_lines_swept"] += 1
            else:
                telemetry["search_stage"]["lines_swept"] += 1
            cps = _run_macro_single_line_sweep(offset, direction, diff_step, weights_all, biases_all)
            if not is_refill:
                telemetry["search_stage"]["sweeps_completed"] += 1
                telemetry["search_stage"]["unfiltered_candidates_yielded"] += len(cps)
            print(
                f"[Macro Sweep] line_seed={line_seed} refill={is_refill} residual={midpoint_residual:.3f} "
                f"found={len(cps)}"
            )
            for cp in cps:
                yield cp

    cps_module.sweep_for_critical_points = macro_sweep_for_critical_points
    recovery_module.sweep_for_critical_points = macro_sweep_for_critical_points

    def macro_gather_ratios(
        args_obj,
        this_layer_critical_point_count,
        critical_points_yielder,
        A,
        B,
        check_fn,
        LAYER,
        COUNT,
        model_obj,
        dimInput,
        dimOfPrevLayer,
        dimOfLayer,
        special,
        eps=1e-6,
    ):
        t0 = time.monotonic()
        phys_before = _queries(bridge)
        log_before = bridge.logical_queries
        manifest["last_reached_stage"] = "ratio_estimation"

        point_index = this_layer_critical_point_count
        new_this_layer_critical_points = []
        new_remaining_critical_points = []
        bad_critical_point_indices = []
        retry_schedule = [grad_eps_decoded, grad_eps_decoded / 2.0, grad_eps_decoded / 4.0]
        try:
            for index in range(len(critical_points_yielder)):
                point = critical_points_yielder[index]
                if not check_fn(point):
                    continue
                telemetry["ratio_stage"]["candidates_attempted"] += 1
                ratio = None
                h_matrix = None
                succeeded_level = None

                for retry_idx, eps_val in enumerate(retry_schedule):
                    try:
                        ratios, h_matrices = ccpv_module.get_ratios_lstsq(
                            args_obj, model_obj, [point], A, B, eps=eps_val
                        )
                        if ratios is not None and len(ratios) > 0:
                            arr = np.asarray(ratios[0], dtype=np.float64).flatten()
                            if np.all(np.isfinite(arr)) and np.max(np.abs(arr)) > 0:
                                ratio = arr
                                h_matrix = h_matrices[0]
                                succeeded_level = retry_idx
                                break
                    except utils_module.AcceptableFailure:
                        print(f"[Macro Ratio] Point {index}: AcceptableFailure at EPS={eps_val}")

                if ratio is None or h_matrix is None:
                    telemetry["ratio_stage"]["acceptable_failures"] += 1
                    continue

                telemetry["ratio_stage"]["candidates_succeeded"] += 1
                telemetry["ratio_stage"]["retry_level_successes"][str(succeeded_level)] += 1
                new_this_layer_critical_points.append(
                    recovery_module.CriticalPoint(point_index, ratio, point, h_matrix)
                )
                point_index += 1
        finally:
            telemetry["ratio_stage"]["physical_queries"] += _queries(bridge) - phys_before
            telemetry["ratio_stage"]["logical_queries"] += bridge.logical_queries - log_before
            telemetry["ratio_stage"]["elapsed_seconds"] = round(
                telemetry["ratio_stage"]["elapsed_seconds"] + (time.monotonic() - t0), 3
            )
        print(
            f"[Macro Ratio] Completed: succeeded={telemetry['ratio_stage']['candidates_succeeded']}/"
            f"{telemetry['ratio_stage']['candidates_attempted']} "
            f"(failures={telemetry['ratio_stage']['acceptable_failures']})"
        )
        return (
            new_this_layer_critical_points,
            new_remaining_critical_points,
            bad_critical_point_indices,
        )

    ccpv_module.gather_ratios = macro_gather_ratios
    recovery_module.gather_ratios = macro_gather_ratios

    # Patch utils.get_second_grad_unsigned when called with eps=1e-4
    orig_utils_second_grad = utils_module.get_second_grad_unsigned

    def macro_utils_second_grad(args_obj, x, direction, weights, biases, eps, eps2):
        if abs(float(eps) - 1e-4) < 1e-12:
            eps = grad_eps_decoded
            eps2 = grad_eps_decoded / 3.0
        return orig_utils_second_grad(args_obj, x, direction, weights, biases, eps, eps2)

    utils_module.get_second_grad_unsigned = macro_utils_second_grad
    recovery_module.get_second_grad_unsigned = macro_utils_second_grad
    ccp_module.get_second_grad_unsigned = macro_utils_second_grad

    # Wrap cluster_critical_points to record telemetry and guard against empty component filtering
    def macro_cluster_critical_points(
        args_obj,
        model_obj,
        critical_points,
        noise_critical_points,
        special_sols,
        ker_spaces,
        good_indices,
        bad_indices,
        expected_neurons,
        LAYER,
        debug=False,
        savePath=None,
    ):
        t0 = time.monotonic()
        phys_before = _queries(bridge)
        log_before = bridge.logical_queries
        manifest["last_reached_stage"] = "clustering"

        good_indices_not_merged = good_indices.copy()
        telemetry["clustering_stage"]["good_critical_points"] = len(good_indices)

        all_ratios = np.array([cp.ratio for cp in critical_points], dtype=np.float64)
        all_criticals = np.array([cp.point for cp in critical_points], dtype=np.float64)

        t_merge_good_CPs_start = time.time()
        components, guessed_rows = ccp_module.graph_solve(
            args_obj,
            model_obj,
            critical_points,
            good_indices_not_merged,
        )
        partial_rows = [guessed_row for guessed_row, _, _ in guessed_rows]
        normalize_errors = [normalize_error for _, _, normalize_error in guessed_rows]

        good_components_indices = []
        for i in range(len(components)):
            if LAYER == 0:
                if normalize_errors[i] > 1 and len(components[i]) < 3:
                    continue
                good_components_indices.append(i)
            else:
                if normalize_errors[i] > 1e-2 and len(components[i]) <= 5:
                    continue
                good_components_indices.append(i)

        t_merge_good_CPs = time.time() - t_merge_good_CPs_start
        components = [components[i] for i in good_components_indices]
        partial_rows = [partial_rows[i] for i in good_components_indices]
        normalize_errors = [normalize_errors[i] for i in good_components_indices]

        telemetry["clustering_stage"]["graph_components_recovered"] = len(components)
        telemetry["clustering_stage"]["witness_counts_before_coord_filter"] = [
            len(c) for c in components
        ]

        t_filter_critical_points_start = time.time()
        if args_obj.layerID == 1 and args_obj.real_attack == 1:
            weights_all, biases_all = utils_module.getAllWeightsAndBiases(model_obj)
            flat_threshold = 1e-5
            flat_num = 2

            for com_index in range(len(components)):
                component = components[com_index]
                removed_indices = []
                for point_index in range(len(component)):
                    point = all_criticals[component[point_index]]
                    bad_quality_points_count = 0
                    for i in range(len(point)):
                        d = np.sign(np.random.normal(0, 1, point.shape))
                        ratio_val = ccp_module.get_second_grad_unsigned(
                            args_obj,
                            point,
                            d,
                            weights_all,
                            biases_all,
                            grad_eps_decoded,
                            grad_eps_decoded / 3.0,
                        )
                        if abs(ratio_val) < flat_threshold:
                            bad_quality_points_count += 1
                            if bad_quality_points_count >= flat_num:
                                removed_indices.append(point_index)
                                break
                filtered_comp = np.delete(component, removed_indices).tolist()
                if len(filtered_comp) == 0 and len(component) > 0:
                    filtered_comp = list(component)
                    telemetry["clustering_stage"]["empty_component_guards_triggered"] += 1
                components[com_index] = filtered_comp
                partial_rows[com_index] = all_ratios[components[com_index][0]]

            guessed_rows = []
            for c_count, component in enumerate(components):
                potential_matrix_rows = all_ratios[list(component)]
                guessed_row = potential_matrix_rows[0]
                max_element = np.max(np.abs(guessed_row))
                guessed_row = guessed_row / max_element
                guessed_rows.append([guessed_row, 0, 0])
            partial_rows = [guessed_row for guessed_row, _, _ in guessed_rows]

        t_filter_critical_points = time.time() - t_filter_critical_points_start
        telemetry["clustering_stage"]["witness_counts_after_coord_filter"] = [
            len(c) for c in components
        ]
        telemetry["clustering_stage"]["physical_queries"] += _queries(bridge) - phys_before
        telemetry["clustering_stage"]["logical_queries"] += bridge.logical_queries - log_before
        telemetry["clustering_stage"]["elapsed_seconds"] = round(
            telemetry["clustering_stage"]["elapsed_seconds"] + (time.monotonic() - t0), 3
        )
        return (
            components,
            partial_rows,
            None,
            None,
            None,
            [t_merge_good_CPs, t_filter_critical_points],
        )

    ccp_module.cluster_critical_points = macro_cluster_critical_points
    recovery_module.cluster_critical_points = macro_cluster_critical_points

    def macro_improve_precision(
        args_obj,
        A,
        B,
        known_A,
        known_B,
        extracted_sequence,
        model_obj,
        dimInput,
        dimOfPrevLayer,
        dimOfLayer,
        critical_groups,
    ):
        weight_improved = np.array(known_A, dtype=np.float64, copy=True)
        bias_improved = np.array(known_B, dtype=np.float64, copy=True)

        for i in range(len(critical_groups)):
            extracted_weight = known_A[:, i].flatten().astype(np.float64)
            pts = np.asarray(critical_groups[i], dtype=np.float64)
            comp_info: dict[str, Any] = {
                "component_index": int(i),
                "num_points_initial": int(pts.shape[0]),
                "num_points_used": int(pts.shape[0]),
                "improved": False,
                "diff_vs_ratio": None,
                "sigma_63": None,
                "sigma_64": None,
            }
            if pts.ndim != 2 or pts.shape[0] < dimInput + 1:
                comp_info["reason"] = "insufficient_points"
                telemetry["precision_stage"]["svd_components_kept_original"] += 1
                telemetry["precision_stage"]["svd_per_component"].append(comp_info)
                continue

            pts_in = pts
            w_norm_init = float(np.max(np.abs(extracted_weight)))
            if w_norm_init > 0:
                proj = -(pts_in @ (extracted_weight / w_norm_init))
                dev_init = np.abs(proj - float(np.median(proj)))
                med_dev = float(np.median(dev_init))
                mask_init = dev_init <= max(5.0 * med_dev, 1e-6)
                if int(np.count_nonzero(mask_init)) >= dimInput + 1:
                    pts_in = pts_in[mask_init]

            x_bar = np.mean(pts_in, axis=0)
            X_c = pts_in - x_bar
            _, S_init, Vh_init = np.linalg.svd(X_c, full_matrices=False)
            w_init = Vh_init[-1, :].copy()
            w_scale_init = float(np.max(np.abs(w_init)))
            if w_scale_init > 0:
                residuals = np.abs(X_c @ (w_init / w_scale_init))
                med_res = float(np.median(residuals))
                cutoff = max(4.0 * med_res, 1e-6)
                inlier_mask = residuals <= cutoff
                if int(np.count_nonzero(inlier_mask)) >= dimInput + 1:
                    pts_in = pts_in[inlier_mask]
                    x_bar = np.mean(pts_in, axis=0)
                    X_c = pts_in - x_bar
                    _, S, Vh = np.linalg.svd(X_c, full_matrices=False)
                else:
                    S, Vh = S_init, Vh_init
            else:
                S, Vh = S_init, Vh_init

            comp_info["num_points_used"] = int(pts_in.shape[0])
            comp_info["sigma_63"] = float(S[-2]) if len(S) >= 2 else None
            comp_info["sigma_64"] = float(S[-1]) if len(S) >= 1 else None

            w_tls = Vh[-1, :].flatten().copy()
            b_tls = -float(np.dot(x_bar, w_tls))
            zero_index = np.where(np.abs(extracted_weight) < 1e-12)[0]
            vector_valid = True
            for l_idx in zero_index:
                if abs(w_tls[l_idx]) > 1e-2:
                    vector_valid = False
                    break
                w_tls[l_idx] = 0.0

            if not vector_valid:
                comp_info["reason"] = "zero_index_mismatch"
                telemetry["precision_stage"]["svd_components_kept_original"] += 1
                telemetry["precision_stage"]["svd_per_component"].append(comp_info)
                continue

            anchor_idx = int(np.argmax(np.abs(extracted_weight)))
            if extracted_weight[anchor_idx] == 0 or w_tls[anchor_idx] == 0:
                comp_info["reason"] = "zero_anchor"
                telemetry["precision_stage"]["svd_components_kept_original"] += 1
                telemetry["precision_stage"]["svd_per_component"].append(comp_info)
                continue

            factor = w_tls[anchor_idx] / extracted_weight[anchor_idx]
            diff = float(np.sum(np.abs(w_tls - factor * extracted_weight)))
            comp_info["diff_vs_ratio"] = diff

            if diff < 1e-2:
                x_opt = w_tls.copy()
                c_opt = float(-np.dot(x_bar, x_opt))
                sign_align = np.sign(extracted_weight[anchor_idx] / x_opt[anchor_idx])
                x_opt = x_opt * sign_align
                c_opt = c_opt * float(sign_align)
                max_abs = float(np.max(np.abs(x_opt)))
                if max_abs > 0:
                    x_opt = x_opt / max_abs
                    c_opt = c_opt / max_abs
                weight_improved[:, i] = x_opt
                bias_improved[i] = c_opt
                comp_info["improved"] = True
                telemetry["precision_stage"]["svd_components_improved"] += 1
            else:
                comp_info["reason"] = f"diff_exceeded_1e-2 ({diff:.6e})"
                telemetry["precision_stage"]["svd_components_kept_original"] += 1
            telemetry["precision_stage"]["svd_per_component"].append(comp_info)

        return weight_improved, bias_improved

    pi_module.improve_precision = macro_improve_precision
    recovery_module.improve_precision = macro_improve_precision

    # ------------------------------------------------------------------
    # Execute macro first-layer pipeline
    # ------------------------------------------------------------------
    output_path.mkdir(parents=True, exist_ok=False)
    manifest_path = output_path / "run_manifest.json"
    log_path = output_path / "liu_stdout.log"
    _write_json(manifest_path, manifest)
    started = time.monotonic()

    try:
        np.random.seed(options.seed)
        random.seed(options.seed)
        GlobalConfig.query_count = 0
        GlobalConfig.crit_query_count = 0
        GlobalConfig.SAVED_QUERIES = []
        GlobalConfig.set_save_queries = False

        stage_args = argparse.Namespace(
            dataset="cifar10",
            model=model_path.stem,
            device="cpu",
            layerID=1,
            tgtNeurons=None,
            eps=8,
            quantized=2,
            signRecoveryMethod="neuronWiggle",
            seed=options.seed,
            Count=options.count,
            cheat_noise=0,
            cheat_sign=0,
            output_index=options.output_index,
            real_attack=1,
            debug=0,
            output_path=str(output_path),
        )
        carrier_weights, carrier_biases = getWhiteboxSignatures(model, 1)
        save_path = output_path / "extract results" / f"Seed{options.seed}-Count{options.count}"
        (save_path / f"output_index_{options.output_index}").mkdir(parents=True, exist_ok=True)
        weights_all, biases_all = utils_module.getAllWeightsAndBiases(model)
        dim_of_input = int(shape[0])
        dim_of_prev_layer = int(shape[0])
        dim_of_layer = int(shape[1])

        with log_path.open("w", encoding="utf-8") as log, \
                contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            # Step 1: Collect Count initial candidate critical points
            manifest["last_reached_stage"] = "critical_point_search"
            t_search0 = time.monotonic()
            phys_search0 = _queries(bridge)
            log_search0 = bridge.logical_queries

            critical_points_yielder = recovery_module.sweep_for_critical_points(
                stage_args, model, 100, "cifar10"
            )
            remaining_critical_points = []
            while len(remaining_critical_points) < options.count:
                point = next(critical_points_yielder)
                remaining_critical_points.append(point)

            telemetry["search_stage"]["initial_candidates_collected"] = len(remaining_critical_points)
            telemetry["search_stage"]["physical_queries"] += _queries(bridge) - phys_search0
            telemetry["search_stage"]["logical_queries"] += bridge.logical_queries - log_search0
            telemetry["search_stage"]["elapsed_seconds"] = round(time.monotonic() - t_search0, 3)

            # Step 2 & 3: Ratio computation, clustering, and initial bias estimation
            (
                extracted_normal,
                extracted_bias,
                extracted_sign,
                extracted_sequence,
                still_remaining_critical_points,
                _T,
                _T_follow,
                clustered_components,
            ) = recovery_module.compute_layer_values(
                stage_args,
                remaining_critical_points,
                critical_points_yielder,
                model,
                carrier_weights,
                carrier_biases,
                1,
                dim_of_input,
                dim_of_prev_layer,
                dim_of_layer,
                {},
                str(save_path),
            )

            pre_precision_normal = np.asarray(extracted_normal, dtype=np.float64).copy()
            pre_precision_bias = np.asarray(extracted_bias, dtype=np.float64).copy()

            np.savez(
                signature_path,
                extracted_normal=pre_precision_normal,
                extracted_bias=pre_precision_bias,
                extracted_sign=np.asarray(extracted_sign),
                extracted_sequence=np.asarray(extracted_sequence),
            )
            manifest["last_reached_stage"] = "signature_checkpoint_saved"
            if options.stop_after_signature:
                raise StopAfterSignature

            # Step 4: Witness refill and precision improvement
            manifest["last_reached_stage"] = "precision_improvement"
            t_prec0 = time.monotonic()
            phys_prec0 = _queries(bridge)
            log_prec0 = bridge.logical_queries

            target_length = int(options.target_length)
            min_length = min(len(comp) for comp in clustered_components)
            telemetry["precision_stage"]["initial_min_witness_count"] = int(min_length)
            print(
                f"[Macro Precision] Initial component witness counts: "
                f"{[len(c) for c in clustered_components]} (min={min_length}, target={target_length})"
            )

            refill_rng = np.random.RandomState(options.seed + 500_000)
            while min_length < target_length:
                point = next(critical_points_yielder)
                telemetry["precision_stage"]["refill_candidates_pulled"] += 1
                output = np.matmul(point, extracted_normal) + extracted_bias
                point_l2 = max(1.0, float(np.linalg.norm(point)))
                membership_tol = float(options.membership_rel_tol) * point_l2

                for element_index in range(len(output)):
                    if abs(float(output[element_index])) < membership_tol:
                        telemetry["precision_stage"]["refill_membership_hits"] += 1
                        if len(clustered_components[element_index]) > target_length:
                            continue

                        filtered = False
                        bad_quality_points_count = 0
                        for _ in range(int(options.refill_check_directions)):
                            d = np.sign(refill_rng.normal(0.0, 1.0, size=point.shape))
                            d[d == 0] = 1.0
                            ratio_val = recovery_module.get_second_grad_unsigned(
                                stage_args,
                                point,
                                d,
                                weights_all,
                                biases_all,
                                grad_eps_decoded,
                                grad_eps_decoded / 3.0,
                            )
                            if abs(float(ratio_val)) < 1e-5:
                                bad_quality_points_count += 1
                                if bad_quality_points_count >= 2:
                                    filtered = True
                                    break
                        if filtered:
                            telemetry["precision_stage"]["refill_quality_rejected"] += 1
                            continue

                        telemetry["precision_stage"]["refill_quality_passed"] += 1
                        clustered_components[element_index].append(point)
                        min_length = min(len(comp) for comp in clustered_components)

            telemetry["precision_stage"]["final_min_witness_count"] = int(min_length)
            telemetry["precision_stage"]["final_witness_counts"] = [
                len(c) for c in clustered_components
            ]

            if target_length > 0:
                extracted_normal, extracted_bias = recovery_module.improve_precision(
                    stage_args,
                    carrier_weights[:-1],
                    carrier_biases[:-1],
                    extracted_normal,
                    extracted_bias,
                    extracted_sequence,
                    model,
                    dim_of_input,
                    dim_of_prev_layer,
                    dim_of_layer,
                    clustered_components,
                )

            telemetry["precision_stage"]["physical_queries"] += _queries(bridge) - phys_prec0
            telemetry["precision_stage"]["logical_queries"] += bridge.logical_queries - log_prec0
            telemetry["precision_stage"]["elapsed_seconds"] = round(time.monotonic() - t_prec0, 3)

            # Step 5: Macro sign recovery
            manifest["last_reached_stage"] = "sign_recovery"
            t_sign0 = time.monotonic()
            phys_sign0 = _queries(bridge)
            log_sign0 = bridge.logical_queries

            sign_rng = np.random.RandomState(options.seed + 900_000)
            recovered_signs = None
            for attempt in range(1, 9):
                telemetry["sign_stage"]["attempts"] = attempt
                point_codes = np.floor(
                    sign_rng.randn(dim_of_input) * options.offset_scale_codes
                ).astype(np.int64)
                continuous_input_point = point_codes.astype(np.float64) / float(q)

                matrix_Y = []
                vector_Z = []
                for _ in range(dim_of_input):
                    random_direction = np.sign(sign_rng.normal(0.0, 1.0, size=dim_of_input))
                    random_direction[random_direction == 0] = 1.0

                    output_point = np.matmul(continuous_input_point, extracted_normal) + extracted_bias
                    final_output_point = utils_module.predict_manual_fast(
                        continuous_input_point, weights_all, biases_all
                    )

                    point_eps = continuous_input_point + sign_eps_decoded * random_direction
                    output_point_eps = np.matmul(point_eps, extracted_normal) + extracted_bias
                    final_output_point_eps = utils_module.predict_manual_fast(
                        point_eps, weights_all, biases_all
                    )

                    matrix_Y.append(output_point_eps - output_point)
                    vector_Z.append(final_output_point_eps - final_output_point)

                matrix_Y_arr = np.asarray(matrix_Y, dtype=np.float64)
                vector_Z_arr = np.asarray(vector_Z, dtype=np.float64).reshape(len(vector_Z), -1)[
                    :, options.output_index
                ]
                soln, *_ = np.linalg.lstsq(matrix_Y_arr, vector_Z_arr, rcond=1e-6)
                fit_vec = matrix_Y_arr @ soln
                rel_res = float(
                    np.linalg.norm(fit_vec - vector_Z_arr)
                    / max(1e-12, float(np.linalg.norm(vector_Z_arr)))
                )
                telemetry["sign_stage"]["lstsq_relative_residual"] = rel_res
                print(f"[Macro Sign] attempt={attempt} lstsq_rel_residual={rel_res:.6e}")
                if rel_res <= 1e-4 or attempt == 8:
                    output_at_base = np.matmul(continuous_input_point, extracted_normal) + extracted_bias
                    original_output_signs = np.sign(output_at_base)
                    original_output_signs[original_output_signs == 0] = 1.0
                    signs_list = []
                    active_count = 0
                    for neuron_id in range(extracted_normal.shape[1]):
                        out_sign = float(original_output_signs[neuron_id])
                        if abs(float(soln[neuron_id])) < 1e-5:
                            signs_list.append(-1.0 * out_sign)
                        else:
                            active_count += 1
                            signs_list.append(1.0 * out_sign)
                    recovered_signs = np.asarray(signs_list, dtype=np.float64)
                    telemetry["sign_stage"]["active_neuron_count_at_probe"] = int(active_count)
                    telemetry["sign_stage"]["signs_recovered_count"] = int(
                        np.count_nonzero(recovered_signs != 0)
                    )
                    telemetry["sign_stage"]["recovered_signs"] = [int(s) for s in recovered_signs]
                    break

            extracted_normal = extracted_normal * recovered_signs
            extracted_bias = extracted_bias * recovered_signs
            extracted_sign = np.asarray(recovered_signs, dtype=np.int64)

            # Also orient the pre-precision checkpoint with the same recovered signs for comparison
            pre_precision_signs = []
            output_at_base_pre = (
                np.matmul(continuous_input_point, pre_precision_normal) + pre_precision_bias
            )
            orig_signs_pre = np.sign(output_at_base_pre)
            orig_signs_pre[orig_signs_pre == 0] = 1.0
            for neuron_id in range(pre_precision_normal.shape[1]):
                out_sign = float(orig_signs_pre[neuron_id])
                if abs(float(soln[neuron_id])) < 1e-5:
                    pre_precision_signs.append(-1.0 * out_sign)
                else:
                    pre_precision_signs.append(1.0 * out_sign)
            pre_precision_signs_arr = np.asarray(pre_precision_signs, dtype=np.float64)

            telemetry["sign_stage"]["physical_queries"] += _queries(bridge) - phys_sign0
            telemetry["sign_stage"]["logical_queries"] += bridge.logical_queries - log_sign0
            telemetry["sign_stage"]["elapsed_seconds"] = round(time.monotonic() - t_sign0, 3)

        np.savez(
            pre_precision_recovered_path,
            extracted_normal=np.asarray(pre_precision_normal * pre_precision_signs_arr),
            extracted_bias=np.asarray(pre_precision_bias * pre_precision_signs_arr),
            extracted_sign=np.asarray(pre_precision_signs_arr, dtype=np.int64),
            extracted_sequence=np.asarray(extracted_sequence),
        )
        np.savez(
            recovered_path,
            extracted_normal=np.asarray(extracted_normal),
            extracted_bias=np.asarray(extracted_bias),
            extracted_sign=np.asarray(extracted_sign),
            extracted_sequence=np.asarray(extracted_sequence),
        )
        manifest["last_reached_stage"] = "completed"
        manifest["status"] = "completed"
    except StopAfterSignature:
        manifest["status"] = "signature_only_completed"
    except AttackBudgetExceeded as exc:
        manifest["status"] = "censored"
        manifest["error_type"] = type(exc).__name__
        manifest["error"] = str(exc)
        manifest["traceback"] = traceback.format_exc()
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["error_type"] = type(exc).__name__
        manifest["error"] = str(exc)
        manifest["traceback"] = traceback.format_exc()
    finally:
        manifest["elapsed_seconds"] = round(time.monotonic() - started, 3)
        manifest["physical_oracle_queries"] = _queries(bridge)
        manifest["logical_liu_queries"] = bridge.logical_queries
        manifest["liu_query_counter"] = GlobalConfig.query_count
        manifest["liu_critical_query_counter"] = GlobalConfig.crit_query_count
        manifest["max_abs_input_code"] = bridge.max_abs_input_code
        manifest["signature_checkpoint_written"] = signature_path.is_file()
        manifest["pre_precision_recovered_written"] = pre_precision_recovered_path.is_file()
        manifest["recovered_written"] = recovered_path.is_file()
        manifest["log_path"] = str(log_path)
        _write_json(manifest_path, manifest)
        print(json.dumps(manifest, indent=2, sort_keys=True))

    if manifest["status"] not in ("completed", "signature_only_completed"):
        sys.exit(1)


if __name__ == "__main__":
    main()
