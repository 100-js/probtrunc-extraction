"""Offline diagnostic of Liu's first-layer ratio estimator at grid-sized eps.

This bypasses critical-point *discovery*: target weights create known first-layer
critical points solely for diagnosis and scoring.  It cannot establish whether
the attacker can find those points.  The actual Liu ``get_ratios_lstsq`` path
still makes every oracle query, chooses random sign directions, resolves their
relative signs, and solves its least-squares system without modifications.

The default only prints the experiment plan.  ``--execute`` performs the
bounded oracle measurements and writes a complete per-case JSON record.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from first_layer_adapter import ContinuousOracleBridge, build_decoy_model
from oracle_bridge import (
    AttackBudgetExceeded,
    LIU_ROOT,
    ROOT,
    ProbTruncBridge,
    install_bridge,
    load_integer_target,
)


HERE = Path(__file__).resolve().parent
DEFAULT_MODEL = ROOT / "checkpoints" / "digits" / "(64, 20, 20, 10)-01.npz"


def _positive_int_list(raw: str) -> list[int]:
    try:
        values = [int(value.strip()) for value in raw.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Expected comma-separated integers") from exc
    if not values or any(value < 1 for value in values) or values != sorted(set(values)):
        raise argparse.ArgumentTypeError("Expected strictly increasing positive integers")
    return values


def _positive_float_list(raw: str) -> list[float]:
    try:
        values = [float(value.strip()) for value in raw.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Expected comma-separated numbers") from exc
    if not values or any(not np.isfinite(value) or value <= 0 for value in values):
        raise argparse.ArgumentTypeError("Expected finite positive numbers")
    if values != sorted(set(values)):
        raise argparse.ArgumentTypeError("Expected strictly increasing numbers")
    return values


def _physical_queries(bridge) -> int:
    return bridge.oracle.total_queries if hasattr(bridge, "oracle") else bridge.total_queries


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _hidden_margin_score(
    point: np.ndarray,
    weights: list[np.ndarray],
    biases: list[np.ndarray],
    q: int,
    neuron_index: int,
    radius: float,
) -> float:
    """Conservative L-infinity clearance from every other ReLU hyperplane.

    It is a selection diagnostic only; the attack function never receives the
    real parameters.  The bound propagates absolute weights through ReLUs.
    """
    value = point
    bound = np.ones_like(point)
    clearances: list[float] = []
    for layer, (weight_int, bias_int) in enumerate(zip(weights[:-1], biases[:-1])):
        weight = weight_int.astype(np.float64) / q
        bias = bias_int.astype(np.float64) / q
        preactivation = value @ weight + bias
        bound = bound @ np.abs(weight)
        ratios = np.abs(preactivation) / np.maximum(radius * bound, np.finfo(float).tiny)
        if layer == 0:
            ratios = np.delete(ratios, neuron_index)
        clearances.extend(float(ratio) for ratio in ratios)
        value = np.maximum(preactivation, 0.0)
    return min(clearances) if clearances else float("inf")


def choose_diagnostic_points(
    weights: list[np.ndarray],
    biases: list[np.ndarray],
    q: int,
    *,
    neuron_index: int,
    seed: int,
    point_scale: float,
    candidate_pool: int,
    point_count: int,
    max_eps: float,
) -> list[dict]:
    """Predeclare CPs by geometry, never by oracle outcome or ratio accuracy."""
    target_weight = weights[0][:, neuron_index].astype(np.float64) / q
    target_bias = float(biases[0][neuron_index]) / q
    squared_norm = float(target_weight @ target_weight)
    if squared_norm == 0:
        raise ValueError("Target neuron has zero first-layer weight")
    rng = np.random.default_rng(seed)
    radius = max_eps + 0.5 / q  # input rounding can move a coordinate by 0.5 LSB
    candidates = []
    for index in range(candidate_pool):
        base = rng.normal(0.0, point_scale, size=weights[0].shape[0])
        point = base - ((base @ target_weight + target_bias) / squared_norm) * target_weight
        score = _hidden_margin_score(point, weights, biases, q, neuron_index, radius)
        candidates.append({
            "candidate_index": index,
            "point": point.tolist(),
            "target_preactivation_residual": float(point @ target_weight + target_bias),
            "other_relu_margin_over_radius_bound": score,
            "other_relus_certified_stable_in_probe_ball": bool(score > 1.0),
        })
    candidates.sort(key=lambda item: (-item["other_relu_margin_over_radius_bound"], item["candidate_index"]))
    return candidates[:point_count]


def prepare_point_centers(
    points: list[dict], weights: list[np.ndarray], biases: list[np.ndarray], q: int,
    *, neuron_index: int, snap_to_grid: bool,
) -> list[dict]:
    """Keep exact diagnostic CPs fixed, but optionally supply rounded centers."""
    target_weight = weights[0][:, neuron_index].astype(np.float64) / q
    target_bias = float(biases[0][neuron_index]) / q
    prepared = []
    for point_record in points:
        exact = np.asarray(point_record["point"], dtype=np.float64)
        snapped_codes = np.rint(exact * q).astype(np.int64)
        snapped = snapped_codes.astype(np.float64) / q
        supplied = snapped if snap_to_grid else exact
        prepared.append({
            **point_record,
            "exact_point": exact.tolist(),
            "snapped_point": snapped.tolist(),
            "snapped_input_codes": snapped_codes.tolist(),
            "supplied_center": supplied.tolist(),
            "exact_target_preactivation_residual": float(exact @ target_weight + target_bias),
            "snapped_target_preactivation_residual": float(snapped @ target_weight + target_bias),
            "supplied_center_target_preactivation_residual": float(supplied @ target_weight + target_bias),
            "snap_linf_input_lsbs": float(np.max(np.abs(snapped - exact)) * q),
        })
    return prepared


def projective_error(ratio: np.ndarray, true_weight: np.ndarray) -> dict:
    """The first-layer ratio is identifiable up to nonzero sign and scale."""
    estimate = np.asarray(ratio, dtype=np.float64).reshape(-1)
    target = np.asarray(true_weight, dtype=np.float64).reshape(-1)
    if estimate.shape != target.shape or not np.all(np.isfinite(estimate)):
        raise ValueError("Ratio has wrong shape or nonfinite entries")
    estimate_norm = float(np.linalg.norm(estimate))
    target_norm = float(np.linalg.norm(target))
    if estimate_norm == 0 or target_norm == 0:
        raise ValueError("Cannot score a zero ratio vector")
    cosine = float(np.clip(abs(estimate @ target) / (estimate_norm * target_norm), 0.0, 1.0))
    return {
        "absolute_cosine": cosine,
        "unsigned_projective_error": float(np.sqrt(max(0.0, 1.0 - cosine * cosine))),
    }


class QueryRouter:
    """Keep one Liu patch installed while isolating counters for every case."""

    def __init__(self, q: int) -> None:
        self.q = q
        self.active_bridge = None
        self.input_codes: list[np.ndarray] = []
        self._global_config = None

    def select(self, bridge) -> None:
        self.active_bridge = bridge
        self.input_codes = []
        bridge._global_config = self._global_config

    def predict_manual_fast(self, x, weights, biases):
        if self.active_bridge is None:
            raise RuntimeError("No active oracle case")
        samples = np.asarray(x, dtype=np.float64)
        if samples.ndim == 1:
            samples = samples[None, :]
        result = self.active_bridge.predict_manual_fast(samples, weights, biases)
        codes = np.rint(samples * self.q).astype(np.int64)
        self.input_codes.extend(codes)
        return result

    def collision_summary(self) -> dict:
        """Count exact input-code collisions in Liu's eps/3 inner pairs."""
        complete_stencils = len(self.input_codes) // 4
        pair_count = 2 * complete_stencils
        vector_collisions = 0
        matching_coordinates = 0
        coordinate_count = 0
        for offset in range(0, complete_stencils * 4, 4):
            for left, right in ((offset, offset + 1), (offset + 2, offset + 3)):
                equal = self.input_codes[left] == self.input_codes[right]
                vector_collisions += bool(np.all(equal))
                matching_coordinates += int(np.sum(equal))
                coordinate_count += int(equal.size)
        return {
            "input_code_sequence_sha256": hashlib.sha256(
                np.ascontiguousarray(np.stack(self.input_codes), dtype=np.int64).tobytes()
                if self.input_codes else b""
            ).hexdigest(),
            "input_code_sequence_length": len(self.input_codes),
            "completed_four_query_stencils": complete_stencils,
            "unpaired_logical_samples": len(self.input_codes) % 4,
            "inner_pairs": pair_count,
            "identical_input_code_pairs": vector_collisions,
            "identical_input_code_pair_fraction": vector_collisions / pair_count if pair_count else None,
            "identical_input_code_coordinate_fraction": (
                matching_coordinates / coordinate_count if coordinate_count else None
            ),
            "note": "Input-code equality, not output equality or zero finite difference.",
        }


def _new_bridge(mode: str, repeats: int, weights, biases, q, shape, options):
    if mode == "pt":
        return ProbTruncBridge(
            weights, biases, q, shape,
            noise_seed=options.noise_seed,
            repeats=repeats,
            max_queries=options.max_queries_per_case,
            max_seconds=options.max_seconds_per_case,
        )
    return ContinuousOracleBridge(
        weights, biases, q, shape,
        grid_input=(mode == "float-grid"),
        max_queries=options.max_queries_per_case,
        max_seconds=options.max_seconds_per_case,
    )


def run_case(
    options,
    *,
    point_record: dict,
    eps_record: dict,
    mode: str,
    repeats: int,
    router: QueryRouter,
    model,
    get_ratios_lstsq,
    acceptable_failure,
    weights,
    biases,
    q: int,
    shape,
) -> dict:
    bridge = _new_bridge(mode, repeats, weights, biases, q, shape, options)
    router.select(bridge)
    point = np.asarray(point_record["supplied_center"], dtype=np.float64)
    true_weight = weights[0][:, options.neuron_index].astype(np.float64) / q
    liu_args = SimpleNamespace(
        dataset="cifar10", real_attack=1, layerID=1,
        output_index=options.output_index, debug=False,
    )
    record = {
        "candidate_index": point_record["candidate_index"],
        "eps_label": eps_record["label"],
        "eps": eps_record["eps"],
        "eps_in_input_lsbs": eps_record["eps_in_input_lsbs"],
        "mode": mode,
        "repeats": repeats,
        "direction_seed": options.direction_seed + point_record["candidate_index"],
        "center_policy": "snapped-input-grid" if options.snap_critical_point_to_grid else "exact-real",
        "exact_target_preactivation_residual": point_record["exact_target_preactivation_residual"],
        "snapped_target_preactivation_residual": point_record["snapped_target_preactivation_residual"],
        "supplied_center_target_preactivation_residual": point_record["supplied_center_target_preactivation_residual"],
        "status": "running",
    }
    attack_rng_state = np.random.get_state()
    np.random.seed(record["direction_seed"])
    stdout = io.StringIO()
    started = time.monotonic()
    try:
        with contextlib.redirect_stdout(stdout):
            ratios, h_matrices = get_ratios_lstsq(
                liu_args, model, [point], [], [], eps=eps_record["eps"]
            )
        if len(ratios) != 1 or len(h_matrices) != 1:
            raise ValueError("Liu returned an unexpected number of ratio systems")
        ratio = np.asarray(ratios[0], dtype=np.float64).reshape(-1)
        record["ratio"] = ratio.tolist()
        record["h_matrix_rank"] = int(np.linalg.matrix_rank(h_matrices[0]))
        if not np.any(ratio):
            # Liu can return an all-zero least-squares vector when every
            # finite difference vanishes on the input grid.  This has no
            # projective direction and is a measured failure, not a harness
            # exception or an extraction success.
            record["status"] = "degenerate_ratio"
        else:
            record.update(projective_error(ratio, true_weight))
            record["status"] = "completed"
    except acceptable_failure as exc:
        record.update(status="liu_acceptable_failure", error=repr(exc))
    except AttackBudgetExceeded as exc:
        record.update(status="budget_exceeded", error=str(exc))
    except Exception as exc:
        record.update(status="error", error=repr(exc), traceback=traceback.format_exc())
    finally:
        np.random.set_state(attack_rng_state)
        record["elapsed_seconds"] = time.monotonic() - started
        record["logical_oracle_queries"] = bridge.logical_queries
        record["physical_oracle_queries"] = _physical_queries(bridge)
        record["input_code_collisions"] = router.collision_summary()
        record["liu_stdout_tail"] = stdout.getvalue()[-2000:]
    return record


def summarize(results: list[dict], eps_records: list[dict], point_count: int, float_tol: float) -> list[dict]:
    """Preserve unconditional denominators and float-qualified paired subsets."""
    summaries = []
    for eps in eps_records:
        rows = [row for row in results if row["eps_label"] == eps["label"]]
        float_rows = {row["candidate_index"]: row for row in rows if row["mode"] == "float"}
        qualified = {
            index for index, row in float_rows.items()
            if row["status"] == "completed" and row["unsigned_projective_error"] <= float_tol
        }
        mode_rows = []
        for mode, repeats in sorted({(row["mode"], row["repeats"]) for row in rows}):
            case_rows = [row for row in rows if row["mode"] == mode and row["repeats"] == repeats]
            successful = [row for row in case_rows if row["status"] == "completed"]
            paired = [row for row in case_rows if row["candidate_index"] in qualified]
            paired_successful = [row for row in paired if row["status"] == "completed"]
            status_counts = {
                status: sum(row["status"] == status for row in case_rows)
                for status in sorted({row["status"] for row in case_rows})
            }
            mode_rows.append({
                "mode": mode,
                "repeats": repeats,
                "status_counts": status_counts,
                "unconditional_attempted": len(case_rows),
                "unconditional_completed": len(successful),
                "total_logical_oracle_queries": sum(row["logical_oracle_queries"] for row in case_rows),
                "total_physical_oracle_queries": sum(row["physical_oracle_queries"] for row in case_rows),
                "unconditional_error_le_float_tolerance": sum(
                    row["unsigned_projective_error"] <= float_tol for row in successful
                ),
                "float_qualified_attempted": len(paired),
                "float_qualified_completed": len(paired_successful),
                "float_qualified_error_le_float_tolerance": sum(
                    row["unsigned_projective_error"] <= float_tol for row in paired_successful
                ),
                "float_qualified_median_projective_error": (
                    float(np.median([row["unsigned_projective_error"] for row in paired_successful]))
                    if paired_successful else None
                ),
            })
        summaries.append({
            "eps_label": eps["label"],
            "eps": eps["eps"],
            "eps_in_input_lsbs": eps["eps_in_input_lsbs"],
            "predeclared_point_count": point_count,
            "float_positive_controls": len(qualified),
            "float_positive_control_max_projective_error": float_tol,
            "cases": mode_rows,
        })
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--neuron-index", type=int, default=0)
    parser.add_argument("--output-index", type=int, default=0)
    parser.add_argument("--point-seed", type=int, default=42)
    parser.add_argument("--point-scale", type=float, default=10.0)
    parser.add_argument("--candidate-pool", type=int, default=64)
    parser.add_argument("--points", type=int, default=5)
    parser.add_argument("--snap-critical-point-to-grid", action="store_true",
                        help="Supply round(q*exact critical point)/q to Liu's ratio code")
    parser.add_argument("--direction-seed", type=int, default=123)
    parser.add_argument("--noise-seed", type=int, default=1337)
    parser.add_argument("--decoy-seed", type=int, default=271828)
    parser.add_argument("--multipliers", type=_positive_float_list, default=[1, 2, 3, 4, 6, 8],
                        help="Comma-separated epsilon multiples of 1/q")
    parser.add_argument("--original-eps", type=float, default=1e-4,
                        help="Liu's first attempted epsilon in the 64-20-20-10 first-layer path")
    parser.add_argument("--pt-repeats", type=_positive_int_list, default=[1, 4, 16])
    parser.add_argument("--float-control-max-error", type=float, default=0.05)
    parser.add_argument("--max-queries-per-case", type=int, default=20_000)
    parser.add_argument("--max-seconds-per-case", type=float, default=60.0)
    parser.add_argument("--output", type=Path, default=HERE / "liu_ratio_grid_epsilon.json")
    parser.add_argument("--execute", action="store_true")
    options = parser.parse_args()

    if (options.points < 1 or options.candidate_pool < options.points or options.point_scale <= 0
            or not np.isfinite(options.point_scale) or options.original_eps <= 0
            or not np.isfinite(options.original_eps) or options.float_control_max_error < 0
            or options.float_control_max_error > 1 or options.max_queries_per_case < 1
            or options.max_seconds_per_case <= 0):
        parser.error("Invalid point count, epsilon, tolerance, or per-case budget")
    model_path = options.model.resolve(strict=True)
    weights, biases, q, shape = load_integer_target(model_path)
    if len(weights) < 2 or not 0 <= options.neuron_index < shape[1]:
        parser.error("Expected a hidden-layer target and valid --neuron-index")
    if not 0 <= options.output_index < shape[-1]:
        parser.error("--output-index must name a target output")

    eps_records = [{
        "label": "liu_first_1e-4",
        "eps": options.original_eps,
        "eps_in_input_lsbs": options.original_eps * q,
    }] + [{
        "label": f"{multiplier:g}_input_lsb",
        "eps": multiplier / q,
        "eps_in_input_lsbs": multiplier,
    } for multiplier in options.multipliers]
    exact_points = choose_diagnostic_points(
        weights, biases, q,
        neuron_index=options.neuron_index,
        seed=options.point_seed,
        point_scale=options.point_scale,
        candidate_pool=options.candidate_pool,
        point_count=options.points,
        max_eps=max(item["eps"] for item in eps_records),
    )
    points = prepare_point_centers(
        exact_points, weights, biases, q,
        neuron_index=options.neuron_index,
        snap_to_grid=options.snap_critical_point_to_grid,
    )
    modes = [("float", 1), ("float-grid", 1)] + [("pt", n) for n in options.pt_repeats]
    result = {
        "status": "planned" if not options.execute else "running",
        "scope": "Known true first-layer critical points; ratio recovery only; no critical-point search.",
        "center_policy": "snapped-input-grid" if options.snap_critical_point_to_grid else "exact-real",
        "center_policy_note": (
            "The exact critical point is target-derived in both modes; snapped mode supplies its nearest 1/q input-grid point to Liu. This does not model critical-point discovery."
        ),
        "mode_semantics": {
            "float": "Unrounded input; decoded integer target weights evaluated by float64 layers.",
            "float-grid": "Input rounded to 1/q, then float64 layers; not deterministic layerwise truncation.",
            "pt": "Input rounded to 1/q, then probabilistic truncation at each affine layer; independent repeated outputs are averaged.",
        },
        "liu_stencil": "Each get_second_grad_unsigned uses four oracle samples; inner offset eps/3, so eps=1/q does not guarantee a full-grid-unit inner displacement.",
        "model": str(model_path),
        "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "shape": shape,
        "q": q,
        "neuron_index": options.neuron_index,
        "output_index": options.output_index,
        "point_selection": {
            "seed": options.point_seed,
            "normal_base_scale": options.point_scale,
            "candidate_pool": options.candidate_pool,
            "selection": "Highest target-derived, conservative clearance from other hidden ReLU boundaries; no oracle outcome used.",
            "selected_points": points,
        },
        "direction_seed_rule": "direction_seed + candidate_index; same directions for all oracle modes and epsilons of a point",
        "direction_seed": options.direction_seed,
        "noise_seed": options.noise_seed,
        "decoy_seed": options.decoy_seed,
        "eps_schedule": eps_records,
        "modes": [{"mode": mode, "repeats": repeats} for mode, repeats in modes],
        "float_positive_control_max_projective_error": options.float_control_max_error,
        "max_physical_queries_per_case": options.max_queries_per_case,
        "max_seconds_per_case": options.max_seconds_per_case,
        "numpy_version": np.__version__,
        "cases": [],
        "summary": [],
    }
    if not options.execute:
        print(json.dumps(result, indent=2, sort_keys=True))
        return

    output_path = options.output.resolve()
    if output_path.exists():
        parser.error(f"Refusing to overwrite existing result: {output_path}")
    model = build_decoy_model(shape, q, options.decoy_seed)
    from signature_recovery.utils import getAllWeightsAndBiases
    decoy_weights, decoy_biases = getAllWeightsAndBiases(model)
    for layer, (decoy_weight, decoy_bias) in enumerate(zip(decoy_weights, decoy_biases)):
        if np.array_equal(decoy_weight, weights[layer].astype(np.float64) / q) or np.array_equal(
            decoy_bias, biases[layer].astype(np.float64) / q
        ):
            raise RuntimeError(f"Decoy layer {layer + 1} accidentally matches target parameters")

    import model_extraction  # noqa: F401 -- load Liu modules before query patching
    from signature_recovery.compute_critical_point_values import get_ratios_lstsq
    from signature_recovery.utils import AcceptableFailure

    router = QueryRouter(q)
    result["query_patched_modules"] = install_bridge(router)
    result["liu_ratio_source_sha256"] = hashlib.sha256(
        (LIU_ROOT / "signature_recovery" /
         "compute_critical_point_values.py").read_bytes()
    ).hexdigest()
    _write_json(output_path, result)
    try:
        for eps_record in eps_records:
            for point in points:
                for mode, repeats in modes:
                    case = run_case(
                        options, point_record=point, eps_record=eps_record,
                        mode=mode, repeats=repeats, router=router, model=model,
                        get_ratios_lstsq=get_ratios_lstsq,
                        acceptable_failure=AcceptableFailure,
                        weights=weights, biases=biases, q=q, shape=shape,
                    )
                    result["cases"].append(case)
                    result["summary"] = summarize(
                        result["cases"], eps_records, len(points), options.float_control_max_error
                    )
                    _write_json(output_path, result)
                    print(
                        f"{eps_record['label']} cp={point['candidate_index']} "
                        f"{mode} r{repeats}: {case['status']} "
                        f"error={case.get('unsigned_projective_error')} "
                        f"logical={case['logical_oracle_queries']} "
                        f"physical={case['physical_oracle_queries']}",
                        file=sys.stderr,
                    )
        result["status"] = "completed"
    except KeyboardInterrupt:
        result["status"] = "interrupted"
    finally:
        result["summary"] = summarize(result["cases"], eps_records, len(points), options.float_control_max_error)
        _write_json(output_path, result)
    print(json.dumps({"status": result["status"], "output": str(output_path), "summary": result["summary"]}, indent=2))


if __name__ == "__main__":
    main()
