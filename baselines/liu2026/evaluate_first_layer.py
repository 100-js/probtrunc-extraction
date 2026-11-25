"""Score Liu's recovered first layer against a saved target, after the attack.

The evaluator is intentionally separate from the attack and oracle bridge.
Target parameters are used only to compute these post-run metrics.  ReLU
neurons are identifiable up to permutation and *positive* rescaling, so the
comparison uses a one-to-one minimum-error assignment of augmented (weight,
bias) vectors after optimizing that rescaling.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment


THRESHOLDS = (0.01, 0.1)


def _load_target(path: Path) -> tuple[np.ndarray, np.ndarray, int]:
    with np.load(path, allow_pickle=False) as data:
        q = int(data["q"])
        weight = np.asarray(data["W_0"], dtype=np.float64)
        bias = np.asarray(data["B_0"], dtype=np.float64).reshape(-1)
    if q <= 0 or weight.ndim != 2 or bias.shape != (weight.shape[1],):
        raise ValueError("Invalid first-layer parameters in target NPZ")
    if not np.all(np.isfinite(weight)) or not np.all(np.isfinite(bias)):
        raise ValueError("Target first layer contains non-finite values")
    return weight / q, bias / q, q


def _load_recovered(path: Path, input_dim: int) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        weight = np.asarray(data["extracted_normal"], dtype=np.float64)
        bias = np.asarray(data["extracted_bias"], dtype=np.float64).reshape(-1)
    if weight.ndim != 2 or weight.shape[0] != input_dim:
        raise ValueError(f"Recovered weight must have shape ({input_dim}, n)")
    if bias.shape != (weight.shape[1],):
        raise ValueError("Recovered bias count does not match recovered neurons")
    if not np.all(np.isfinite(weight)) or not np.all(np.isfinite(bias)):
        raise ValueError("Recovered first layer contains non-finite values")
    return weight, bias


def evaluate(
    target_weight: np.ndarray,
    target_bias: np.ndarray,
    recovered_weight: np.ndarray,
    recovered_bias: np.ndarray,
    *,
    unsigned: bool = False,
) -> dict:
    """Return scale/permutation invariant first-layer recovery metrics.

    For true augmented vector ``t`` and recovered vector ``e``, the pairwise
    error is ``min_{a >= 0} ||a*e-t||_2 / ||t||_2``.  A reversed sign is
    therefore an error even though it defines the same hyperplane, because
    it changes which side of a ReLU is active.  With ``unsigned=True``, use
    ``a`` of either sign for the intermediate signature checkpoint before
    Liu's sign-recovery stage. Missing true neurons receive error 1.0.
    """

    target_weight = np.asarray(target_weight, dtype=np.float64)
    recovered_weight = np.asarray(recovered_weight, dtype=np.float64)
    target_bias = np.asarray(target_bias, dtype=np.float64).reshape(-1)
    recovered_bias = np.asarray(recovered_bias, dtype=np.float64).reshape(-1)
    if (
        target_weight.ndim != 2
        or recovered_weight.ndim != 2
        or target_weight.shape[0] != recovered_weight.shape[0]
        or target_bias.shape != (target_weight.shape[1],)
        or recovered_bias.shape != (recovered_weight.shape[1],)
    ):
        raise ValueError("Incompatible first-layer dimensions")
    if target_weight.shape[1] == 0:
        raise ValueError("Target first layer must contain at least one neuron")
    if not all(np.all(np.isfinite(array)) for array in (
        target_weight, target_bias, recovered_weight, recovered_bias
    )):
        raise ValueError("First-layer parameters must be finite")

    true_vectors = np.vstack((target_weight, target_bias[None, :]))
    recovered_vectors = np.vstack((recovered_weight, recovered_bias[None, :]))
    true_norm = np.linalg.norm(true_vectors, axis=0)
    if np.any(true_norm == 0):
        raise ValueError("Zero target neuron has no defined projective error")

    true_count = true_vectors.shape[1]
    recovered_count = recovered_vectors.shape[1]
    all_errors = np.ones(true_count, dtype=np.float64)
    matches = []
    if recovered_count:
        recovered_norm = np.linalg.norm(recovered_vectors, axis=0)
        unit_recovered = np.divide(
            recovered_vectors,
            recovered_norm[None, :],
            out=np.zeros_like(recovered_vectors),
            where=recovered_norm[None, :] > 0,
        )
        cosine = (true_vectors / true_norm[None, :]).T @ unit_recovered
        cosine = np.clip(np.abs(cosine) if unsigned else cosine, 0.0, 1.0)
        pair_errors = np.sqrt(np.maximum(0.0, 1.0 - cosine**2))
        true_indices, recovered_indices = linear_sum_assignment(pair_errors)
        all_errors[true_indices] = pair_errors[true_indices, recovered_indices]
        matches = [
            {
                "true_index": int(i),
                "recovered_index": int(j),
                "projective_error": float(pair_errors[i, j]),
            }
            for i, j in zip(true_indices, recovered_indices)
        ]

    matched_true = {item["true_index"] for item in matches}
    coverage = {
        f"error_le_{threshold:g}": sum(
            item["projective_error"] <= threshold for item in matches
        ) / true_count
        for threshold in THRESHOLDS
    }
    return {
        "metric": (
            "unsigned_augmented_l2_projective_error" if unsigned
            else "positive_scale_augmented_l2_projective_error"
        ),
        "target_neurons": true_count,
        "recovered_neurons": recovered_count,
        "unmatched_true_indices": [i for i in range(true_count) if i not in matched_true],
        "matches": matches,
        "coverage": coverage,
        "error_summary": {
            "median": float(np.median(all_errors)),
            "p90": float(np.percentile(all_errors, 90)),
            "max": float(np.max(all_errors)),
        },
    }


def compare_reference(candidate: dict, reference: dict, tolerance: float) -> dict:
    """Check a declared absolute error tolerance against a float baseline."""

    if tolerance < 0 or not np.isfinite(tolerance):
        raise ValueError("Comparison tolerance must be finite and nonnegative")
    if candidate["target_first_layer_sha256"] != reference.get("target_first_layer_sha256"):
        raise ValueError("Float reference was evaluated on a different target first layer")
    if candidate["metric"] != reference.get("metric"):
        raise ValueError("Float reference uses a different evaluation metric")
    coverage_ok = {
        key: candidate["coverage"][key] >= value
        for key, value in reference["coverage"].items()
    }
    error_ok = {
        key: candidate["error_summary"][key] <= value + tolerance
        for key, value in reference["error_summary"].items()
    }
    reference_nontrivial = reference["coverage"].get("error_le_0.1", 0.0) > 0
    return {
        "absolute_error_tolerance": tolerance,
        "reference_nontrivial": reference_nontrivial,
        "coverage_at_least_reference": coverage_ok,
        "error_within_reference_plus_tolerance": error_ok,
        "matches_float_reference": (
            reference_nontrivial and all(coverage_ok.values()) and all(error_ok.values())
        ),
    }


def _target_digest(weight: np.ndarray, bias: np.ndarray, q: int) -> str:
    digest = hashlib.sha256()
    for array in (np.asarray([q], dtype="<i8"), weight, bias):
        value = np.ascontiguousarray(array, dtype="<f8" if array.dtype.kind == "f" else "<i8")
        digest.update(np.asarray(value.shape, dtype="<i8").tobytes())
        digest.update(value.tobytes())
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="Target NPZ with W_0, B_0, q")
    parser.add_argument("--recovered", type=Path, required=True, help="Liu recovered_1.npz")
    parser.add_argument("--reference-metrics", type=Path, help="JSON from a float-reference evaluation")
    parser.add_argument("--match-tolerance", type=float, default=0.001,
                        help="Absolute normalized-error allowance above float reference")
    parser.add_argument("--unsigned", action="store_true",
                        help="Evaluate a pre-sign-recovery signature checkpoint")
    parser.add_argument("--output", type=Path, help="Optional destination JSON; otherwise stdout only")
    args = parser.parse_args()

    model_path = args.model.resolve(strict=True)
    recovered_path = args.recovered.resolve(strict=True)
    true_weight, true_bias, q = _load_target(model_path)
    recovered_weight, recovered_bias = _load_recovered(recovered_path, true_weight.shape[0])
    result = evaluate(true_weight, true_bias, recovered_weight, recovered_bias, unsigned=args.unsigned)
    result.update({
        "model": str(model_path),
        "recovered": str(recovered_path),
        "q": q,
        "target_first_layer_sha256": _target_digest(true_weight, true_bias, q),
    })
    if args.reference_metrics is not None:
        reference_path = args.reference_metrics.resolve(strict=True)
        reference = json.loads(reference_path.read_text(encoding="utf-8"))
        result["float_reference"] = str(reference_path)
        result["reference_comparison"] = compare_reference(
            result, reference, args.match_tolerance
        )
    output = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        destination = args.output.resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(output, encoding="utf-8")
    print(output, end="")


if __name__ == "__main__":
    main()
