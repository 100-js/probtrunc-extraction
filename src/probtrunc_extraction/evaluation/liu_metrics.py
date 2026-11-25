"""Independent post-hoc evaluation for Liu et al. (2026) first-layer recovery outputs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment


def _load_first_layer(path: Path, weight_key: str, bias_key: str) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        weight = np.asarray(archive[weight_key], dtype=np.float64)
        bias = np.asarray(archive[bias_key], dtype=np.float64).reshape(-1)
    if weight.ndim != 2 or min(weight.shape) == 0 or bias.shape != (weight.shape[1],):
        raise ValueError(f"Invalid first-layer weight or bias shape in {path}")
    if not np.isfinite(weight).all() or not np.isfinite(bias).all():
        raise ValueError(f"Non-finite first-layer parameters in {path}")
    return weight, bias


def match_directions(
    target_weight: np.ndarray,
    recovered_weight: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Match recovered neuron weight vectors to target columns up to sign via Hungarian assignment."""
    if target_weight.shape[0] != recovered_weight.shape[0]:
        raise ValueError("Input dimension mismatch between target and recovered first layer")
    target_norm = np.linalg.norm(target_weight, axis=0)
    recovered_norm = np.linalg.norm(recovered_weight, axis=0)
    if np.any(target_norm == 0) or np.any(recovered_norm == 0):
        raise ValueError("First-layer directions require nonzero weight vectors")
    target_unit = (target_weight / target_norm).T
    recovered_unit = (recovered_weight / recovered_norm).T
    chordal = np.minimum(
        np.linalg.norm(target_unit[:, None] - recovered_unit[None, :], axis=2),
        np.linalg.norm(target_unit[:, None] + recovered_unit[None, :], axis=2),
    )
    target_idx, rec_idx = linear_sum_assignment(chordal)
    return target_idx, rec_idx, chordal[target_idx, rec_idx]


def _find_snapshot(run_dir: Path, filename: str) -> Path | None:
    direct = run_dir / filename
    if direct.is_file():
        return direct
    matches = sorted(run_dir.rglob(filename))
    return matches[0] if matches else None


def score_liu_run(
    target_path: str | Path,
    *,
    run_dir: str | Path | None = None,
    signature_path: str | Path | None = None,
    recovered_path: str | Path | None = None,
    manifest_path: str | Path | None = None,
    direction_tolerance: float = 0.05,
) -> dict:
    """Score a live Liu first-layer run directory (or explicit NPZ snapshots) against a target."""
    if not np.isfinite(direction_tolerance) or not 0.0 < direction_tolerance < 1.0:
        raise ValueError("direction_tolerance must be in (0, 1)")

    target_file = Path(target_path)
    target_sha256 = hashlib.sha256(target_file.read_bytes()).hexdigest()
    target_w, target_b_codes = _load_first_layer(target_file, "W_0", "B_0")

    sig_file = Path(signature_path) if signature_path is not None else None
    rec_file = Path(recovered_path) if recovered_path is not None else None
    man_file = Path(manifest_path) if manifest_path is not None else None

    if run_dir is not None:
        base = Path(run_dir)
        if man_file is None and (base / "run_manifest.json").is_file():
            man_file = base / "run_manifest.json"
        if sig_file is None:
            sig_file = _find_snapshot(base, "signature_checkpoint_1.npz")
        if rec_file is None:
            rec_file = _find_snapshot(base, "recovered_1.npz")

    if sig_file is None and rec_file is None:
        raise FileNotFoundError("Provide --run-dir containing Liu NPZ outputs or explicit --signature/--recovered paths")

    report: dict = {
        "target": str(target_file),
        "target_sha256": target_sha256,
        "target_neurons": int(target_w.shape[1]),
    }

    if man_file is not None and man_file.is_file():
        manifest = json.loads(man_file.read_text(encoding="utf-8"))
        report["manifest"] = {
            "path": str(man_file),
            "status": manifest.get("status"),
            "mode": manifest.get("mode"),
            "repeats": manifest.get("repeats"),
            "seed": manifest.get("seed"),
            "physical_oracle_queries": manifest.get("physical_oracle_queries"),
            "logical_liu_queries": manifest.get("logical_liu_queries"),
            "elapsed_seconds": manifest.get("elapsed_seconds"),
            "telemetry": manifest.get("telemetry"),
        }

    if sig_file is not None and sig_file.is_file():
        sig_w, _ = _load_first_layer(sig_file, "extracted_normal", "extracted_bias")
        _, _, sig_errors = match_directions(target_w, sig_w)
        report["signature"] = {
            "path": str(sig_file),
            "recovered_neurons": int(sig_w.shape[1]),
            "matching_directions": int(np.count_nonzero(sig_errors <= direction_tolerance)),
            "unsigned_chordal_tolerance": float(direction_tolerance),
            "median_chordal_error": float(np.median(sig_errors)),
            "max_chordal_error": float(np.max(sig_errors)),
        }

    if rec_file is not None and rec_file.is_file():
        final_w, final_b = _load_first_layer(rec_file, "extracted_normal", "extracted_bias")
        target_idx, final_idx, final_errors = match_directions(target_w, final_w)
        matched_target = target_w[:, target_idx]
        matched_final = final_w[:, final_idx]
        scale = np.sum(matched_target * matched_final, axis=0) / np.sum(matched_final**2, axis=0)
        positive_sign_matches = int(np.count_nonzero(scale > 0))
        bias_errors = np.abs(scale * final_b[final_idx] - target_b_codes[target_idx])
        report["final_first_layer"] = {
            "path": str(rec_file),
            "recovered_neurons": int(final_w.shape[1]),
            "matching_directions": int(np.count_nonzero(final_errors <= direction_tolerance)),
            "positive_sign_matches": positive_sign_matches,
            "median_chordal_error": float(np.median(final_errors)),
            "max_chordal_error": float(np.max(final_errors)),
            "absolute_bias_error_lsb": {
                "min": float(np.min(bias_errors)),
                "median": float(np.median(bias_errors)),
                "max": float(np.max(bias_errors)),
            },
        }

    return report
