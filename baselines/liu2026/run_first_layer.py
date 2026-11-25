"""Run only Liu's first-layer extraction against a float or PT oracle.

Setup is checked without issuing queries by default.  ``--execute`` starts
the bounded extraction.  The target weights are held only by the oracle;
Liu's Keras carrier has independent decoy parameters of the same shapes.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import random
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np

from first_layer_adapter import (
    ContinuousOracleBridge,
    build_decoy_model,
    install_first_layer_guards,
)
from oracle_bridge import ROOT, ProbTruncBridge, install_bridge, load_integer_target


HERE = Path(__file__).resolve().parent


class StopAfterSignature(Exception):
    """Requested checkpoint at the end of Liu's signature-recovery stage."""


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _queries(bridge) -> int:
    return bridge.oracle.total_queries if hasattr(bridge, "oracle") else bridge.total_queries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        default=ROOT / "checkpoints" / "digits" / "(64, 20, 20, 10)-01.npz",
    )
    parser.add_argument("--mode", choices=("pt", "float", "float-grid"), default="pt")
    parser.add_argument("--repeats", type=int, default=1, help="Independent PT responses averaged per Liu query")
    parser.add_argument("--seed", type=int, default=42, help="Liu attack RNG seed")
    parser.add_argument("--noise-seed", type=int, default=1337)
    parser.add_argument("--decoy-seed", type=int, default=271828)
    parser.add_argument("--output-index", type=int, default=0)
    parser.add_argument("--count", type=int, default=500, help="Initial critical-point target")
    parser.add_argument("--max-queries", type=int, default=2_000_000, help="Physical oracle calls")
    parser.add_argument("--max-seconds", type=float, default=3_600)
    parser.add_argument("--output-path", type=Path)
    parser.add_argument("--stop-after-signature", action="store_true",
                        help="Save unsigned signatures and skip precision/sign stages")
    parser.add_argument("--execute", action="store_true", help="Start the first-layer extraction")
    options = parser.parse_args()

    if options.repeats < 1 or options.count < 1 or options.max_queries < 1 or options.max_seconds <= 0:
        parser.error("repeats, count, max-queries, and max-seconds must be positive")
    if options.mode != "pt" and options.repeats != 1:
        parser.error("--repeats applies only to the stochastic PT oracle")

    model_path = options.model.resolve(strict=True)
    weights, biases, q, shape = load_integer_target(model_path)
    if len(weights) < 2 or not 0 <= options.output_index < shape[-1]:
        parser.error("Expected at least one hidden layer and an output-index in range")

    # The dummy Keras carrier supplies only public layer sizes.  Its weights
    # differ from every true target layer by construction and are never used
    # for oracle answers.
    model = build_decoy_model(shape, q, options.decoy_seed)
    from signature_recovery.utils import getAllWeightsAndBiases

    decoy_weights, decoy_biases = getAllWeightsAndBiases(model)
    for i, (decoy_w, decoy_b) in enumerate(zip(decoy_weights, decoy_biases)):
        if np.array_equal(decoy_w, weights[i].astype(np.float64) / q) or np.array_equal(
            decoy_b, biases[i].astype(np.float64) / q
        ):
            raise RuntimeError(f"Decoy layer {i + 1} accidentally matches target parameters")

    # Import first so every upstream copy of predict_manual_fast is patched.
    import model_extraction  # noqa: F401
    from global_vars import GlobalConfig
    import signature_recovery.signature_recovery as recovery_module
    from signature_recovery.signature_recovery import recoverCritPts_Signature
    from whitebox import getWhiteboxSignatures

    if options.mode == "pt":
        bridge = ProbTruncBridge(
            weights, biases, q, shape,
            noise_seed=options.noise_seed,
            repeats=options.repeats,
            max_queries=options.max_queries,
            max_seconds=options.max_seconds,
        )
    else:
        bridge = ContinuousOracleBridge(
            weights, biases, q, shape,
            grid_input=options.mode == "float-grid",
            max_queries=options.max_queries,
            max_seconds=options.max_seconds,
        )
    query_patches = install_bridge(bridge)
    guard_patches = install_first_layer_guards()

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    output_path = (
        options.output_path.resolve()
        if options.output_path is not None
        else HERE / "runs_first_layer" / model_path.stem / f"{options.mode}_r{options.repeats}_s{options.seed}_{stamp}"
    )
    recovered_path = output_path / "recovered_1.npz"
    signature_path = output_path / "signature_checkpoint_1.npz"
    manifest = {
        "status": "ready" if not options.execute else "running",
        "attack_executed": bool(options.execute),
        "first_layer_only": True,
        "stop_after_signature": options.stop_after_signature,
        "model": str(model_path),
        "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "shape": shape,
        "q": q,
        "mode": options.mode,
        "repeats": options.repeats,
        "seed": options.seed,
        "noise_seed": options.noise_seed if options.mode == "pt" else None,
        "decoy_seed": options.decoy_seed,
        "carrier_contains_target_parameters": False,
        "dataset_branch": "cifar10",
        "dataset_branch_reason": "Liu's multi-output path is required for this 10-logit target",
        "output_index": options.output_index,
        "count": options.count,
        "real_attack": 1,
        "cheat_sign": 0,
        "cheat_noise": 0,
        "source_patch": "width-20 tol default, empty diagnostic list guard, explicit no-component error, and multi-output sign-solver axis fix",
        "quality_checker": "disabled; all component critical points retained",
        "true_sign_fallback": "abort if reached",
        "max_queries": options.max_queries,
        "max_seconds": options.max_seconds,
        "physical_oracle_queries": 0,
        "logical_liu_queries": 0,
        "output_path": str(output_path),
        "recovered_path": str(recovered_path),
        "signature_checkpoint_path": str(signature_path),
        "query_patches": query_patches,
        "guard_patches": guard_patches,
    }
    if not options.execute:
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return

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
            dataset="cifar10", model=model_path.stem, device="cpu",
            layerID=1, tgtNeurons=None, eps=8, quantized=2,
            signRecoveryMethod="neuronWiggle", seed=options.seed,
            Count=options.count, cheat_noise=0, cheat_sign=0,
            output_index=options.output_index, real_attack=1, debug=0,
            output_path=str(output_path),
        )
        # These are decoy parameters.  Their values cannot reveal target W/B.
        carrier_weights, carrier_biases = getWhiteboxSignatures(model, 1)
        save_path = output_path / "extract results" / f"Seed{options.seed}-Count{options.count}"
        original_compute_layer_values = recovery_module.compute_layer_values

        def capture_signature(*call_args, **call_kwargs):
            values = original_compute_layer_values(*call_args, **call_kwargs)
            np.savez(
                signature_path,
                extracted_normal=np.asarray(values[0]),
                extracted_bias=np.asarray(values[1]),
                extracted_sign=np.asarray(values[2]),
                extracted_sequence=np.asarray(values[3]),
            )
            if options.stop_after_signature:
                raise StopAfterSignature
            return values

        recovery_module.compute_layer_values = capture_signature
        with log_path.open("w", encoding="utf-8") as log, \
                contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            result = recoverCritPts_Signature(
                stage_args, model, carrier_weights, carrier_biases,
                1, 2, dataset="cifar10", savePath=str(save_path),
            )
        extracted_normal, extracted_bias, extracted_sign, extracted_sequence = result
        np.savez(
            recovered_path,
            extracted_normal=np.asarray(extracted_normal),
            extracted_bias=np.asarray(extracted_bias),
            extracted_sign=np.asarray(extracted_sign),
            extracted_sequence=np.asarray(extracted_sequence),
        )
        manifest["status"] = "completed"
    except StopAfterSignature:
        manifest["status"] = "signature_only_completed"
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
        manifest["log_path"] = str(log_path)
        _write_json(manifest_path, manifest)
        print(json.dumps(manifest, indent=2, sort_keys=True))
    if manifest["status"] not in ("completed", "signature_only_completed"):
        sys.exit(1)


if __name__ == "__main__":
    main()
