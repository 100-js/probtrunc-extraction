"""One-query connection check; deliberately never invokes Liu's attack."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from oracle_bridge import (
    ROOT,
    ProbTruncBridge,
    build_liu_model,
    install_bridge,
    load_integer_target,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        default=ROOT / "checkpoints" / "digits" / "(64, 20, 20, 10)-01.npz",
        help="Existing quantized target checkpoint NPZ",
    )
    parser.add_argument("--noise-seed", type=int, default=1337)
    args = parser.parse_args()

    weights, biases, q, shape = load_integer_target(args.model)
    model = build_liu_model(weights, biases, q)

    # Importing model_extraction loads all modules that hold copied references
    # to the upstream query function.  No attack entry point is called.
    import model_extraction  # noqa: F401
    from global_vars import GlobalConfig
    from signature_recovery import utils

    bridge = ProbTruncBridge(weights, biases, q, shape, noise_seed=args.noise_seed)
    bridge.validate_model(model)
    patched = install_bridge(bridge)
    assert callable(utils.predict_manual_fast)
    assert model_extraction.predict_manual_fast.__self__ is bridge

    sample = np.zeros((1, shape[0]), dtype=np.float64)
    attack_rng_before = np.random.get_state()
    result = utils.predict_manual_fast(sample, *utils.getAllWeightsAndBiases(model))
    attack_rng_after = np.random.get_state()
    assert all(np.array_equal(a, b) for a, b in zip(attack_rng_before, attack_rng_after))
    assert result.shape == (1, shape[-1])
    assert np.all(result * q == np.rint(result * q))
    assert bridge.oracle.total_queries == GlobalConfig.query_count == 1

    digest = hashlib.sha256(args.model.read_bytes()).hexdigest()
    print(json.dumps({
        "status": "connection_ok",
        "attack_executed": False,
        "model": str(args.model.resolve()),
        "model_sha256": digest,
        "q": q,
        "shape": shape,
        "noise_seed": args.noise_seed,
        "patched_modules": patched,
        "oracle_queries": bridge.oracle.total_queries,
        "sample_output_integer": np.rint(result[0] * q).astype(np.int64).tolist(),
    }, indent=2))


if __name__ == "__main__":
    main()
