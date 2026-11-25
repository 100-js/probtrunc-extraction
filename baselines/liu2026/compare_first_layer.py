"""Compare Liu first-layer recovery on a float oracle and repeated PT queries.

The default prints a plan without running an attack.  With ``--execute``,
each case runs in a fresh process with the same target, attack seed and output
logit.  Parity is measured against the float run after both extractions end.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from oracle_bridge import ROOT


HERE = Path(__file__).resolve().parent


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _parse_repeats(value: str) -> list[int]:
    try:
        repeats = [int(item.strip()) for item in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Expected comma-separated positive integers") from exc
    if not repeats or repeats[0] != 1 or any(n < 1 for n in repeats):
        raise argparse.ArgumentTypeError("Repeat schedule must start at 1 and contain positive integers")
    if repeats != sorted(set(repeats)):
        raise argparse.ArgumentTypeError("Repeat schedule must be strictly increasing")
    return repeats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        default=ROOT / "checkpoints" / "digits" / "(64, 20, 20, 10)-01.npz",
    )
    parser.add_argument("--repeats", type=_parse_repeats, default=[1, 2, 4, 8, 16])
    parser.add_argument("--reference-mode", choices=("float", "float-grid"), default="float")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--noise-seed", type=int, default=1337)
    parser.add_argument("--output-index", type=int, default=0)
    parser.add_argument("--count", type=int, default=500)
    parser.add_argument("--max-queries", type=int, default=2_000_000, help="Physical calls per case")
    parser.add_argument("--max-seconds", type=float, default=3_600, help="Wall seconds per case")
    parser.add_argument("--match-tolerance", type=float, default=0.001)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--stop-after-signature", action="store_true",
                        help="Compare Liu's unsigned signature stage only")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    model = args.model.resolve(strict=True)
    if args.count < 1 or args.max_queries < 1 or args.max_seconds <= 0 or args.match_tolerance < 0:
        parser.error("count, max-queries, max-seconds and match-tolerance must be valid positive limits")
    cases = [(args.reference_mode, 1)] + [("pt", n) for n in args.repeats]
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else HERE / "runs_first_layer" / model.stem / f"comparison_{stamp}"
    )
    summary = {
        "status": "planned" if not args.execute else "running",
        "model": str(model),
        "reference_mode": args.reference_mode,
        "repeat_schedule": args.repeats,
        "same_attack_seed": args.seed,
        "noise_seed": args.noise_seed,
        "output_index": args.output_index,
        "count": args.count,
        "stop_after_signature": args.stop_after_signature,
        "physical_query_budget_per_case": args.max_queries,
        "seconds_budget_per_case": args.max_seconds,
        "match_tolerance": args.match_tolerance,
        "cases": [{"mode": mode, "repeats": n} for mode, n in cases],
        "output_dir": str(output_dir),
        "minimal_repeats_matching_float": None,
        "minimal_repeats_matching_float_signature": None,
    }
    if not args.execute:
        print(json.dumps(summary, indent=2, sort_keys=True))
        return

    output_dir.mkdir(parents=True, exist_ok=False)
    summary_path = output_dir / "comparison_summary.json"
    _write_json(summary_path, summary)
    reference_metrics: Path | None = None
    reference_signature_metrics: Path | None = None
    reference_completed = False
    for mode, repeats in cases:
        if mode == "pt" and repeats > 1 and not reference_completed and reference_signature_metrics is None:
            summary["status"] = "reference_incomplete"
            break
        case_dir = output_dir / (mode if mode != "pt" else f"pt_r{repeats}")
        command = [
            sys.executable, str(HERE / "run_first_layer.py"),
            "--model", str(model), "--mode", mode, "--repeats", str(repeats),
            "--seed", str(args.seed), "--noise-seed", str(args.noise_seed),
            "--output-index", str(args.output_index), "--count", str(args.count),
            "--max-queries", str(args.max_queries), "--max-seconds", str(args.max_seconds),
            "--output-path", str(case_dir), "--execute",
        ]
        if args.stop_after_signature:
            command.append("--stop-after-signature")
        timed_out = False
        with (output_dir / f"{case_dir.name}_process_stdout.log").open("w", encoding="utf-8") as stdout, \
                (output_dir / f"{case_dir.name}_process_stderr.log").open("w", encoding="utf-8") as stderr:
            try:
                process = subprocess.run(
                    command, stdout=stdout, stderr=stderr, check=False,
                    timeout=args.max_seconds + 120,
                )
                returncode = process.returncode
            except subprocess.TimeoutExpired:
                timed_out = True
                returncode = None
        manifest_path = case_dir / "run_manifest.json"
        case = {
            "mode": mode, "repeats": repeats, "returncode": returncode,
            "manifest": str(manifest_path),
        }
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            case.update({
                "status": "hard_timeout" if timed_out else manifest["status"],
                "physical_oracle_queries": manifest.get("physical_oracle_queries"),
                "logical_liu_queries": manifest.get("logical_liu_queries"),
                "elapsed_seconds": manifest.get("elapsed_seconds"),
                "error": manifest.get("error"),
            })
            signature_path = Path(manifest["signature_checkpoint_path"])
            if signature_path.is_file():
                signature_metrics_path = case_dir / "signature_metrics.json"
                signature_cmd = [
                    sys.executable, str(HERE / "evaluate_first_layer.py"),
                    "--model", str(model), "--recovered", str(signature_path),
                    "--output", str(signature_metrics_path), "--unsigned",
                    "--match-tolerance", str(args.match_tolerance),
                ]
                if mode == "pt" and reference_signature_metrics is not None:
                    signature_cmd.extend(["--reference-metrics", str(reference_signature_metrics)])
                signature_evaluation = subprocess.run(
                    signature_cmd, capture_output=True, text=True, check=False
                )
                if signature_evaluation.returncode == 0:
                    signature_metrics = json.loads(signature_metrics_path.read_text(encoding="utf-8"))
                    case["signature_metrics"] = str(signature_metrics_path)
                    case["signature_coverage"] = signature_metrics["coverage"]
                    case["signature_error_summary"] = signature_metrics["error_summary"]
                    if mode != "pt":
                        reference_signature_metrics = signature_metrics_path
                    elif reference_signature_metrics is not None:
                        case["signature_matches_float_reference"] = signature_metrics[
                            "reference_comparison"
                        ]["matches_float_reference"]
                else:
                    case["signature_evaluation_error"] = signature_evaluation.stderr[-3000:]
            if manifest["status"] == "completed":
                metrics_path = case_dir / "first_layer_metrics.json"
                evaluate_cmd = [
                    sys.executable, str(HERE / "evaluate_first_layer.py"),
                    "--model", str(model), "--recovered", manifest["recovered_path"],
                    "--output", str(metrics_path), "--match-tolerance", str(args.match_tolerance),
                ]
                if mode == "pt" and reference_metrics is not None:
                    evaluate_cmd.extend(["--reference-metrics", str(reference_metrics)])
                evaluation = subprocess.run(evaluate_cmd, capture_output=True, text=True, check=False)
                case["evaluation_returncode"] = evaluation.returncode
                if evaluation.returncode != 0:
                    case["evaluation_error"] = evaluation.stderr[-3000:]
                else:
                    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                    case["metrics"] = str(metrics_path)
                    case["coverage"] = metrics["coverage"]
                    case["error_summary"] = metrics["error_summary"]
                    if mode != "pt":
                        reference_metrics = metrics_path
                        reference_completed = True
                    elif reference_metrics is not None:
                        case["matches_float_reference"] = metrics["reference_comparison"]["matches_float_reference"]
        else:
            case["status"] = "hard_timeout" if timed_out else "missing_manifest"
        summary["cases_completed"] = summary.get("cases_completed", []) + [case]
        _write_json(summary_path, summary)
        if mode == "pt" and case.get("matches_float_reference"):
            summary["minimal_repeats_matching_float"] = repeats
            summary["status"] = "parity_reached"
            break
        if mode == "pt" and case.get("signature_matches_float_reference"):
            summary["minimal_repeats_matching_float_signature"] = repeats
            if not reference_completed:
                summary["status"] = "signature_parity_reached"
                break

    if summary["status"] == "running":
        completed = summary.get("cases_completed", [])
        if args.stop_after_signature and reference_signature_metrics is not None:
            summary["status"] = (
                "signature_not_reached_within_schedule"
                if len(completed) == len(cases) and all(
                    item.get("status") == "signature_only_completed" for item in completed
                ) else "signature_inconclusive_budget_or_failure"
            )
        elif not reference_completed:
            summary["status"] = (
                "full_reference_incomplete_signature_available"
                if reference_signature_metrics is not None else "reference_incomplete"
            )
        elif len(completed) < len(cases):
            summary["status"] = "incomplete"
        elif any(item.get("status") != "completed" for item in completed):
            summary["status"] = "inconclusive_budget_or_failure"
        else:
            summary["status"] = "not_reached_within_schedule"
    _write_json(summary_path, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
