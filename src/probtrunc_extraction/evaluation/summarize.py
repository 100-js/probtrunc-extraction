"""Aggregate freshly executed benchmark run directories into paper tables (CSV & Markdown)."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Sequence


def _write_csv(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return path


def _markdown_table(columns: Sequence[str], rows: Sequence[Sequence]) -> list[str]:
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    body = ["| " + " | ".join(map(str, row)) + " |" for row in rows]
    return [header, divider, *body, ""]


def summarize_runs(run_dirs: Sequence[str | Path], output_dir: str | Path) -> dict:
    """Read live ``summary.json`` files from completed runs and emit CSV & Markdown reports."""
    if not run_dirs:
        raise ValueError("Provide at least one completed benchmark run directory")

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    bench_rows: list[dict] = []
    sir_rows: list[dict] = []
    bw_rows: list[dict] = []

    for run_dir in run_dirs:
        summary_path = Path(run_dir) / "summary.json"
        if not summary_path.is_file():
            raise FileNotFoundError(f"Missing summary.json in {run_dir}")
        data = json.loads(summary_path.read_text(encoding="utf-8"))

        bias = data.get("bias_max_lsb") or [None, None, None]
        q_mean = data.get("queries_mean")
        q_sd = data.get("queries_population_sd")
        bench_rows.append({
            "configuration": data["configuration"],
            "architecture": "-".join(map(str, data["architecture"])),
            "S": data["fraction_bits"],
            "W1": data["W1"],
            "W2": data["W2"],
            "W3": data["W3"],
            "B1_max_lsb": "" if bias[0] is None else bias[0],
            "B2_max_lsb": "" if bias[1] is None else bias[1],
            "B3_max_lsb": "" if bias[2] is None else bias[2],
            "queries_mean_M": "" if q_mean is None else f"{q_mean / 1_000_000:.2f}",
            "queries_sd_M": "" if q_sd is None else f"{q_sd / 1_000_000:.2f}",
            "completed_trials": data["completed_trials"],
            "expected_trials": data["expected_trials"],
        })

        for row in data.get("sir_summary", []):
            sir_rows.append({"configuration": data["configuration"], **row})

        bw = data.get("bitwidth_summary")
        if bw:
            for layer_row in bw["layers"]:
                bw_rows.append({"configuration": data["configuration"], **layer_row})

    written: list[Path] = [_write_csv(destination / "benchmark.csv", bench_rows)]
    if sir_rows:
        written.append(_write_csv(destination / "sir.csv", sir_rows))
    if bw_rows:
        written.append(_write_csv(destination / "bitwidth.csv", bw_rows))

    md_lines = [
        "# Live Benchmark Summary",
        "",
        "Generated directly from executed trial records (`summary.json`).",
        "",
        "## Model Extraction Benchmark",
        "",
    ]
    md_lines.extend(
        _markdown_table(
            ("Config", "Architecture", "S", "W1", "W2", "W3", "B1", "B2", "B3", "Queries (M)"),
            [
                (
                    row["configuration"],
                    row["architecture"],
                    row["S"],
                    row["W1"],
                    row["W2"],
                    row["W3"],
                    row["B1_max_lsb"] if row["B1_max_lsb"] != "" else "—",
                    row["B2_max_lsb"] if row["B2_max_lsb"] != "" else "—",
                    row["B3_max_lsb"] if row["B3_max_lsb"] != "" else "—",
                    f"{row['queries_mean_M']} ± {row['queries_sd_M']}"
                    if row["queries_mean_M"] != ""
                    else "—",
                )
                for row in bench_rows
            ],
        )
    )

    if sir_rows:
        md_lines.extend(["## Layer-2 Sign-Recovery SIR", ""])
        md_lines.extend(
            _markdown_table(
                ("Config", "Outcome", "Trials", "Mean", "Pop. SD", "Min", "Max"),
                [
                    (
                        r["configuration"],
                        r["outcome"],
                        r["trials"],
                        f"{r['mean']:.2f}",
                        f"{r['population_sd']:.2f}",
                        f"{r['minimum']:.2f}",
                        f"{r['maximum']:.2f}",
                    )
                    for r in sir_rows
                ],
            )
        )

    if bw_rows:
        md_lines.extend(["## Observed Signed Bit-Widths", ""])
        md_lines.extend(
            _markdown_table(
                ("Config", "Layer", "Reports", "Pre-Trunc Max", "Pre Bits", "Post-Trunc Max", "Post Bits"),
                [
                    (
                        r["configuration"],
                        f"L{r['layer']}",
                        r["reports"],
                        f"{r['pre_trunc_max_abs']:,}",
                        r["pre_trunc_signed_bits"],
                        f"{r['post_trunc_max_abs']:,}",
                        r["post_trunc_signed_bits"],
                    )
                    for r in bw_rows
                ],
            )
        )

    md_path = destination / "summary.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8", newline="\n")
    written.append(md_path)
    return {"files": [str(p) for p in written], "benchmarks": bench_rows}
