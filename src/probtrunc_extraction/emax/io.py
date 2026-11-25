"""Histogram export and loader utilities for empirical E_max simulations."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def export_to_dat(
    data: np.ndarray,
    filename: str,
    *,
    output_dir: str | Path = "runs/emax",
    bins: int = 50,
    density: bool = True,
) -> Path:
    filepath = Path(output_dir) / filename
    filepath.parent.mkdir(parents=True, exist_ok=True)

    counts, bin_edges = np.histogram(data, bins=bins, density=density)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    with filepath.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("x y\n")
        for x, y in zip(bin_centers, counts):
            handle.write(f"{x:.6f} {y:.6f}\n")

    print(f"Data exported to {filepath}. (Samples: {len(data):,})")
    return filepath


def export_latex_histogram(
    data: np.ndarray,
    filename: str,
    *,
    output_dir: str | Path = "runs/emax",
    bins: int = 50,
    density: bool = True,
    color: str = "cyan!60",
    legend_label: str = "",
) -> Path:
    filepath = Path(output_dir) / filename
    filepath.parent.mkdir(parents=True, exist_ok=True)

    counts, bin_edges = np.histogram(data, bins=bins, density=density)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    coords = [f"({x:.6f}, {y:.6f})" for x, y in zip(bin_centers, counts) if y > 0]
    coords_str = "\n        ".join(coords)
    tex_code = (
        "\\addplot[\n"
        "    ybar,\n"
        "    bar width=3pt,\n"
        f"    fill={color},\n"
        "    draw=black\n"
        "] coordinates {\n"
        f"        {coords_str}\n"
        "};"
    )
    if legend_label:
        tex_code += f"\n\\addlegendentry{{{legend_label}}}\n"

    filepath.write_text(tex_code, encoding="utf-8", newline="\n")
    print(f"LaTeX plot exported to {filepath}. (Valid coordinates: {len(coords)})")
    return filepath


def load_dat(
    filename: str,
    *,
    output_dir: str | Path = "runs/emax",
) -> tuple[np.ndarray | None, np.ndarray | None]:
    filepath = Path(output_dir) / filename
    if not filepath.is_file():
        print(f"Warning: File {filepath} not found. Data load aborted.")
        return None, None
    data = np.loadtxt(filepath, skiprows=1)
    return data[:, 0], data[:, 1]
