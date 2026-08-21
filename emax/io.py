import os
import numpy as np


BASE_DIR = "emax_res"


def _ensure_dir(filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)


def export_to_dat(data, filename, bins=50, density=True):
    filepath = os.path.join(BASE_DIR, filename)
    _ensure_dir(filepath)

    counts, bin_edges = np.histogram(data, bins=bins, density=density)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    with open(filepath, 'w') as f:
        f.write("x y\n")
        for x, y in zip(bin_centers, counts):
            f.write(f"{x:.6f} {y:.6f}\n")

    print(f"Data exported to {filepath}. (Samples: {len(data):,})")


def export_latex_histogram(data, filename, bins=50, density=True, color="cyan!60", legend_label=""):
    filepath = os.path.join(BASE_DIR, filename)
    _ensure_dir(filepath)

    counts, bin_edges = np.histogram(data, bins=bins, density=density)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    coords = []
    for x, y in zip(bin_centers, counts):
        if y > 0:
            coords.append(f"({x:.6f}, {y:.6f})")

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

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(tex_code)

    print(f"LaTeX plot exported to {filepath}. (Valid coordinates: {len(coords)})")


def load_dat(filename):
    filepath = os.path.join(BASE_DIR, filename)
    if not os.path.exists(filepath):
        print(f"Warning: File {filepath} not found. Data load aborted.")
        return None, None

    data = np.loadtxt(filepath, skiprows=1)
    return data[:, 0], data[:, 1]