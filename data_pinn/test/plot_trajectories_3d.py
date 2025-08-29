#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, re, ast, math
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.colors import Normalize


def parse_args():
    ap = argparse.ArgumentParser(
        description="Plot 3D trajectories from simpleflight_fig8_adaptive_*.csv, color by wind speed."
    )
    ap.add_argument("--dir", type=str, default=".",
                    help="Directory that contains the CSVs (default: current directory)")
    ap.add_argument("--pattern", type=str, default="simpleflight_fig8_adaptive_*.csv",
                    help="Filename glob pattern (default: simpleflight_fig8_adaptive_*.csv)")
    ap.add_argument("--save", type=str, default="test_trajectories_3d.png",
                    help="Output image file (default: test_trajectories_3d.png)")
    ap.add_argument("--dpi", type=int, default=220, help="Save figure DPI (default: 220)")
    return ap.parse_args()


def extract_speed_and_label(name: str):
    """
    从文件名提取风速数值和展示标签
    - nowind         → (0.0, "0 m/s")
    - 10wind         → (10.0, "10.0 m/s")
    - 10sint         → (10.3, "10.0 m/s sinusoid")  # 颜色扰动 +0.3
    - 13p5wind       → (13.5, "13.5 m/s")
    - 13p5sint       → (13.8, "13.5 m/s sinusoid")
    """
    base = Path(name).stem.lower()
    if "nowind" in base:
        return 0.0, "0 m/s"

    # 支持整数/小数 (p 表示小数点)
    m = re.match(r".*?(\d+(?:p\d+)?)(wind|sint|wid)", base)
    if m:
        raw = m.group(1)
        spd = float(raw.replace("p", "."))  # 13p5 → 13.5
        if m.group(2) in ["wind", "wid"]:
            return spd, f"{spd:.1f} m/s"
        elif m.group(2) == "sint":
            return spd + 0.3, f"{spd:.1f} m/s sinusoid"

    # 默认：尝试找数字
    m2 = re.search(r"(\d+(?:\.\d+)?)", base)
    if m2:
        spd = float(m2.group(1))
        return spd, f"{spd:.1f} m/s"

    return float("nan"), name


def parse_vec3(cell: str) -> np.ndarray:
    if cell is None or (isinstance(cell, float) and math.isnan(cell)):
        return None
    try:
        v = ast.literal_eval(str(cell))
        if isinstance(v, (list, tuple)) and len(v) >= 3:
            return np.array([float(v[0]), float(v[1]), float(v[2])], dtype=float)
    except Exception:
        return None
    return None


def load_xyz_from_csv(fp: Path):
    df = pd.read_csv(fp, engine="python")
    if "p" not in df.columns:
        raise ValueError(f"{fp.name} 缺少列 'p'")
    pts = [parse_vec3(c) for c in df["p"]]
    pts = [p for p in pts if p is not None]
    if not pts:
        raise ValueError(f"{fp.name} 中 'p' 列为空或不可解析")

    arr = np.vstack(pts)
    arr[:, 2] = -arr[:, 2]  # Z 数值取反，高度正向
    return arr[:, 0], arr[:, 1], arr[:, 2]


def main():
    args = parse_args()

    # === 全局字体放大 ===
    plt.rcParams.update({
        "font.size": 16,
        "axes.titlesize": 18,
        "axes.labelsize": 16,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 14
    })

    root = Path(args.dir)
    files = sorted(root.glob(args.pattern))
    if not files:
        raise SystemExit(f"未在 {root.resolve()} 下找到匹配 {args.pattern} 的文件")

    entries = []
    for f in files:
        spd, label = extract_speed_and_label(f.name)
        entries.append((f, spd, label))

    entries.sort(key=lambda x: (math.isnan(x[1]), x[1]))

    speeds = [s for _, s, _ in entries if not math.isnan(s)]
    vmin, vmax = (min(speeds), max(speeds)) if speeds else (0.0, 1.0)
    if speeds and math.isclose(vmin, vmax):
        vmin, vmax = vmin - 1.0, vmax + 1.0

    norm = Normalize(vmin=vmin, vmax=vmax, clip=True)
    cmap = get_cmap("plasma")

    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")

    for fp, spd, label in entries:
        try:
            X, Y, Z = load_xyz_from_csv(fp)
        except Exception as e:
            print(f"[跳过] {fp.name}: {e}")
            continue

        color = (0.4, 0.4, 0.4, 0.9) if math.isnan(spd) else cmap(norm(spd))
        ax.plot(X, Y, Z, lw=1.8, alpha=0.9, label=label, color=color)

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("SimpleFlight Test Trajectories", pad=10)

    # 调整刻度字体
    ax.tick_params(axis="both", which="major", labelsize=14)
    ax.tick_params(axis="z", which="major", labelsize=14)

    if speeds:
        mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        mappable.set_array([])
        cbar = fig.colorbar(mappable, ax=ax, shrink=0.72, pad=0.08)
        cbar.set_label("Wind speed (m/s)", fontsize=16)
        cbar.ax.tick_params(labelsize=14)

    ax.legend(loc="best", fontsize=14)
    fig.tight_layout()
    fig.savefig(args.save, dpi=args.dpi, bbox_inches="tight")
    print(f"Saved figure -> {args.save}")
    plt.show()


if __name__ == "__main__":
    main()
