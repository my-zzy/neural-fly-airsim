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
        description="Plot 3D trajectories from simpleflight_random_adaptive_*.csv, color by wind speed."
    )
    ap.add_argument("--dir", type=str, default=".",
                    help="Directory that contains the CSVs (default: current directory)")
    ap.add_argument("--pattern", type=str, default="simpleflight_random_adaptive_*.csv",
                    help="Filename glob pattern")
    ap.add_argument("--save", type=str, default="trajectories_3d.png",
                    help="Output image file (default: trajectories_3d.png)")
    ap.add_argument("--dpi", type=int, default=220, help="Save figure DPI (default: 220)")
    return ap.parse_args()


def extract_speed_and_label(name: str):
    """
    从文件名提取风速数值和展示标签
    - nowind → (0.0, "0 m/s")
    - 10wind → (10.0, "10 m/s")
    - 10sint → (10.3, "10 m/s sinusoid")  # 注意颜色扰动
    """
    base = Path(name).stem.lower()
    if "nowind" in base:
        return 0.0, "0 m/s"

    m = re.match(r".*?(\d+)(wind|sint)", base)
    if m:
        spd = float(m.group(1))
        if m.group(2) == "wind":
            return spd, f"{spd:.0f} m/s"
        elif m.group(2) == "sint":
            return spd + 0.3, f"{spd:.0f} m/s sinusoid"  # 加一点偏移区分颜色

    m2 = re.search(r"(\d+(?:\.\d+)?)", base)
    if m2:
        spd = float(m2.group(1))
        return spd, f"{spd:.0f} m/s"
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


def load_xyz_from_csv(fp: Path, max_points=2000):
    df = pd.read_csv(fp, engine="python")
    if "p" not in df.columns:
        raise ValueError(f"{fp.name} 缺少列 'p'")
    pts = [parse_vec3(c) for c in df["p"]]
    pts = [p for p in pts if p is not None]
    if not pts:
        raise ValueError(f"{fp.name} 中 'p' 列为空或不可解析")

    arr = np.vstack(pts)

    # --- 只取中间 max_points 个 ---
    N = len(arr)
    if N > max_points:
        start = (N - max_points) // 2
        arr = arr[start:start+max_points]

    # --- Z数值取反 ---
    arr[:, 2] = -arr[:, 2]

    return arr[:, 0], arr[:, 1], arr[:, 2]



def main():
    args = parse_args()

    # === 全局字体加大 ===
    plt.rcParams.update({
        "font.size": 16,
        "axes.titlesize": 20,
        "axes.labelsize": 18,
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
    cmap = get_cmap("viridis")

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    for fp, spd, label in entries:
        try:
            X, Y, Z = load_xyz_from_csv(fp, max_points=4000)
        except Exception as e:
            print(f"[跳过] {fp.name}: {e}")
            continue

        color = (0.4, 0.4, 0.4, 0.9) if math.isnan(spd) else cmap(norm(spd))
        ax.plot(X, Y, Z, lw=1.8, alpha=0.9, label=label, color=color)

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("SimpleFlight Random Trajectories", pad=20)  # 标题和图保持距离

    # --- Z轴字体放大 ---
    ax.tick_params(axis="z", which="major", labelsize=14)

    if speeds:
        mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        mappable.set_array([])
        # 调整 pad=0.1 让热力条和Z轴更远
        cbar = fig.colorbar(mappable, ax=ax, shrink=0.72, pad=0.1)
        cbar.set_label("Wind speed (m/s)", fontsize=16)
        cbar.ax.tick_params(labelsize=14)

    ax.legend(loc="best", fontsize=14)
    fig.tight_layout()
    fig.savefig(args.save, dpi=args.dpi, bbox_inches="tight")
    print(f"Saved figure -> {args.save}")
    plt.show()



if __name__ == "__main__":
    main()
