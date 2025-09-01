#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Analyze & Plot Heatmap Matrix (local-centered/global axis) + Export Metrics (t >= t0)

- 输入 CSV 至少包含列：t, p, p_d（其中 p/p_d 为 "[x, y, z]" 字符串）
- 文件名约定：simpleflight_fig8_<METHOD>_<CONDITION>.csv
- 输出：
  1) plots_dir/error_heatmaps_matrix.png
  2) plots_dir/avg_tracking_error_after20s.csv
"""

import os, ast, argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 论文友好样式
plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 13,
    "axes.labelsize": 10,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 11,
    "axes.linewidth": 1.0,
})

# 列（风况）与行（方法）顺序
COLUMN_ORDER = ['nowind','5wind','10wind','12wind','13p5wind','18sint','15wind']
METHOD_ORDER = ['pid','adaptive','nnadaptive','pinnadaptive']  # 最后一行为 MetaPINN
METHOD_DISPLAY = {'pid':'PID','adaptive':'Adaptive','nnadaptive':'NNAdaptive','pinnadaptive':'MetaPINN'}
METHOD_ALIASES  = {'metapinn':'pinnadaptive','metapinnadaptive':'pinnadaptive','pinn':'pinnadaptive'}

# ---------------- 基础IO ----------------
def parse_position_data(s):
    try:
        return np.array(ast.literal_eval(s), dtype=float)
    except Exception:
        s = str(s).strip().strip('[]')
        parts = [p for p in s.replace(' ', '').split(',') if p]
        return np.array([float(x) for x in parts[:3]], dtype=float)

def load_flight_csv(csv_path: Path):
    df = pd.read_csv(csv_path)
    time = df['t'].to_numpy(dtype=float)
    pos  = np.vstack([parse_position_data(x) for x in df['p']])
    posd = np.vstack([parse_position_data(x) for x in df['p_d']])
    err  = pos - posd
    se   = np.sum(err**2, axis=1)
    return {'time':time,'position':pos,'desired_position':posd,'error':err,
            'squared_error':se,'filename':csv_path.stem,'path':str(csv_path)}

def extract_method_condition(stem: str):
    parts = stem.lower().split('_')
    if len(parts) < 4: return None, None
    method = METHOD_ALIASES.get(parts[2], parts[2])
    cond = '_'.join(parts[3:])
    return method, cond

# ---------------- 坐标范围工具 ----------------
def global_xy_limits(flight_data_list, pad_ratio=0.06):
    xs, ys = [], []
    for d in flight_data_list:
        p, pd = d['position'], d['desired_position']
        xs.extend([p[:,0].min(), p[:,0].max(), pd[:,0].min(), pd[:,0].max()])
        ys.extend([p[:,1].min(), p[:,1].max(), pd[:,1].min(), pd[:,1].max()])
    xmin, xmax = float(np.min(xs)), float(np.max(xs))
    ymin, ymax = float(np.min(ys)), float(np.max(ys))
    xr, yr = xmax - xmin, ymax - ymin
    xmin -= xr * pad_ratio; xmax += xr * pad_ratio
    ymin -= yr * pad_ratio; ymax += yr * pad_ratio
    return xmin, xmax, ymin, ymax

def centered_local_limits(pos, posd, pad_ratio=0.06):
    # 用该子图的数据/期望轨迹计算中心 & 正方形范围
    x_all = np.concatenate([pos[:,0], posd[:,0]])
    y_all = np.concatenate([pos[:,1], posd[:,1]])
    cx = 0.5 * (x_all.min() + x_all.max())
    cy = 0.5 * (y_all.min() + y_all.max())
    half_x = 0.5 * (x_all.max() - x_all.min())
    half_y = 0.5 * (y_all.max() - y_all.min())
    R = max(half_x, half_y) * (1.0 + pad_ratio)  # 正方形半径，保证图形居中不被裁切
    if R <= 1e-6: R = 1.0
    return (cx - R, cx + R, cy - R, cy + R)

def symmetric_ticks(lo, hi, n=5):
    # 生成对称刻度（适合比较），保持端点在范围边界上
    return np.linspace(lo, hi, num=n)

# ---------------- 绘图主函数 ----------------
def plot_error_heatmaps_matrix(flight_data_list, plots_dir,
                               vclip_percentile=95,
                               axis_mode='local-centered',
                               n_ticks=5,
                               pad_ratio=0.06):
    """
    axis_mode:
      - 'local-centered'：每个子图居中且范围随数据（默认，满足“8字在中间，范围缩放随着数据”）
      - 'global'：全局统一坐标范围与刻度
    """
    methods_data = {m: [] for m in METHOD_ORDER}
    all_sqerr = []

    for d in flight_data_list:
        method, cond = extract_method_condition(d['filename'])
        if (method in methods_data) and (cond is not None):
            methods_data[method].append((d, cond))
            all_sqerr.extend(d['squared_error'])

    if not all_sqerr:
        print("[WARN] 无误差数据，跳过绘图。"); return

    vmin = float(np.min(all_sqerr))
    vmax = float(np.percentile(all_sqerr, vclip_percentile))
    print(f"[ColorScale] vmin={vmin:.6g}, vmax={vmax:.6g} (p{vclip_percentile})")

    # 全局范围（仅 global 模式使用）
    if axis_mode == 'global':
        gxmin, gxmax, gymin, gymax = global_xy_limits(flight_data_list, pad_ratio=pad_ratio)
        g_xticks = symmetric_ticks(gxmin, gxmax, n=n_ticks)
        g_yticks = symmetric_ticks(gymin, gymax, n=n_ticks)

    n_rows, n_cols = len(METHOD_ORDER), len(COLUMN_ORDER)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(3.3 * n_cols, 3.3 * n_rows),
                             constrained_layout=False)

    if n_rows == 1: axes = np.array([axes])
    if n_cols == 1: axes = axes.reshape(n_rows, 1)

    first_scatter = None

    for r, method in enumerate(METHOD_ORDER):
        # cond -> data
        cond_map = {c: None for c in COLUMN_ORDER}
        for d, c in methods_data.get(method, []):
            if c in cond_map: cond_map[c] = d

        for c, cond in enumerate(COLUMN_ORDER):
            ax = axes[r, c]
            dd = cond_map[cond]
            if dd is None:
                ax.set_visible(False)
                continue

            pos, posd = dd['position'], dd['desired_position']
            sqerr = dd['squared_error']

            sc = ax.scatter(pos[:,0], pos[:,1], c=sqerr,
                            cmap='turbo', s=1.0, vmin=vmin, vmax=vmax, zorder=2)
            if first_scatter is None: first_scatter = sc

            # 期望轨迹（黑虚线）
            ax.plot(posd[:,0], posd[:,1], 'k--', linewidth=1.1, alpha=0.9, zorder=3)

            # 坐标范围与刻度
            if axis_mode == 'global':
                xmin, xmax, ymin, ymax = gxmin, gxmax, gymin, gymax
                xticks, yticks = g_xticks, g_yticks
            else:  # local-centered
                xmin, xmax, ymin, ymax = centered_local_limits(pos, posd, pad_ratio=pad_ratio)
                xticks = symmetric_ticks(xmin, xmax, n=n_ticks)
                yticks = symmetric_ticks(ymin, ymax, n=n_ticks)

            ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
            ax.set_xticks(xticks);   ax.set_yticks(yticks)
            ax.set_aspect('equal', adjustable='box')

            # 轴名与网格
            ax.set_xlabel("X (m)")
            ax.set_ylabel("Y (m)")
            ax.grid(True, alpha=0.25, linewidth=0.8)
            ax.tick_params(length=5, width=1.0)

            # 列首标题
            if r == 0:
                ax.set_title(cond.upper(), fontsize=13, fontweight='bold', pad=8)

    # 左侧页边距写方法名（不与轴标签冲突）
    plt.subplots_adjust(left=0.10, right=0.92, top=0.92, bottom=0.08,
                        wspace=0.35, hspace=0.35)
    for r, method in enumerate(METHOD_ORDER):
        disp = METHOD_DISPLAY.get(method, method.upper())
        bbox = axes[r, 0].get_position()
        ymid = (bbox.y0 + bbox.y1) / 2.0
        plt.gcf().text(0.04, ymid, disp, rotation=90, va='center', ha='center',
                       fontsize=13, fontweight='bold')

    # 右侧 colorbar
    if first_scatter is not None:
        cbar_ax = plt.gcf().add_axes([0.94, 0.15, 0.015, 0.7])
        cbar = plt.colorbar(first_scatter, cax=cbar_ax, orientation='vertical')
        cbar.set_label(f'Squared Error (m²) [clipped @ p{vclip_percentile}]', fontsize=11)
        cbar.ax.tick_params(labelsize=10)

    plt.suptitle('Squared Position Error Heatmaps', fontsize=16, fontweight='bold', y=0.97)

    out_path = os.path.join(plots_dir, 'error_heatmaps_matrix.png')
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved heatmap: {out_path}")

# ---------------- 指标：t >= t0 ----------------
def compute_metrics_after_t0(flight_data_list, t0=20.0):
    rows = []
    for d in flight_data_list:
        time, err = d['time'], d['error']
        mask = (time >= t0)
        if not np.any(mask):
            print(f"[WARN] {d['filename']} 无 t >= {t0}s 数据，跳过。"); continue
        e = err[mask]
        e_abs = np.abs(e); e_norm = np.linalg.norm(e, axis=1)
        mae_x, mae_y, mae_z = np.mean(e_abs, axis=0)
        mae_3d = float(np.mean(e_norm))
        rmse_3d = float(np.sqrt(np.mean(e_norm**2)))
        method, cond = extract_method_condition(d['filename'])
        rows.append({
            'filename': d['filename'],
            'method': METHOD_DISPLAY.get(METHOD_ALIASES.get(method, method), (method or '').upper()),
            'condition': cond or '',
            'n_samples_after_t0': int(e.shape[0]),
            't0_seconds': float(t0),
            'mae_x': float(mae_x),
            'mae_y': float(mae_y),
            'mae_z': float(mae_z),
            'mae_3d': mae_3d,
            'rmse_3d': rmse_3d,
        })
    return rows

def export_metrics_csv(rows, out_csv):
    if not rows:
        print("[WARN] 没有可导出的指标。"); return
    df = pd.DataFrame(rows)
    # 友好排序
    method_rank = {METHOD_DISPLAY[m]: i for i, m in enumerate(METHOD_ORDER)}
    cond_rank   = {c.upper(): i for i, c in enumerate(COLUMN_ORDER)}
    df['method_rank'] = df['method'].map(method_rank).fillna(999).astype(int)
    df['cond_rank']   = df['condition'].str.upper().map(cond_rank).fillna(999).astype(int)
    df = df.sort_values(['method_rank','cond_rank','filename']).drop(columns=['method_rank','cond_rank'])
    df.to_csv(out_csv, index=False, float_format="%.6f")
    print(f"[OK] Saved metrics CSV: {out_csv}")

# ---------------- Main ----------------
def main():
    ap = argparse.ArgumentParser(description="Plot heatmap matrix (local-centered/global) and export metrics.")
    ap.add_argument("--data_dir",  type=str, default="data_baseline/test0901")
    ap.add_argument("--plots_dir", type=str, default="plots/0901")
    ap.add_argument("--t0",        type=float, default=20.0, help="Use samples with t >= t0 (seconds)")
    ap.add_argument("--vclip",     type=float, default=95.0, help="Percentile for color clipping (0-100)")
    ap.add_argument("--axis_mode", type=str, choices=["local-centered","global"], default="local-centered",
                    help="Axis scaling mode: 'local-centered' keeps 8-shape centered per subplot; 'global' unifies range.")
    ap.add_argument("--ticks",     type=int, default=5, help="Number of ticks per axis")
    ap.add_argument("--pad",       type=float, default=0.06, help="Padding ratio around data range")
    args = ap.parse_args()

    data_dir  = Path(args.data_dir)
    plots_dir = Path(args.plots_dir); plots_dir.mkdir(parents=True, exist_ok=True)

    if not data_dir.exists():
        print(f"[ERR] Data dir not found: {data_dir}"); return
    csv_files = sorted(data_dir.glob("*.csv"))
    if not csv_files:
        print(f"[ERR] No CSV files in {data_dir}"); return

    print(f"[INFO] Found {len(csv_files)} CSV files:")
    for f in csv_files: print("  -", f.name)

    # 读取
    flight_data_list = []
    for f in csv_files:
        try: flight_data_list.append(load_flight_csv(f))
        except Exception as e: print(f"[WARN] Failed to load {f.name}: {e}")

    if not flight_data_list:
        print("[ERR] No datasets loaded, abort."); return

    # 绘图
    plot_error_heatmaps_matrix(
        flight_data_list, str(plots_dir),
        vclip_percentile=args.vclip,
        axis_mode=args.axis_mode,
        n_ticks=args.ticks,
        pad_ratio=args.pad,
    )

    # 指标
    rows = compute_metrics_after_t0(flight_data_list, t0=args.t0)
    export_metrics_csv(rows, os.path.join(str(plots_dir), "avg_tracking_error_after20s.csv"))
    print("[DONE] All tasks completed.")

if __name__ == "__main__":
    main()
