#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, ast, math
import numpy as np
import pandas as pd

def parse_args():
    ap = argparse.ArgumentParser(
        description="计算CSV中两列向量之间的RMSE（逐轴 & 3D）。默认 p vs p_d。"
    )
    ap.add_argument("--csv", required=True, help="CSV 文件路径")
    ap.add_argument("--xcol", default="p", help="被测列名（默认 p）")
    ap.add_argument("--ycol", default="p_d", help="参考列名（默认 p_d）")
    ap.add_argument("--tcol", default="t", help="时间列名（默认 t）")
    ap.add_argument("--tmin", type=float, default=None, help="仅统计 t >= tmin 的样本")
    ap.add_argument("--tmax", type=float, default=None, help="仅统计 t <= tmax 的样本")
    ap.add_argument("--skip", type=int, default=0, help="跳过前N行样本（默认0）")
    ap.add_argument("--save", default=None, help="将结果另存为CSV（可选）")
    return ap.parse_args()

def to_vec(x):
    """把形如 '[a, b, c]' 的字符串安全解析为 numpy 向量。"""
    if isinstance(x, (list, tuple, np.ndarray)):
        return np.asarray(x, dtype=float)
    if isinstance(x, str):
        try:
            v = ast.literal_eval(x)
            return np.asarray(v, dtype=float)
        except Exception:
            pass
    # 单值也兼容
    try:
        return np.asarray([float(x)], dtype=float)
    except Exception:
        return np.asarray([np.nan])

def main():
    args = parse_args()
    df = pd.read_csv(args.csv)

    # 时间过滤（可选）
    if args.tcol in df.columns:
        if args.tmin is not None:
            df = df[df[args.tcol] >= args.tmin]
        if args.tmax is not None:
            df = df[df[args.tcol] <= args.tmax]

    # 跳过前N行（可选）
    if args.skip > 0:
        df = df.iloc[args.skip:].copy()

    # 基本检查
    for col in (args.xcol, args.ycol):
        if col not in df.columns:
            raise ValueError(f"找不到列: {col}")

    # 解析向量列
    X = df[args.xcol].map(to_vec).to_list()
    Y = df[args.ycol].map(to_vec).to_list()

    # 对齐维度，并丢弃解析失败/维度不一致的行
    clean_X, clean_Y, idx_keep = [], [], []
    for i, (x, y) in enumerate(zip(X, Y)):
        if x.ndim == 1 and y.ndim == 1 and len(x) == len(y) and not (np.any(np.isnan(x)) or np.any(np.isnan(y))):
            clean_X.append(x)
            clean_Y.append(y)
            idx_keep.append(i)

    if len(clean_X) == 0:
        raise RuntimeError("没有可用样本（可能是列内容解析失败或全部含NaN/维度不一致）。")

    X = np.vstack(clean_X)  # [N, D]
    Y = np.vstack(clean_Y)  # [N, D]
    E = X - Y               # 误差 [N, D]

    # 逐轴 RMSE
    rmse_axes = np.sqrt(np.mean(E**2, axis=0))  # [D]
    # 3D RMSE：sqrt(mean(||e||^2))
    rmse_3d = math.sqrt(np.mean(np.sum(E**2, axis=1)))

    # 输出
    axes_names = ["X", "Y", "Z", "W1", "W2", "W3", "W4"]  # 够用的标签池
    names = [axes_names[i] if i < len(axes_names) else f"Dim{i}" for i in range(rmse_axes.shape[0])]

    print(f"样本数: {X.shape[0]} | 维度: {X.shape[1]}")
    for i, v in enumerate(rmse_axes):
        print(f"RMSE[{names[i]}] = {v:.6f}")
    print(f"RMSE[3D] = {rmse_3d:.6f}")

    # 可选保存
    if args.save:
        out = {
            "samples": X.shape[0],
            "dims": X.shape[1],
            **{f"RMSE[{names[i]}]": float(rmse_axes[i]) for i in range(len(names))},
            "RMSE[3D]": float(rmse_3d),
            "xcol": args.xcol,
            "ycol": args.ycol,
            "tmin": args.tmin,
            "tmax": args.tmax,
            "skip": args.skip,
        }
        pd.DataFrame([out]).to_csv(args.save, index=False)
        print(f"结果已保存到: {args.save}")

if __name__ == "__main__":
    main()
