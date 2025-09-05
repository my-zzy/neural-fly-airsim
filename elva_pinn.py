#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, time, csv, argparse
from typing import Optional, Tuple, List, Dict

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn

# ========= 项目内模块 =========
from meta_pinn import MetaPINN, load_data, format_data, get_uav_physics
from meta_pinn.utils import set_seed, StandardScaler
from meta_pinn.dataio import convert_dataset_to_numpy, build_condition_vector_for_dataset
from meta_pinn.configbaselines import DEFAULT_OPTIONS      # ★ 参数以此为准

# ========= 评测超参（可用命令行覆盖）=========
BATCH        = 8192
WARMUP_ITERS = 3

# ----------------------------------
# 小工具
# ----------------------------------
def _cuda_sync_if_needed(device: torch.device) -> None:
    if device.type == 'cuda':
        torch.cuda.synchronize()

def _fit_scalers_and_set_to_model(Data, options: Dict, model: MetaPINN, device: torch.device) -> StandardScaler:
    """
    拟合 X/FA/C 的 scaler；把 FA/C 的 mean/std 写回模型缓冲区（供物理量反归一化使用）。
    返回 x_scaler。
    """
    X_all, F_all, C_all = [], [], []
    cond_dim = options.get('cond_dim', 1)
    per_task_stats = []

    for i in range(len(Data)):
        Xi, Vi, Ai, Alpi, Fi, Fni = convert_dataset_to_numpy(Data[i], options)
        X_all.append(Xi); F_all.append(Fi)
        C_all.append(build_condition_vector_for_dataset(Data[i], cond_dim))
        per_task_stats.append((i, Fi.mean(axis=0), Fi.std(axis=0), np.sqrt((Fi**2).mean(axis=0))))

    X_all = np.vstack(X_all)
    F_all = np.vstack(F_all)
    C_all = np.vstack(C_all)

    # 打印训练集 fa 统计
    fa_mean = F_all.mean(axis=0); fa_std = F_all.std(axis=0)
    fa_rms  = np.sqrt((F_all**2).mean(axis=0))
    print("[fa stats / train all] mean =", np.round(fa_mean,4),
          "| std =", np.round(fa_std,4),
          "| rms =", np.round(fa_rms,4),
          "| ||std|| =", round(float(np.linalg.norm(fa_std)),4))

    x_scaler = StandardScaler().fit(X_all)
    f_scaler = StandardScaler().fit(F_all)
    c_scaler = StandardScaler().fit(C_all)

    # 写入模型
    model.set_force_scaler(
        torch.tensor(f_scaler.mean, dtype=torch.float32, device=device),
        torch.tensor(f_scaler.std,  dtype=torch.float32, device=device)
    )
    model.set_condition_scaler(
        torch.tensor(c_scaler.mean, dtype=torch.float32, device=device),
        torch.tensor(c_scaler.std,  dtype=torch.float32, device=device)
    )

    # 归一化 sanity check
    Fn = (F_all - f_scaler.mean) / (f_scaler.std + 1e-12)
    print("[fa stats / normalized] mean ≈", np.round(Fn.mean(axis=0),3),
          "| std ≈", np.round(Fn.std(axis=0),3))
    print("---- per-task fa (mean | std | rms) ----")
    for i, m, s, r in per_task_stats:
        print(f"  task {i}: mean={np.round(m,3)} | std={np.round(s,3)} | rms={np.round(r,3)}")

    return x_scaler

def _compute_train_f_mean(DataTrain, options: Dict) -> np.ndarray:
    """训练集上计算 fa 的全局均值，作为 Mean Predictor。"""
    F_all = []
    for i in range(len(DataTrain)):
        _,_,_,_,Fi,_ = convert_dataset_to_numpy(DataTrain[i], options)
        F_all.append(Fi)
    return np.vstack(F_all).mean(axis=0)  # (3,)

@torch.no_grad()
def _pred_block(model: MetaPINN, x_batch: torch.Tensor, task_j: int, c_batch: Optional[torch.Tensor]=None) -> torch.Tensor:
    """
    不改 MetaPINN：统一调用 _pred_physical；外部不传 task_idx。
    """
    # 安全保护：task_j 不超过训练时的 num_tasks-1
    max_tid = model.task_embeddings.shape[0] - 1
    safe_tid = int(task_j) if int(task_j) <= max_tid else max_tid
    if safe_tid != task_j:
        print(f"[warn] task_id {task_j} 越界，已用 {safe_tid} 代替。")
    return model._pred_physical(x_batch, task_id=safe_tid, c_in=c_batch)

def _vector_rmse(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    e = y_pred - y_true
    return torch.sqrt(torch.mean(torch.sum(e * e, dim=-1))).item()

# ------- 适配所需（仅在本文件内定义，避免改库）-------
@torch.no_grad()
def backup_task_embeddings(model: MetaPINN) -> torch.Tensor:
    return model.task_embeddings.detach().clone()

@torch.no_grad()
def restore_task_embeddings(model: MetaPINN, backup: torch.Tensor) -> None:
    model.task_embeddings.copy_(backup)

def _augment_support(x: torch.Tensor, repeats: int = 1, noise_std: float = 0.0) -> torch.Tensor:
    if repeats <= 1 and noise_std <= 0: return x
    xs = [x]
    for _ in range(repeats-1):
        xs.append(x + noise_std * torch.randn_like(x) if noise_std > 0 else x.clone())
    return torch.cat(xs, dim=0)

def pick_support_indices(score_t: torch.Tensor, K: int, top_frac: float = 1.0) -> torch.Tensor:
    N = score_t.numel()
    if N == 0: return torch.tensor([], dtype=torch.long, device=score_t.device)
    K = int(min(max(1, K), N))
    if top_frac >= 1.0: return torch.topk(score_t, k=K).indices
    topK = max(1, int(K * float(top_frac)))
    idx_top = torch.topk(score_t, k=min(topK, N)).indices
    mask = torch.ones(N, dtype=torch.bool, device=score_t.device); mask[idx_top]=False
    idx_pool = torch.nonzero(mask, as_tuple=False).squeeze(-1)
    if idx_pool.numel() > 0:
        perm = idx_pool[torch.randperm(idx_pool.numel(), device=score_t.device)]
        idx_rand = perm[:max(0, K - idx_top.numel())]
        idx = torch.unique(torch.cat([idx_top, idx_rand]))[:K]
    else:
        idx = idx_top[:K]
    return idx

def adapt_task_embedding(model: MetaPINN, x_k, v_k, a_lp_k, f_k, fn_k, c_k, mass, task_id:int,
                         steps=150, lr=5e-2,
                         w_newton=0.02, w_resid=0.05, w_bias=0.005,
                         use_cond_mod=True, cond_mod_from='target', w_beta_reg=0.0008,
                         adapt_repeats=1, adapt_noise_std=0.0, adapt_prox_weight=0.003,
                         adapt_phys_gain=1.0, wind_threshold=5.0,
                         adapt_w_newton_scale=1.0, adapt_w_resid_scale=1.0,
                         adapt_axis_scale=(1.0, 1.0, 1.0),
                         adapt_w_beta_reg_scale=1.0):
    """
    仅调 task_embeddings / task_beta_logscale；其余权重保持冻结。
    （评测主流程里做了 model.requires_grad_(False)，这里需临时开启这两个参数）
    """
    dev = next(model.parameters()).device

    # ---------- 临时开启需要微调的参数 ----------
    prev_req = {
        "emb": bool(model.task_embeddings.requires_grad),
        "beta": bool(model.task_beta_logscale.requires_grad),
    }
    model.task_embeddings.requires_grad_(True)
    model.task_beta_logscale.requires_grad_(True)
    model.train()

    # --------- 准备数据 ----------
    x_k, v_k, a_lp_k = x_k.to(dev), v_k.to(dev), a_lp_k.to(dev)
    f_k, fn_k = f_k.to(dev), fn_k.to(dev)
    c_k = c_k.to(dev) if c_k is not None else None

    # ---------- 线性初始化 task_beta_logscale（可选） ----------
    try:
        with torch.no_grad():
            model.task_beta_logscale[task_id].zero_()
            if (c_k is not None) and (model.cond2beta is not None):
                beta_base = model._beta_from_c(c_k, task_id=task_id)
                beta_base_med = beta_base.median(dim=0).values
            else:
                beta_base_med = torch.ones(3, device=dev)

            f_pred0 = model._pred_physical(x_k, task_id=task_id, c_in=c_k)
            y = f_k - f_pred0
            X = fn_k
            lam = 1e-3
            beta_ls = []
            for ax in range(3):
                xax = X[:, ax:ax+1]; yax = y[:, ax:ax+1]
                num = (xax * yax).sum(); den = (xax * xax).sum() + lam
                b = (num / (den + 1e-8)).clamp(min=model.beta_min, max=model.beta_max)
                beta_ls.append(b)
            beta_ls = torch.stack(beta_ls)
            logscale = torch.log((beta_ls + 1e-6) / (beta_base_med + 1e-6))
            model.task_beta_logscale[task_id].copy_(logscale.clamp(-1.5, 1.5))
    except Exception:
        pass

    # ---------- 只优化这两个参数 ----------
    opt = torch.optim.SGD([model.task_embeddings, model.task_beta_logscale], lr=lr)

    # 物理项放大（强风时）
    phys_scale = 1.0
    if c_k is not None:
        wind_mag = c_k.abs().mean().item()
        if wind_mag >= wind_threshold:
            phys_scale = 1.0 + adapt_phys_gain
    wN = w_newton * phys_scale * adapt_w_newton_scale
    wR = w_resid  * phys_scale * adapt_w_resid_scale
    wB = w_bias
    wBeta = w_beta_reg * float(adapt_w_beta_reg_scale)

    axis_scale_t = torch.tensor(adapt_axis_scale, dtype=torch.float32, device=dev).view(1,3)

    # ---------- 适配循环 ----------
    with torch.enable_grad():   # 确保梯度跟踪开启
        e_anchor = model.task_embeddings[task_id].detach().clone()
        for _ in range(int(steps)):
            opt.zero_grad(set_to_none=True)

            # 数据扩增（可选）
            def _augment(x):
                if adapt_repeats <= 1 and adapt_noise_std <= 0:
                    return x
                xs = [x]
                for _ in range(adapt_repeats-1):
                    xs.append(x + (adapt_noise_std * torch.randn_like(x) if adapt_noise_std > 0 else x.clone()))
                return torch.cat(xs, dim=0)

            x_aug = _augment(x_k)
            rep = max(1, x_aug.shape[0] // max(1, x_k.shape[0]))
            def tile(t): return t.repeat((rep,1)) if (t is not None and t.ndim==2) else t
            v_aug = tile(v_k); a_aug = tile(a_lp_k); f_aug = tile(f_k); fn_aug = tile(fn_k); c_aug = tile(c_k)
            if a_aug is not None: a_aug = a_aug * axis_scale_t

            total, _ = model.total_loss(
                x_aug, v_aug, a_aug, mass, f_aug, fn_aug, task_id,
                w_newton=wN, w_resid=wR, w_bias=wB,
                c_target=c_aug,
                use_cond_mod=use_cond_mod, cond_mod_from=cond_mod_from, w_beta_reg=wBeta
            )

            # 近端正则：避免嵌入漂移过大
            e_cur = model.task_embeddings[task_id]
            l_prox = adapt_prox_weight * torch.sum((e_cur - e_anchor)**2)

            (total + l_prox).backward()

            # 只保留本 task 的梯度
            with torch.no_grad():
                if model.task_embeddings.grad is not None:
                    mask = torch.zeros_like(model.task_embeddings); mask[task_id] = 1.0
                    model.task_embeddings.grad *= mask
                if model.task_beta_logscale.grad is not None:
                    maskb = torch.zeros_like(model.task_beta_logscale); maskb[task_id] = 1.0
                    model.task_beta_logscale.grad *= maskb

            opt.step()

    # ---------- 恢复 requires_grad 状态 ----------
    model.task_embeddings.requires_grad_(prev_req["emb"])
    model.task_beta_logscale.requires_grad_(prev_req["beta"])

# ----------------------------------
# 只计“模型前向推理”的时间（总耗时 & 平均每样本耗时）
# ----------------------------------
@torch.no_grad()
def measure_inference_time(model: MetaPINN,
                           x_all: torch.Tensor,
                           c_all: Optional[torch.Tensor],
                           task_j: int,
                           batch: int,
                           warmup_iters: int,
                           device: torch.device) -> Tuple[float, float]:
    """
    只计模型对给定数据的一次完整推理时间（总耗时、平均每样本耗时）。
    - 不包含拟合/适配时间
    - 不包含数据搬运/缩放/索引选择等准备
    """
    N = int(x_all.shape[0])
    if N == 0:
        return 0.0, 0.0

    # warmup
    for _ in range(max(0, warmup_iters)):
        for s in range(0, N, batch):
            e = min(s + batch, N)
            xb = x_all[s:e]
            cb = c_all[s:e] if c_all is not None else None
            _ = _pred_block(model, xb, task_j=task_j, c_batch=cb)
    _cuda_sync_if_needed(device)

    # timing (单次完整遍历)
    t0 = time.perf_counter()
    for s in range(0, N, batch):
        e = min(s + batch, N)
        xb = x_all[s:e]
        cb = c_all[s:e] if c_all is not None else None
        _ = _pred_block(model, xb, task_j=task_j, c_batch=cb)
    _cuda_sync_if_needed(device)
    total_ms = (time.perf_counter() - t0) * 1000.0
    return total_ms, total_ms / N

# ----------------------------------
# Neural-Fly 风格三列指标评测（时间仅统计推理）
# ----------------------------------
def evaluate_neuralfly_style(model: MetaPINN, DataTrain, TestData, options: Dict,
                             device: torch.device, save_csv_path: str) -> None:
    os.makedirs(os.path.dirname(save_csv_path), exist_ok=True)

    x_scaler = _fit_scalers_and_set_to_model(DataTrain, options, model, device)
    f_mean_train = _compute_train_f_mean(DataTrain, options)
    f_mean_train_t = torch.tensor(f_mean_train, dtype=torch.float32, device=device).view(1, 3)

    # 预热（首次上卡）
    if len(TestData) > 0 and WARMUP_ITERS > 0:
        X0, _, _, _, _, _ = convert_dataset_to_numpy(TestData[0], options)
        C0 = build_condition_vector_for_dataset(TestData[0], model.cond_dim)
        X0n = x_scaler.transform(X0)
        xb = torch.tensor(X0n[:min(1024, len(X0n))], dtype=torch.float32, device=device)
        cb = torch.tensor(C0[:xb.shape[0]], dtype=torch.float32, device=device)
        with torch.no_grad():
            for _ in range(WARMUP_ITERS):
                _ = _pred_block(model, xb, task_j=0, c_batch=cb)
        _cuda_sync_if_needed(device)

    rows = []
    mass = options.get('UAV_mass', 1.0)
    tot_before_time, tot_before_N = 0.0, 0
    tot_after_time,  tot_after_N  = 0.0, 0

    print()
    for j in range(len(TestData)):
        meta = getattr(TestData[j], 'meta', {}) or {}
        cond_txt = meta.get('condition', '')
        print(f"**** : {cond_txt} ****")

        X, V, A, Alp, F, Fnom = convert_dataset_to_numpy(TestData[j], options)
        C = build_condition_vector_for_dataset(TestData[j], model.cond_dim)
        Xn = x_scaler.transform(X)

        x_all = torch.tensor(Xn, dtype=torch.float32, device=device)
        v_all = torch.tensor(V,  dtype=torch.float32, device=device)
        a_all = torch.tensor(Alp, dtype=torch.float32, device=device)
        f_all = torch.tensor(F,  dtype=torch.float32, device=device)
        fn_all= torch.tensor(Fnom, dtype=torch.float32, device=device)
        c_all = torch.tensor(C,  dtype=torch.float32, device=device) if C is not None else None
        N = x_all.shape[0]

        # ----- 1) Before learning -----
        total_before_ms, ms_before = measure_inference_time(
            model, x_all, c_all, task_j=j, batch=BATCH, warmup_iters=WARMUP_ITERS, device=device
        )
        tot_before_time += (total_before_ms / 1000.0)  # 秒
        tot_before_N    += N

        # 前向用于 MSE
        with torch.no_grad():
            preds_before = []
            for s in range(0, N, BATCH):
                e = min(s + BATCH, N)
                xb = x_all[s:e]
                cb = c_all[s:e] if c_all is not None else None
                preds_before.append(_pred_block(model, xb, task_j=j, c_batch=cb))
        y_before  = torch.cat(preds_before, dim=0)
        mse_before = torch.mean((y_before - f_all) ** 2).item()

        # ----- 2) Mean predictor -----
        y_mean = f_mean_train_t.expand(N, -1)
        mse_mean = torch.mean((y_mean - f_all) ** 2).item()

        # ----- 3) After learning φ(x) -----
        # 选择支持集（K-shot），做适配（不计入“推理时间”）
        if options.get('kshot_score', 'fn') == 'vnorm':
            score_vec = torch.linalg.norm(v_all, dim=-1)
        else:
            score_vec = fn_all.abs().sum(-1)
        idxK = pick_support_indices(score_vec, K=options.get('K_shot', 2500),
                                    top_frac=options.get('eval_top_frac', 1.0))

        mask_query = torch.ones(N, dtype=torch.bool, device=device); mask_query[idxK] = False
        xK, vK, aK, fK, fnK = x_all[idxK], v_all[idxK], a_all[idxK], f_all[idxK], fn_all[idxK]
        cK = c_all[idxK] if c_all is not None else None
        xQ, fQ, cQ = x_all[mask_query], f_all[mask_query], (c_all[mask_query] if c_all is not None else None)

        emb_bak  = backup_task_embeddings(model)
        beta_bak = model.task_beta_logscale.detach().clone()

        adapt_task_embedding(
            model, xK, vK, aK, fK, fnK, cK, mass, task_id=j,
            steps=options.get('adapt_steps', 250),
            lr=options.get('adapt_lr', 0.2),
            w_newton=options.get('w_newton', 0.01),
            w_resid=options.get('w_resid', 0.02),
            w_bias=options.get('w_bias', 0.005),
            use_cond_mod=options.get('use_cond_mod', True),
            cond_mod_from=options.get('cond_mod_from', 'target'),
            w_beta_reg=options.get('w_beta_reg', 0.0008),
            adapt_repeats=options.get('eval_adapt_repeats', 1),
            adapt_noise_std=options.get('eval_adapt_noise_std', 0.0),
            adapt_prox_weight=options.get('adapt_prox_weight', 0.003),
            adapt_phys_gain=options.get('adapt_phys_gain', 1.0),
            wind_threshold=options.get('adapt_wind_threshold', 5.0),
            adapt_w_newton_scale=options.get('adapt_w_newton_scale', 0.5),
            adapt_w_resid_scale=options.get('adapt_w_resid_scale', 0.5),
            adapt_axis_scale=options.get('adapt_axis_scale', (1.0, 1.0, 1.3)),
            adapt_w_beta_reg_scale=1.0
        )

        # 只对“查询集”计推理时间（不含适配）
        total_after_ms, ms_after = measure_inference_time(
            model, xQ, cQ, task_j=j, batch=BATCH, warmup_iters=WARMUP_ITERS, device=device
        )
        tot_after_time += (total_after_ms / 1000.0)  # 秒
        tot_after_N    += xQ.shape[0]

        # 推理得到 MSE(after)
        with torch.no_grad():
            preds_after = []
            M = xQ.shape[0]
            for s in range(0, M, BATCH):
                e = min(s + BATCH, M)
                xb = xQ[s:e]
                cb = cQ[s:e] if cQ is not None else None
                preds_after.append(_pred_block(model, xb, task_j=j, c_batch=cb))
            y_after = torch.cat(preds_after, dim=0)
            mse_after = torch.mean((y_after - fQ) ** 2).item()

        restore_task_embeddings(model, emb_bak)
        with torch.no_grad():
            model.task_beta_logscale.copy_(beta_bak)

        # ----- 打印（只保留推理时间：总耗时 & 平均每样本）-----
        print(f"Before learning:       MSE is {mse_before:.2f}   | infer_total={total_before_ms:.3f} ms | infer_avg={ms_before:.6f} ms/sample")
        print(f"Mean predictor:        MSE is {mse_mean:.2f}")
        print(f"After learning phi(x): MSE is {mse_after:.2f} | infer_total={total_after_ms:.3f} ms | infer_avg={ms_after:.6f} ms/sample\n")

        rows.append({
            'task_idx': j,
            'condition': cond_txt,
            'N_total': int(N),
            'N_query': int(xQ.shape[0]),
            'MSE_before': float(mse_before),
            'MSE_mean_predictor': float(mse_mean),
            'MSE_after_phi': float(mse_after),
            'before_infer_total_ms': float(total_before_ms),
            'before_infer_avg_ms_per_sample': float(ms_before),
            'after_infer_total_ms':  float(total_after_ms),
            'after_infer_avg_ms_per_sample':  float(ms_after),
            'improve_vs_before_abs': float(mse_before - mse_after),
            'improve_vs_before_pct': float(100.0 * (mse_before - mse_after) / max(1e-12, mse_before)),
            'improve_vs_mean_abs': float(mse_mean - mse_after),
            'improve_vs_mean_pct': float(100.0 * (mse_mean - mse_after) / max(1e-12, mse_mean)),
        })

    if len(rows) > 0:
        with open(save_csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    # Overall 汇总（仅推理）
    overall_before = 1000.0 * tot_before_time / max(1, tot_before_N)  # ms/sample
    overall_after  = 1000.0 * tot_after_time  / max(1, tot_after_N)   # ms/sample
    print("=== Inference Time Summary (forward only) ===")
    print(f"Overall avg BEFORE: {overall_before:.3f} ms/sample")
    print(f"Overall avg AFTER : {overall_after:.3f} ms/sample")

    with open(save_csv_path.replace('.csv', '_overall_time.txt'), 'w') as f:
        f.write(f"overall_before_ms_per_sample: {overall_before:.6f}\n")
        f.write(f"overall_after_ms_per_sample:  {overall_after:.6f}\n")

# ----------------------------------
# 主入口
# ----------------------------------
def parse_args():
    ap = argparse.ArgumentParser(description="Evaluate Meta-PINN on TestData (Neural-Fly style).")
    ap.add_argument("--weights", type=str,
                    default="saved_models/meta_pinn_baselinedata/meta_pinn_last.pth",
                    help="Path to trained weights (.pth)")
    ap.add_argument("--save_dir", type=str,
                    default="saved_models/meta_pinn_baselinedata",
                    help="Directory to save eval csv")
    ap.add_argument("--train_dir", type=str, default="data/training")
    ap.add_argument("--test_dir",  type=str, default="data/experiment")
    ap.add_argument("--test_exp",  type=str, default="(baseline_)([0-9]*|no)wind",
                    help="Regex for TestData expnames")
    ap.add_argument("--batch", type=int, default=BATCH)
    ap.add_argument("--warmup", type=int, default=WARMUP_ITERS)
    return ap.parse_args()

def main():
    args = parse_args()
    global BATCH, WARMUP_ITERS
    BATCH = int(args.batch); WARMUP_ITERS = int(args.warmup)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(DEFAULT_OPTIONS.get('seed', 42),
             deterministic=DEFAULT_OPTIONS.get('deterministic', True))
    print(f"SEED={DEFAULT_OPTIONS.get('seed',42)}, "
          f"deterministic={DEFAULT_OPTIONS.get('deterministic',True)}, device={device}")

    # 物理常量（保持一致）
    phys = get_uav_physics(device=device)
    DEFAULT_OPTIONS['drag_box'] = phys['drag_box']

    # 加载数据（features/output 与训练一致）
    raw_train = load_data(args.train_dir)
    raw_test  = load_data(args.test_dir, expnames=args.test_exp)
    Data      = format_data(raw_train, features=DEFAULT_OPTIONS['features'], output='fa')
    TestData  = format_data(raw_test,  features=DEFAULT_OPTIONS['features'], output='fa')

    for i,d in enumerate(Data):
        print(f"[Train Task {i}] {d.meta.get('method','')}_{d.meta.get('condition','')} <- {d.meta.get('filename','')}")
    for i,d in enumerate(TestData):
        print(f"[Test  Task {i}]  {d.meta.get('method','')}_{d.meta.get('condition','')} <- {d.meta.get('filename','')}")

    # 构建模型（结构参数与训练一致）
    input_dim = Data[0].X.shape[1]
    model = MetaPINN(
        input_dim=input_dim,
        num_tasks=len(Data),                   # 以训练任务数为准
        task_dim=128,
        hidden_dim=384,
        use_uncertainty=True,
        cond_dim=DEFAULT_OPTIONS['cond_dim'],
        use_cond_mod=DEFAULT_OPTIONS['use_cond_mod'],
        cond_mod_from=DEFAULT_OPTIONS['cond_mod_from'],
        beta_min=DEFAULT_OPTIONS['beta_min'],
        beta_max=DEFAULT_OPTIONS['beta_max']
    ).to(device)

    # 加载权重
    assert os.path.isfile(args.weights), f"找不到权重文件: {args.weights}"
    ckpt = torch.load(args.weights, map_location=device)
    missing, unexpected = model.load_state_dict(ckpt, strict=False)
    print(f"[Load] from {args.weights} | missing={len(missing)} | unexpected={len(unexpected)}")

    # 评测（Neural-Fly 样式）
    model.requires_grad_(False)
    csv_path = os.path.join(args.save_dir, "neuralfly_style_eval.csv")
    evaluate_neuralfly_style(model, Data, TestData, DEFAULT_OPTIONS, device, csv_path)
    print(f"[Saved] {csv_path}")

if __name__ == "__main__":
    main()
