
import os, math, numpy as np, torch, torch.nn.functional as Fnn
from .utils import StandardScaler, linear_warm, save_checkpoint
from .dataio import prepare_dataloader, convert_dataset_to_numpy, build_condition_vector_for_dataset
from .adapt import backup_task_embeddings, restore_task_embeddings, pick_support_indices, adapt_task_embedding

def _fit_scalers(Data, options, model, device):
    import numpy as np
    X_all, F_all, C_all_list, Ns = [], [], [], []
    cond_dim = options.get('cond_dim', 1)

    # 逐任务收集
    per_task_stats = []
    for i in range(len(Data)):
        Xi,Vi,Ai,Alpi,Fi,Fni = convert_dataset_to_numpy(Data[i], options)
        X_all.append(Xi); F_all.append(Fi); Ns.append(len(Xi))
        C_all_list.append(build_condition_vector_for_dataset(Data[i], cond_dim))
        # 任务级统计（可选）
        per_task_stats.append((
            i,
            Fi.mean(axis=0),           # (3,)
            Fi.std(axis=0),            # (3,)
            np.sqrt((Fi**2).mean(axis=0))  # RMS per axis
        ))

    X_all = np.vstack(X_all)
    F_all = np.vstack(F_all)
    C_all = np.vstack(C_all_list)

    # === 全局 fa 统计 ===
    fa_mean = F_all.mean(axis=0)             # (3,)
    fa_std  = F_all.std(axis=0)              # (3,)
    fa_rms  = np.sqrt((F_all**2).mean(axis=0))
    fa_std_norm = np.linalg.norm(fa_std)

    # 打印（四舍五入避免刷屏）
    print("[fa stats / train all] mean =", np.round(fa_mean, 4),
          "| std =", np.round(fa_std, 4),
          "| rms =", np.round(fa_rms, 4),
          "| ||std|| =", round(float(fa_std_norm), 4))

    # 小标准差告警（避免 scaler 爆炸）
    tiny = (fa_std < 1e-6)
    if tiny.any():
        bad_axes = [idx for idx, ok in enumerate(~tiny) if not ok]
        print("[warn] Some fa dims have ~zero std; dims:", bad_axes,
              "| raw std:", np.round(fa_std, 8))

    # === 拟合 scaler ===
    x_scaler = StandardScaler().fit(X_all)
    f_scaler = StandardScaler().fit(F_all)
    c_scaler = StandardScaler().fit(C_all)

    # 归一化后再核对一次（应 ~ N(0,1)）
    F_all_norm = (F_all - f_scaler.mean) / (f_scaler.std + 1e-12)
    fa_n_mean = F_all_norm.mean(axis=0)
    fa_n_std  = F_all_norm.std(axis=0)
    print("[fa stats / normalized] mean ≈", np.round(fa_n_mean, 3),
          "| std ≈", np.round(fa_n_std, 3))

    # 可选：逐任务简表
    print("---- per-task fa (mean | std) ----")
    for i, m, s, r in per_task_stats:
        print(f"  task {i}: mean={np.round(m,3)} | std={np.round(s,3)} | rms={np.round(r,3)}")

    # 把 scaler 参数喂给模型（模型内部在 total_loss / pred 时用）
    model.set_force_scaler(
        torch.tensor(f_scaler.mean, dtype=torch.float32, device=device),
        torch.tensor(f_scaler.std,  dtype=torch.float32, device=device)
    )
    model.set_condition_scaler(
        torch.tensor(c_scaler.mean, dtype=torch.float32, device=device),
        torch.tensor(c_scaler.std,  dtype=torch.float32, device=device)
    )

    return x_scaler, Ns

def train_meta_pinn_multitask(model, Data, TestData, mass, options, save_path=None):
    device = next(model.parameters()).device
    opt = torch.optim.AdamW(model.parameters(), lr=options['learning_rate'], weight_decay=1e-4)
    scheduler = None

    x_scaler, Ns = _fit_scalers(Data, options, model, device)

    if options.get('scheduler','onecycle') == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=options['num_epochs'])
    else:
        bsz = options.get('batch_size',128)
        steps_per_epoch = int(sum(math.ceil(n/bsz) for n in Ns))
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=options.get('max_lr', max(1e-5, options['learning_rate']*0.1)),
            epochs=options['num_epochs'], steps_per_epoch=max(1,steps_per_epoch),
            pct_start=0.1, div_factor=10.0, final_div_factor=100.0
        )

    hist = {'train_total':[], 'train_mse':[], 'train_newton':[], 'train_residual':[], 'train_beta_reg':[], 'test_avg_mse':[]}

    beta_reg_schedule = options.get('beta_reg_schedule', None)
    if not callable(beta_reg_schedule):
        beta_reg_schedule = lambda epoch: options.get('w_beta_reg', 0.0008)

    warmup_epochs = options.get('warmup_epochs', 40)
    warmup_start  = options.get('warmup_start', 0.03)
    warmup_end    = options.get('warmup_end', 1.0)
    use_cond_mod  = options.get('use_cond_mod', True)
    cond_mod_from = options.get('cond_mod_from', 'target')

    best_mse = float('inf')
    ckpt_last = os.path.join(save_path, "meta_pinn_last.pth") if save_path else None
    ckpt_best = os.path.join(save_path, "meta_pinn_best.pth") if save_path else None

    for epoch in range(options['num_epochs']):
        w_beta_reg = float(beta_reg_schedule(epoch))
        warm = linear_warm((epoch + 1) / max(1, warmup_epochs), warmup_start, warmup_end)

        w_newton = warm * options.get('w_newton',0.01)
        w_resid  = max(warm * options.get('w_resid',0.02), warm * 0.1)

        model.train()
        total_sum = mse_sum = newton_sum = resid_sum = beta_sum = 0.0; n_batches=0

        for tid in range(len(Data)):
            X,V,A,Alp,F,Fnom = convert_dataset_to_numpy(Data[tid], options)
            Xn = x_scaler.transform(X)
            C  = build_condition_vector_for_dataset(Data[tid], model.cond_dim)
            loader = prepare_dataloader(
                Xn,V,Alp,F,Fnom, task_id=tid, C=C,
                batch_size=options.get('batch_size',128), shuffle=True,
                seed=options.get('seed', 42)
            )
            for batch in loader:
                x = batch['x'].to(device); v = batch['v'].to(device); a_lp = batch['a_lp'].to(device)
                f = batch['f'].to(device); fn = batch['fn'].to(device); c_tgt = batch['c'].to(device)
                ttid = batch['task_id']

                opt.zero_grad()
                total, comps = model.total_loss(
                    x, v, a_lp, mass, f, fn, ttid,
                    w_newton=w_newton, w_resid=w_resid, 
                    c_target=c_tgt,
                    use_cond_mod=use_cond_mod, cond_mod_from=cond_mod_from, w_beta_reg=w_beta_reg
                )
                total.backward(); opt.step()
                if isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR): scheduler.step()
                total_sum += total.item(); mse_sum += comps['mse'].item(); newton_sum += comps['newton'].item()
                resid_sum += comps['residual'].item(); beta_sum += comps['beta_reg'].item(); n_batches+=1

        if isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR): scheduler.step()

        hist['train_total'].append(total_sum/max(1,n_batches))
        hist['train_mse'].append(mse_sum/max(1,n_batches))
        hist['train_newton'].append(newton_sum/max(1,n_batches))
        hist['train_residual'].append(resid_sum/max(1,n_batches))
        hist['train_beta_reg'].append(beta_sum/max(1,n_batches))

        # ===== evaluation (per-task K-shot) =====
        model.eval(); emb_bak = backup_task_embeddings(model)
        beta_bak = model.task_beta_logscale.detach().clone()
        avg_test_mse = 0.0
        for j in range(len(TestData)):
            Xv,Vv,Av,Alpv,Fv,Fnomv = convert_dataset_to_numpy(TestData[j], options)
            Xv_n = x_scaler.transform(Xv)
            Cv   = build_condition_vector_for_dataset(TestData[j], model.cond_dim)
            x_all  = torch.tensor(Xv_n, dtype=torch.float32, device=device)
            v_all  = torch.tensor(Vv,   dtype=torch.float32, device=device)
            al_all = torch.tensor(Alpv, dtype=torch.float32, device=device)
            f_all  = torch.tensor(Fv,   dtype=torch.float32, device=device)
            fn_all = torch.tensor(Fnomv,dtype=torch.float32, device=device)
            c_all  = torch.tensor(Cv,   dtype=torch.float32, device=device)

            K_task = options.get('K_shot', 300)
            if options.get('kshot_score', 'fn') == 'vnorm':
                score_vec = torch.linalg.norm(v_all, dim=-1)
            else:
                score_vec = fn_all.abs().sum(-1)

            idxK  = pick_support_indices(score_vec, K=K_task, top_frac=options.get('eval_top_frac',1.0))

            mask_query = torch.ones(x_all.shape[0], dtype=torch.bool, device=device)
            mask_query[idxK] = False
            xK, vK, aK, fK, fnK, cK = x_all[idxK], v_all[idxK], al_all[idxK], f_all[idxK], fn_all[idxK], c_all[idxK]
            xQ, fQ, cQ = x_all[mask_query], f_all[mask_query], c_all[mask_query]

            with torch.no_grad():
                f_pred = model._pred_physical(xQ, task_id=j, c_in=cQ)
                mse_before = Fnn.mse_loss(f_pred, fQ).item()

            adapt_task_embedding(
                model, xK, vK, aK, fK, fnK, cK, mass, task_id=j,
                steps=options.get('adapt_steps',150),
                lr=options.get('adapt_lr', 5e-2),
                w_newton=w_newton, w_resid=w_resid, 
                use_cond_mod=use_cond_mod, cond_mod_from=cond_mod_from, w_beta_reg=options.get('w_beta_reg',0.0008),
                adapt_repeats=options.get('eval_adapt_repeats',1),
                adapt_noise_std=options.get('eval_adapt_noise_std',0.0),
                adapt_prox_weight=options.get('adapt_prox_weight',0.003),
                adapt_phys_gain=options.get('adapt_phys_gain', 1.0),
                wind_threshold=options.get('adapt_wind_threshold', 5.0),
                adapt_w_newton_scale=options.get('adapt_w_newton_scale', 0.5),
                adapt_w_resid_scale=options.get('adapt_w_resid_scale', 0.5),
                adapt_axis_scale=options.get('adapt_axis_scale',(1.0,1.0,1.3)),
                adapt_w_beta_reg_scale=1.0
            )

            with torch.no_grad():
                f_pred = model._pred_physical(xQ, task_id=j, c_in=cQ)
                mse_after = Fnn.mse_loss(f_pred, fQ).item()
                avg_test_mse += mse_after

            if options.get('print_task_mse', True):
                print(f"  Task {j}: K-shot before MSE={mse_before:.4f} -> after={mse_after:.4f}")

            restore_task_embeddings(model, emb_bak)
            with torch.no_grad(): model.task_beta_logscale.copy_(beta_bak)

        avg_test_mse /= max(1,len(TestData))
        hist['test_avg_mse'].append(avg_test_mse)
        print(f"[Epoch {epoch}] warmup={warm:.2f} | Train total: {hist['train_total'][-1]:.4f} | MSE {hist['train_mse'][-1]:.4f} Newton {hist['train_newton'][-1]:.4f} Resid {hist['train_residual'][-1]:.4f} | BetaReg {hist['train_beta_reg'][-1]:.4f} || Test MSE {avg_test_mse:.4f}")

        if save_path:
            save_checkpoint(model, ckpt_last)
            if avg_test_mse < best_mse:
                best_mse = avg_test_mse
                save_checkpoint(model, ckpt_best)

    return hist
