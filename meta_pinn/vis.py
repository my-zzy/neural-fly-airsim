# meta_pinn/vis.py
import numpy as np
import torch
import matplotlib.pyplot as plt

from .dataio import convert_dataset_to_numpy, build_condition_vector_for_dataset
from .utils import StandardScaler
from .adapt import adapt_task_embedding, backup_task_embeddings, restore_task_embeddings

def fit_x_scaler(train_datasets, options):
    """和训练时一致：用训练集的 X 拟合输入标准化。"""
    X_all = []
    for d in train_datasets:
        X,_,_,_,_,_ = convert_dataset_to_numpy(d, options)
        X_all.append(X)
    X_all = np.vstack(X_all)
    return StandardScaler().fit(X_all)

@torch.no_grad()
def _predict(model, x_norm, c_phys, task_id):
    return model._pred_physical(x_norm, task_id=task_id, c_in=c_phys)

# --- vis_validation: 验证阶段更激进的自适应 ---
def vis_validation(
    *,
    model,
    train_datasets,
    test_data,
    task_id,
    options,
    idx_adapt_start=0,
    idx_adapt_end=1000,
    idx_val_start=1000,
    idx_val_end=2000,
    x_scaler=None,
    title_prefix="",
    # 验证阶段适应的默认增强（也可在 options 里配置同名键覆盖）
    # val_lr_mul=5.0,            # 学习率放大倍数
    # val_lr_max=0.25,           # 学习率上限
    # val_steps_mul=3.0,         # 适应步数乘子
    # val_w_beta_reg_scale=0.3,  # β 正则缩放（更小=更自由）
    # val_prox_weight=0.0,       # 近端正则（默认关闭）
    # 验证阶段更强适应
    val_lr_mul=8.0,            # 学习率放大倍数
    val_lr_max=0.35,           # 学习率上限
    val_steps_mul=4.0,         # 适应步数乘子
    val_w_beta_reg_scale=0.2,  # β 正则缩放（更小=更自由）
    val_prox_weight=0.0,       # 近端正则（默认关闭）
    val_w_newton_scale=0.7,    # Newton 权重缩放（略降低）
    val_w_resid_scale=1.5,    # Residual 权重缩放（略提高）

    #val_w_newton_scale=0.8,    # Newton 权重缩放（略降低）
    #val_w_resid_scale=1.25,    # Residual 权重缩放（略提高）
    val_repeats=1,             # 支持集增强：重复次数（>1 会引入随机性）
    val_noise_std=0.0,         # 支持集增强：噪声强度
    val_axis_scale=None        # 轴向强调，默认沿用 options / (1,1,1.3)
):
    import numpy as np
    import torch
    import matplotlib.pyplot as plt
    from .dataio import convert_dataset_to_numpy, build_condition_vector_for_dataset
    from .utils import StandardScaler
    from .adapt import adapt_task_embedding, backup_task_embeddings, restore_task_embeddings

    device = next(model.parameters()).device
    # 允许用 options 覆盖验证超参
    lr_mul   = options.get('val_lr_mul',   val_lr_mul)
    lr_max   = options.get('val_lr_max',   val_lr_max)
    steps_mul= options.get('val_steps_mul',val_steps_mul)
    wbr_scale= options.get('val_w_beta_reg_scale', val_w_beta_reg_scale)
    prox_w   = options.get('val_prox_weight', val_prox_weight)
    wN_scale = options.get('val_w_newton_scale', val_w_newton_scale)
    wR_scale = options.get('val_w_resid_scale',  val_w_resid_scale)
    reps     = options.get('val_repeats', val_repeats)
    nstd     = options.get('val_noise_std', val_noise_std)
    axis_scale = (
        val_axis_scale if val_axis_scale is not None
        else options.get('val_axis_scale', options.get('adapt_axis_scale', (1.0, 1.0, 1.3)))
    )

    if x_scaler is None:
        # 与训练一致的输入标准化
        X_all = []
        for d in train_datasets:
            X,_,_,_,_,_ = convert_dataset_to_numpy(d, options)
            X_all.append(X)
        X_all = np.vstack(X_all)
        x_scaler = StandardScaler().fit(X_all)

    # 准备派生量
    X,V,A,Alp,F,Fn = convert_dataset_to_numpy(test_data, options)
    C = build_condition_vector_for_dataset(test_data, model.cond_dim)

    N = len(X)
    sA, eA = max(0, idx_adapt_start), min(N, idx_adapt_end)
    sV, eV = max(0, idx_val_start),   min(N, idx_val_end)
    assert sA < eA and sV < eV, "adapt/val 区间为空，请检查索引"

    T = lambda a: torch.tensor(a, dtype=torch.float32, device=device)
    x_all_n = T(x_scaler.transform(X))
    v_all   = T(V)
    a_all   = T(Alp)
    f_all   = T(F)
    fn_all  = T(Fn)
    c_all   = T(C)

    # 适应前预测
    with torch.no_grad():
        f_pred_before = model._pred_physical(x_all_n[sV:eV], task_id=task_id, c_in=c_all[sV:eV])

    # ===== 验证阶段：更激进的 K-shot 适应 =====
    emb_bak = backup_task_embeddings(model)
    beta_bak = model.task_beta_logscale.detach().clone()

    base_lr    = options.get('adapt_lr', 5e-2)
    adapt_lr   = min(base_lr * float(lr_mul), float(lr_max))
    base_steps = int(options.get('adapt_steps', 150))
    adapt_steps= max(1, int(base_steps * float(steps_mul)))
    wN = options.get('w_newton', 0.01)
    wR = options.get('w_resid',  0.02)
    w_beta = options.get('w_beta_reg', 0.0008) * float(wbr_scale)

    adapt_task_embedding(
        model,
        x_k   = x_all_n[sA:eA], v_k=v_all[sA:eA], a_lp_k=a_all[sA:eA],
        f_k   = f_all[sA:eA],   fn_k=fn_all[sA:eA], c_k=c_all[sA:eA],
        mass  = options['UAV_mass'], task_id=task_id,
        steps = adapt_steps,
        lr    = adapt_lr,
        w_newton=wN, w_resid=wR,
        use_cond_mod=options.get('use_cond_mod', True),
        cond_mod_from=options.get('cond_mod_from','target'),
        w_beta_reg=w_beta,
        adapt_repeats=reps,
        adapt_noise_std=nstd,
        adapt_prox_weight=prox_w,
        adapt_phys_gain=options.get('adapt_phys_gain', 1.0),
        wind_threshold=options.get('adapt_wind_threshold', 5.0),
        adapt_w_newton_scale=float(wN_scale),
        adapt_w_resid_scale =float(wR_scale),
        adapt_axis_scale=axis_scale,
        adapt_w_beta_reg_scale=1.0
    )

    # 适应后预测
    with torch.no_grad():
        f_pred_after = model._pred_physical(x_all_n[sV:eV], task_id=task_id, c_in=c_all[sV:eV])

    # 还原，避免污染外部模型
    restore_task_embeddings(model, emb_bak)
    with torch.no_grad():
        model.task_beta_logscale.copy_(beta_bak)

    # 绘图
    import matplotlib.pyplot as plt
    t = test_data.meta['t'][sV:eV]
    y = F[sV:eV]
    fig, axs = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
    labs = ['Fx','Fy','Fz']
    for k, ax in enumerate(axs):
        ax.plot(t, y[:,k], label='GT', linewidth=1.2)
        ax.plot(t, f_pred_before.detach().cpu().numpy()[:,k], label='pred (before)', linewidth=1.0)
        ax.plot(t, f_pred_after.detach().cpu().numpy()[:,k],  label='pred (after)',  linewidth=1.0)
        ax.set_ylabel(labs[k]); ax.grid(True, alpha=0.3)
        if k == 0:
            ax.set_title(f"{title_prefix} {test_data.meta.get('condition','')}  |  adapt: [{sA},{eA})  val: [{sV},{eV})")
    axs[-1].set_xlabel('time [s]')
    axs[0].legend(ncol=3, fontsize=9)
    plt.tight_layout(); plt.show()

    mse_before = torch.mean((f_pred_before - f_all[sV:eV])**2).item()
    mse_after  = torch.mean((f_pred_after  - f_all[sV:eV])**2).item()
    return mse_before, mse_after


# --- error_statistics: 与上面一致地使用验证增强超参 ---
def error_statistics(
    *,
    model,
    train_datasets,
    test_data,
    task_id,
    options,
    x_scaler=None,
    idx_adapt_start=0,
    idx_adapt_end=None,
    eval_slice=None,
    # 验证阶段增强
    # val_lr_mul=5.0,
    # val_lr_max=0.25,
    # val_steps_mul=3.0,
    # val_w_beta_reg_scale=0.3,
    # val_prox_weight=0.0,
    # val_w_newton_scale=0.8,
    # val_w_resid_scale=1.25,
    val_lr_mul=8.0,            # 学习率放大倍数
    val_lr_max=0.35,           # 学习率上限
    val_steps_mul=4.0,         # 适应步数乘子
    val_w_beta_reg_scale=0.2,  # β 正则缩放（更小=更自由）
    val_prox_weight=0.0,       # 近端正则（默认关闭）
    val_w_newton_scale=0.7,    # Newton 权重缩放（略降低）
    val_w_resid_scale=1.5,    # Residual 权重缩放（略提高）
    
    val_repeats=1,
    val_noise_std=0.0,
    val_axis_scale=None
):
    import numpy as np
    import torch
    from .dataio import convert_dataset_to_numpy, build_condition_vector_for_dataset
    from .utils import StandardScaler
    from .adapt import adapt_task_embedding, backup_task_embeddings, restore_task_embeddings

    device = next(model.parameters()).device
    # options 覆盖
    lr_mul   = options.get('val_lr_mul',   val_lr_mul)
    lr_max   = options.get('val_lr_max',   val_lr_max)
    steps_mul= options.get('val_steps_mul',val_steps_mul)
    wbr_scale= options.get('val_w_beta_reg_scale', val_w_beta_reg_scale)
    prox_w   = options.get('val_prox_weight', val_prox_weight)
    wN_scale = options.get('val_w_newton_scale', val_w_newton_scale)
    wR_scale = options.get('val_w_resid_scale',  val_w_resid_scale)
    reps     = options.get('val_repeats', val_repeats)
    nstd     = options.get('val_noise_std', val_noise_std)
    axis_scale = (
        val_axis_scale if val_axis_scale is not None
        else options.get('val_axis_scale', options.get('adapt_axis_scale', (1.0, 1.0, 1.3)))
    )

    if x_scaler is None:
        X_all = []
        for d in train_datasets:
            X,_,_,_,_,_ = convert_dataset_to_numpy(d, options)
            X_all.append(X)
        X_all = np.vstack(X_all)
        x_scaler = StandardScaler().fit(X_all)

    X,V,A,Alp,F,Fn = convert_dataset_to_numpy(test_data, options)
    C = build_condition_vector_for_dataset(test_data, model.cond_dim)

    N = len(X)
    sA = max(0, idx_adapt_start)
    eA = min(N, idx_adapt_end if idx_adapt_end is not None else sA + min(1000, N - sA))
    if eval_slice is None:
        sE, eE = eA, N
    else:
        sE, eE = max(0, eval_slice[0]), min(N, eval_slice[1])
    assert sA < eA and sE < eE, "adapt/评估区间为空，请检查索引"

    T = lambda a: torch.tensor(a, dtype=torch.float32, device=device)
    x_all_n = T(x_scaler.transform(X))
    v_all   = T(V)
    a_all   = T(Alp)
    f_all   = T(F)
    fn_all  = T(Fn)
    c_all   = T(C)

    # before
    with torch.no_grad():
        f_pred_before = model._pred_physical(x_all_n[sE:eE], task_id=task_id, c_in=c_all[sE:eE])
        mse_before = torch.mean((f_pred_before - f_all[sE:eE])**2).item()

    # mean predictor（训练集 F 全局均值）
    F_train = []
    for d in train_datasets:
        _,_,_,_,Ftr,_ = convert_dataset_to_numpy(d, options)
        F_train.append(Ftr)
    F_mu = torch.tensor(np.vstack(F_train).mean(axis=0), device=device)  # [3]
    mse_mean = torch.mean((f_all[sE:eE] - F_mu)**2).item()

    # after: 更激进的验证自适应
    emb_bak = backup_task_embeddings(model)
    beta_bak = model.task_beta_logscale.detach().clone()

    base_lr    = options.get('adapt_lr', 5e-2)
    adapt_lr   = min(base_lr * float(lr_mul), float(lr_max))
    base_steps = int(options.get('adapt_steps', 150))
    adapt_steps= max(1, int(base_steps * float(steps_mul)))
    wN = options.get('w_newton', 0.01)
    wR = options.get('w_resid',  0.02)
    w_beta = options.get('w_beta_reg', 0.0008) * float(wbr_scale)

    adapt_task_embedding(
        model,
        x_k   = x_all_n[sA:eA], v_k=v_all[sA:eA], a_lp_k=a_all[sA:eA],
        f_k   = f_all[sA:eA],   fn_k=fn_all[sA:eA], c_k=c_all[sA:eA],
        mass  = options['UAV_mass'], task_id=task_id,
        steps = adapt_steps,
        lr    = adapt_lr,
        w_newton=wN, w_resid=wR,
        use_cond_mod=options.get('use_cond_mod', True),
        cond_mod_from=options.get('cond_mod_from','target'),
        w_beta_reg=w_beta,
        adapt_repeats=reps,
        adapt_noise_std=nstd,
        adapt_prox_weight=prox_w,
        adapt_phys_gain=options.get('adapt_phys_gain', 1.0),
        wind_threshold=options.get('adapt_wind_threshold', 5.0),
        adapt_w_newton_scale=float(wN_scale),
        adapt_w_resid_scale =float(wR_scale),
        adapt_axis_scale=axis_scale,
        adapt_w_beta_reg_scale=1.0
    )
    with torch.no_grad():
        f_pred_after = model._pred_physical(x_all_n[sE:eE], task_id=task_id, c_in=c_all[sE:eE])
        mse_after = torch.mean((f_pred_after - f_all[sE:eE])**2).item()

    restore_task_embeddings(model, emb_bak)
    with torch.no_grad():
        model.task_beta_logscale.copy_(beta_bak)

    return mse_before, mse_mean, mse_after
