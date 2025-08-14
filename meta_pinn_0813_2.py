# meta_pinn.py — Meta-PINN (Scheme A)
# SG smoothing, low-freq Newton, robust Residual
# nominal drag (tiny thrust), beta_drag calibration
# OneCycleLR + CNM (conditional nominal modulation)
# task-specific beta bias (adaptable per task in K-shot)
# + per-task adapt LR (adapt_lr_task) + epoch-wise beta_reg schedule
# + All-task booster: closed-form beta warm-start + informative K-shot + aggressive hyper overrides
# + Deterministic training/eval & checkpoint saving

import os, math, random, numpy as np, torch
import torch.nn as nn
import torch.nn.functional as Fnn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# ------------------ Repro helpers ------------------
def set_seed(seed: int = 42, deterministic: bool = True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")  # or ":4096:8"
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======================= Utils =======================

class StandardScaler:
    def __init__(self): self.mean=None; self.std=None
    def fit(self, x: np.ndarray):
        self.mean = x.mean(axis=0, keepdims=True)
        self.std  = x.std(axis=0, keepdims=True) + 1e-8
        return self
    def transform(self, x: np.ndarray): return (x - self.mean)/self.std
    def inverse_transform(self, x: np.ndarray): return x*self.std + self.mean

def _ma_torch(a_t: torch.Tensor, k: int) -> torch.Tensor:
    if k <= 1: return a_t
    x = a_t.T.unsqueeze(0)  # [1,3,N]
    pad = k - 1
    xpad = Fnn.pad(x, (pad, 0), mode='replicate')
    w = torch.ones(1, 1, k, device=a_t.device, dtype=a_t.dtype) / k
    y = Fnn.conv1d(xpad, w, groups=3)
    return y.squeeze(0).T

def smooth_acc_torch(a_t: torch.Tensor, window: int = 15, poly: int = 3) -> torch.Tensor:
    try:
        from scipy.signal import savgol_filter
        a_np = a_t.detach().cpu().numpy()
        win = max(5, window | 1)
        poly = min(poly, win - 1)
        a_low = savgol_filter(a_np, window_length=win, polyorder=poly, axis=0, mode='interp')
        return torch.as_tensor(a_low, device=a_t.device, dtype=a_t.dtype)
    except Exception:
        k = max(5, window | 1)
        return _ma_torch(a_t, k)

# ---------- Condition helpers ----------

def _parse_condition_to_vec(meta: dict, cond_dim: int = 1) -> np.ndarray:
    if 'wind_vec' in meta and meta['wind_vec'] is not None:
        v = np.asarray(meta['wind_vec'], dtype=np.float32).reshape(-1)
    elif 'wind_speed' in meta:
        v = np.array([float(meta['wind_speed'])], dtype=np.float32)
    else:
        v = 0.0
        cond = str(meta.get('condition', '')).lower()
        if 'wind' in cond:
            import re
            m = re.search(r'(\d+(\.\d+)?)', cond)
            v = float(m.group(1)) if m else 0.0
        v = np.array([v], dtype=np.float32)
    if v.size < cond_dim: v = np.pad(v, (0, cond_dim - v.size))
    elif v.size > cond_dim: v = v[:cond_dim]
    return v.astype(np.float32)

def build_condition_vector_for_dataset(dobj, cond_dim: int = 1) -> np.ndarray:
    c_vec = _parse_condition_to_vec(dobj.meta, cond_dim=cond_dim)
    N = dobj.X.shape[0]
    return np.repeat(c_vec.reshape(1, -1), N, axis=0).astype(np.float32)

# ======================= Nominal aero model (pure aerodynamics) =======================

def compute_nominal_force(
    X: np.ndarray,
    V: np.ndarray,
    feature_len: dict,
    *,
    rho=1.225,
    C_T=0.109919,
    D=0.2286,
    rpm_max=6396.667,
    mass=1.0,
    hover_throttle=None,     # None / scalar / [N]
    hover_pwm_norm=0.5,
    use_hover_calib=True,
    drag_box_np=None,
    beta_drag: float = 1.0,  # 阻力缩放（可由 epoch 0 自动回归）
) -> np.ndarray:
    # build column slices from feature_len order in utils.format_data
    idx = 0; slices = {}
    for name, d in feature_len.items():
        if d <= 0: continue
        slices[name] = slice(idx, idx + d); idx += d

    N = V.shape[0]
    # PWM block -> 4 motors (ensure 0..1)
    if 'pwm' in slices:
        pwm_blk = X[:, slices['pwm']].astype(np.float32)
        if pwm_blk.max() > 5.0:  # looks like 1000~2000
            pwm_blk = (pwm_blk - 1000.0) / 1000.0
        pwm_norm = pwm_blk[:, :4] if pwm_blk.shape[1] >= 4 else np.repeat(pwm_blk, 4, axis=1)
        pwm_norm = np.clip(pwm_norm, 0.0, 1.0)
    else:
        pwm_norm = np.zeros((N, 4), dtype=np.float32)

    # PWM -> RPM (linear), optional hover calibration so that sum(T) ≈ m g
    rpm = pwm_norm * rpm_max  # [N,4]

    # 悬停标定：sum(T) ≈ m g（优先使用低速样本）
    if use_hover_calib:
        if hover_throttle is not None:
            h = np.asarray(hover_throttle, dtype=np.float32)
            if h.ndim == 0:
                rpm_hover = np.ones_like(pwm_norm) * (float(h) * rpm_max)
            else:
                h = h.reshape(-1,1) if h.ndim == 1 else h[:, :4]
                rpm_hover = np.repeat(h, 4, axis=1) * rpm_max
        else:
            rpm_hover = np.ones_like(pwm_norm) * (hover_pwm_norm * rpm_max)

        n_hover = rpm_hover / 60.0
        T_hover_per = C_T * rho * (n_hover**2) * (D**4)
        T_sum = T_hover_per.sum(axis=1)  # [N]

        # 只用低速样本估计 scale，减少风/机动干扰
        vmag = np.linalg.norm(V, axis=1)
        sel = vmag < np.percentile(vmag, 30)
        if not np.any(sel):
            sel = slice(None)

        target = mass * 9.80665
        scale = target / (np.median(T_sum[sel]) + 1e-8)
        scale = float(np.clip(scale, 0.2, 5.0))  # safe clamp
        rpm = rpm * scale

    # thrust (+z) —— 对“残差标签”场景，保持极小推力项
    n = rpm / 60.0
    T_per = C_T * rho * (n**2) * (D**4)                 # [N,4]
    T_per = np.clip(T_per, 0.0, 10.0)                   # safety cap per motor
    T_sum = T_per.sum(axis=1)                           # [N]
    alpha_thrust = 1e-10
    T_sum = alpha_thrust * T_sum

    if drag_box_np is not None:
        # 支持 beta_drag: float 或 [3]
        beta = np.asarray(beta_drag, dtype=np.float32)
        if beta.ndim == 0:
            beta_vec = np.array([float(beta)]*3, dtype=np.float32)
        else:
            assert beta.shape[0] == 3, "beta_drag vector must have length 3"
            beta_vec = beta.astype(np.float32)

        # F_drag = - (beta ⊙ drag_box ⊙ rho) ⊙ (v|v|)
        Z = V * np.abs(V)                          # [N,3]
        coeff = (drag_box_np.astype(np.float32) * float(rho)) * beta_vec  # [3]
        F_drag = - Z * coeff                       # [N,3]
    else:
        F_drag = - 0.325 * V

    f_nom = np.zeros_like(V, dtype=np.float32)
    f_nom[:, 0] = -F_drag[:, 0]
    f_nom[:, 1] = -F_drag[:, 1]
    f_nom[:, 2] = T_sum - F_drag[:, 2]
    return f_nom


# ======================= Model =======================

class MetaPINN(nn.Module):
    def __init__(self, input_dim, num_tasks, task_dim=8, hidden_dim=64, use_uncertainty=True,
                 # condition scalers
                 cond_dim=1,
                 # CNM
                 use_cond_mod=True, cond_mod_from='target',
                 beta_min=0.15, beta_max=8.0,
                 # FiLM（默认关闭）
                 use_film=False, film_gamma_min=0.6, film_gamma_max=1.6, film_delta_max=0.2):
        super().__init__()
        self.num_tasks = num_tasks
        self.use_uncertainty = use_uncertainty

        # ---- force predictor ----
        self.task_embeddings = nn.Parameter(torch.randn(num_tasks, task_dim) * 0.1)
        self.fc1 = nn.Linear(input_dim + task_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 3)
        self.act = nn.Tanh()

        if use_uncertainty:
            self.log_vars = nn.Parameter(torch.zeros(4))  # data, newton, resid, balance

        # buffers for inverse-transform of force
        self.register_buffer('f_mean_t', torch.zeros(1,3))
        self.register_buffer('f_std_t',  torch.ones(1,3))

        # condition scalers
        self.cond_dim = cond_dim
        self.register_buffer('c_mean_t', torch.zeros(1,cond_dim))
        self.register_buffer('c_std_t',  torch.ones(1,cond_dim))

        # ---- CNM: c -> beta ----
        self.use_cond_mod = bool(use_cond_mod)
        self.cond_mod_from = str(cond_mod_from)
        self.beta_min = float(beta_min)
        self.beta_max = float(beta_max)
        if self.use_cond_mod:
            self.cond2beta = nn.Sequential(
                nn.Linear(cond_dim, max(8, cond_dim*2)),
                nn.Tanh(),
                nn.Linear(max(8, cond_dim*2), 3)
            )
        else:
            self.register_module('cond2beta', None)

        # ---- 任务专属 β 偏置：log-scale（可在 K-shot 中适配）----
        self.task_beta_logscale = nn.Parameter(torch.zeros(num_tasks, 3))

        # ---- FiLM(beta)（默认关闭）----
        self.use_film = bool(use_film)
        self.film_gamma_min = float(film_gamma_min)
        self.film_gamma_max = float(film_gamma_max)
        self.film_delta_max = float(film_delta_max)
        if self.use_film:
            self.beta2film = nn.Sequential(
                nn.Linear(3, 16),
                nn.Tanh(),
                nn.Linear(16, 2*hidden_dim)  # -> [gamma|delta]
            )
        else:
            self.register_module('beta2film', None)

    # ----- scalers -----
    @torch.no_grad()
    def set_force_scaler(self, mean_t: torch.Tensor, std_t: torch.Tensor):
        self.f_mean_t.copy_(mean_t); self.f_std_t.copy_(std_t)
    @torch.no_grad()
    def set_condition_scaler(self, mean_t: torch.Tensor, std_t: torch.Tensor):
        self.c_mean_t.copy_(mean_t); self.c_std_t.copy_(std_t)

    # ----- helpers -----
    @staticmethod
    def _sigmoid_range(x, lo, hi):
        return lo + (hi - lo) * torch.sigmoid(x)

    def _beta_from_c(self, c_phys: torch.Tensor, task_id: int = None) -> torch.Tensor:
        c_n = (c_phys - self.c_mean_t) / (self.c_std_t + 1e-8)
        raw = self.cond2beta(c_n)
        beta = self._sigmoid_range(raw, self.beta_min, self.beta_max)  # [B,3]
        # 任务专属 logscale 偏置（可适配）
        if (task_id is not None):
            beta = beta * torch.exp(self.task_beta_logscale[task_id].unsqueeze(0))
            beta = torch.clamp(beta, self.beta_min, self.beta_max)
        return beta

    def _film_from_beta(self, beta: torch.Tensor, hidden_dim: int):
        gd = self.beta2film(beta)   # [B, 2H]
        gamma_raw, delta_raw = torch.split(gd, hidden_dim, dim=-1)
        gamma = self._sigmoid_range(gamma_raw, self.film_gamma_min, self.film_gamma_max)  # [B,H]
        delta = self.film_delta_max * torch.tanh(delta_raw)                                # [B,H]
        return gamma, delta

    # ----- forward with optional FiLM -----
    def _pred_physical(self, x, task_id: int, c_in: torch.Tensor = None):
        B = x.shape[0]
        e = self.task_embeddings[task_id].expand(B, -1)
        h = torch.cat([x, e], dim=-1)
        h = self.act(self.fc1(h))

        if self.use_film and (c_in is not None) and (self.beta2film is not None) and (self.cond2beta is not None):
            beta = self._beta_from_c(c_in, task_id=task_id)
            gamma, delta = self._film_from_beta(beta, h.shape[-1])
            h = gamma * h + delta  # FiLM modulation

        h = self.act(self.fc2(h))
        f_n = self.fc3(h)
        return f_n * self.f_std_t + self.f_mean_t

    # ----- losses -----
    def compute_components(self, x, v, a_lp, m, f_actual_phys, f_nom_phys, task_id: int,
                           w_balance=0.0, c_target=None,
                           use_cond_mod=False, cond_mod_from='target', w_beta_reg=0.01):
        # choose c input (for CNM & FiLM)
        c_in = None
        if (cond_mod_from == 'target') and (c_target is not None):
            c_in = c_target

        # CNM: beta on F_nom（带任务偏置）
        beta_reg = x.new_tensor(0.0)
        fn_phys = f_nom_phys
        if use_cond_mod and (self.cond2beta is not None) and (c_in is not None):
            beta = self._beta_from_c(c_in, task_id=task_id)
            fn_phys = beta * f_nom_phys
            # β 偏离 1 的惩罚 + 任务偏置的 L2 正则（很小）
            beta_reg = ((beta - 1.0)**2).mean() + 0.01 * (self.task_beta_logscale[task_id]**2).mean()

        # force prediction with (optional) FiLM
        f_pred = self._pred_physical(x, task_id, c_in=c_in)

        # data loss
        l_mse = Fnn.mse_loss(f_pred, f_actual_phys)

        # robust residual
        resid = (f_pred + fn_phys) - f_actual_phys
        scale = resid.detach().abs().median().clamp(min=1.0)
        l_resid = Fnn.mse_loss(resid / scale, torch.zeros_like(resid))

        # low-frequency Newton
        l_newton = Fnn.mse_loss(f_pred, m * a_lp)

        # optional balance
        l_balance = Fnn.mse_loss(f_pred, torch.zeros_like(f_pred)) if w_balance > 0 else f_pred.new_tensor(0.0)

        return f_pred, l_mse, l_newton, l_resid, l_balance, beta_reg, fn_phys

    def total_loss(self, x, v, a_lp, m, f_actual_phys, f_nom_phys, task_id: int,
                   w_newton=0.01, w_resid=0.02, w_balance=0.0, w_bias=0.01,
                   c_target=None, use_cond_mod=False, cond_mod_from='target', w_beta_reg=0.01):

        f_pred, l_mse, l_newton, l_resid, l_bal, l_beta, fn_phys_used = self.compute_components(
            x, v, a_lp, m, f_actual_phys, f_nom_phys, task_id,
            w_balance=w_balance, c_target=c_target,
            use_cond_mod=use_cond_mod, cond_mod_from=cond_mod_from, w_beta_reg=w_beta_reg
        )

        # 批量均值偏置惩罚（把 resid 的均值往 0 拉）
        resid_batch = (f_pred + fn_phys_used) - f_actual_phys
        l_bias = (resid_batch.mean(dim=0) ** 2).sum()

        # uncertainty weighting（仅对基础项）
        if self.use_uncertainty:
            with torch.no_grad(): self.log_vars.clamp_(-5.0, 5.0)
            s = self.log_vars
            base_terms   = [l_mse, l_newton, l_resid, l_bal]
            base_weights = [1.0,   w_newton, w_resid,  w_balance]
            total = 0.0
            for i, (Li, wi) in enumerate(zip(base_terms, base_weights)):
                if wi == 0.0: continue
                s_idx = min(i, len(s)-1)
                term_value = 0.5 * torch.exp(-s[s_idx]) * Li + 0.5 * s[s_idx]
                total += wi * torch.clamp(term_value, min=0.0)
        else:
            total = l_mse + w_newton*l_newton + w_resid*l_resid + w_balance*l_bal

        total = total + (w_bias * l_bias) + (w_beta_reg * l_beta if use_cond_mod else 0.0)

        return total, {
            'mse': l_mse.detach(),
            'newton': l_newton.detach(),
            'residual': l_resid.detach(),
            'balance': l_bal.detach(),
            'beta_reg': l_beta.detach(),
            'bias': l_bias.detach(),
        }

# ======================= Dataset & loaders =======================

class PINNDataset(Dataset):
    def __init__(self, X, V, A_lp, F_phys, F_nom_phys, task_id:int=0, C=None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.V = torch.tensor(V, dtype=torch.float32)
        self.A_lp = torch.tensor(A_lp, dtype=torch.float32)
        self.F = torch.tensor(F_phys, dtype=torch.float32)
        self.F_nom = torch.tensor(F_nom_phys, dtype=torch.float32)
        self.task_id = task_id
        if C is None: C = np.zeros((len(X), 1), dtype=np.float32)
        self.C = torch.tensor(C, dtype=torch.float32)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx):
        return {'x': self.X[idx], 'v': self.V[idx], 'a_lp': self.A_lp[idx],
                'f': self.F[idx], 'fn': self.F_nom[idx], 'c': self.C[idx],
                'task_id': self.task_id}

def prepare_dataloader(X,V,A_lp,F_phys,F_nom_phys, task_id:int, C=None, batch_size=128, shuffle=True, seed: int = 42):
    g = torch.Generator()
    g.manual_seed(int(seed) + int(task_id))  # 每个任务不同但可复现
    return DataLoader(
        PINNDataset(X,V,A_lp,F_phys,F_nom_phys,task_id,C),
        batch_size=batch_size, shuffle=shuffle, drop_last=False,
        generator=g, num_workers=0, persistent_workers=False
    )

# ======================= K-shot adaptation（支持集增强 + 近端正则 + 可适配 β） =======================

@torch.no_grad()
def backup_task_embeddings(model: MetaPINN): return model.task_embeddings.detach().clone()

@torch.no_grad()
def restore_task_embeddings(model: MetaPINN, backup: torch.Tensor): model.task_embeddings.copy_(backup)

def _augment_support(x, repeats=1, noise_std=0.0):
    if repeats <= 1 and noise_std <= 0: return x
    xs = [x]
    for _ in range(repeats-1):
        if noise_std > 0:
            xs.append(x + noise_std * torch.randn_like(x))
        else:
            xs.append(x.clone())
    return torch.cat(xs, dim=0)

def _pick_support_indices(score_t: torch.Tensor, K: int, top_frac: float = 0.7):
    """Hybrid K-shot: 前 top_frac*K 用“信息量高”的样本（score 大），其余随机补齐。
       当 top_frac>=1.0 时，完全取确定性的前 K。"""
    N = score_t.numel()
    if N == 0: return torch.tensor([], dtype=torch.long, device=score_t.device)
    K = int(min(max(1, K), N))
    if top_frac >= 1.0:
        return torch.topk(score_t, k=K).indices
    topK = max(1, int(K * float(top_frac)))
    idx_top = torch.topk(score_t, k=min(topK, N)).indices
    # 其余随机补齐
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
                         steps=150, lr=5e-2, w_newton=0.02, w_resid=0.05, w_balance=0.0,
                         use_cond_mod=True, cond_mod_from='target', w_beta_reg=0.0008,
                         # robustification
                         adapt_repeats=4, adapt_noise_std=0.015, adapt_prox_weight=0.003,
                         # phys gain for high-wind tasks
                         adapt_phys_gain=1.0, wind_threshold=5.0,
                         # adapt-phase physics downscale
                         adapt_w_newton_scale=1.0, adapt_w_resid_scale=1.0,
                         # axis emphasis (e.g., boost z-axis Newton)
                         adapt_axis_scale=(1.0, 1.0, 1.0),
                         # scale beta reg during adapt (e.g., for freedom)
                         adapt_w_beta_reg_scale=1.0):
    """
    适应阶段：先做一次闭式 β 预热，然后联合微调 task embedding + β-logscale。
    """
    model.train()
    dev = model.task_embeddings.device
    x_k, v_k, a_lp_k, f_k, fn_k = x_k.to(dev), v_k.to(dev), a_lp_k.to(dev), f_k.to(dev), fn_k.to(dev)
    c_k = c_k.to(dev) if c_k is not None else None

    # ===== 1) 闭式 β 初始化（ridge / per-axis）=====
    try:
        with torch.no_grad():
            # 保存 & 清零本任务的 logscale，先拿到 base β
            model.task_beta_logscale[task_id].zero_()

            # base β（由 cond2beta 给出），取中位数稳定
            if (c_k is not None) and (model.cond2beta is not None):
                beta_base = model._beta_from_c(c_k, task_id=task_id)  # [K,3]
                beta_base_med = beta_base.median(dim=0).values        # [3]
            else:
                beta_base_med = torch.ones(3, device=x_k.device)

            # 目标校正量 y = f_actual - f_pred（不含 β）
            f_pred0 = model._pred_physical(x_k, task_id=task_id, c_in=c_k)
            y = f_k - f_pred0                       # [K,3]
            X = fn_k                                 # [K,3]
            lam = 1e-3
            beta_ls = []
            for ax in range(3):
                xax = X[:, ax:ax+1]; yax = y[:, ax:ax+1]
                num = (xax * yax).sum()
                den = (xax * xax).sum() + lam
                b = (num / (den + 1e-8)).clamp(min=model.beta_min, max=model.beta_max)
                beta_ls.append(b)
            beta_ls = torch.stack(beta_ls)  # [3]

            # 需要的 logscale：beta_base_med * exp(logscale) ≈ beta_ls
            logscale = torch.log((beta_ls + 1e-6) / (beta_base_med + 1e-6))
            model.task_beta_logscale[task_id].copy_(logscale.clamp(-1.5, 1.5))
    except Exception:
        pass

    # ===== 2) 近端 anchor（限制嵌入漂移）=====
    with torch.no_grad():
        e_anchor = model.task_embeddings[task_id].detach().clone()

    # 只更新 embedding & β-logscale（对其他任务做 mask）
    opt = torch.optim.SGD([model.task_embeddings, model.task_beta_logscale], lr=lr)

    # 物理项缩放
    phys_scale = 1.0
    if c_k is not None:
        wind_mag = c_k.abs().mean().item()
        if wind_mag >= wind_threshold:
            phys_scale = 1.0 + adapt_phys_gain  # e.g., 2.0 when gain=1.0
    wN = w_newton * phys_scale * adapt_w_newton_scale
    wR = w_resid  * phys_scale * adapt_w_resid_scale
    wB = w_balance
    wBeta = w_beta_reg * float(adapt_w_beta_reg_scale)

    axis_scale_t = torch.tensor(adapt_axis_scale, dtype=torch.float32, device=dev).view(1,3)

    # ===== 3) 微调循环 =====
    for _ in range(int(steps)):
        opt.zero_grad()
        # support augmentation (on normalized x)
        x_aug = _augment_support(x_k, repeats=adapt_repeats, noise_std=adapt_noise_std)
        rep = x_aug.shape[0] // x_k.shape[0]
        def tile(t): return t.repeat((rep,1)) if (t is not None and t.ndim==2) else t
        v_aug = tile(v_k); a_aug = tile(a_lp_k); f_aug = tile(f_k); fn_aug = tile(fn_k); c_aug = tile(c_k)

        # Newton 轴向强调（例如 z 轴 ×1.3）
        if a_aug is not None:
            a_aug = a_aug * axis_scale_t

        total,_ = model.total_loss(
            x_aug, v_aug, a_aug, mass, f_aug, fn_aug, task_id,
            w_newton=wN, w_resid=wR, w_balance=wB,
            c_target=c_aug,
            use_cond_mod=use_cond_mod, cond_mod_from=cond_mod_from, w_beta_reg=wBeta
        )

        # proximal penalty on the task embedding（限制 embedding 过度漂移）
        e_cur = model.task_embeddings[task_id]
        l_prox = adapt_prox_weight * torch.sum((e_cur - e_anchor)**2)
        (total + l_prox).backward()

        # 只更新当前任务的 embedding 与 β 偏置
        with torch.no_grad():
            if model.task_embeddings.grad is not None:
                mask = torch.zeros_like(model.task_embeddings); mask[task_id]=1.0
                model.task_embeddings.grad *= mask
            if model.task_beta_logscale.grad is not None:
                maskb = torch.zeros_like(model.task_beta_logscale); maskb[task_id] = 1.0
                model.task_beta_logscale.grad *= maskb
        opt.step()

# ======================= Data conversion (build a_lp + f_nom) =======================

def convert_dataset_to_numpy(data, options):
    """
    Returns: X, V, A (raw), A_lp (smoothed for Newton), F_phys, F_nom_phys
    Assumes data.X stacks features (v, q, pwm...), and data.Y is fa (pure aerodynamics residual).
    """
    import utils
    X = data.X.astype(np.float32)
    V = data.X[:, :3].astype(np.float32)  # assume first 3 are v (body frame)
    # true dt from timestamps
    t = np.asarray(data.meta['t'], dtype=np.float64)
    dt = np.diff(t, prepend=t[0])
    if not np.any(dt>0): dt[:] = 0.05
    else: dt[dt<=0] = np.median(dt[dt>0])
    A = np.zeros_like(V); A[1:] = (V[1:] - V[:-1]) / dt[1:,None]; A[0] = A[1]

    # low-freq acceleration for Newton
    A_lp = smooth_acc_torch(torch.from_numpy(A), window=options.get('sg_window',15), poly=options.get('sg_poly',3)).cpu().numpy()

    F_phys = data.Y.astype(np.float32)

    # drag box from options (torch or np)
    drag_box_opt = options.get('drag_box', None)
    drag_box_np  = drag_box_opt.detach().cpu().numpy() if isinstance(drag_box_opt, torch.Tensor) else drag_box_opt

    # hover_throttle if present
    hover_thr = data.meta.get('hover_throttle', None)
    beta_opt = options.get('beta_drag_vec', options.get('beta_drag', 1.0))

    F_nom = compute_nominal_force(
        X, V, utils.feature_len,
        rho=options.get('air_density',1.225),
        C_T=options.get('UAV_rotor_C_T',0.109919),
        D=options.get('UAV_propeller_diameter',0.2286),
        rpm_max=options.get('UAV_rotor_max_rpm',6396.667),
        mass=options.get('UAV_mass',1.0),
        hover_throttle=hover_thr,
        hover_pwm_norm=options.get('hover_pwm_norm',0.5),
        use_hover_calib=True,
        drag_box_np=drag_box_np,
        beta_drag=beta_opt,
    )

    return X, V, A, A_lp, F_phys, F_nom

# ======================= Train / Evaluate =======================

def _linear_warm(warmup_progress: float, start: float, end: float) -> float:
    """Linearly interpolate warm factor between start and end with progress in [0,1]."""
    return float(start + (end - start) * np.clip(warmup_progress, 0.0, 1.0))

def _save_checkpoint(model, path):
    torch.save(model.state_dict(), path)


def train_meta_pinn_multitask(model: MetaPINN, Data, TestData, mass, options, save_path=None):
    import utils
    num_tasks, num_test = len(Data), len(TestData)

    # Fit scalers on train X, F, and condition C
    X_all, F_all, Ns, C_all_list = [], [], [], []
    cond_dim = options.get('cond_dim', 1)
    for i in range(num_tasks):
        Xi,Vi,Ai,Alpi,Fi,Fni = convert_dataset_to_numpy(Data[i], options)
        X_all.append(Xi); F_all.append(Fi); Ns.append(len(Xi))
        C_all_list.append(build_condition_vector_for_dataset(Data[i], cond_dim))
    X_all = np.vstack(X_all); F_all = np.vstack(F_all)
    x_scaler = StandardScaler().fit(X_all); f_scaler = StandardScaler().fit(F_all)
    C_all = np.vstack(C_all_list); c_scaler = StandardScaler().fit(C_all)

    dev = next(model.parameters()).device
    opt = torch.optim.AdamW(model.parameters(), lr=options['learning_rate'], weight_decay=1e-4)

    # force & condition scaler buffers
    model.set_force_scaler(
        torch.tensor(f_scaler.mean, dtype=torch.float32, device=dev),
        torch.tensor(f_scaler.std,  dtype=torch.float32, device=dev)
    )
    model.set_condition_scaler(
        torch.tensor(c_scaler.mean, dtype=torch.float32, device=dev),
        torch.tensor(c_scaler.std,  dtype=torch.float32, device=dev)
    )

    # scheduler
    scheduler = None
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

    hist = {'train_total':[], 'train_mse':[], 'train_newton':[], 'train_residual':[], 'train_balance':[], 'train_beta_reg':[], 'test_avg_mse':[]}

    # dynamic beta reg schedule fallback
    beta_reg_schedule = options.get('beta_reg_schedule', None)
    if not callable(beta_reg_schedule):
        # default: keep constant
        beta_reg_schedule = lambda epoch: options.get('w_beta_reg', 0.0008)

    warmup_epochs = options.get('warmup_epochs', 40)
    warmup_start  = options.get('warmup_start', 0.03)
    warmup_end    = options.get('warmup_end', 1.0)

    use_cond_mod    = options.get('use_cond_mod', True)
    cond_mod_from   = options.get('cond_mod_from', 'target')

    best_mse = float('inf')
    ckpt_last = os.path.join(save_path, "meta_pinn_last.pth") if save_path else None
    ckpt_best = os.path.join(save_path, "meta_pinn_best.pth") if save_path else None

    for epoch in range(options['num_epochs']):
        # ===== epoch-wise updates =====
        options['w_beta_reg'] = float(beta_reg_schedule(epoch))
        w_beta_reg = options['w_beta_reg']

        # warm-up factor (0..1), then mapped to [start,end]
        warm_progress = (epoch + 1) / max(1, warmup_epochs)
        warm = _linear_warm(warm_progress, warmup_start, warmup_end)

        w_newton = warm * options.get('w_newton',0.01)
        w_resid  = warm * options.get('w_resid',0.02)
        w_bal    = warm * options.get('w_balance',0.0)
        # 保证 residual 能爬到 0.05 * warm
        resid_target = 0.1
        w_resid = max(w_resid, warm * resid_target)

        model.train()
        total_sum = mse_sum = newton_sum = resid_sum = bal_sum = beta_sum = 0.0; n_batches=0

        for tid in range(num_tasks):
            X,V,A,Alp,F,Fnom = convert_dataset_to_numpy(Data[tid], options)
            Xn = x_scaler.transform(X)
            C  = build_condition_vector_for_dataset(Data[tid], model.cond_dim)
            loader = prepare_dataloader(
                Xn,V,Alp,F,Fnom, task_id=tid, C=C,
                batch_size=options.get('batch_size',128), shuffle=True,
                seed=options.get('seed', 42)
            )
            for batch in loader:
                x = batch['x'].to(dev); v = batch['v'].to(dev); a_lp = batch['a_lp'].to(dev)
                f = batch['f'].to(dev); fn = batch['fn'].to(dev); c_tgt = batch['c'].to(dev)
                ttid = batch['task_id']

                opt.zero_grad()
                total, comps = model.total_loss(
                    x, v, a_lp, mass, f, fn, ttid,
                    w_newton=w_newton, w_resid=w_resid, w_balance=w_bal,
                    c_target=c_tgt,
                    use_cond_mod=use_cond_mod, cond_mod_from=cond_mod_from, w_beta_reg=w_beta_reg
                )
                total.backward(); opt.step()
                if scheduler is not None and isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR): scheduler.step()
                total_sum += total.item(); mse_sum += comps['mse'].item(); newton_sum += comps['newton'].item()
                resid_sum += comps['residual'].item(); bal_sum += comps['balance'].item(); beta_sum += comps['beta_reg'].item(); n_batches+=1

        if scheduler is not None and isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR): scheduler.step()

        hist['train_total'].append(total_sum/max(1,n_batches))
        hist['train_mse'].append(mse_sum/max(1,n_batches))
        hist['train_newton'].append(newton_sum/max(1,n_batches))
        hist['train_residual'].append(resid_sum/max(1,n_batches))
        hist['train_balance'].append(bal_sum/max(1,n_batches))
        hist['train_beta_reg'].append(beta_sum/max(1,n_batches))

        # ===== evaluation with per-task adaptation（闭式 β 预热 + 信息量更高的 K-shot + ALL-task aggressive hypers） =====
        model.eval(); emb_bak = backup_task_embeddings(model)
        beta_bak = model.task_beta_logscale.detach().clone()
        avg_test_mse = 0.0
        for j in range(num_test):
            Xv,Vv,Av,Alpv,Fv,Fnomv = convert_dataset_to_numpy(TestData[j], options)
            Xv_n = x_scaler.transform(Xv)
            Cv   = build_condition_vector_for_dataset(TestData[j], model.cond_dim)
            x_all  = torch.tensor(Xv_n, dtype=torch.float32, device=dev)
            v_all  = torch.tensor(Vv,   dtype=torch.float32, device=dev)
            al_all = torch.tensor(Alpv, dtype=torch.float32, device=dev)
            f_all  = torch.tensor(Fv,   dtype=torch.float32, device=dev)
            fn_all = torch.tensor(Fnomv,dtype=torch.float32, device=dev)
            c_all  = torch.tensor(Cv,   dtype=torch.float32, device=dev)

            # ===== before MSE =====
            K_task = options.get('K_shot_task0', options.get('K_shot', 300))  # 统一用“task0”配置

            # 选择评分方式：fn 强度 或 速度范数
            score_mode = options.get('kshot_score', 'fn')  # 'fn' | 'vnorm'
            if score_mode == 'vnorm':
                score_vec = torch.linalg.norm(v_all, dim=-1)
            else:
                score_vec = fn_all.abs().sum(-1)

            # 评估阶段：完全确定的 top-K
            top_frac_eval = options.get('eval_top_frac', 1.0)
            idxK  = _pick_support_indices(score_vec, K=K_task, top_frac=top_frac_eval)

            # 支持/查询划分：K 用挑选的，query 用其余
            mask_query = torch.ones(x_all.shape[0], dtype=torch.bool, device=dev)
            mask_query[idxK] = False
            xK, vK, aK, fK, fnK, cK = x_all[idxK], v_all[idxK], al_all[idxK], f_all[idxK], fn_all[idxK], c_all[idxK]
            xQ, fQ, cQ = x_all[mask_query], f_all[mask_query], c_all[mask_query]

            with torch.no_grad():
                f_pred = model._pred_physical(xQ, task_id=j, c_in=cQ)
                mse_before = Fnn.mse_loss(f_pred, fQ).item()

            # per-task adapt LR（对所有任务使用 Task0 的更激进上限/倍率）
            adapt_lr_map = options.get('adapt_lr_task', {})
            adapt_lr_used = adapt_lr_map.get(j, adapt_lr_map.get(0, options.get('adapt_lr', 5e-2)))
            mul   = options.get('adapt_lr_task0_mul', 1.5)
            lrmax = options.get('adapt_lr_task0_max', 0.12)
            adapt_lr_used = min(adapt_lr_used * mul, lrmax)

            # 适应步数 / 近端 / β 正则缩放 / 轴向强调（统一使用 Task0 设定）
            adapt_steps_used = int(options.get('adapt_steps', 150) * options.get('adapt_steps_mul_task0', 2.0))
            prox_w = options.get('adapt_prox_weight', 0.003) * options.get('adapt_prox_weight_task0_scale', 0.5)
            w_beta_reg_used = options.get('w_beta_reg', 0.0008) * options.get('adapt_w_beta_reg_scale_task0', 0.3)
            axis_scale = (1.0, 1.0, options.get('adapt_z_gain_task0', 1.3))

            # 评估阶段：不加噪声、不重复，确保可复现
            adapt_repeats_eval = options.get('eval_adapt_repeats', 1)
            adapt_noise_std_eval = options.get('eval_adapt_noise_std', 0.0)

            adapt_task_embedding(
                model, xK, vK, aK, fK, fnK, cK, mass, task_id=j,
                steps=adapt_steps_used,
                lr=adapt_lr_used,
                w_newton=w_newton, w_resid=w_resid, w_balance=w_bal,
                use_cond_mod=use_cond_mod, cond_mod_from=cond_mod_from, w_beta_reg=w_beta_reg_used,
                adapt_repeats=adapt_repeats_eval,
                adapt_noise_std=adapt_noise_std_eval,
                adapt_prox_weight=prox_w,
                adapt_phys_gain=options.get('adapt_phys_gain', 1.0),
                wind_threshold=options.get('adapt_wind_threshold', 5.0),
                adapt_w_newton_scale=options.get('adapt_w_newton_scale', 0.5),
                adapt_w_resid_scale=options.get('adapt_w_resid_scale', 0.5),
                adapt_axis_scale=axis_scale,
                adapt_w_beta_reg_scale=1.0  # 已直接体现在 w_beta_reg_used
            )

            with torch.no_grad():
                f_pred = model._pred_physical(xQ, task_id=j, c_in=cQ)
                mse_after = Fnn.mse_loss(f_pred, fQ).item()
                avg_test_mse += mse_after

            if options.get('print_task_mse', True):
                print(f"  Task {j}: K-shot before MSE={mse_before:.4f} -> after={mse_after:.4f}")

            restore_task_embeddings(model, emb_bak)
            with torch.no_grad():
                model.task_beta_logscale.copy_(beta_bak)

        avg_test_mse /= max(1,num_test)
        hist['test_avg_mse'].append(avg_test_mse)
        print(f"[Epoch {epoch}] warmup={warm:.2f} | Train total: {hist['train_total'][-1]:.4f} | "
              f"MSE {hist['train_mse'][-1]:.4f} Newton {hist['train_newton'][-1]:.4f} "
              f"Resid {hist['train_residual'][-1]:.4f} | BetaReg {hist['train_beta_reg'][-1]:.4f} || Test MSE {avg_test_mse:.4f}")

        # ----- save checkpoints -----
        if save_path:
            _save_checkpoint(model, ckpt_last)
            if avg_test_mse < best_mse:
                best_mse = avg_test_mse
                _save_checkpoint(model, ckpt_best)

    return hist

# ======================= Plot helpers =======================

def plot_training_curves(hist):
    plt.figure(figsize=(12,4))
    plt.subplot(1,3,1); plt.plot(hist['train_total'],label='total'); plt.plot(hist['test_avg_mse'],label='test MSE'); plt.legend(); plt.title('Total & Test'); plt.xlabel('epoch')
    plt.subplot(1,3,2); plt.plot(hist['train_mse'],label='data MSE'); plt.plot(hist['train_newton'],label='Newton (LF)'); plt.plot(hist['train_residual'],label='Residual (robust)')
    if 'train_balance' in hist and max(hist['train_balance'])>0: plt.plot(hist['train_balance'],label='Balance')
    if 'train_beta_reg' in hist and max(hist['train_beta_reg'])>0: plt.plot(hist['train_beta_reg'],label='Beta Reg')
    plt.legend(); plt.title('Loss components'); plt.xlabel('epoch')
    plt.subplot(1,3,3); plt.axis('off'); plt.text(0.02,0.6,'Scheme A + CNM + per-task β-bias\nClosed-form β warm-start + informative K-shot\nAggressive adapt hypers for ALL tasks\nDeterministic eval + checkpoints',fontsize=9)
    plt.tight_layout(); plt.show()

# ======================= Main =======================

if __name__ == '__main__':
    import utils
    from physics_params import get_uav_physics

    # ---- seed & determinism ----
    SEED = 42
    set_seed(SEED, deterministic=True)
    print(f"SEED={SEED}, deterministic=True")

    phys = get_uav_physics(device=device)
    drag_box = phys["drag_box"]
    inertia_diag = phys["inertia_diag"]

    # ---- options (edit here) ----
    options = {
        'seed': SEED,
        'deterministic': True,

        'learning_rate': 1e-2,
        'num_epochs': 20,
        'batch_size': 512,

        # === K-shot ===
        'K_shot': 2500,
        'K_shot_task0': 2500,              # 现在对所有任务生效
        'kshot_score': 'fn',               # 'fn' | 'vnorm'
        'kshot_top_frac': 0.6,
        'kshot_top_frac_task0': 0.9,       # train时仍可混入随机补齐
        'eval_top_frac': 1.0,              # eval：纯 top-K，完全可复现

        # === Adapt 阶段参数 ===
        'adapt_steps': 250,
        'adapt_steps_mul_task0': 4.0,      # 现在对所有任务生效
        # 分任务 adapt 学习率（再统一乘以 task0 的倍数与上限）
        'adapt_lr_task': {0: 20e-1, 1: 20e-1, 2: 20e-1, 3: 20e-1},
        'adapt_lr_task0_mul': 1.5,         # 现在对所有任务生效
        'adapt_lr_task0_max': 0.3,         # 现在对所有任务生效

        'task_dim': 128,

        # === 物理 loss 权重 + warmup（线性从 start→end） ===
        'w_newton': 0.01,
        'w_resid':  0.02,
        'w_balance': 0.0,
        'w_bias': 0.005,
        'warmup_epochs': 20,
        'warmup_start': 0.03,
        'warmup_end': 0.80,
        'patience': 60, 'min_delta': 0.0,
        'print_task_mse': True,

        # === scheduler ===
        'scheduler': 'onecycle',
        'max_lr': 1e-4,

        # === 特征和数据平滑 ===
        'features': ['v','q','pwm'],
        'sg_window': 10, 'sg_poly': 3,
        'hover_pwm_norm': 0.5,

        # === UAV params (pure aerodynamics; NO gravity here) ===
        'UAV_mass': 1.0,
        'UAV_rotor_C_T': 0.109919,
        'UAV_rotor_C_P': 0.040164,
        'air_density': 1.225,
        'UAV_rotor_max_rpm': 6396.667,
        'UAV_propeller_diameter': 0.2286,

        # === drag box (torch tensor OK) ===
        'drag_box' : drag_box,

        # === 阻力缩放（epoch 0 会自动回归，初始给 1.0） ===
        'beta_drag': 1.0,

        # ====== condition / CNM ======
        'cond_dim': 1,
        'use_cond_mod': True,
        'cond_mod_from': 'target',
        'beta_min': 0.15, 'beta_max': 8.0,

        # β 正则的初值（会被 schedule 改写）
        'w_beta_reg': 0.0008,

        # 动态 β 正则（每个 epoch 调用）；前 50 轮保持，之后指数衰减
        'beta_reg_schedule': (lambda epoch: (0.0015 if epoch < 50 else 0.0015 * (0.85 ** (epoch - 50)))),

        # === Adapt 阶段物理损失缩放（降低物理主导，利于适应）===
        'adapt_w_newton_scale': 0.5,
        'adapt_w_resid_scale' : 0.5,

        # === 适配稳健化（统一使用 Task0 设定） ===
        'adapt_repeats':10,
        'adapt_noise_std': 0.015,
        'adapt_prox_weight': 0.003,
        'adapt_prox_weight_task0_scale': 0.5,   # 对所有任务生效
        'adapt_w_beta_reg_scale_task0': 0.3,    # 对所有任务生效
        'adapt_z_gain_task0': 1.5,              # 对所有任务生效
        'adapt_phys_gain': 1.0,
        'adapt_wind_threshold': 5.0,

        # Eval 去随机
        'eval_adapt_repeats': 1,
        'eval_adapt_noise_std': 0.0,

        # FiLM 关闭
        'use_film': False,
    }

    # load data
    raw_train = utils.load_data('data/training')
    raw_test  = utils.load_data('data/experiment', expnames='(baseline_)([0-9]*|no)wind')
    Data      = utils.format_data(raw_train, features=options['features'], output='fa')
    TestData  = utils.format_data(raw_test,  features=options['features'], output='fa')

    # pretty print mapping
    for i,d in enumerate(Data):     print(f"[Train Task {i}] {d.meta.get('method','')}_{d.meta.get('condition','')} <- {d.meta.get('filename','')}")
    for i,d in enumerate(TestData): print(f"[Test  Task {i}] {d.meta.get('method','')}_{d.meta.get('condition','')} <- {d.meta.get('filename','')}")

    num_tasks = len(Data); input_dim = Data[0].X.shape[1]
    model = MetaPINN(input_dim=input_dim, num_tasks=num_tasks,
                     task_dim=options['task_dim'], hidden_dim=384, use_uncertainty=True,
                     cond_dim=options.get('cond_dim',1),
                     use_cond_mod=options.get('use_cond_mod', True),
                     cond_mod_from=options.get('cond_mod_from', 'target'),
                     beta_min=options.get('beta_min',0.15), beta_max=options.get('beta_max',8.0),
                     use_film=options.get('use_film', False),
                     film_gamma_min=options.get('film_gamma_min',0.6) if 'film_gamma_min' in options else 0.6,
                     film_gamma_max=options.get('film_gamma_max',1.6) if 'film_gamma_max' in options else 1.6,
                     film_delta_max=options.get('film_delta_max',0.2) if 'film_delta_max' in options else 0.2)
    model.to(device)

    save_dir = 'saved_models/meta_pinn'
    os.makedirs(save_dir, exist_ok=True)

    # ---------- epoch 0: 分轴线性回归 beta_drag_vec（仅做一次，在 train 内也有保护，这里提前做一次更直观） ----------
    # 这里不保存，只打印，实际以 train 内第一次 epoch 的校准为准
    try:
        Xi,Vi,Ai,Alpi,Fi,Fni = convert_dataset_to_numpy(Data[0], options)
        vmag = np.linalg.norm(Vi, axis=1)
        sel = (vmag < np.percentile(vmag, 70))
        if np.any(sel) and isinstance(options['drag_box'], torch.Tensor):
            drag_box_np = options['drag_box'].detach().cpu().numpy()
            c = (options.get('air_density', 1.225) * drag_box_np).astype(np.float32)
            Z = Vi[sel] * np.abs(Vi[sel])
            beta_vec = []
            lam = 1e-3
            for k in range(3):
                Xk = -(Z[:, k:k+1] * c[k])
                yk = Fi[sel, k:k+1]
                XtX = (Xk.T @ Xk) + lam*np.eye(1)
                Xty = Xk.T @ yk
                beta_k = float(np.linalg.solve(XtX, Xty))
                beta_vec.append(beta_k)
            beta_vec = np.clip(np.array(beta_vec, dtype=np.float32), options.get('beta_min',0.15), options.get('beta_max',8.0))
            print(f"  >>> calibrated beta_drag_vec = {np.round(beta_vec,3)}")
            options['beta_drag_vec'] = beta_vec.tolist()
            options['calibrated_beta'] = True
    except Exception:
        pass

    hist = train_meta_pinn_multitask(model, Data, TestData, options['UAV_mass'], options, save_path=save_dir)
    # save last
    torch.save(model.state_dict(),os.path.join(save_dir, "meta_pinn_last.pth"))

    # 也把训练曲线存起来，方便复现实验
    np.save(os.path.join(save_dir, "history_train_total.npy"), np.array(hist['train_total']))
    np.save(os.path.join(save_dir, "history_test_mse.npy"),   np.array(hist['test_avg_mse']))

    plot_training_curves(hist)
