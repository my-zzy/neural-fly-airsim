
import torch, torch.nn as nn, torch.nn.functional as Fnn

@torch.no_grad()
def backup_task_embeddings(model): return model.task_embeddings.detach().clone()
@torch.no_grad()
def restore_task_embeddings(model, backup): model.task_embeddings.copy_(backup)

def _augment_support(x, repeats=1, noise_std=0.0):
    if repeats <= 1 and noise_std <= 0: return x
    xs = [x]
    for _ in range(repeats-1):
        xs.append(x + noise_std * torch.randn_like(x) if noise_std > 0 else x.clone())
    return torch.cat(xs, dim=0)

def pick_support_indices(score_t: torch.Tensor, K: int, top_frac: float = 1.0):
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

def adapt_task_embedding(model, x_k, v_k, a_lp_k, f_k, fn_k, c_k, mass, task_id:int,
                         steps=150, lr=5e-2, w_newton=0.02, w_resid=0.05,
                         use_cond_mod=True, cond_mod_from='target', w_beta_reg=0.0008,
                         adapt_repeats=1, adapt_noise_std=0.0, adapt_prox_weight=0.003,
                         adapt_phys_gain=1.0, wind_threshold=5.0,
                         adapt_w_newton_scale=1.0, adapt_w_resid_scale=1.0,
                         adapt_axis_scale=(1.0, 1.0, 1.0),
                         adapt_w_beta_reg_scale=1.0):
    model.train()
    dev = model.task_embeddings.device
    x_k, v_k, a_lp_k, f_k, fn_k = x_k.to(dev), v_k.to(dev), a_lp_k.to(dev), f_k.to(dev), fn_k.to(dev)
    c_k = c_k.to(dev) if c_k is not None else None

    try:
        with torch.no_grad():
            model.task_beta_logscale[task_id].zero_()
            if (c_k is not None) and (model.cond2beta is not None):
                beta_base = model._beta_from_c(c_k, task_id=task_id)
                beta_base_med = beta_base.median(dim=0).values
            else:
                beta_base_med = torch.ones(3, device=x_k.device)

            f_pred0 = model._pred_physical(x_k, task_id=task_id, c_in=c_k)
            y = f_k - f_pred0
            X = fn_k
            lam = 1e-3
            beta_ls = []
            for ax in range(3):
                xax = X[:, ax:ax+1]; yax = y[:, ax:ax+1]
                num = (xax * yax).sum()
                den = (xax * xax).sum() + lam
                b = (num / (den + 1e-8)).clamp(min=model.beta_min, max=model.beta_max)
                beta_ls.append(b)
            beta_ls = torch.stack(beta_ls)
            logscale = torch.log((beta_ls + 1e-6) / (beta_base_med + 1e-6))
            model.task_beta_logscale[task_id].copy_(logscale.clamp(-1.5, 1.5))
    except Exception:
        pass

    with torch.no_grad():
        e_anchor = model.task_embeddings[task_id].detach().clone()

    opt = torch.optim.SGD([model.task_embeddings, model.task_beta_logscale], lr=lr)

    phys_scale = 1.0
    if c_k is not None:
        wind_mag = c_k.abs().mean().item()
        if wind_mag >= wind_threshold:
            phys_scale = 1.0 + adapt_phys_gain
    wN = w_newton * phys_scale * adapt_w_newton_scale
    wR = w_resid  * phys_scale * adapt_w_resid_scale
    
    wBeta = w_beta_reg * float(adapt_w_beta_reg_scale)

    axis_scale_t = torch.tensor(adapt_axis_scale, dtype=torch.float32, device=dev).view(1,3)

    for _ in range(int(steps)):
        opt.zero_grad()
        x_aug = _augment_support(x_k, repeats=adapt_repeats, noise_std=adapt_noise_std)
        rep = x_aug.shape[0] // x_k.shape[0]
        def tile(t): return t.repeat((rep,1)) if (t is not None and t.ndim==2) else t
        v_aug = tile(v_k); a_aug = tile(a_lp_k); f_aug = tile(f_k); fn_aug = tile(fn_k); c_aug = tile(c_k)

        if a_aug is not None: a_aug = a_aug * axis_scale_t

        total,_ = model.total_loss(
            x_aug, v_aug, a_aug, mass, f_aug, fn_aug, task_id,
            w_newton=wN, w_resid=wR, 
            c_target=c_aug,
            use_cond_mod=use_cond_mod, cond_mod_from=cond_mod_from, w_beta_reg=wBeta
        )

        e_cur = model.task_embeddings[task_id]
        l_prox = adapt_prox_weight * torch.sum((e_cur - e_anchor)**2)
        (total + l_prox).backward()

        with torch.no_grad():
            if model.task_embeddings.grad is not None:
                mask = torch.zeros_like(model.task_embeddings); mask[task_id]=1.0
                model.task_embeddings.grad *= mask
            if model.task_beta_logscale.grad is not None:
                maskb = torch.zeros_like(model.task_beta_logscale); maskb[task_id] = 1.0
                model.task_beta_logscale.grad *= maskb
        opt.step()
