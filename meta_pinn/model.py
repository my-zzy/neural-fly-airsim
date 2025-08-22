
import torch, torch.nn as nn, torch.nn.functional as Fnn

class MetaPINN(nn.Module):
    def __init__(self, input_dim, num_tasks, task_dim=16, hidden_dim=128, use_uncertainty=True,
                 cond_dim=1, use_cond_mod=True, cond_mod_from='target', beta_min=0.15, beta_max=8.0):
        super().__init__()
        self.num_tasks = num_tasks
        self.use_uncertainty = use_uncertainty

        self.task_embeddings = nn.Parameter(torch.randn(num_tasks, task_dim) * 0.1)
        self.fc1 = nn.Linear(input_dim + task_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 3)
        self.act = nn.Tanh()

        if use_uncertainty:
            self.log_vars = nn.Parameter(torch.zeros(4))

        self.register_buffer('f_mean_t', torch.zeros(1,3))
        self.register_buffer('f_std_t',  torch.ones(1,3))

        self.cond_dim = cond_dim
        self.register_buffer('c_mean_t', torch.zeros(1,cond_dim))
        self.register_buffer('c_std_t',  torch.ones(1,cond_dim))

        self.use_cond_mod = bool(use_cond_mod)
        self.cond_mod_from = str(cond_mod_from)
        self.beta_min = float(beta_min); self.beta_max = float(beta_max)
        if self.use_cond_mod:
            self.cond2beta = nn.Sequential(
                nn.Linear(cond_dim, max(8, cond_dim*2)),
                nn.Tanh(),
                nn.Linear(max(8, cond_dim*2), 3)
            )
        else:
            self.register_module('cond2beta', None)

        self.task_beta_logscale = nn.Parameter(torch.zeros(num_tasks, 3))

    @torch.no_grad()
    def set_force_scaler(self, mean_t: torch.Tensor, std_t: torch.Tensor):
        self.f_mean_t.copy_(mean_t); self.f_std_t.copy_(std_t)
    @torch.no_grad()
    def set_condition_scaler(self, mean_t: torch.Tensor, std_t: torch.Tensor):
        self.c_mean_t.copy_(mean_t); self.c_std_t.copy_(std_t)

    @staticmethod
    def _sigmoid_range(x, lo, hi): return lo + (hi - lo) * torch.sigmoid(x)

    def _beta_from_c(self, c_phys: torch.Tensor, task_id: int = None) -> torch.Tensor:
        c_n = (c_phys - self.c_mean_t) / (self.c_std_t + 1e-8)
        raw = self.cond2beta(c_n)
        beta = self._sigmoid_range(raw, self.beta_min, self.beta_max)
        if (task_id is not None):
            beta = beta * torch.exp(self.task_beta_logscale[task_id].unsqueeze(0))
            beta = torch.clamp(beta, self.beta_min, self.beta_max)
        return beta
    def _pred_physical(self, x, task_id: int, c_in: torch.Tensor = None):
        B = x.shape[0]
        e = self.task_embeddings[task_id].expand(B, -1)
        h = torch.cat([x, e], dim=-1)
        h = self.act(self.fc1(h))
        h = self.act(self.fc2(h))
        f_n = self.fc3(h)
        return f_n * self.f_std_t + self.f_mean_t

    def compute_components(self, x, v, a_lp, m, f_actual_phys, f_nom_phys, task_id: int,
                           c_target=None,
                           use_cond_mod=False, cond_mod_from='target', w_beta_reg=0.01):
        c_in = None
        if (cond_mod_from == 'target') and (c_target is not None):
            c_in = c_target

        beta_reg = x.new_tensor(0.0)
        fn_phys = f_nom_phys
        if use_cond_mod and (self.cond2beta is not None) and (c_in is not None):
            beta = self._beta_from_c(c_in, task_id=task_id)
            fn_phys = beta * f_nom_phys
            beta_reg = ((beta - 1.0)**2).mean() + 0.01 * (self.task_beta_logscale[task_id]**2).mean()

        f_pred = self._pred_physical(x, task_id, c_in=c_in)
        l_mse = Fnn.mse_loss(f_pred, f_actual_phys)

        resid = (f_pred + fn_phys) - f_actual_phys
        scale = resid.detach().abs().median().clamp(min=1.0)
        l_resid = Fnn.mse_loss(resid / scale, torch.zeros_like(resid))

        l_newton = Fnn.mse_loss(f_pred, m * a_lp)

        return f_pred, l_mse, l_newton, l_resid, beta_reg, fn_phys

    def total_loss(self, x, v, a_lp, m, f_actual_phys, f_nom_phys, task_id: int,
                   w_newton=0.01, w_resid=0.02, w_bias=0.01,
                   c_target=None, use_cond_mod=False, cond_mod_from='target', w_beta_reg=0.01):

        f_pred, l_mse, l_newton, l_resid, l_beta, fn_phys_used = self.compute_components(
            x, v, a_lp, m, f_actual_phys, f_nom_phys, task_id,  c_target=c_target,
            use_cond_mod=use_cond_mod, cond_mod_from=cond_mod_from, w_beta_reg=w_beta_reg
        )

        resid_batch = (f_pred + fn_phys_used) - f_actual_phys
        l_bias = (resid_batch.mean(dim=0) ** 2).sum()

        if self.use_uncertainty:
            with torch.no_grad(): self.log_vars.clamp_(-5.0, 5.0)
            s = self.log_vars
            base_terms   = [l_mse, l_newton, l_resid]
            base_weights = [1.0,   w_newton, w_resid]
            total = 0.0
            for i, (Li, wi) in enumerate(zip(base_terms, base_weights)):
                if wi == 0.0: continue
                s_idx = min(i, len(s)-1)
                term_value = 0.5 * torch.exp(-s[s_idx]) * Li + 0.5 * s[s_idx]
                total += wi * torch.clamp(term_value, min=0.0)
        else:
            total = l_mse + w_newton*l_newton + w_resid*l_resid

        total = total + (w_bias * l_bias) + (w_beta_reg * l_beta if use_cond_mod else 0.0)

        return total, {
            'mse': l_mse.detach(),
            'newton': l_newton.detach(),
            'residual': l_resid.detach(),
            'beta_reg': l_beta.detach(),
            'bias': l_bias.detach(),
        }