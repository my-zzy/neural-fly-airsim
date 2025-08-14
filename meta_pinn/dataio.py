
import numpy as np, torch
from torch.utils.data import Dataset, DataLoader
from .utils import smooth_acc_torch

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
    g.manual_seed(int(seed) + int(task_id))
    return DataLoader(
        PINNDataset(X,V,A_lp,F_phys,F_nom_phys,task_id,C),
        batch_size=batch_size, shuffle=shuffle, drop_last=False,
        generator=g, num_workers=0, persistent_workers=False
    )

def convert_dataset_to_numpy(data, options):
    """
    Returns: X, V, A (raw), A_lp (smoothed), F_phys, F_nom_phys
    Assumes data.X stacks features (v, q, pwm...), and data.Y is fa (pure aerodynamics residual).
    """
    from .extutils import feature_len  # user's existing utils (feature_len, load/format)
    from .physics import compute_nominal_force
    X = data.X.astype(np.float32)
    V = data.X[:, :3].astype(np.float32)  # assume first 3 are v (body frame)

    t = np.asarray(data.meta['t'], dtype=np.float64)
    dt = np.diff(t, prepend=t[0])
    if not np.any(dt>0): dt[:] = 0.05
    else: dt[dt<=0] = np.median(dt[dt>0])
    A = np.zeros_like(V); A[1:] = (V[1:] - V[:-1]) / dt[1:,None]; A[0] = A[1]

    A_lp = smooth_acc_torch(torch.from_numpy(A), window=options.get('sg_window',15), poly=options.get('sg_poly',3)).cpu().numpy()
    F_phys = data.Y.astype(np.float32)

    drag_box_opt = options.get('drag_box', None)
    drag_box_np  = drag_box_opt.detach().cpu().numpy() if hasattr(drag_box_opt, 'detach') else drag_box_opt

    hover_thr = data.meta.get('hover_throttle', None)
    beta_opt = options.get('beta_drag_vec', options.get('beta_drag', 1.0))

    F_nom = compute_nominal_force(
        X, V, feature_len,
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
