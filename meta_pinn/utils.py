
import os, random, numpy as np, torch
import torch.nn.functional as Fnn

def set_seed(seed: int = 42, deterministic: bool = True):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")
        try: torch.use_deterministic_algorithms(True)
        except Exception: pass
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

class StandardScaler:
    def __init__(self): self.mean=None; self.std=None
    def fit(self, x):
        self.mean = x.mean(axis=0, keepdims=True)
        self.std  = x.std(axis=0, keepdims=True) + 1e-8
        return self
    def transform(self, x): return (x - self.mean)/self.std
    def inverse_transform(self, x): return x*self.std + self.mean

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

def linear_warm(progress: float, start: float, end: float) -> float:
    return float(start + (end - start) * max(0.0, min(1.0, progress)))

def save_checkpoint(model, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
