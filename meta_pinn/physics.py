
import numpy as np, torch

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
    hover_throttle=None,
    hover_pwm_norm=0.5,
    use_hover_calib=True,
    drag_box_np=None,
    beta_drag: float = 1.0,
) -> np.ndarray:
    idx = 0; slices = {}
    for name, d in feature_len.items():
        if d <= 0: continue
        slices[name] = slice(idx, idx + d); idx += d

    N = V.shape[0]
    if 'pwm' in slices:
        pwm_blk = X[:, slices['pwm']].astype(np.float32)
        if pwm_blk.max() > 5.0:
            pwm_blk = (pwm_blk - 1000.0) / 1000.0
        pwm_norm = pwm_blk[:, :4] if pwm_blk.shape[1] >= 4 else np.repeat(pwm_blk, 4, axis=1)
        pwm_norm = np.clip(pwm_norm, 0.0, 1.0)
    else:
        pwm_norm = np.zeros((N, 4), dtype=np.float32)

    rpm = pwm_norm * rpm_max

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
        T_sum = T_hover_per.sum(axis=1)

        vmag = np.linalg.norm(V, axis=1)
        sel = vmag < np.percentile(vmag, 30)
        if not np.any(sel): sel = slice(None)

        target = mass * 9.80665
        scale = target / (np.median(T_sum[sel]) + 1e-8)
        scale = float(np.clip(scale, 0.2, 5.0))
        rpm = rpm * scale

    n = rpm / 60.0
    T_per = C_T * rho * (n**2) * (D**4)
    T_per = np.clip(T_per, 0.0, 10.0)
    T_sum = T_per.sum(axis=1)
    alpha_thrust = 1e-10
    T_sum = alpha_thrust * T_sum

    if drag_box_np is not None:
        beta = np.asarray(beta_drag, dtype=np.float32)
        if beta.ndim == 0: beta_vec = np.array([float(beta)]*3, dtype=np.float32)
        else: beta_vec = beta.astype(np.float32)
        Z = V * np.abs(V)
        coeff = (drag_box_np.astype(np.float32) * float(rho)) * beta_vec
        F_drag = - Z * coeff
    else:
        F_drag = - 0.325 * V

    f_nom = np.zeros_like(V, dtype=np.float32)
    f_nom[:, 0] = -F_drag[:, 0]
    f_nom[:, 1] = -F_drag[:, 1]
    f_nom[:, 2] = T_sum - F_drag[:, 2]
    return f_nom

def calibrate_beta_drag_vec(V: np.ndarray, F_phys: np.ndarray, drag_box_np: np.ndarray, air_density: float, beta_min: float, beta_max: float):
    """Closed-form per-axis beta calibration using ridge (very lightweight)."""
    try:
        c = (air_density * drag_box_np).astype(np.float32)
        Z = V * np.abs(V)
        beta_vec = []
        lam = 1e-3
        for k in range(3):
            Xk = -(Z[:, k:k+1] * c[k])
            yk = F_phys[:, k:k+1]
            XtX = (Xk.T @ Xk) + lam*np.eye(1)
            Xty = Xk.T @ yk
            beta_k = float(np.linalg.solve(XtX, Xty))
            beta_vec.append(beta_k)
        beta_vec = np.clip(np.array(beta_vec, dtype=np.float32), beta_min, beta_max)
        return beta_vec
    except Exception:
        return None
