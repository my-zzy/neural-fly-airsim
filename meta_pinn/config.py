DEFAULT_OPTIONS = {
    'seed': 42,
    'deterministic': True,
    'learning_rate': 1e-2,
    'num_epochs': 120,              # 延长训练时间
    'batch_size': 512,

    # === K-shot ===
    'K_shot': 200,
    'kshot_score': 'fn',
    'eval_top_frac': 1.0,

    # === Adapt phase ===
    'adapt_steps': 250,
    'adapt_lr': 0.2,
    'adapt_repeats': 1,
    'adapt_noise_std': 0.0,
    'adapt_prox_weight': 0.003,
    'adapt_w_newton_scale': 0.5,
    'adapt_w_resid_scale': 0.5,
    'adapt_axis_scale': (1.0, 1.0, 1.3),
    'adapt_phys_gain': 1.0,
    'adapt_wind_threshold': 5.0,

    # === Physical loss + warmup ===
    'w_newton': 0.0,                # 初始为 0
    'w_resid':  0.0,                # 初始为 0
    'w_bias': 0.005,
    'warmup_epochs': 20,
    'warmup_start': 0.03,
    'warmup_end': 0.80,

    # === scheduler ===
    'scheduler': 'cosine',          # 余弦退火 (更平滑)
    'max_lr': 5e-4,                 # 稍大一点的上限，防止后期停滞

    # === features & smoothing ===
    'features': ['v','q','pwm'],
    'sg_window': 10, 'sg_poly': 3,
    'hover_pwm_norm': 0.5,

    # === UAV params ===
    'UAV_mass': 1.0,
    'UAV_rotor_C_T': 0.109919,
    'UAV_rotor_C_P': 0.040164,
    'air_density': 1.225,
    'UAV_rotor_max_rpm': 6396.667,
    'UAV_propeller_diameter': 0.2286,

    # === drag ===
    'drag_box' : None,
    'beta_drag': 1.0,

    # ====== condition / CNM ======
    'cond_dim': 1,
    'use_cond_mod': True,
    'cond_mod_from': 'target',
    'beta_min': 0.15, 'beta_max': 8.0,

    # β regularization
    'w_beta_reg': 0.0008,
    'beta_reg_schedule': None,

    # === Eval K-shot determinism ===
    'eval_adapt_repeats': 1,
    'eval_adapt_noise_std': 0.0,

    # === Physical loss schedule (新增) ===
    'phys_loss_schedule': {
        'start_epoch': 60,          # 从第 60 epoch 开始逐步引入
        'end_epoch': 100,           # 在第 100 epoch 达到目标权重
        'w_newton_target': 0.01,
        'w_resid_target': 0.02,
    },
}
