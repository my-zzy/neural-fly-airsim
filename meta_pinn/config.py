DEFAULT_OPTIONS = {
    'seed': 42,
    'deterministic': True,
    'learning_rate': 1e-2,
    'num_epochs': 60,                 # 更少的训练轮数
    'batch_size': 512,

    # === K-shot ===
    'K_shot': 200,                    # 温和一些，避免适配过拟合
    'kshot_score': 'fn',
    'eval_top_frac': 1.0,

    # === Adapt phase ===
    'adapt_steps': 100,
    'adapt_lr': 0.1,
    'adapt_repeats': 1,
    'adapt_noise_std': 0.0,
    'adapt_prox_weight': 0.003,
    'adapt_w_newton_scale': 0.5,
    'adapt_w_resid_scale': 0.5,
    'adapt_axis_scale': (1.0, 1.0, 1.3),
    'adapt_phys_gain': 1.0,
    'adapt_wind_threshold': 5.0,

    # === Physical loss + warmup ===
    # 目标权重写在这里，前 warmup_epochs 由 warmup 从 0→1 平滑放大
    'w_newton': 0.005,
    'w_resid':  0.010,
    'w_bias': 0.005,
    'warmup_epochs': 15,              # 前 15 个 epoch 做 warmup
    'warmup_start': 0.0,              # 从 0 开始
    'warmup_end': 1.0,                # 拉满为 1.0

    # === scheduler ===
    'scheduler': 'cosine',
    'max_lr': 5e-4,

    # === features & smoothing ===
    'features': ['v','q','pwm'],      # 与当前数据对齐（总维度=11）
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
    'drag_box': None,
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

    # 不再使用单独的物理 schedule，完全交给 warmup
    'phys_loss_schedule': None,
}
