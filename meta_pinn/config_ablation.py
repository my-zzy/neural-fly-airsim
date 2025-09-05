DEFAULT_OPTIONS = {
    'seed': 42,
    'deterministic': True,

    # === 训练参数 ===
    'learning_rate': 1e-3,         # 稳定收敛
    'num_epochs': 60,              # 总训练 60 轮
    'batch_size': 64,              # 小 batch，提高梯度多样性

    # === K-shot ===
    'K_shot': 100,
    'adapt_steps': 10,             # 减少适应步数
    'adapt_lr': 0.003,             # 更温和的内循环更新
    'adapt_repeats': 1,
    'adapt_noise_std': 0.0,
    'adapt_prox_weight': 0.001,
    'adapt_w_newton_scale': 0,
    'adapt_w_resid_scale': 0,

    # === Physical loss + warmup ===
    'w_newton': 0,             # 更轻的物理项
    'w_resid':  0,
    'w_bias':   0,
    'warmup_epochs': 60,           # 前 45 轮主要学 MSE
    'warmup_start': 0.0,
    'warmup_end': 1.0,

    # === scheduler ===
    'scheduler': 'cosine',
    'max_lr': 5e-4,                # 防止高 lr 抖动

    # === features & smoothing ===
    'features': ['v','q','pwm'],
    'sg_window': 15, 'sg_poly': 3,
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
    'w_beta_reg': 0.0005,
    'beta_reg_schedule': None,

    # === Eval K-shot determinism ===
    'eval_adapt_repeats': 2,
    'eval_adapt_noise_std': 0.0,

    'phys_loss_schedule': None,
}
