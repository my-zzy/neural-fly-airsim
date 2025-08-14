
# Meta-PINN 
## Layout
```
meta_pinn_refactor/
├─ meta_pinn/
│  ├─ __init__.py
│  ├─ config.py          # Default options
│  ├─ utils.py           # seed, scaler, smoothing, warmup, checkpoint
│  ├─ physics.py         # nominal force & beta calibration
│  ├─ model.py           # MetaPINN model and losses
│  ├─ dataio.py          # dataset, dataloader, numpy conversion, condition vectors
│  ├─ adapt.py           # K-shot utilities + adaptation loop
│  └─ train.py           # training + evaluation (with K-shot adapt)
├─ main_meta_pinn.py               # example entrypoint (uses your existing utils & physics_params)
└─ README.md
```

## Usage

```bash
python main_meta_pinn.py
```

Customize hyperparameters in `meta_pinn/config.py` or pass your own `options` into `train_meta_pinn_multitask`.
