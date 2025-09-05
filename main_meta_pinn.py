import os, numpy as np, torch
from meta_pinn import MetaPINN, train_meta_pinn_multitask, plot_training_curves
from meta_pinn.utils import set_seed
from meta_pinn.config import DEFAULT_OPTIONS
from meta_pinn.physics import calibrate_beta_drag_vec
from meta_pinn.dataio import convert_dataset_to_numpy
from meta_pinn import load_data, format_data
from meta_pinn import get_uav_physics

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SEED = DEFAULT_OPTIONS['seed']
    set_seed(SEED, deterministic=True)
    print(f"SEED={SEED}, deterministic=True, device={device}")

    phys = get_uav_physics(device=device)
    DEFAULT_OPTIONS['drag_box'] = phys['drag_box']

    # load data via user's existing helpers
    raw_train = load_data('data/training')
    raw_test  = load_data('data/experiment', expnames='(baseline_)([0-9]*|no)wind')
    Data      = format_data(raw_train, features=DEFAULT_OPTIONS['features'], output='fa')
    TestData  = format_data(raw_test,  features=DEFAULT_OPTIONS['features'], output='fa')

    for i,d in enumerate(Data):     print(f"[Train Task {i}] {d.meta.get('method','')}_{d.meta.get('condition','')} <- {d.meta.get('filename','')}")
    for i,d in enumerate(TestData): print(f"[Test  Task {i}] {d.meta.get('method','')}_{d.meta.get('condition','')} <- {d.meta.get('filename','')}")

    num_tasks = len(Data); input_dim = Data[0].X.shape[1]
    model = MetaPINN(input_dim=input_dim, num_tasks=num_tasks,
                     task_dim=128, hidden_dim=384, use_uncertainty=True,
                     cond_dim=DEFAULT_OPTIONS.get('cond_dim',1),
                     use_cond_mod=DEFAULT_OPTIONS.get('use_cond_mod', True),
                     cond_mod_from=DEFAULT_OPTIONS.get('cond_mod_from', 'target'),
                     beta_min=DEFAULT_OPTIONS.get('beta_min',0.15), beta_max=DEFAULT_OPTIONS.get('beta_max',8.0))
    model.to(device)

    # Optional: one-shot calibration of beta_drag_vec (uses first train task's calmer speeds)
    try:
        Xi,Vi,Ai,Alpi,Fi,Fni = convert_dataset_to_numpy(Data[0], DEFAULT_OPTIONS)
        vmag = np.linalg.norm(Vi, axis=1)
        sel = (vmag < np.percentile(vmag, 70))
        if np.any(sel) and hasattr(DEFAULT_OPTIONS['drag_box'], 'detach'):
            drag_box_np = DEFAULT_OPTIONS['drag_box'].detach().cpu().numpy()
            beta_vec = calibrate_beta_drag_vec(Vi[sel], Fi[sel], drag_box_np,
                                               DEFAULT_OPTIONS.get('air_density',1.225),
                                               DEFAULT_OPTIONS.get('beta_min',0.15),
                                               DEFAULT_OPTIONS.get('beta_max',8.0))
            if beta_vec is not None:
                print(f"  >>> calibrated beta_drag_vec = {np.round(beta_vec,3)}")
                DEFAULT_OPTIONS['beta_drag_vec'] = beta_vec.tolist()
                DEFAULT_OPTIONS['calibrated_beta'] = True
    except Exception:
        pass

    save_dir = 'saved_models/meta_pinn'; os.makedirs(save_dir, exist_ok=True)
    hist = train_meta_pinn_multitask(model, Data, TestData, DEFAULT_OPTIONS['UAV_mass'], DEFAULT_OPTIONS, save_path=save_dir)
    torch.save(model.state_dict(), os.path.join(save_dir, 'meta_pinn_last.pth'))
    np.save(os.path.join(save_dir, 'history_train_total.npy'), np.array(hist['train_total']))
    np.save(os.path.join(save_dir, 'history_test_mse.npy'),   np.array(hist['test_avg_mse']))
    plot_training_curves(hist)

if __name__ == '__main__':
    main()
