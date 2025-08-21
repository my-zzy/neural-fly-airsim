
import matplotlib.pyplot as plt

def plot_training_curves(hist):
    fig = plt.figure(figsize=(12,4))
    plt.subplot(1,3,1); plt.plot(hist['train_total'],label='total'); plt.plot(hist['test_avg_mse'],label='test MSE'); plt.legend(); plt.title('Total & Test'); plt.xlabel('epoch')
    plt.subplot(1,3,2); plt.plot(hist['train_mse'],label='data MSE'); plt.plot(hist['train_newton'],label='Newton (LF)'); plt.plot(hist['train_residual'],label='Residual (robust)')
    if 'train_balance' in hist and max(hist['train_balance'])>0: plt.plot(hist['train_balance'],label='Balance')
    if 'train_beta_reg' in hist and max(hist['train_beta_reg'])>0: plt.plot(hist['train_beta_reg'],label='Beta Reg')
    plt.legend(); plt.title('Loss components'); plt.xlabel('epoch')
    plt.subplot(1,3,3); plt.axis('off'); plt.text(0.02,0.6,'Meta-PINN Scheme A\nCNM + per-task \u03b2-bias\nClosed-form \u03b2 warm-start + K-shot\nDeterministic eval + checkpoints',fontsize=9)
    plt.tight_layout(); plt.show()
    
    return fig