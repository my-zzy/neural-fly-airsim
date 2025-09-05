import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
torch.set_default_tensor_type('torch.DoubleTensor')
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import random_split

import utils
import mlmodel

import sys
if sys.platform == 'win32':
    NUM_WORKERS = 0 # Windows does not support multiprocessing
else:
    NUM_WORKERS = 2
print('running on ' + sys.platform + ', setting ' + str(NUM_WORKERS) + ' workers')
dim_a = 4
features = ['v', 'q', 'pwm']
label = 'fa'

# Training data collected from the neural-fly drone
dataset = 'neural-fly' 
dataset_folder = 'data/training'
hover_pwm_ratio = 1.

# # Training data collected from an intel aero drone
# dataset = 'neural-fly-transfer'
# dataset_folder = 'data/training-transfer'
# hover_pwm = 910 # mean hover pwm for neural-fly drone
# intel_hover_pwm = 1675 # mean hover pwm for intel-aero drone
# hover_pwm_ratio = hover_pwm / intel_hover_pwm # scaling ratio from system id

modelname = f"{dataset}_dim-a-{dim_a}_{'-'.join(features)}" # 'intel-aero_fa-num-Tsp_v-q-pwm'
print(modelname)

RawData = utils.load_data(dataset_folder)
Data = utils.format_data(RawData, features=features, output=label)
print(Data[0][0][0])
testdata_folder = 'data/experiment'
RawData = utils.load_data(testdata_folder, expnames='(baseline_)([0-9]*|no)wind')
TestData = utils.format_data(RawData, features=features, output=label, hover_pwm_ratio=hover_pwm_ratio) # wind condition label, C, will not make sense for this data - that's okay since C is only used in the training process

for data in Data:
    utils.plot_subdataset(data, features, title_prefix="(Training data)")
    break

for data in TestData:
    utils.plot_subdataset(data, features, title_prefix="(Testing Data)")
    break

options = {}
options['dim_x'] = Data[0].X.shape[1]
options['dim_y'] = Data[0].Y.shape[1]
options['num_c'] = len(Data)
print(len(Data))
print('dims of (x, y) are', (options['dim_x'], options['dim_y']))
print('there are ' + str(options['num_c']) + ' different conditions')
print(Data[0].Y[0])

# Set hyperparameters
options['features'] = features
options['dim_a'] = dim_a
options['loss_type'] = 'crossentropy-loss'

options['shuffle'] = True # True: shuffle trajectories to data points
options['K_shot'] = 32 # number of K-shot for least square on a
options['phi_shot'] = 256 # batch size for training phi

options['alpha'] = 0.01 # adversarial regularization loss
options['learning_rate'] = 5e-4
options['frequency_h'] = 2 # how many times phi is updated between h updates, on average
options['SN'] = 2. # maximum single layer spectral norm of phi
options['gamma'] = 10. # max 2-norm of a
options['num_epochs'] = 200

print(features)
print(dim_a)

# Trainset = []
# Adaptset = []
Trainloader = []
Adaptloader = []
for i in range(options['num_c']):
    fullset = mlmodel.MyDataset(Data[i].X, Data[i].Y, Data[i].C)
    
    l = len(Data[i].X)
    if options['shuffle']:
        trainset, adaptset = random_split(fullset, [int(2/3*l), l-int(2/3*l)])
    else:
        trainset = mlmodel.MyDataset(Data[i].X[:int(2/3*l)], Data[i].Y[:int(2/3*l)], Data[i].C) 
        adaptset = mlmodel.MyDataset(Data[i].X[int(2/3*l):], Data[i].Y[int(2/3*l):], Data[i].C)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=options['phi_shot'], shuffle=options['shuffle'], num_workers=NUM_WORKERS)
    adaptloader = torch.utils.data.DataLoader(adaptset, batch_size=options['K_shot'], shuffle=options['shuffle'], num_workers=NUM_WORKERS)
   
    # Trainset.append(trainset)
    # Adaptset.append(adaptset)
    Trainloader.append(trainloader) # for training phi
    Adaptloader.append(adaptloader) # for LS on a

# Store the model class definition in an external file so they can be referenced outside this script
phi_net = mlmodel.Phi_Net(options)
h_net = mlmodel.H_Net_CrossEntropy(options)
criterion = nn.MSELoss()
criterion_h = nn.CrossEntropyLoss()
optimizer_h = optim.Adam(h_net.parameters(), lr=options['learning_rate'])
optimizer_phi = optim.Adam(phi_net.parameters(), lr=options['learning_rate'])
model_save_freq = 200 # How often to save the model

# Create some arrays to save training statistics
Loss_f = [] # combined force prediction loss
Loss_c = [] # combined adversarial loss

# Loss for each subdataset 
Loss_test_nominal = [] # loss without any learning
Loss_test_mean = [] # loss with mean predictor
Loss_test_phi = [] # loss with NN
for i in range(len(TestData)):
    Loss_test_nominal.append([])
    Loss_test_mean.append([])
    Loss_test_phi.append([])

# Training!
for epoch in range(options['num_epochs']):
    # Randomize the order in which we train over the subdatasets
    arr = np.arange(options['num_c'])
    np.random.shuffle(arr)

    # Running loss over all subdatasets
    running_loss_f = 0.0
    running_loss_c = 0.0

    for i in arr:
        with torch.no_grad():
            adaptloader = Adaptloader[i]
            kshot_data = next(iter(adaptloader))
            trainloader = Trainloader[i]
            data = next(iter(trainloader))
        
        optimizer_phi.zero_grad()
        
        '''
        Least-square to get $a$ from K-shot data
        '''
        X = kshot_data['input'] # K x dim_x
        Y = kshot_data['output'] # K x dim_y
        Phi = phi_net(X) # K x dim_a
        Phi_T = Phi.transpose(0, 1) # dim_a x K
        A = torch.inverse(torch.mm(Phi_T, Phi)) # dim_a x dim_a
        a = torch.mm(torch.mm(A, Phi_T), Y) # dim_a x dim_y
        if torch.norm(a, 'fro') > options['gamma']:
            a = a / torch.norm(a, 'fro') * options['gamma']
            
        '''
        Batch training \phi_net
        '''
        inputs = data['input'] # B x dim_x
        labels = data['output'] # B x dim_y
        
        c_labels = data['c'].type(torch.long)
            
        # forward + backward + optimize
        outputs = torch.mm(phi_net(inputs), a)
        loss_f = criterion(outputs, labels)
        temp = phi_net(inputs)
        
        loss_c = criterion_h(h_net(temp), c_labels)
            
        loss_phi = loss_f - options['alpha'] * loss_c
        loss_phi.backward()
        optimizer_phi.step()
        
        '''
        Discriminator training
        '''
        if np.random.rand() <= 1.0 / options['frequency_h']:
            optimizer_h.zero_grad()
            temp = phi_net(inputs)
            
            loss_c = criterion_h(h_net(temp), c_labels)
            
            loss_h = loss_c
            loss_h.backward()
            optimizer_h.step()
        
        '''
        Spectral normalization
        '''
        if options['SN'] > 0:
            for param in phi_net.parameters():
                M = param.detach().numpy()
                if M.ndim > 1:
                    s = np.linalg.norm(M, 2)
                    if s > options['SN']:
                        param.data = param / s * options['SN']
         
        running_loss_f += loss_f.item()
        running_loss_c += loss_c.item()
    
    # Save statistics
    Loss_f.append(running_loss_f / options['num_c'])
    Loss_c.append(running_loss_c / options['num_c'])
    if epoch % 10 == 0:
        print('[%d] loss_f: %.2f loss_c: %.2f' % (epoch, running_loss_f / options['num_c'], running_loss_c / options['num_c']))

        
    with torch.no_grad():
        for j in range(len(TestData)):
            loss_nominal, loss_mean, loss_phi = mlmodel.error_statistics(TestData[j].X, TestData[j].Y, phi_net, h_net, options=options)
            Loss_test_nominal[j].append(loss_nominal)
            Loss_test_mean[j].append(loss_mean)
            Loss_test_phi[j].append(loss_phi)

    # if epoch % model_save_freq == 0:
    #     mlmodel.save_model(phi_net=phi_net, h_net=h_net, modelname=modelname + '-epoch-' + str(epoch), options=options)

mlmodel.save_model(phi_net=phi_net, h_net=h_net, modelname=modelname + '-epoch-' + str(epoch), options=options)

plt.subplot(2, 1, 1)
plt.plot(Loss_f)
plt.xlabel('epoch')
plt.ylabel('f-loss [N]')
plt.title('training f loss')
plt.subplot(2, 1, 2)
plt.plot(Loss_c)
plt.title('training c loss')
plt.xlabel('epoch')
plt.ylabel('c-loss')
plt.tight_layout()

# for j in range(len(TestData)):
#     plt.figure()
#     # plt.plot(Loss_test_nominal[j], label='nominal')
#     plt.plot(Loss_test_mean[j], label='mean')
#     plt.plot(np.array(Loss_test_phi[j]), label='phi*a')
#     # plt.plot(np.array(Loss_test_exp_forgetting[j]), label='exp forgetting')
#     plt.legend()
#     plt.title(f'Test data set {j} - {TestData[j].meta["condition"]}')

plt.show()

# Choose final model
stopping_epoch = 199
options['num_epochs'] = stopping_epoch
final_model = mlmodel.load_model(modelname = modelname + '-epoch-' + str(stopping_epoch))
print('Final model loaded:', final_model)
print()
print(final_model.phi(torch.zeros(11)))
import time, numpy as np, torch, torch.nn as nn

def _cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def error_statistics_with_time(data_input: np.ndarray,
                               data_output: np.ndarray,
                               phi_net, h_net, options):
    """
    返回: (error_before, error_mean, error_after, t_after_total_ms, t_after_per_sample_ms)
    说明:
      - t_after_* 含最小二乘拟合 + φ 前向 + 线性头
      - 采用 torch.cholesky_solve 拟合 a
    """
    criterion = nn.MSELoss()
    device = next(phi_net.parameters()).device if hasattr(phi_net, "parameters") else torch.device("cpu")
    phi_net.eval()

    lam = float(options.get("lam", 0.0))
    X_t = torch.from_numpy(data_input).to(device)
    Y_t = torch.from_numpy(data_output).to(device)
    N   = X_t.shape[0]

    # ---------- Before ----------
    y_before = torch.zeros_like(Y_t)
    error_1 = criterion(Y_t, y_before).item()

    # ---------- Mean ----------
    y_mean = Y_t.mean(dim=0, keepdim=True).expand_as(Y_t)
    error_2 = criterion(Y_t, y_mean).item()

    # ---------- After (含拟合 + 验证前向) ----------
    _cuda_sync()
    t0 = time.perf_counter()
    with torch.no_grad():
        Phi   = phi_net(X_t)                         # [N, dim_a]
        Phi_T = Phi.transpose(0, 1)
        dim_a = Phi.shape[1]
        I     = torch.eye(dim_a, device=Phi.device, dtype=Phi.dtype)
        A = Phi_T @ Phi + lam * I
        B = Phi_T @ Y_t
        L = torch.linalg.cholesky(A)
        a = torch.cholesky_solve(B, L)               # [dim_a, dim_y]
        y_after = Phi @ a
    _cuda_sync()
    elapsed = time.perf_counter() - t0

    error_3 = criterion(Y_t, y_after).item()
    t_after_total_ms = 1000.0 * elapsed
    t_after_per_sample_ms = t_after_total_ms / max(1, N)

    return error_1, error_2, error_3, t_after_total_ms, t_after_per_sample_ms
for data in TestData:
    e1, e2, e3, t_total, t_avg = error_statistics_with_time(
        data.X, data.Y, final_model.phi, final_model.h, options
    )
    print(f"[{data.meta.get('condition')}] MSE -> before={e1:.4f}, mean={e2:.4f}, after={e3:.4f}")
    print(f"[{data.meta.get('condition')}] 时间 -> 总耗时={t_total:.3f} ms, 平均={t_avg:.6f} ms/sample")
