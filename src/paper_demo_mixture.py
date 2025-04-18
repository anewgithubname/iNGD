# %%
import torch
torch.manual_seed(42)
# device = 'cpu'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# %%
import numpy as np
from util import MMD, median_distance
from sklearn.datasets import make_moons, make_s_curve
from run_tsm import run_ntkNGD
from cometitors import run_rkl_wgf, run_mmdgf

# read commandline first argument for dimension
import sys
d = 5
# d = int(sys.argv[1]) if len(sys.argv) > 1 else 5
print(f"Data {d} dimension")

def gendata(n = 100):    
    x1 = torch.randn(100000, d, device=device) * .5 - 2
    x2 = torch.randn(100000, d, device=device) * .5 + 2
    x = torch.vstack([x1, x2])
    x = x[torch.randperm(x.shape[0]), :]
    
    return x[:n, :]
# %%
runexp = True
# read commandline second argument for run experiments
trial = int(sys.argv[2]) if len(sys.argv) > 2 else 10
print(f"Running {trial} trials")

if runexp:
    print("Running experiments")
    for seed in range(trial):
        print(f"Running trial {seed}")
        torch.manual_seed(seed)
        x1 = gendata() 
        x0 = torch.randn_like(x1)
        
        print("ntKiNG")
        xt_NGD, traj_NGD = run_ntkNGD(x1.clone(), x0.clone(), eta = 1, maxiter = 100, method='ntkNGD')
        print("KiNG")
        xt_kNGD, traj_kNGD = run_ntkNGD(x1.clone(), x0.clone(), eta = 1, maxiter = 100, method='kNGD')
        print("WGF")
        xt_KDE, traj_KDE = run_rkl_wgf(x1.clone(), x0.clone(), 1, 100)
        print("MMD")
        xt_MMD, traj_MMD = run_mmdgf(x1.clone(), x0.clone(), 100, 100) # we forget to divde the MMD gradient by sample size!
        
        np.save(f'res/Target_{seed}_{x1.shape[1]}.npy', x1.cpu().numpy())
        np.save(f'res/NGD_{seed}_{x1.shape[1]}.npy', traj_NGD)
        np.save(f'res/kNGD_{seed}_{x1.shape[1]}.npy', traj_kNGD)
        np.save(f'res/KDE_{seed}_{x1.shape[1]}.npy', traj_KDE)
        np.save(f'res/MMD_{seed}_{x1.shape[1]}.npy', traj_MMD)
        print()
else: 
    print("Loading experimental results from files")
# %%
import torch
from util import KLIEP
import numpy as np
from matplotlib import pyplot as plt

trial = 10

res_NGD = np.zeros((trial, 101))
res_kNGD = np.zeros((trial, 101))
res_KDE = np.zeros((trial, 101))
res_MMD = np.zeros((trial, 101))

for seed in range(trial):
    x1_test = gendata(1000)
    
    # load data from files
    traj_NGD = np.load(f'res/NGD_{seed}_{x1.shape[1]}.npy')
    traj_kNGD = np.load(f'res/kNGD_{seed}_{x1.shape[1]}.npy')
    traj_KDE = np.load(f'res/KDE_{seed}_{x1.shape[1]}.npy')
    traj_MMD = np.load(f'res/MMD_{seed}_{x1.shape[1]}.npy')
    
    for i in range(len(traj_NGD)):
        xt_NGD = torch.tensor(traj_NGD[i], device=device)
        med = median_distance(xt_NGD)
        mmd = MMD(x1_test, xt_NGD, med).item()
        res_NGD[seed, i] = mmd
        
        xt_kNGD = torch.tensor(traj_kNGD[i], device=device)
        med = median_distance(xt_kNGD)
        mmd = MMD(x1_test, xt_kNGD, med).item()
        res_kNGD[seed, i] = mmd
        
        xt_KDE = torch.tensor(traj_KDE[i], device=device)
        med = median_distance(xt_KDE)
        mmd = MMD(x1_test, xt_KDE, med).item()
        res_KDE[seed, i] = mmd
        
        xt_MMD = torch.tensor(traj_MMD[i], device=device)
        med = median_distance(xt_MMD)
        mmd = MMD(x1_test, xt_MMD, med).item()
        res_MMD[seed, i] = mmd

    # plt.plot(NGD_list, label='NGD', c = 'r')
    # plt.plot(kNGD_list, label='kNGD', c = 'g')
    # plt.plot(KDE_list, label='KDE', c = 'b')

mean_NGD = res_NGD.mean(0)
std_NGD = res_NGD.std(0)/np.sqrt(trial)
mean_kNGD = res_kNGD.mean(0)
std_kNGD = res_kNGD.std(0)/np.sqrt(trial)
mean_KDE = res_KDE.mean(0)
std_KDE = res_KDE.std(0)/np.sqrt(trial)
mean_MMD = res_MMD.mean(0)
std_MMD = res_MMD.std(0)/np.sqrt(trial)

plt.figure(figsize=(4, 4))
plt.plot(mean_NGD, label='ntKiNG', c='r')
plt.fill_between(np.arange(res_NGD.shape[1]), mean_NGD - std_NGD, mean_NGD + std_NGD, color='r', alpha=0.3)

plt.plot(mean_kNGD, label='KiNG', c='g')
plt.fill_between(np.arange(res_kNGD.shape[1]), mean_kNGD - std_kNGD, mean_kNGD + std_kNGD, color='g', alpha=0.3)

plt.plot(mean_KDE, label='WGF', c='b')
plt.fill_between(np.arange(res_KDE.shape[1]), mean_KDE - std_KDE, mean_KDE + std_KDE, color='b', alpha=0.3)

plt.plot(mean_MMD, label='MMD', c='k')
plt.fill_between(np.arange(res_MMD.shape[1]), mean_MMD - std_MMD, mean_MMD + std_MMD, color='k', alpha=0.3)

plt.xlabel('Iteration', fontsize=14)
plt.ylabel('MMD', fontsize=14)
plt.yscale('log')
# plt.grid()
plt.legend(fontsize=14)
plt.savefig(f'res/mixture_{x1.shape[1]}d.png')
# %%

# plt.scatter(traj_NGD[-1][:, 0], traj_NGD[-1][:, 1], c = 'r', label='NGD')