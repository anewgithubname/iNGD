# %%
import torch
from matplotlib import pyplot as plt

from util import MMD, median_distance
torch.manual_seed(987)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from cometitors import run_rkl_wgf
from cometitors import run_mmdgf


# %%
from torchvision import transforms
from IPython import display
from torch.utils.data import DataLoader
from sklearn.datasets import make_moons
from sklearn.datasets import make_s_curve
from traindenoisingsm import EnergyBasedModel
from traindenoisingsm import train_energy_based_model

X = make_s_curve(n_samples=10000, noise=0.01, random_state=42)[0][:, [0, 2]]
d = X.shape[1]

X = torch.tensor(X, dtype=torch.float32).to(device)
xt = X[:200, :]
xt = xt + torch.randn_like(xt) * 0.3

plt.figure(figsize=(4, 4))
plt.scatter(X[:200, 0].cpu(), X[:200, 1].cpu(), c='b', s=1)
plt.scatter(xt[:, 0].cpu(), xt[:, 1].cpu(), c='r', s=1)
plt.xlim(-2, 2); plt.ylim(-3, 3)
plt.title("Noisy samples")

# %%
import os
def gendata(n = 200):
    # take a random sample from the X
    ret = X[torch.randint(0, X.shape[0], (n,)), :]
    return ret

model = EnergyBasedModel(d).to(device)

if not os.path.exists("sm_model.pt"):
    model.train()
    X = gendata(n = 10000) 
    dataloader = DataLoader(X, batch_size=777, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    train_energy_based_model(model, dataloader, optimizer, sigma = .3, epochs=1003)
    model.eval()
    torch.save(model.state_dict(), "sm_model.pt")
else:
    model.load_state_dict(torch.load("sm_model.pt", map_location=device))
    model.eval()

# %%
from run_tsm import run_ntkNGD

x1 = gendata(n = 10000)[-200:]
xt_ngd, xt_traj_ngd = run_ntkNGD(x1, xt.clone(), None, 1e-1, None, 1000, 'ntkNGD', kern ='nn', kernmodel=model)
print("mmd: ", MMD(xt_ngd, X[:200, :], median_distance(xt)).item())

# %%
x1 = gendata(n = 10000) 
x1 = x1.detach().clone()[-200:]
xt_wgf, xt_traj_wgf = run_rkl_wgf(x1, xt.clone(), eta = 1, niter=1000)

# %%
plt.figure(figsize=(12, 4))
plt.subplots_adjust(wspace=0.05)  # Reduce horizontal space between subplots

plt.subplot(1, 3, 1)
plt.scatter(x1[:, 0].cpu(), x1[:, 1].cpu(), c='b', s=16)
plt.scatter(xt[:, 0].cpu(), xt[:, 1].cpu(), c='r', s=16)
# plt.legend(loc='upper right', fontsize=16)
plt.title("red: $X_0$, blue: $Y$", fontsize=16)
plt.xticks([])
plt.yticks([])

plt.subplot(1, 3, 3)
plt.scatter(xt_wgf[:, 0].cpu(), xt_wgf[:, 1].cpu(), c='r', s=16)
plt.xlim(-2, 2); plt.ylim(-3, 3)
plt.title("$X_{1000}$: WGF", fontsize=16)
plt.xticks([])
plt.yticks([])

plt.subplot(1, 3, 2)
plt.scatter(xt_ngd[:, 0].cpu(), xt_ngd[:, 1].cpu(), c='r', s=16)
plt.xlim(-2, 2); plt.ylim(-3, 3)
plt.title("$X_{1000}$: ntKiNG + pretrained EBM", fontsize=16)
plt.xticks([])
plt.yticks([])
plt.show()
# plt.savefig("res/scurve_ntkng_wgf.pdf", bbox_inches='tight', dpi=300)
# %%
