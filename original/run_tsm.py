# %%
from sklearn.datasets import make_moons, make_s_curve
import torch
from torch.utils.data import DataLoader, Dataset
torch.manual_seed(42)
# device = 'cpu'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# %%
from nn import SimpleCNN
from matplotlib import pyplot as plt
from traindenoisingsm import EnergyBasedModel, train_energy_based_model
from tsm import kNGD, kNGD_old
from torch.func import functional_call, vmap, vjp, jvp, jacrev
from neurald2k import empirical_ntk_jacobian_contraction as ntk_jac
from scipy.stats import norm
from util import median_distance, kernel
from util import MMD
from tsm2_ntk import SimpleNet, SimpleNet2, ntkNGD
from util import KLIEP

def run_ntkNGD(gendata, x0, xeval=None, eta = 0.1, updatedim = None, maxiter=30, method='ntkNGD', 
                                    kern='rbf', kernmodel = None, callback = None):
    xt = x0
    d = xt.shape[1]
    traj = [xt.detach().cpu().numpy()]

    # catch keyboard interrupt
    try:
        for iter in range(maxiter):
            
            if not callable(gendata):
                x1 = gendata
            else:
                x1 = gendata()
            
            if callback is not None:
                callback(x1, xt)
            
            if updatedim is None: 
                updatedim = range(d)
            
            net = SimpleNet(indim=d, hiddim=31, outdim=d).to(device)

            xb = torch.vstack([xt, x1])
            med = median_distance(xb[:, updatedim])  
            xb = xb[torch.randperm(xb.shape[0]), :]
            xb = xb[:200, :]
            
            def f(x):
                if kern == 'rbf':
                    # rbf kernel
                    k1 = kernel(x, xb, med)
                    return k1
                elif kern == 'poly':
                    # polynomial kernel with degree 2
                    k2 = (x @ xb.T + 1) ** 2
                    return k2
                elif kern == 'nn':
                    # neural network kernel
                    return kernmodel(x, penultimate=True, flattened=True)
                else:
                    return kernmodel(x, xb, med)
            
            if method == 'ntkNGD':
                fx = ntkNGD(x1, xt, 1e-3, f, net, 0).detach()
            elif method == 'kNGD':
                fx = kNGD(x1, xt, 1e-3, f, 0).detach()
            elif method == 'kNGD_old':
                fx = kNGD_old(x1, xt, 1e-3, f, net, 0).detach()
                
            xt[:, updatedim] = xt[:, updatedim] + eta * fx[:, updatedim]
            if xeval is not None:
                xeval[:, updatedim] = xeval[:, updatedim] + eta * fx[:, updatedim]
            # xt = torch.clamp(xt, 0, 1)

            traj.append(xt.detach().cpu().numpy())
            
            if callback is None:
                print("mmd: ", MMD(xt[:, updatedim], x1[:, updatedim], med).item())
                # print("KLIEP: ", KLIEP(x1, xt).item())
    
    except KeyboardInterrupt:
        print("Interrupted, will output the current xt")
        pass
        
    return xt, traj

# %%

if __name__ == "__main__":
    from IPython import display
    torch.manual_seed(42)
    d = 2
    
    model = EnergyBasedModel(2).to(device)
    xt = torch.randn(100, 2, device=device)
    MAXITER = 20
    for i in range(MAXITER):
        sigma = .01 * 1.25 ** (MAXITER - 1 - i)
        def gendata(n = 100):
            # xData = make_moons(n_samples= n, noise=0.01)[0]
            xData = make_s_curve(n_samples= n , noise=0.01)[0][:, [0, 2]]
            xData = torch.tensor(xData, dtype=torch.float32).to(device)
            return xData + torch.randn_like(xData)* sigma
        
        x1 = gendata(n = 10000) 
        # x0 = torch.rand(100, 2, device=device)*5 - 2.5
        x0 = gendata(n = 10000) + torch.randn_like(x1)* .1 
        model.train()
        X = torch.vstack([x0, x1])
        dataloader = DataLoader(X, batch_size=777, shuffle=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        train_energy_based_model(model, dataloader, optimizer, sigma = sigma, epochs=103)
        model.eval()
        
        def plotscatter(x1, xt):
            plt.scatter(xt[:, 0].cpu(), xt[:, 1].cpu(), c='r', s=1)
            plt.scatter(x1[:, 0].cpu(), x1[:, 1].cpu(), c='b', s=1)
            plt.xlim(-5, 5); plt.ylim(-5, 5)
            plt.show()
            display.clear_output(wait=True)

        xt, xt_traj = run_ntkNGD(gendata, xt, 1e-1, 200, kern ='nn', kernmodel=model, callback=plotscatter)
        print("mmd: ", MMD(xt, x1, median_distance(xt)).item())

    print("mean: ", torch.mean(xt, 0))
    print("std: ", torch.std(xt, 0))
    
    plt.figure(figsize=(5, 4))
    plt.scatter(xt[:, 0].cpu(), xt[:, 1].cpu(), c='r', s=1)
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    
    
# %%
