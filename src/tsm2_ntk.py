# %%
from d2k import gaussian_rbf_second_deriv_xy
import torch
from torch import nn as torchnn
from torch import sum, mean, cov, einsum

from torch.distributions import Normal
from neurald2k import empirical_ntk_jacobian_contraction as ntk_jac
from torch.autograd.functional import jacobian
from torch.func import functional_call

# %%

def ntkNGD(x1, xt, lam, f, nn, lam2):
    xt = xt.detach().clone()
    xt.requires_grad = True
    fx1 = f(x1)
    fxt = f(xt)

    # Compute full Jacobian dfxt in one step
    dfxt = jacobian(lambda x: f(x).sum(dim=0), xt, create_graph=True)
    dfxt = dfxt.permute(1, 2, 0).detach()  # Shape: [nt, d, b]
    xt.requires_grad = False
    
    nabla_L = (mean(fx1, 0) - mean(fxt, 0)).detach()
    F = cov(fxt.T)
    
    # ker = f(torch.vstack([xt, x1]))
    F = F + lam2 * torch.eye(F.shape[0], device=F.device, dtype=torch.float32)
    
    params = {k: v.detach() for k, v in nn.named_parameters()}

    def fnet_single(params, x):
        return functional_call(nn, params, (x.unsqueeze(0),)).squeeze(0)
    
    # d2k shape: [nt, nt, d, d], jac shape: [nt, d, params]
    d2k = ntk_jac(fnet_single, params, xt, xt, compute='diagonal')
    # d2k = torch.diag_embed(d2k)
    # einstein sum, d2k and dfxt, over the dimension of xt
    # d2k_dfxt shape: [nt, nt, d, b]
    d2k_dfxt = einsum('ijk,ikm->ijkm', d2k, dfxt)
    # einstein sum, d2k_dfxt and dfxt, over the dimension of xt
    # d2k_dfxt_dfxt shape: [nt, nt, b, b]
    d2k_dfxt_dfxt = einsum('ijlm,jln->ijmn', d2k_dfxt, dfxt)
    
    # KK dimension: [b, b]
    KK = lam * F + sum(d2k_dfxt_dfxt, (0, 1)) / xt.shape[0]**2 + 1e-3 * torch.eye(F.shape[0], device=F.device, dtype=torch.float32)
    inv_KK_nabla_L = torch.linalg.solve(KK, nabla_L.T)
 
    fx = torch.zeros(xt.shape[0], xt.shape[1], device=x1.device, dtype=torch.float32)
    for i in range(xt.shape[1]):
        d2k_dfxt_i = mean(d2k_dfxt[:,:,i,:], 0) # shape: [nt, b]
        # print((nabla_L.T @ torch.inverse(KK)).shape)
        fx[:, i] = d2k_dfxt_i @ inv_KK_nabla_L
    
    return fx

class SimpleNet(torchnn.Module):
    def __init__(self, indim = 1, hiddim = 121, outdim = 1):
        super(SimpleNet, self).__init__()
        self.fc1 = torchnn.Linear(indim, hiddim)
        self.fc2 = torchnn.Linear(hiddim, hiddim)
        self.fc3 = torchnn.Linear(hiddim, outdim)
    
    def forward(self, x):
        # x = self.fc1(x)
        x = torchnn.functional.tanh(self.fc1(x))
        x = torchnn.functional.tanh(self.fc2(x))
        x = self.fc3(x)
        return x
    
class SimpleNet2(torchnn.Module):
    def __init__(self, indim = 1, hiddim = 121, outdim = 1):
        super(SimpleNet2, self).__init__()
        self.fc1 = torchnn.Linear(indim, hiddim)
        self.fc2 = torchnn.Linear(hiddim, hiddim)
        self.fc3 = torchnn.Linear(hiddim, hiddim)
        self.fc4 = torchnn.Linear(hiddim, outdim)
    
    def forward(self, x):
        # x = self.fc1(x)
        x = torchnn.functional.tanh(self.fc1(x))
        x = torchnn.functional.tanh(self.fc2(x))
        x = torchnn.functional.tanh(self.fc3(x))
        x = self.fc4(x)
        return x

net = SimpleNet(indim=1, hiddim=61, outdim=2)

f = lambda x: torch.hstack([x, x**2])

fx = ntkNGD(torch.randn(1000, 1, dtype=torch.float32)-2, torch.randn(1000, 1, dtype=torch.float32), 1e-3, f, net, 0)
print(mean(fx, 0))
# %%
