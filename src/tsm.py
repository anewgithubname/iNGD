from d2k import gaussian_rbf_second_deriv_xy
import torch
from torch import nn

from torch.distributions import Normal

from util import median_distance, kernel, dist2
# %%

def kNGD(x1, xt, lam, f, lam2):
    xt = xt.detach().clone()
    xt.requires_grad = True
    fx1 = f(x1)
    fxt = f(xt)

    dfxt = torch.zeros(fxt.shape[0], xt.shape[1], fxt.shape[1], device=fxt.device)
    for i in range(fxt.shape[1]):
        # autograd fxt with respect to each dimension
        dfxt[:,:,i] = torch.autograd.grad(fxt[:,i].sum(), xt, create_graph=True)[0]
    # dfxt shape: [nt, d, b]
    dfxt = dfxt.detach()

    med = median_distance(xt)
    sigma = med * 1
    
    nabla_L = (torch.mean(fx1, 0) - torch.mean(fxt, 0)).detach()
    F = torch.cov(fxt.T)
    F = F + lam2 * torch.eye(F.shape[0], device=F.device)
    
    # d2k shape: [nt, nt, d, d]
    d2k = gaussian_rbf_second_deriv_xy(xt, xt, sigma)
    # einstein sum, d2k and dfxt, over the dimension of xt
    # d2k_dfxt shape: [nt, nt, d, b]
    d2k_dfxt = torch.einsum('ijkl,ikm->ijlm', d2k, dfxt)
    # einstein sum, d2k_dfxt and dfxt, over the dimension of xt
    # d2k_dfxt_dfxt shape: [nt, nt, b, b]
    d2k_dfxt_dfxt = torch.einsum('ijlm,jln->ijmn', d2k_dfxt, dfxt)
    
    # KK dimension: [b, b]
    KK = lam * F + torch.sum(d2k_dfxt_dfxt, (0, 1)) / xt.shape[0]**2 + 1e-3 * torch.eye(fxt.shape[1], device=fxt.device)
    
    # fx = nabla_L.T @ torch.inverse(KK) @ torch.einsum('ijk,iljm->klm', dfxt, d2k) / xt.shape[0] 
    
    invKK_nabla_L = torch.linalg.solve(KK, nabla_L.T)
    fx = torch.zeros(xt.shape[0], xt.shape[1], device=xt.device)
    for i in range(xt.shape[1]):
        d2k_dfxt_i = torch.mean(d2k_dfxt[:,:,i,:], 0) # shape: [nt, b]
        # print((nabla_L.T @ torch.inverse(KK)).shape)
        fx[:, i] = d2k_dfxt_i @ invKK_nabla_L
    
    return fx

f = lambda x: x
fx = kNGD(torch.randn(1000, 2)-1, torch.randn(1000, 2), 0, f, 0)
print(torch.mean(fx, 0))
# %%
from torch.func import functional_call
from neurald2k import empirical_ntk_jacobian_contraction as ntk_jac

def kNGD_old(x1, xt, lam, f, nn, lam2):
    xt = xt.detach().clone()
    xt.requires_grad = True
    fx1 = f(x1)
    fxt = f(xt)


    dfxt = torch.zeros(fxt.shape[0], xt.shape[1], fxt.shape[1], device=fxt.device)
    for i in range(fxt.shape[1]):
        # autograd fxt with respect to each dimension
        dfxt[:,:,i] = torch.autograd.grad(fxt[:,i].sum(), xt, create_graph=True)[0]
    dfxt = dfxt.detach()


    med = median_distance(xt)
    k = kernel(xt, xt, med * 1 )
    
    # params = {k: v.detach() for k, v in nn.named_parameters()}

    # def fnet_single(params, x):
    #     return functional_call(nn, params, (x.unsqueeze(0),)).squeeze(0)
    
    # k = ntk_jac(fnet_single, params, xt, xt, compute='trace') / xt.shape[1]
    
    nabla_L = (torch.mean(fx1, 0) - torch.mean(fxt, 0)).detach()
    F = torch.cov(fxt.T)
    F = F + lam2 * torch.eye(F.shape[0], device=F.device)
    
    fx = torch.zeros(xt.shape[0], xt.shape[1], device=xt.device)
    for i in range(xt.shape[1]):
        difxt = dfxt[:, i, :].squeeze() 
        K = difxt.T @ k @ difxt / xt.shape[0]**2
        fx[:, i] = (nabla_L.T @ torch.inverse(lam * F + K + 1e-3 * torch.eye(fxt.shape[1], device=fxt.device) ) @ (difxt.T @ k /xt.shape[0] )).T
    
    return fx
# %%

# plt.plot(torch.linspace(-5,5,100), f(torch.linspace(-5,5,100))[:,2])

# def fourierfea(b, t):
#     freq = (torch.arange(b, dtype=torch.float32) + 1)*2
#     fourier_features = torch.cat([freq * torch.cos(freq * t), -freq *torch.sin(freq * t)], dim=-1)

#     return fourier_features

# def dfourierfea(b, t):
#     freq = (torch.arange(b, dtype=torch.float32) + 1)*2
#     fourier_features = torch.cat([-freq**2 *torch.sin(freq * t), -freq**2 *torch.cos(freq * t)], dim=-1)
    
#     return fourier_features

def dphit(t, dfea):
    b = 1
    feat = torch.zeros(b * dfea, dfea)
    for i in range(dfea):
        # feat[i*b:(i+1)*b, i] = fourierfea(b//2, t)
        feat[i*b:(i+1)*b, i] = torch.ones(b)

    return feat

def d2phit(t, dfea):
    b = 1
    feat = torch.zeros(b * dfea, dfea)
    for i in range(dfea):
        # feat[i*b:(i+1)*b, i] = dfourierfea(b//2, t)
        feat[i*b:(i+1)*b, i] = torch.zeros(b)

    return feat

def dthetat(generator, timestep, n, tstart = 0, tend = 1, lmb = 1e-2):
    data = []
    
    t = torch.linspace(tstart, tend, timestep + 1)
    for i in range(timestep + 1):
        x = generator(t[i])
        data.append((x, t[i]))
    
    n, d = data[0][0].shape
    dfea = f(data[0][0]).shape[1]
    dtime = dphit(0, dfea).shape[0]
    C = torch.zeros(dtime, dtime)
    b = torch.zeros(dtime, 1)
    
    for x, t in data:
        dphi = dphit(t, dfea)
        d2phi = d2phit(t, dfea)
        t = torch.ones(n)*t
        
        fx = f(x)
        fxbar = fx - torch.mean(fx, 0)
        dphi_fxbar = fxbar @ dphi.T
        dphi_fx = fx @ dphi.T
        d2phi_fx =  fx @ d2phi.T
        
        Ct = dphi_fxbar.T * g2(t, tstart, tend) @ dphi_fxbar / n
        bt = dphi_fx.T @ dg2(t, tstart, tend).reshape(-1, 1) / n
        bt = bt + d2phi_fx.T @ g2(t, tstart, tend).reshape(-1, 1) / n
        
        C = C + Ct
        b = b + bt
    
    return torch.inverse(C + torch.eye(dtime)*lmb) @ b * -1

if __name__ == '__main__':
    from nn import MLP
    from matplotlib import pyplot as plt
    torch.manual_seed(42)
    generator = MLP()

    tstart = .1
    tend = .2

    print(dthetat(generator, 1000, 1000, tstart, tend, lmb = 1e-1))
    tmarks = torch.linspace(tstart, tend, 100)
    alpha = dthetat(generator, 1000, 1000, tstart, tend, lmb = 0)

    dthetathat =[]
    for t in tmarks:
        dthe = dphit(t, 2).T @ alpha
        print(dthe)
        dthetathat.append(dthe[0,0].item())

    plt.plot(tmarks, dthetathat)

    def dthetastar():
        t = torch.linspace(tstart, tend, 100)
        gradt = []
        t.requires_grad = True
        for i in range(100):
            grad = torch.autograd.grad( generator.mu(t[i])*generator.sigma(t[i])**2, t, create_graph=True)[0]
            
            # grad = torch.autograd.grad(-1 * generator.sigma(t[i])**2 / 2, t, create_graph=True)[0]
            gradt.append(grad[i].item())
        
        return gradt

    plt.plot(tmarks, dthetastar())
    plt.show()