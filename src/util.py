# %%
import torch
# %%
def median_distance(x):
    d2 = dist2(x, x)
    return torch.sqrt(torch.median(d2[d2 > 0])/2)

def dist2(x, y):
    return torch.sum(x**2, 1, keepdim=True) + \
           torch.sum(y**2, 1, keepdim=True).T - \
           2 * x @ y.T

def kernel(x, y, sigma):
    # kernel function between two n by d data matrices with bandwidth sigma
    # compute the pairwise squared Euclidean distance

    return torch.exp(-dist2(x, y) / (2 * sigma **2))

# X = torch.randn(10, 2)
# Y = torch.randn(100, 2)

# print(kernel(X, Y, median_distance(X)*10))
# %%
import numpy as np
def MMD(x, y, sigma):
    # compute the MMD between two sets of samples
    nx = x.shape[0]
    ny = y.shape[0]
    kxx = kernel(x, x, sigma)
    kyy = kernel(y, y, sigma)
    kxy = kernel(x, y, sigma)
    return torch.sqrt(kxx.sum() / nx**2 + kyy.sum() / ny**2 - 2 * kxy.sum() / nx / ny)


def KLIEP_obj(theta, kp, kq):
        
    o1 = torch.mean(kp @ theta, 0)
    o2 = torch.logsumexp(kq @ theta, 0) - np.log(kq.shape[0] * 1.0)
    
    return - o1 + o2

def KLIEP(xp, xq):
    xptest = xp[:50, :]
    xqtest = xq[:50, :]
    
    xp = xp[50:, :]
    xq = xq[50:, :]
    
    theta = torch.randn(50, device = xp.device)
    theta.requires_grad = True
    
    med = median_distance(xp)
    xb = xp[:50, :]
    
    kp = kernel(xp, xb, med)
    kq = kernel(xq, xb, med)
    
    kptest = kernel(xptest, xb, med)
    kqtest = kernel(xqtest, xb, med)
    
    optimizer = torch.optim.Adam([theta], lr=1e-2)
    
    for i in range(1000):
        optimizer.zero_grad()
        loss = KLIEP_obj(theta, kp, kq)
        loss.backward()
        optimizer.step()

    return -KLIEP_obj(theta, kptest, kqtest)
    
xp = torch.randn(500, 2) + 0
xq = torch.randn(500, 2)

print(KLIEP(xp, xq))

# %%
