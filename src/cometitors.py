# %%
import torch
from tqdm import tqdm
from util import dist2, kernel, median_distance, MMD

from util import kernel as rbf_kernel
# %%

import torch

def kde_log_density_gradient(
    x: torch.Tensor,
    y: torch.Tensor,
    bandwidth: float = 1.0,
    eps: float = 1e-12
) -> torch.Tensor:
    """
    Compute the gradient of the log of the kernel density estimate (log-KDE)
    at points `y`, given sample points `x`, using a Gaussian kernel with a
    specified bandwidth.
    
    Args:
        x (torch.Tensor): Shape (n, d), the sample points.
        y (torch.Tensor): Shape (m, d), the evaluation/query points.
        bandwidth (float): The bandwidth (sigma) for the Gaussian kernel.
        eps (float): A small constant to avoid division by zero.
    
    Returns:
        torch.Tensor: Shape (m, d), the gradients of the log-density at each
                      of the m points in y.
    """
    # x: (n, d)
    # y: (m, d)
    n, d = x.shape
    m = y.shape[0]
    
    # Compute pairwise differences: shape (m, n, d)
    diff = y.unsqueeze(1) - x.unsqueeze(0)
    
    # Squared distances: shape (m, n)
    sqdist = diff.pow(2).sum(dim=-1)
    
    # Gaussian kernel weights (no normalization factor needed; it cancels out in ratio)
    # shape (m, n)
    kernel = torch.exp(-0.5 * sqdist / (bandwidth**2))
    
    # Sum of kernel weights for each y_j, shape (m, 1)
    denom = kernel.sum(dim=1, keepdim=True) + eps  # add eps to avoid / zero
    
    # Weighted differences: shape (m, n, d)
    weighted_diff = diff * kernel.unsqueeze(-1)
    
    # Sum over n sample points to get the numerator, shape (m, d)
    numerator = weighted_diff.sum(dim=1)
    
    # Gradient of log density:
    #   = -1/sigma^2 * ( sum_i (y - x_i) * k(y, x_i) ) / sum_i k(y, x_i)
    grad_log_density = - (1.0 / (bandwidth**2)) * (numerator / denom)
    
    return grad_log_density

def rkl_wgf(x, y, xt, sigma):
    return kde_log_density_gradient(x, xt, sigma) - kde_log_density_gradient(y, xt, sigma)

def run_rkl_wgf(x1, xt, eta = 0.1, niter = 1000):
    traj = [xt.detach().cpu().numpy()]
    try:
        for i in tqdm(range(niter)):
            med = median_distance(xt)
            grad = rkl_wgf(x1, xt, xt, med)
            xt = xt + eta * grad
            traj.append(xt.detach().cpu().numpy())
            # print("mmd: ", MMD(xt, x1, med).item())
    except KeyboardInterrupt:
        print("Interrupted, will output the current xt")
        pass
    return xt, traj

def mmd_gradient_flow_update(x1, xt, sigma=1.0, dt=1e-2):
    """
    Perform one gradient flow update on the particle distribution xt so as to minimize MMD^2
    with respect to the target distribution x1.
    
    Args:
        xt: Tensor of shape (N, d) representing the current particle positions.
        x1: Tensor of shape (M, d) representing target samples.
        sigma: Bandwidth of the RBF kernel.
        dt: Time step (learning rate) for the gradient flow update.
    
    Returns:
        Updated particle positions xt (Tensor of shape (N, d)).
    """
    N = xt.shape[0]
    M = x1.shape[0]
    
    # --- Particle-particle interaction ---
    # Compute the kernel matrix among particles (shape: (N, N))
    K_xx = rbf_kernel(xt, xt, sigma)
    # Compute pairwise differences: for each i,j, diff[i,j] = x_i - x_j, shape: (N, N, d)
    diff_xx = xt.unsqueeze(1) - xt.unsqueeze(0)
    # Sum over the j index:
    grad_self = (K_xx.unsqueeze(-1) * diff_xx).sum(dim=1)
    # Scale factor from differentiation (note the 1/(N^2 sigma^2) factor):
    grad_self = grad_self / (N**2 * sigma**2)
    
    # --- Particle-target interaction ---
    # Compute the kernel matrix between particles and target samples (shape: (N, M))
    K_xy = rbf_kernel(xt, x1, sigma)
    # Compute differences: for each i,l, diff[i,l] = x_i - y_l, shape: (N, M, d)
    diff_xy = xt.unsqueeze(1) - x1.unsqueeze(0)
    grad_cross = (K_xy.unsqueeze(-1) * diff_xy).sum(dim=1)
    # Scale factor from differentiation (note the 2/(N*M sigma^2) factor):
    grad_cross = grad_cross * (2 / (N * M * sigma**2))
    
    # The gradient of MMD^2 with respect to xt is:
    #   grad = -grad_self + grad_cross
    grad = -grad_self + grad_cross
    
    return -grad 

    # # A gradient flow update is:
    # #   xt_new = xt - dt * grad(MMD^2)
    # xt_new = xt - dt * grad
    # return xt_new

def compute_mmd_energy(xt, x1):
    """
    Compute the (biased) MMD energy using the kernel k(x,y) = -||x-y||.
    
    The energy is defined as:
      E =  (1/N^2) ∑_{i,j} k(xt_i, xt_j)
         + (1/M^2) ∑_{i,j} k(x1_i, x1_j)
         - (2/(N*M)) ∑_{i,j} k(xt_i, x1_j).
         
    Since x1 is fixed, only the terms involving xt will contribute to the gradient.
    
    Args:
        xt: Tensor of shape (N, d) representing the particle positions.
        x1: Tensor of shape (M, d) representing the target samples.
    
    Returns:
        energy: A scalar tensor representing the MMD energy.
    """
    N, M = xt.shape[0], x1.shape[0]
    
    # Compute pairwise Euclidean distances using torch.cdist.
    # torch.cdist returns a matrix of pairwise distances.
    dist_xx = torch.cdist(xt, xt, p=2)  # shape: (N, N)
    dist_yy = torch.cdist(x1, x1, p=2)  # shape: (M, M)
    dist_xy = torch.cdist(xt, x1, p=2)  # shape: (N, M)
    
    # Apply the kernel: k(x,y) = -||x-y||
    K_xx = -dist_xx
    K_yy = -dist_yy
    K_xy = -dist_xy
    
    energy = K_xx.mean() + K_yy.mean() - 2 * K_xy.mean()
    return energy

def run_mmdgf(x1, xt, eta = 0.1, niter = 1000):
    traj = [xt.detach().cpu().numpy()]
    try:
        for i in tqdm(range(niter)):
            med = median_distance(xt)
            # grad = mmd_gradient_flow_update(x1, xt, med)
            # grad = compute_gradient(xt, x1)
            # autograd compute_mmd_energy
            xt_copy = xt.detach().clone()
            xt_copy.requires_grad = True
            grad = -torch.autograd.grad(compute_mmd_energy(xt_copy, x1), xt_copy)[0]
            xt = (xt + eta * grad).detach()
            
            traj.append(xt.detach().cpu().numpy())
            # print("mmd: ", MMD(xt, x1, med).item())
    except KeyboardInterrupt:
        print("Interrupted, will output the current xt")
        pass
    return xt, traj
# %%
if __name__ == "__main__":
    from matplotlib import pyplot as plt
    x1 = torch.randn(1000, 2) * .5 + 1
    xt = torch.randn(1000, 2) * 2

    med = median_distance(xt) * 1
    # print(run_mmdgf(x1, xt, med))

    xt, _ = run_mmdgf(x1, xt, 1000, 100)
    
    plt.scatter(x1[:, 0], x1[:, 1], label='x1')
    plt.scatter(xt[:, 0], xt[:, 1], label='xt')


# print(kde_log_density_gradient(xt, torch.ones(1, 2), med))
# plot the gradient of the kernel density estimator using quiver plot

# xmarks = torch.linspace(-3, 3, 10)
# ymarks = torch.linspace(-3, 3, 10)
# X, Y = torch.meshgrid(xmarks, ymarks)
# points = torch.vstack([X.flatten(), Y.flatten()]).T

# # grad1 = kde_grad(x1, points, med)
# # gradt = kde_grad(xt, points, med)
# # grad = grad1 - gradt
# grad = rkl_wgf(x1, xt, points, med)
# # reshape into 10 by 10 by 2 tensor
# grad = grad.reshape(10, 10, 2)

# import matplotlib.pyplot as plt
# plt.quiver(X, Y, grad[:, :, 0], grad[:, :, 1])
# plt.show()


# %%
