# %%
import torch

def gaussian_rbf_second_deriv_xy_loop(x, y, sigma):
    """
    For-loop version of nabla_x nabla_y k(x,y) for testing/validation.
    This code explicitly loops over n1 and n2.
    
    Parameters
    ----------
    x : torch.Tensor, shape (n1, d)
    y : torch.Tensor, shape (n2, d)
    sigma : float
    
    Returns
    -------
    second_deriv : torch.Tensor, shape (d, d, n1, n2)
        The (d x d) matrix of second partial derivatives for each pair (x_i, y_j).
    """
    n1, d = x.shape
    n2 = y.shape[0]
    
    # Prepare an output tensor of shape (d, d, n1, n2)
    second_deriv_xy = torch.zeros(n1, n2, d, d, dtype=x.dtype, device=x.device)
    
    # Precompute identity matrix
    I = torch.eye(d, dtype=x.dtype, device=x.device)
    
    # Loop over all (i, j)
    for i in range(n1):
        for j in range(n2):
            # diff = (x_i - y_j), shape (d,)
            diff_ij = x[i] - y[j]  # shape (d,)
            
            # dist_sq = ||x_i - y_j||^2
            dist_sq = diff_ij.pow(2).sum()
            
            # Gaussian kernel scalar
            k_ij = torch.exp(-0.5 * dist_sq / (sigma**2))
            
            # Outer product diff_ij (d x 1) * diff_ij (1 x d) => (d x d)
            outer_ij = diff_ij.unsqueeze(-1) * diff_ij.unsqueeze(-2)  # shape (d, d)
            
            # The matrix inside brackets:
            #   (1/sigma^2)*I - (1/sigma^4)*(diff_ij)(diff_ij)^T
            bracket = (1.0 / sigma**2) * I - (1.0 / sigma**4) * outer_ij
            
            # Multiply by k_ij
            second_deriv_xy[i, j, :, :] = k_ij * bracket
    
    return second_deriv_xy

def gaussian_rbf_second_deriv_xy(x, y, sigma):
    """
    Compute nabla_x nabla_y k(x, y) for the Gaussian (RBF) kernel
        k(x,y) = exp(-||x - y||^2 / (2 sigma^2))
    for all pairs (x_i, y_j).

    Parameters
    ----------
    x : torch.Tensor, shape (n1, d)
    y : torch.Tensor, shape (n2, d)
    sigma : float

    Returns
    -------
    second_deriv : torch.Tensor, shape (d, d, n1, n2)
        The (d x d) matrix of second partial derivatives for each pair (x_i, y_j).
    """

    # Shapes:
    #   x: [n1, d]
    #   y: [n2, d]

    # 1) Compute the pairwise differences (x_i - y_j)
    #    result shape: [n1, n2, d]
    diff = x[:, None, :] - y[None, :, :]

    # 2) Compute pairwise squared distances
    #    result shape: [n1, n2]
    dist_sq = diff.pow(2).sum(dim=-1)

    # 3) Compute the Gaussian kernel for each pair: k(x_i, y_j)
    #    result shape: [n1, n2]
    K = torch.exp(-0.5 * dist_sq / sigma**2)

    # 4) Build the outer product (x_i - y_j)(x_i - y_j)^T for each pair
    #    result shape: [n1, n2, d, d]
    outer_diff = diff.unsqueeze(-1) * diff.unsqueeze(-2)

    # 5) Construct the [n1, n2, d, d] expression for
    #      [1/sigma^2 * I - 1/sigma^4 * (x_i - y_j)(x_i - y_j)^T].
    #    First, get an identity matrix of shape [d, d] and broadcast it:
    I = torch.eye(x.shape[-1], device=x.device, dtype=x.dtype)  # shape (d, d)
    I_broadcast = I.view(1, 1, x.shape[-1], x.shape[-1])        # shape (1, 1, d, d)

    bracket_term = (1.0 / sigma**2) * I_broadcast - (1.0 / sigma**4) * outer_diff

    # 6) Multiply by the scalar kernel K, broadcasting over [n1, n2] -> [n1, n2, 1, 1]
    K_expanded = K.unsqueeze(-1).unsqueeze(-1)  # shape (n1, n2, 1, 1)

    # 7) Final shape wanted: (d, d, n1, n2).
    #    Currently bracket_term is (n1, n2, d, d).
    #    Multiply and then permute:
    second_deriv_xy = K_expanded * bracket_term  # shape (n1, n2, d, d)
    # second_deriv_xy = second_deriv_xy.permute(2, 3, 0, 1)  # shape (d, d, n1, n2)

    return second_deriv_xy

# %%
# ---------------------------------------------------------------------
# Example usage
if __name__ == "__main__":
    n1, n2, d = 5, 5, 2
    sigma = 2.0

    torch.manual_seed(0)
    # Random example data
    y = torch.randn(n1, d)
    x = torch.randn(n2, d)

    # Compute the second partial derivatives
    result = gaussian_rbf_second_deriv_xy(x, y, sigma)
    result2 = gaussian_rbf_second_deriv_xy_loop(x, y, sigma)
    
    # Check if the results are close
    print("Max difference:", (result - result2).abs().max().item())
    if (result - result2).abs().max() > 1e-6:
        print("impementation wrong!!!!")

    print("Result shape:", result.shape)  # should be (d, d, n1, n2)
    # For example, check one element:
    print("Example [0,0,0,0]:", result[0, 0, 0, 0])
# %%
