# %%
import torch
import torch.nn as nn
from torch.func import functional_call, vmap, vjp, jvp, jacrev
device = 'cuda' if torch.cuda.device_count() > 0 else 'cpu'

class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(3, 111)
        self.fc2 = nn.Linear(111, 2)
    
    def forward(self, x):
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x
                                            
x_train = torch.randn(200, 3, device=device)
x_test = torch.randn(200, 3, device=device)


# %%

net = SimpleNet().to(device)

# Detaching the parameters because we won't be calling Tensor.backward().
params = {k: v.detach() for k, v in net.named_parameters()}

def fnet_single(params, x):
    return functional_call(net, params, (x.unsqueeze(0),)).squeeze(0)

def empirical_ntk_jacobian_contraction(fnet_single, params, x1, x2, compute='full'):
    # Compute J(x1)
    jac1 = vmap(jacrev(fnet_single), (None, 0))(params, x1)
    jac1 = jac1.values()
    jac1 = [j.flatten(2) for j in jac1]

    # Compute J(x2)
    jac2 = vmap(jacrev(fnet_single), (None, 0))(params, x2)
    jac2 = jac2.values()
    jac2 = [j.flatten(2) for j in jac2]

    # Compute J(x1) @ J(x2).T
    einsum_expr = None
    if compute == 'full':
        einsum_expr = 'Naf,Mbf->NMab'
    elif compute == 'trace':
        einsum_expr = 'Naf,Maf->NM'
    elif compute == 'diagonal':
        einsum_expr = 'Naf,Maf->NMa'
    else:
        assert False

    result = torch.stack([torch.einsum(einsum_expr, j1, j2) for j1, j2 in zip(jac1, jac2)])
    result = result.sum(0)
    return result

# result = empirical_ntk_jacobian_contraction(fnet_single, params, x_train, x_test, 'trace')
# print(result.shape)

# def empirical_ntk_ntk_vps(func, params, x1, x2, compute='full'):
#     def get_ntk(x1, x2):
#         def func_x1(params):
#             return func(params, x1)

#         def func_x2(params):
#             return func(params, x2)

#         output, vjp_fn = vjp(func_x1, params)

#         def get_ntk_slice(vec):
#             # This computes ``vec @ J(x2).T``
#             # `vec` is some unit vector (a single slice of the Identity matrix)
#             vjps = vjp_fn(vec)
#             # This computes ``J(X1) @ vjps``
#             _, jvps = jvp(func_x2, (params,), vjps)
#             return jvps

#         # Here's our identity matrix
#         basis = torch.eye(output.numel(), dtype=output.dtype, device=output.device).view(output.numel(), -1)
#         return vmap(get_ntk_slice)(basis)

#     # ``get_ntk(x1, x2)`` computes the NTK for a single data point x1, x2
#     # Since the x1, x2 inputs to ``empirical_ntk_ntk_vps`` are batched,
#     # we actually wish to compute the NTK between every pair of data points
#     # between {x1} and {x2}. That's what the ``vmaps`` here do.
#     result = vmap(vmap(get_ntk, (None, 0)), (0, None))(x1, x2)

#     if compute == 'full':
#         return result
#     if compute == 'trace':
#         return torch.einsum('NMKK->NM', result)
#     if compute == 'diagonal':
#         return torch.einsum('NMKK->NMK', result)

# # Disable TensorFloat-32 for convolutions on Ampere+ GPUs to sacrifice performance in favor of accuracy
# with torch.backends.cudnn.flags(allow_tf32=False):
#     result_from_jacobian_contraction = empirical_ntk_jacobian_contraction(fnet_single, params, x_test, x_train)
#     result_from_ntk_vps = empirical_ntk_ntk_vps(fnet_single, params, x_test, x_train)

# assert torch.allclose(result_from_jacobian_contraction, result_from_ntk_vps, atol=1e-5)


# def empirical_ntk_jacobian_contraction(fnet_single, params, x1, x2):
#     # Compute J(x1)
#     jac1 = vmap(jacrev(fnet_single), (None, 0))(params, x1)
#     jac1 = jac1.values()
#     jac1 = [j.flatten(2) for j in jac1]

#     # Compute J(x2)
#     jac2 = vmap(jacrev(fnet_single), (None, 0))(params, x2)
#     jac2 = jac2.values()
#     jac2 = [j.flatten(2) for j in jac2]

#     # Compute J(x1) @ J(x2).T
#     result = torch.stack([torch.einsum('Naf,Mbf->NMab', j1, j2) for j1, j2 in zip(jac1, jac2)])
#     result = result.sum(0)
#     return result

result = empirical_ntk_jacobian_contraction(fnet_single, params, x_train, x_test)
print(result.shape)

# %%
