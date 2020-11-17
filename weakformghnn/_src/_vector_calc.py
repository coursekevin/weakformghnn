import torch

__all__ = ['divergence', 'curl']


def divergence(f, x):
    """ Computes the divergence for a function f at points x
        INPUTS:
            f < tensor > : vector function values (mxn)
            x < tensor > : (mxn) input vector 
        OUTPUTS
            divergence < tensor > : (m,) 
    """
    div = []
    for j in range(x.shape[1]):
        grad_f = torch.autograd.grad(
            f[:, j].sum(), x, create_graph=True, allow_unused=True)[0]
        div.append(grad_f[:, j])
    return torch.stack(div).sum(dim=0)


def curl(f, x):
    """ Computes the curl for a function f at points x
        INPUTS:
            f < tensor > : vector function values (mxn)
            x < tensor > : mxn input vector 

        OUTPUTS:
            curl < tensor > : mx(n*(n-1)/2)
    """
    N = x.shape[1]
    grad_f_array = []
    for i in range(N):
        grad_f = torch.autograd.grad(
            f[:, i].sum(), x, allow_unused=True, create_graph=True)[0]
        grad_f_array.append(grad_f)
    cu = []
    for i in range(N):
        for j in range(N):
            if i >= j:
                continue
            else:
                c_ij = grad_f_array[j][:, i] - grad_f_array[i][:, j]
                cu.append(c_ij)
    return torch.stack(cu, dim=1)
