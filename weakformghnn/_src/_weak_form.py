import torch

__all__ = ['gauss_rbf',
           'poly_bf',
           'weak_form_loss']


def gauss_rbf(t, c, eps, minimum=0.):
    """ Returns the gaussian rbf for a collection of points t
        INPUTS:
            t < torch.Tensor (m,) > : inputs to gaussian rbfs
            c < torch.Tensor (M,) > : centers of rbfs (where M is the number of GRBFS)
            eps < float > : shape paramter of gaussian radial basis functions
            minimum < float > : minimum grbf function value
        RETURNS:
            gauss_rbfs < torch.Tensor(m,M) > : grbfs evaluated at t's
            gauss_rbf_derivs < torch.Tensor(m,M) > grbf derivs evaluted at ts
    """
    diff = t.view(-1, 1) - c.view(1, -1)  # m x M
    grbf = torch.exp(-eps**2 * torch.pow(diff, 2)) + minimum
    grbf_deriv = torch.mul(-2*eps**2*diff, grbf)
    return grbf, grbf_deriv


def poly_bf(t, c, deg):
    """ Returns polynomial basis funcs and their derivatives centered at c evaluated at points t
        INPUTS:
            t < torch.Tensor (m,) > : inputs to gaussian rbfs
            c < torch.Tensor (M,) > : centers of rbfs (where M is the number of GRBFS)
            deg < int > : polynomial bf degree (note should be < ~ 10)
        RETURNS:
            poly_bfs < torch.Tensor(m,M) > : poly basis functions evaluated at t's (M = c(deg + 1))
            poly_bf_derivs < torch.Tensor(m,M) > : poly basis function derivatives evaluated at times
    """
    diff = t.view(-1, 1) - c.view(1, -1)
    poly = torch.cat([torch.pow(diff, j) for j in range(deg+1)], dim=1)
    poly_deriv = torch.cat(
        [torch.ones(diff.shape)*j for j in range(deg+1)], dim=1)
    poly_deriv[:, 2*diff.shape[1]:] = poly_deriv[:, 2*diff.shape[1]:] * \
        poly[:, diff.shape[1]:-diff.shape[1]]
    return poly, poly_deriv


def weak_form_loss(dx_est, x, t, psi, psi_dot):
    """ Returns the weak-form ode model loss
        INPUTS: 
            dx_est < torch.Tensor, (Bs, m, ndim) > : estimate for derivative
            x < torch.Tensor, (Bs, m, ndim) > : state training point
            t < torch.Tensor, (m) > integration times
            psi < torch.Tensor, (m,M) > : test functions evaluated at measurment times
            psi_dot < torch.Tensor, (m,M) > : test function derivatives evaluated at measurement times
        OUTPUTS:
            weak_form_loss < torch.float > : weak-form squared loss
    """
    # boundary term
    RH_bound = torch.einsum('in,m->inm', x[:, -1, :], psi[-1])
    LH_bound = torch.einsum('in,m->inm', x[:, 0, :], psi[0])
    B = RH_bound - LH_bound

    x_psi_dot = torch.einsum('imn,ml->imnl', x, psi_dot)
    f_psi = torch.einsum('imn,ml->imnl', dx_est, psi)

    L = torch.trapz(f_psi + x_psi_dot, t, dim=1) - B
    return L.pow(2).sum(-1).mean()
