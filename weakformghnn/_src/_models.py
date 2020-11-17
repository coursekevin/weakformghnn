import torch
import torch.nn as nn
from torch.utils.data import Dataset
from ._vector_calc import curl
import math

__all__ = ['PosLinear',
           'ConvexFunc',
           'ConvexFuncZero',
           'ConcaveFunc',
           'ConcaveFuncZero',
           'ScalarFunc',
           'ScalarFuncZero',
           'ScalarFuncZeroPos',
           'ScalarFuncPosUnbnd',
           'ZeroDivMat',
           'GHNN',
           'HNN',
           'ODEFCN',
           'ReHU',
           'GHNNwHPrior']


class PosLinear(nn.Module):
    """ Linear layer with positive weights only
    """
    __constants__ = ['in_features', 'out_features']

    def __init__(self, in_features, out_features):
        super(PosLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.parameter.Parameter(
            torch.Tensor(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, input):
        return input @ torch.pow(self.weight, 2).T

    def extra_repr(self):
        return 'in_features={}, out_features={}'.format(
            self.in_features, self.out_features
        )


class ConvexFunc(nn.Module):
    def __init__(self, ndim, nhidden, nlayers):
        super(ConvexFunc, self).__init__()
        self.linears = nn.ModuleList([nn.Linear(ndim, nhidden)])
        self.poslinear = nn.ModuleList()
        hidden_lin_layers = [nn.Linear(ndim, nhidden)
                             for j in range(nlayers - 1)]
        hidden_pos_layers = [PosLinear(nhidden, nhidden)
                             for j in range(nlayers-1)]
        self.linears.extend(hidden_lin_layers)
        self.poslinear.extend(hidden_pos_layers)
        self.linears.extend([nn.Linear(ndim, 1)])
        self.poslinear.extend([PosLinear(nhidden, 1)])
        self.activation = nn.Softplus()
        self.ndim = ndim

    def forward(self, x):
        z = self.activation(self.linears[0](x))
        for j in range(len(self.poslinear)):
            z = self.activation(self.poslinear[j](z) + self.linears[j+1](x))
        return z


class ConvexFuncZero(ConvexFunc):
    def __init__(self, ndim, nhidden, nlayers):
        super(ConvexFuncZero, self).__init__(ndim, nhidden, nlayers)
        self.conv_forward = super(ConvexFuncZero, self).forward
        self.ndim
        self.register_buffer('zero_input', torch.zeros(1, ndim))

    def forward(self, x):
        return self.conv_forward(x) - self.conv_forward(self.zero_input)


class ConcaveFunc(ConvexFunc):
    def __init__(self, ndim, nhidden, nlayers):
        super(ConcaveFunc, self).__init__(ndim, nhidden, nlayers)

    def forward(self, x):
        conv_out = super(ConcaveFunc, self).forward(x)
        return -1*conv_out


class ConcaveFuncZero(ConcaveFunc):
    def __init__(self, ndim, nhidden, nlayers):
        super(ConcaveFuncZero, self).__init__(ndim, nhidden, nlayers)
        self.conc_forward = super(ConcaveFuncZero, self).forward
        self.ndim = ndim
        self.register_buffer('zero_input', torch.zeros(1, ndim))

    def forward(self, x):
        return self.conc_forward(x) - self.conc_forward(self.zero_input)


class ScalarFunc(nn.Module):
    def __init__(self, ndim, nhidden, nlayers):
        super(ScalarFunc, self).__init__()
        if ndim == 0:
            self.weight = nn.parameter.Parameter(torch.Tensor(1, 1))
            self._res_weight()
            self.mlp = lambda x: torch.ones(
                x.shape[0], device=x.device)*self.weight
        else:
            self.linears = nn.ModuleList([nn.Linear(ndim, nhidden)])
            hidden_layers = [nn.Linear(nhidden, nhidden)
                             for j in range(nlayers - 1)]
            self.linears.extend(hidden_layers)
            self.activation = nn.Softplus()
            self.out = nn.Linear(nhidden, 1)
            self.mlp = self._mlp_forward

    def _mlp_forward(self, x):
        for _, l in enumerate(self.linears):
            x = self.activation(l(x))
        return self.out(x)

    def forward(self, x):
        return self.mlp(x)

    def _res_weight(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))


class ScalarFuncZero(ScalarFunc):
    def __init__(self, ndim, nhidden, nlayers):
        super(ScalarFuncZero, self).__init__(ndim, nhidden, nlayers)
        self.ndim = ndim
        self.scalar = super(ScalarFuncZero, self).forward
        self.register_buffer('zero_input', torch.zeros(1, ndim))

    def forward(self, x):
        return self.scalar(x) - self.scalar(self.zero_input)


class ScalarFuncZeroPos(ScalarFunc):
    def __init__(self, ndim, nhidden, nlayers):
        super(ScalarFuncZeroPos, self).__init__(ndim, nhidden, nlayers)
        self.ndim = ndim
        self.scalar = super(ScalarFuncZeroPos, self).forward
        self.register_buffer('zero_input', torch.zeros(1, ndim))
        self.make_pos = ReHU(1.)

    def sc_zero_pos(self, x):
        return self.make_pos(self.scalar(x)-self.scalar(self.zero_input))
        # return torch.pow(self.scalar(x)-self.scalar(self.zero_input), 2)

    def forward(self, x):
        return self.sc_zero_pos(x) + 0.01*torch.pow(x, 2).sum(dim=-1).unsqueeze(-1)


class ScalarFuncPosUnbnd(ScalarFunc):
    def __init__(self, ndim, nhidden, nlayers):
        super(ScalarFuncPosUnbnd, self).__init__(ndim, nhidden, nlayers)
        self.ndim = ndim
        self.scalar = super(ScalarFuncPosUnbnd, self).forward
        self.register_buffer('zero_input', torch.zeros(1, ndim))
        self.make_pos = nn.Softplus()

    def sc_zero_pos(self, x):
        return self.make_pos(self.scalar(x))

    def forward(self, x):
        return self.sc_zero_pos(x) + 0.01*torch.pow(x, 2).sum(dim=-1).unsqueeze(-1) - self.sc_zero_pos(self.zero_input)


class ZeroDivMat(nn.Module):
    """ Class for constructing a J matrix for generalized Hamiltonian
        TODO: write in terms of masked NN rather than a collection of NNs
    """

    def __init__(self, ndim, nhidden, nlayers):
        super(ZeroDivMat, self).__init__()
        self.ndim = ndim
        self.J = nn.ModuleDict()
        self.idx_list = []
        self.rel_terms = []
        for j in range(ndim):
            for i in range(ndim):
                if i >= j:
                    continue
                else:
                    self.rel_terms.append([ind for ind in range(
                        self.ndim) if ind != i and ind != j])
                    self.idx_list.append((i, j))
                    self.J.update(
                        [['{},{}'.format(i, j), ScalarFunc(ndim-2, nhidden, nlayers)]])

    def forward(self, x):
        J_ret = torch.zeros(x.shape[0], self.ndim, self.ndim, device=x.device)
        for ind, (i, j) in enumerate(self.idx_list):
            J_ret[:, i, j] = self.J['{},{}'.format(i, j)](
                x[:, self.rel_terms[ind]]).view(-1,)
        return J_ret - J_ret.transpose(2, 1)


class GHNN(nn.Module):
    """ Main generalized Hamiltonian neural nets class

        INPUTS
            ndim < int > : number of input dimensions
            nhidden < int > : number of hidden units
            nlayers < int > : number of hidden layers
            prior < dict > : {
                                'H': choices = ['H0', 'H1', None], default = 'H0',
                                'dHdt': choices = ['decreasing', 'constant', None], default = None
                             }
        Desciption of prior information dictionary:
            H (strong priors on the form of the Hamiltonian)
                'H0': H(x) -> infty as x -> infty, H(x) + H(0) >= 0, H(0) = 0. 
                    - use if energy is bounded locally (ie. damped duffing oscilator)
                    - makes sense in most cases
                'H1': H(x) -> infty as x -> infty, H(0) = 0, H(x) > 0 forall x != 0 
                    - use if globally stable at x=0 
            dHdt (strong priors on the energy flux rate)
                'decreasing' : dHdt(x) < 0 along trajectories following dxdt = f(x)
                'constant' : dHdt(x) = 0 along trajectories following dxdt = f(x)

        Regularization shemes: 
            The forward function takes in an optional argument reg_schemes. This tuple
            should contain the names of regularization schemes you would like to use in training.

            The choices for regularization schemes are:
                - curl (use when performing curl regularization)
                - dhdt (use when imposing a soft energy flux rate prior) 

        NOTE: for known gen. Hamiltonian use GHNNwHPrior class
    """

    def __init__(self, ndim, nhidden, nlayers, prior={'H': 'H0'}):
        super(GHNN, self).__init__()
        self.H_prior = prior.get('H', None)
        self.dHdt_prior = prior.get('dHdt', None)

        if self.H_prior == 'H0':
            self.H = ScalarFuncPosUnbnd(ndim, nhidden, nlayers)
        elif self.H_prior == 'H1':
            self.H = ScalarFuncZeroPos(ndim, nhidden, nlayers)
        else:
            self.H = ScalarFuncZero(ndim, nhidden, nlayers)

        self.J = ZeroDivMat(
            ndim, max(1, int(nhidden/(ndim*(ndim-1)/2))), nlayers)
        self.ndim = ndim

        self.hh_strict = False
        self.conservative = False
        if self.dHdt_prior == 'decreasing':
            self.v = ConcaveFuncZero(ndim, nhidden, nlayers)
        elif self.dHdt_prior == 'constant':
            self.conservative = True
        else:
            self.hh_strict = True
            self.fd = ScalarFuncZero(ndim, nhidden, nlayers)

        tri_l_ind = torch.ones(ndim, ndim).tril(-1) == 1
        self.register_buffer('tri_l_ind', tri_l_ind)

    def forward(self, t, x, reg_schemes=()):
        in_shape = x.shape
        x = x.reshape(-1, self.ndim).clone()  # note: clone likely unecessary
        with torch.enable_grad():
            if not x.requires_grad:
                x.requires_grad = True
            grad_H, J_grad_H = self.conservative_forward(x)

            if self.conservative:
                grad_v, R_grad_H = torch.zeros(
                    grad_H.shape), torch.zeros(J_grad_H.shape)
            else:
                grad_v, R_grad_H = self.nonconservative_forward(x, grad_H)
            dxdt = J_grad_H + R_grad_H
            dxdt = dxdt.reshape(in_shape)
            if not reg_schemes:
                out = dxdt
            else:
                reg_vals = []
                for reg in reg_schemes:
                    if reg == 'dhdt':
                        reg_vals.append(torch.einsum(
                            'ij,ij->i', grad_H, R_grad_H).reshape(in_shape[:-1]))
                    elif reg == 'curl':
                        reg_vals.append(curl(R_grad_H, x))
                        # reg_vals.append(self._curl_R_grad_H(x, grad_H, grad_v))
                    else:
                        print('Reg. scheme: {} not recognized'.format(reg))
                out = (dxdt, reg_vals)
        return out

    def forward_odeint(self, t, x):
        """ note: only kept around for compatibility with ghnn v0.0.2 please use forward()
            forward method for ode integration tools
        """
        if not x.requires_grad:
            x.requires_grad = True
        out = self.forward(t, x)
        return out.detach()

    def conservative_forward(self, x):
        grad_H = torch.autograd.grad(self.H(x).sum(), x, create_graph=True)[0]
        J_grad_H = torch.einsum('ijk,ik->ij', self.J(x), grad_H)
        return grad_H, J_grad_H

    def nonconservative_forward(self, x, grad_H):
        if self.hh_strict:
            div_f = torch.autograd.grad(
                self.fd(x).sum(), x, create_graph=True)[0]
            R_grad_H = div_f
            grad_v = None
        elif self.conservative:
            grad_v = torch.zeros(grad_H.shape, requires_grad=True)*x
            R_grad_H = torch.zeros(grad_H.shape, requires_grad=True)*x
        else:
            grad_v = torch.autograd.grad(
                self.v(x).sum(), x, create_graph=True)[0]
            R_grad_H = torch.autograd.grad(
                grad_v, x, create_graph=True, grad_outputs=grad_H)[0]
        return grad_v, R_grad_H

    def _curl_R_grad_H(self, x, grad_H, grad_v):
        """ Note: needs serious optimization
        """
        if self.hh_strict:
            print('Warning, R*gradH is is stable by construction.')
            return torch.zeros(x.shape, device=x.device)
        else:
            hess_H = self.hessian(grad_H, x)
            hess_v = self.hessian(grad_v, x)
            VtrH = torch.einsum('ijk,ikl->ijl', hess_H, hess_v)
            curl_mat = VtrH - VtrH.transpose(1, 2)
            return curl_mat[:, self.tri_l_ind]

    def hessian(self, grad_f, x):
        hess = []
        for j in range(x.shape[1]):
            hess.append(torch.autograd.grad(
                grad_f[:, j].sum(), x, create_graph=True)[0])
        return torch.stack(hess, dim=2)

    def _set_requires_grad_true(self, x):
        if not x.requires_grad:
            x.requires_grad = True

    def _set_requires_grad_false(self, x):
        if x.requires_grad:
            x.requires_grad = False


class GHNNwHPrior(nn.Module):
    """     
        INPUTS
            ndim < int > : number of input dimensions
            nhidden < int > : number of hidden units
            nlayers < int > : number of hidden layers
            H < function (tensor) -> (tensor) > : known gen. Hamiltonian

        Example:
            model = GHNNwHPrior(2, 200, 3, H)
            (training model)
            x = torch.randn(1,2)
            # get value of derivative at x
            dxdt = model(0., x) 
            # get J at x
            J = model.J()
            # get R at x
            R = model.R()
    """

    def __init__(self, n_dim, n_hidden, n_layers, H):
        super(GHNNwHPrior, self).__init__()
        self.n_dim = n_dim
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.H = H
        # setting up linear layers
        layers = [nn.Linear(n_dim, n_hidden), ]
        for j in range(n_layers-1):
            layers.append(nn.Softplus())
            layers.append(nn.Linear(n_hidden, n_hidden))
        layers.append(nn.Softplus())
        num_out = int(n_dim**2)
        layers.append(nn.Linear(n_hidden, num_out))
        self.mlp = nn.Sequential(*layers)
        # setting up W matrix
        # gives full array indices lol
        self.ind = torch.tril_indices(n_dim, n_dim, offset=n_dim)
        self.register_buffer('W', torch.zeros(n_dim, n_dim))
        self.W_tmp = None

    def forward(self, t, input):
        in_shape = input.shape
        x = input.reshape(-1, in_shape[-1])
        W = self.W.repeat(x.shape[0], 1, 1)
        W[:, self.ind[0], self.ind[1]] = self.mlp(x)
        self.W_tmp = W.detach().clone()
        with torch.enable_grad():
            if not x.requires_grad:
                x.requires_grad = True
            grad_H = torch.autograd.grad(
                self.H(x).sum(), x, create_graph=True)[0]
            out = (W @ grad_H.unsqueeze(-1)).squeeze(-1)
        return out.reshape(in_shape)

    def J(self):
        return 0.5*(self.W_tmp - self.W_tmp.transpose(2, 1))

    def R(self):
        return 0.5*(self.W_tmp + self.W_tmp.transpose(2, 1))


class HNN(nn.Module):
    "Assumes ndim = 2n where n is the dimension of q"

    def __init__(self, ndim, nhidden, nlayers):
        super(HNN, self).__init__()
        self.H = ScalarFuncZero(ndim, nhidden, nlayers)
        self.ndim = ndim
        n = int(ndim/2)
        top = torch.cat([torch.zeros(n, n), torch.eye(n)], dim=1)
        bot = torch.cat([-torch.eye(n), torch.zeros(n, n)], dim=1)
        self.register_buffer('J', torch.cat([top, bot], dim=0))

    def forward(self, t, x):
        # assumes x is the form [[q1, q2, ..., qn, p1, p2, ..., pn], ...]
        in_shape = x.shape
        x = x.view(-1, self.ndim).clone()
        with torch.enable_grad():
            if not x.requires_grad:
                x.requires_grad = True
            H = self.H(x)
            gradH = torch.autograd.grad(H.sum(), x, create_graph=True)[0]
            dxdt = gradH @ self.J.T

        return dxdt.reshape(in_shape)


class ODEFCN(nn.Module):
    def __init__(self, ndim, nhidden, nlayers):
        super(ODEFCN, self).__init__()
        self.linears = nn.ModuleList([nn.Linear(ndim, nhidden)])
        hidden_layers = [nn.Linear(nhidden, nhidden)
                         for j in range(nlayers - 1)]
        self.linears.extend(hidden_layers)
        self.activation = nn.Softplus()
        self.out = nn.Linear(nhidden, ndim)
        self.mlp = self._mlp_forward

    def _mlp_forward(self, x):
        for _, l in enumerate(self.linears):
            x = self.activation(l(x))
        return self.out(x)

    def forward(self, t, x):
        return self.mlp(x)

    def H(self, x):
        return 0.*x.sum(dim=-1)


class ReHU(nn.Module):
    """ Rectified Huber unit
        from: https://github.com/locuslab/stable_dynamics/blob/master/models/stabledynamics.py
    """

    def __init__(self, d):
        super(ReHU, self).__init__()
        self.a = 1/d
        self.b = -d/2

    def forward(self, x):
        return torch.max(torch.clamp(torch.sign(x)*self.a/2*x**2, min=0, max=-self.b), x+self.b)
