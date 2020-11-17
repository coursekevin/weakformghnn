"""
Unit testing for generalized Hamiltonian learning
"""
from weakformghnn import *
import torch
import torch.nn as nn
import pytest
import numpy as np
import pickle
from distutils import dir_util
import os
from torchdiffeq import odeint
import matplotlib.pyplot as plt

# Constants
NDIM = 3
NHIDDEN = 50
NLAYERS = 3


@pytest.fixture
def GHNN_model():
    return GHNN(NDIM, NHIDDEN, NLAYERS)


@pytest.fixture
def convex_concave_data():
    t = torch.linspace(0., 10., 100)
    y_conv = torch.pow(t, 2)+10
    y_conc = -y_conv
    loss = torch.nn.MSELoss()
    return (t.view(-1, 1), y_conv.view(-1, 1), y_conc.view(-1, 1), loss)


@pytest.fixture
def ivp_integration_data(datadir):
    y0 = torch.tensor([np.pi - np.pi/32, 0.0], requires_grad=True)
    t_eval = torch.linspace(0., 20., 2000)
    data_path = datadir.join('pendulum_test_data.pkl')
    with open(data_path, 'rb') as handle:
        y = torch.tensor(pickle.load(handle))
    return (y0, t_eval, y)


@pytest.fixture
def vector_calc_data():
    def f(x):
        return torch.stack([2*x[:, 0]**3 + x[:, 1],
                            x[:, 0]*torch.sin(x[:, 1]) + x[:, 2]**2, x[:, 2]**4], dim=1)

    def div_f(x):
        return 6*x[:, 0]**2 + x[:, 0]*torch.cos(x[:, 1]) + 4*x[:, 2]**3

    def curl_f(x):
        return torch.stack([torch.sin(x[:, 1])-1., torch.zeros(x[:, 0].shape), -2*x[:, 2]], dim=1)

    X = torch.randn(20, 3, requires_grad=True)
    f_val = f(X)
    div_val = div_f(X)
    curl_val = curl_f(X)

    return (f_val, div_val, curl_val, X)
# Tests


def test_PosLinear():
    pos_func = PosLinear(5, 1)
    x = torch.pow(torch.randn(20, 5), 2)
    np.testing.assert_array_less(torch.zeros(
        20, 1), pos_func(x).detach().numpy())
    np.testing.assert_array_less(
        pos_func(-x).detach().numpy(), torch.zeros(20, 1))


def test_ScalarFuncZero():
    zero_func = ScalarFuncZero(NDIM, NHIDDEN, NLAYERS)
    x = torch.zeros(10, 20, NDIM)
    np.testing.assert_almost_equal(torch.zeros(
        10, 20, 1), zero_func(x).detach().numpy())


def test_ScalarFuncPos():
    zero_func_pos = ScalarFuncZeroPos(NDIM, NHIDDEN, NLAYERS)
    x = 10.*torch.randn(10, 20, NDIM)-5.
    np.testing.assert_array_less(torch.zeros(
        10, 20, 1), zero_func_pos(x).detach().numpy())

    z = torch.zeros(10, NDIM)
    np.testing.assert_array_almost_equal(torch.zeros(
        10, 1).numpy(), zero_func_pos(z).detach().numpy())


def test_ScalarFunc():
    scalar_func = ScalarFunc(0, NHIDDEN, NLAYERS)
    out1 = scalar_func(torch.randn(10, 0))
    out2 = scalar_func(torch.randn(10, 0))
    np.testing.assert_array_equal(out1.detach().numpy(), out2.detach().numpy())


def test_ZeroDivMat():
    J = ZeroDivMat(NDIM, NHIDDEN, NLAYERS)
    x = torch.randn(10, NDIM)
    np.testing.assert_array_equal((10, NDIM, NDIM), J(x).shape)

    J = ZeroDivMat(2, NHIDDEN, NLAYERS)
    x = torch.randn(10, 2)
    J_out = J(x)
    np.testing.assert_array_equal(
        J_out[0].detach().numpy(), J_out[1].detach().numpy())


def test_GHNN(GHNN_model):
    # note this function just tests that number of computed outputs is correct
    model = GHNN_model
    x = torch.randn(20, NDIM, requires_grad=True)
    np.testing.assert_array_equal((20, NDIM), model(0., x).shape)

    # test additional dimensions are handled correctly
    x = torch.randn(3, 5, 10, NDIM)
    x_shape = x.shape
    out_loop = []
    for i in range(x.shape[0]):
        out_tmp = []
        for j in range(x.shape[1]):
            out_tmp.append(model(0., x[i, j, :, :]))
        out_loop.append(torch.stack(out_tmp))
    out_loop = torch.stack(out_loop)
    np.testing.assert_array_almost_equal(
        out_loop.detach().numpy(), model(0., x).detach().numpy())


def test_hessian(GHNN_model):
    model = GHNN_model

    def gr_f(xin):
        x = xin[:, 0]
        y = xin[:, 1]
        z = xin[:, 2]
        return torch.stack([2*x*y + z, torch.pow(x, 2), x + 3*torch.pow(z, 2)]).T

    def h(xin):
        x = xin[:, 0]
        y = xin[:, 1]
        z = xin[:, 2]
        hess = []
        for j in range(xin.shape[0]):
            hess.append(torch.tensor(
                [[2*y[j], 2*x[j], 1.], [2*x[j], 0., 0.], [1., 0., 6*z[j]]]))
        return torch.stack(hess)
    x = torch.randn(20, NDIM, requires_grad=True)

    np.testing.assert_array_almost_equal(
        model.hessian(gr_f(x), x).detach().numpy(), h(x))


def test_ConvexFunc(convex_concave_data):
    t, yconv, yconc, loss = convex_concave_data
    conv_func1 = ConvexFunc(1, 30, 3)
    conv_func2 = ConvexFunc(1, 30, 3)
    optimizer1 = torch.optim.Adam(conv_func1.parameters(), lr=1e-1)
    optimizer2 = torch.optim.Adam(conv_func2.parameters(), lr=1e-1)
    for _ in range(500):
        L1 = loss(conv_func1(t), yconv)
        L1.backward()
        optimizer1.step()
        optimizer1.zero_grad()
        L2 = loss(conv_func2(t), yconc)
        L2.backward()
        optimizer2.step()
        optimizer2.zero_grad()
    assert L1 < 0.5
    assert L2 > 1000


def test_ConcaveFunc(convex_concave_data):
    t, yconv, yconc, loss = convex_concave_data
    conc_func1 = ConcaveFunc(1, 30, 3)
    conc_func2 = ConcaveFunc(1, 30, 3)
    optimizer1 = torch.optim.Adam(conc_func1.parameters(), lr=1e-1)
    optimizer2 = torch.optim.Adam(conc_func2.parameters(), lr=1e-1)
    for _ in range(500):
        L1 = loss(conc_func1(t), yconv)
        L1.backward()
        optimizer1.step()
        optimizer1.zero_grad()
        L2 = loss(conc_func2(t), yconc)
        L2.backward()
        optimizer2.step()
        optimizer2.zero_grad()
    assert L2 < 0.5
    assert L1 > 1000


def test_ConcaveZero(convex_concave_data):
    t, _, yconc, loss = convex_concave_data
    conv_zero = ConcaveFuncZero(1, 30, 3)
    optimizer = torch.optim.Adam(conv_zero.parameters(), lr=1e-1)
    for _ in range(500):
        L = loss(conv_zero(t), yconc)
        L.backward()
        optimizer.step()
        optimizer.zero_grad()
    x_in = torch.zeros(20, 1)
    np.testing.assert_array_almost_equal(
        x_in, conv_zero(x_in).detach().numpy(), decimal=4)


def test_gauss_rbf():
    t = torch.tensor([1., 2., 3.])
    c = torch.tensor([4., 5.])
    eps = 0.3

    t.requires_grad = True

    grbf, grbf_deriv = gauss_rbf(t, c, eps)

    true_grbf = torch.tensor([[0.4449, 0.2369],
                              [0.6977, 0.4449],
                              [0.9139, 0.6977]])

    true_grad_grbf = []
    for i in range(grbf.shape[1]):
        true_grad_grbf.append(torch.autograd.grad(
            grbf[:, i].sum(), t, retain_graph=True)[0].reshape(-1, 1))
    true_grad_grbf = torch.cat(true_grad_grbf, dim=1)

    # testing grbf val
    np.testing.assert_array_almost_equal(
        grbf.detach().numpy(), true_grbf.numpy(), decimal=4)
    # testing grbf time deriv
    np.testing.assert_array_almost_equal(
        grbf_deriv.detach().numpy(), true_grad_grbf.detach().numpy(), decimal=4)


def test_poly_bf():
    deg = 3
    t = torch.tensor([1., 2., 3.])
    c = torch.tensor([4., 5.])

    t.requires_grad = True
    poly, poly_deriv = poly_bf(t, c, deg)

    expected = torch.tensor([[1.,   1.,  -3.,  -4.,   9.,  16., -27., -64.],
                             [1.,   1.,  -2.,  -3.,   4.,   9.,  -8., -27.],
                             [1.,   1.,  -1.,  -2.,   1.,   4.,  -1.,  -8.]])
    grad_expected = []
    for i in range(poly.shape[1]):
        grad_expected.append(torch.autograd.grad(
            poly[:, i].sum(), t, retain_graph=True)[0].reshape(-1, 1))
    grad_expected = torch.cat(grad_expected, dim=1)

    # testing poly bf val
    np.testing.assert_almost_equal(poly.detach().numpy(), expected.numpy())
    # testing poly bf time deriv.
    np.testing.assert_almost_equal(
        poly_deriv.detach().numpy(), grad_expected.detach().numpy())


def test_divergence_calc(vector_calc_data):
    f_val, div_val, _, X = vector_calc_data

    div_calc = divergence(f_val, X)

    np.testing.assert_array_equal(
        div_calc.detach().numpy(), div_val.detach().numpy())


def test_curl_calc(vector_calc_data):
    f_val, _, curl_val, X = vector_calc_data

    curl_calc = curl(f_val, X)
    np.testing.assert_array_equal(
        curl_calc.detach().numpy(), curl_val.detach().numpy())


if __name__ == "__main__":
    pytest.main()
