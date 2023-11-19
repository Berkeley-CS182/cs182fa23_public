import numpy as np
import torch
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.linear_model import LogisticRegression

import ipywidgets as widgets


###############################################
#### ndarray/Tensor manipulation functions ####
def to_numpy(x):
    return x.detach().numpy()


def to_tensor(x):
    return torch.tensor(x)


def print_dict_initialization(params_dict, dict_name):
    for key in params_dict:
        print(str(key) + '=' + str(dict_name) + "['" + str(key) + '"]')


def my_sign_tensor(x):
    y = torch.sign(x)
    y[y == 0] = 1
    return y.int()


def my_sign_numpy(x):
    y = np.sign(x)
    y[y == 0] = 1
    return y.astype('int')


# Use this version of my_sign
def my_sign(x):
    if torch.is_tensor(x):
        return my_sign_tensor(x)
    return my_sign_numpy(x)


###################################
#### Data generation functions ####
def add_label_noise(y, noise_prob):
    y = y * np.random.choice([-1, 1], p=[noise_prob, 1 - noise_prob], size=y.shape)
    return y


def featurize_fourier(x, d, normalize=False):
    assert (d - 1) % 2 == 0, "d must be odd"
    max_r = int((d - 1) / 2)
    n = len(x)
    A = np.zeros((n, d))
    A[:, 0] = 1
    for d_ in range(1, max_r + 1):
        A[:, 2 * (d_ - 1) + 1] = np.sin(d_ * x * np.pi)
        A[:, 2 * (d_ - 1) + 2] = np.cos(d_ * x * np.pi)
    if normalize:
        A[:, 0] *= (1 / np.sqrt(2))
        A *= np.sqrt(2)
    return A


def featurize(x, d, phi_type, normalize=True):
    function_map = {
        # 'polynomial':featurize_vandermonde,
        'fourier': featurize_fourier}
    return function_map[phi_type](x, d, normalize)


def generate_x(n, x_type, x_low=-1, x_high=1):
    if x_type == 'grid':
        x = np.linspace(x_low, x_high, n, endpoint=False).astype(np.float64)
    elif x_type == 'uniform_random':
        x = np.sort(np.random.uniform(x_low, x_high, n).astype(np.float64))
        # Note that for making it easy for plotting we sort the randomly sampled x in ascending order
    else:
        raise ValueError
    return x


def generate_y(features, k_idx, k_val):
    # y as linear combination of features
    return np.sum(features[:, k_idx] * k_val, 1)


##########################################
#### Closed-form regression functions ####
def solve_ls(phi, y, weights=None):
    d = phi.shape[1]
    if weights is None:
        weights = np.ones(d)
    phi_weighted = weights * phi
    LR = LinearRegression(fit_intercept=False, normalize=False)
    LR.fit(phi_weighted, y)
    coeffs_weighted = LR.coef_
    alpha = coeffs_weighted * weights
    loss = np.mean((y - phi @ alpha.T)**2)

    return alpha.T, loss


def solve_ridge(phi, y, lambda_ridge, weights=None):
    d = phi.shape[1]
    if weights is None:
        weights = np.ones(d)
    phi_weighted = weights * phi

    Rdg = Ridge(fit_intercept=False, normalize=False, alpha=lambda_ridge)
    Rdg.fit(phi_weighted, y)
    coeffs_weighted = Rdg.coef_
    alpha = coeffs_weighted * weights
    loss = np.mean((y - phi @ alpha.T)**2) + lambda_ridge * np.sum((coeffs_weighted)**2)
    return alpha, loss


############################################
#### sklearn logistic regression solver ####
def solve_logistic(phi, z, weights=None):
    # print(z)
    # raise ValueError
    d = phi.shape[1]
    if weights is None:
        weights = np.ones(d)
    phi_weighted = weights * phi
    clf = LogisticRegression(tol=1e-4, verbose=False, solver='lbfgs', random_state=0, fit_intercept=False, C=1e6).fit(phi_weighted, z)

    coeffs_weighted = clf.coef_
    alpha = coeffs_weighted * weights

    z_pred = my_sign(phi @ alpha.T)[:, 0]
    loss = np.mean(z != z_pred)
    # print(loss)
    return alpha.T, loss


#######################################
#### Model class for use with MAML ####
class DummyModel(torch.nn.Module):
    def __init__(self, d):
        super(DummyModel, self).__init__()
        self.feature_weights = torch.nn.Parameter(torch.ones(d).double())
        self.coeffs = torch.nn.Parameter(torch.zeros(d).double())

    def forward(self, F):
        return (F * self.feature_weights) @ self.coeffs


##############################
#### ipywidget generators ####
def generate_int_widget(desc, min_, val, max_, step=1):
    return widgets.IntSlider(
        value=val,
        min=min_,
        max=max_,
        step=step,
        description=desc,
        disabled=False,
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='d')


def generate_float_widget(desc, min_, val, max_, step):
    return widgets.FloatSlider(
        value=val,
        min=min_,
        max=max_,
        step=step,
        description=desc,
        disabled=False,
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='.1f',
    )


def generate_floatlog_widget(desc, min_, step, val, max_, base=10):
    return widgets.FloatLogSlider(
        value=val,
        base=base,
        min=min_,  # max exponent of base
        max=max_,  # min exponent of base
        step=step,  # exponent step
        description=desc
    )
