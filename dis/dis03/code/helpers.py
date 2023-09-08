import torch
import matplotlib.pyplot as plt
import numpy as np
import copy
from ipywidgets import fixed, interactive, widgets

def to_torch(x):
    return torch.from_numpy(x).float()


def to_numpy(x):
    return x.detach().numpy()


def plot_data(X, y, X_test, y_test):
    clip_bound = 2.5
    plt.xlim(0, 1)
    plt.ylim(-clip_bound, clip_bound)
    plt.scatter(X[:, 0], y, c='darkorange', s=40.0, label='training data points')
    plt.plot(X_test, y_test, '--', color='royalblue', linewidth=2.0, label='Ground truth')


def plot_relu(bias, slope):
    plt.scatter([-bias / slope], 0, c='darkgrey', s=40.0)
    if slope > 0 and bias < 0:
        plt.plot([0, -bias / slope, 1], [0, 0, slope * (1 - bias)], ':')
    elif slope < 0 and bias > 0:
        plt.plot([0, -bias / slope, 1], [-bias * slope, 0, 0], ':')


def plot_relus(params):
    slopes = to_numpy(params[0]).ravel()
    biases = to_numpy(params[1])
    for relu in range(biases.size):
        plot_relu(biases[relu], slopes[relu])


def plot_function(X_test, net):
    y_pred = net(to_torch(X_test))
    plt.plot(X_test, to_numpy(y_pred), '-', color='forestgreen', label='prediction')


def plot_update(X, y, X_test, y_test, net, state=None, optim='sgd'):
    if state is not None:
        net.load_state_dict(state)
    plt.figure(figsize=(10, 7))
    plot_relus(list(net.parameters()))
    plot_function(X_test, net)
    plot_data(X, y, X_test, y_test)
    plt.legend()
    plt.title(optim)
    plt.show()


def train_network(X, y, X_test, y_test, net: torch.nn.Module, optim, n_steps, save_every, initial_weights=None, verbose=False, optimizer='sgd', seed=0, ckpt_dir=None):
    loss = torch.nn.MSELoss()
    y_torch = to_torch(y.reshape(-1, 1))
    X_torch = to_torch(X)
    history = {}
    
    # if initial_weights is not None:
    #     net.load_state_dict(initial_weights)
    

    for s in range(n_steps):
        subsample = np.random.choice(y.size, y.size // 5)
        step_loss = loss(y_torch[subsample], net(X_torch[subsample, :]))
        optim.zero_grad()
        step_loss.backward()
        optim.step()
        if (s + 1) % save_every == 0 or s == 0:
            # plot_update(X, y, X_test, y_test, net)
            history[s + 1] = {}
            history[s + 1]['state'] = copy.deepcopy(net.state_dict())
            with torch.no_grad():
                test_loss = loss(to_torch(y_test.reshape(-1, 1)), net(to_torch(X_test)))
            history[s + 1]['train_error'] = to_numpy(step_loss).item()
            history[s + 1]['test_error'] = to_numpy(test_loss).item()
            if verbose:
                print(f"{optimizer} Iteration %d" % (s + 1))
                print("\tTrain Loss: %.3f" % to_numpy(step_loss).item())
                print("\tTest Loss: %.3f" % to_numpy(test_loss).item())
            

    if ckpt_dir != None:
        torch.save(history, f'{ckpt_dir}/ckpt_and_history.pt')   

    return history


def plot_test_train_errors(history, optim='', plot_test=False, plot_train=False):
    sample_points = np.array(list(history.keys()))
    etrain = [history[s]['train_error'] for s in history]
    etest = [history[s]['test_error'] for s in history]
    if plot_train:
      plt.plot(sample_points / 1e3, etrain, label=f'Train Error {optim}')
    if plot_test:
      plt.plot(sample_points / 1e3, etest, label=f'Test Error {optim}')
    plt.xlabel("Iterations (1000's)")
    plt.ylabel("MSE")
    plt.yscale('log')
    plt.legend()
    # plt.show();

TRANS = [0, 1, -1]

def plot_with_error_bar(
        list_of_history, 
        optim='', 
        plot_test=False, 
        plot_train=False,
        idx=0
    ):
    sample_points = np.array(list(list_of_history[0].keys()))
    etrain = [[hist[s]['train_error'] for s in hist] for hist in list_of_history]
    etest = [[hist[s]['test_error'] for s in hist] for hist in list_of_history]
    etrain, etest = np.array(etrain), np.array(etest)
    med_etrain, lower_train, upper_train \
        = np.median(etrain, axis=0), np.percentile(etrain, q=0.75, axis=0), np.percentile(etrain, q=0.25, axis=0)
    med_etest, lower_test, upper_test \
        = np.median(etest, axis=0), np.percentile(etest, q=0.75, axis=0), np.percentile(etest, q=0.25, axis=0)
    
    quantile_train = np.row_stack((lower_train, upper_train))
    quantile_test = np.row_stack((lower_test, upper_test))
    
    if plot_train:
        plt.errorbar(
            sample_points / 1e3 + 0.2 * TRANS[idx], med_etrain, 
            yerr=quantile_train, fmt='-o', 
            label=f'Train Error {optim}', capsize=5
        )
    if plot_test:
        plt.errorbar(
            sample_points / 1e3 + 0.2 * TRANS[idx], med_etest, 
            yerr=quantile_test, fmt='-o', 
            label=f'Test Error {optim}', capsize=5
        )
    
    plt.xlabel("Iterations (1000's)")
    plt.ylabel("MSE")
    plt.yscale('log')
    plt.legend()


def make_iter_slider(iters):
    # print(iters)
    return widgets.SelectionSlider(
        options=iters,
        value=1,
        description='SGD Iterations: ',
        disabled=False
    )


def history_interactive(history, idx, X, y, X_test, y_test, net):
    plot_update(X, y, X_test, y_test, net, state=history[idx]['state'])
    plt.show()
    print("Train Error: %.3f" % history[idx]['train_error'])
    print("Test Error: %.3f" % history[idx]['test_error'])


def make_history_interactive(history, X, y, X_test, y_test, net):
    sample_points = list(history.keys())
    return interactive(history_interactive,
                       history=fixed(history),
                       idx=make_iter_slider(sample_points),
                       X=fixed(X),
                       y=fixed(y),
                       X_test=fixed(X_test),
                       y_test=fixed(y_test),
                       net=fixed(net))
