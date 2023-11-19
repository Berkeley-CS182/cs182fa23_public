import numpy as np
import torch
import higher
from utils import *
from utils_visualization import *


##############################################################################################################################
#   Meta learning using closed form
##############################################################################################################################
def closed_form_ls(F_t, y_t):
    # Returns min norm least squares solution
    w_t = F_t.T @ torch.inverse(F_t @ F_t.T) @ y_t
    return w_t


def meta_update_reg(w_t, feature_weights_t, n_train_meta, phi_type, d, k_idx, k_val, noise_std):
    x_meta = generate_x(n_train_meta, 'uniform_random')
    features_meta = featurize(x_meta, phi_type=phi_type, d=d, normalize=True)
    features_meta_t = torch.tensor(features_meta)
    y_meta = generate_y(features_meta, k_idx, k_val)
    y_meta += np.random.normal(0, noise_std, y_meta.shape)

    y_meta_t = torch.tensor(y_meta)
    y_meta_pred_t = features_meta_t @ w_t

    criterion = torch.nn.MSELoss(reduction='sum')
    loss = criterion(y_meta_t, y_meta_pred_t)

    return loss


def get_post_data_reg(x_type, phi_type, k_idx, num_tasks_test, d, n_train_post_range, n_test, noise_std):
    k_val_test = np.random.uniform(-1, 1, size=(len(k_idx), num_tasks_test))
    k_val_test /= np.linalg.norm(k_val_test, axis=0)

    data_dict = {'train': {}, 'test': {}}

    x_test = generate_x(n_test, 'uniform_random')
    features_test = featurize(x_test, phi_type=phi_type, d=d, normalize=True)
    data_dict['test']['features'] = features_test

    data_dict['test']['x'] = x_test
    data_dict['test']['y'] = []
    for i in range(num_tasks_test):
        k_val_post = k_val_test[:, i]
        y_test = generate_y(features_test, k_idx, k_val_post)
        data_dict['test']['y'].append(y_test)
    for n_train_post in n_train_post_range:
        data_dict['train'][n_train_post] = {}

        x_train_post = generate_x(n_train_post, x_type)
        features_post = featurize(x_train_post, phi_type=phi_type, d=d, normalize=True)

        data_dict['train'][n_train_post]['x'] = x_train_post
        data_dict['train'][n_train_post]['features'] = features_post

        data_dict['train'][n_train_post]['y'] = []
        for i in range(num_tasks_test):
            k_val_post = k_val_test[:, i]
            y_post = generate_y(features_post, k_idx, k_val_post)
            y_post += np.random.normal(0, noise_std, y_post.shape)
            data_dict['train'][n_train_post]['y'].append(y_post)

    return data_dict


def test_oracle_reg(data_dict, x_type, feature_weights, min_n_train_post=0):
    # plt.plot(feature_weights, 'o-')
    # plt.show()
    # print(feature_weights)
    k_idx = np.where(feature_weights != 0)[0]
    feature_weights = None
    test_loss_matrix = []

    n_train_post_range = np.sort(np.array(list(data_dict['train'].keys())))
    n_train_post_range = n_train_post_range[n_train_post_range >= min_n_train_post]

    test_data_dict = data_dict['test']
    features_test_post = test_data_dict['features']

    for n_train_post in n_train_post_range:
        # print("n", n_train_post)
        train_data_dict = data_dict['train'][n_train_post]

        features_post = train_data_dict['features'][:, k_idx]
        cfeatures_test_post = features_test_post[:, k_idx]

        feature_norms = np.linalg.norm(features_post, axis=0)
        r_idx = np.where(feature_norms > 1e-6)[0]

        features_post = features_post[:, r_idx]
        cfeatures_test_post = cfeatures_test_post[:, r_idx]

        test_loss_array = []

        for i in range(len(train_data_dict['y'])):
            y_post = train_data_dict['y'][i]
            # print(features_post.shape, y_post.shape)
            if x_type == 'grid':
                # Use ridge with small reguralizer to avoid crazy effects of poor conditioning
                w_post, loss = solve_ridge(features_post, y_post, lambda_ridge=1e-12, weights=feature_weights)
            else:
                w_post, loss = solve_ls(features_post, y_post, weights=feature_weights)

            y_post_pred = features_post @ w_post
            y_test_post = test_data_dict['y'][i]
            y_test_post_pred = cfeatures_test_post @ w_post

            # Compute the regression loss
            test_loss = np.mean((y_test_post - y_test_post_pred)**2)
            test_loss_array.append(test_loss)

        test_loss_matrix.append(test_loss_array)

    test_loss_matrix = np.array(test_loss_matrix).T
    avg_test_loss = np.mean(test_loss_matrix, 0)

    top_10_loss = np.percentile(test_loss_matrix, 90, axis=0)
    bot_10_loss = np.percentile(test_loss_matrix, 10, axis=0)

    return n_train_post_range, avg_test_loss, top_10_loss, bot_10_loss


def test_reg(data_dict, feature_weights, min_n_train_post=0):
    # plt.plot(feature_weights, 'o-')
    # plt.show()
    # print(feature_weights)
    test_loss_matrix = []

    n_train_post_range = np.sort(np.array(list(data_dict['train'].keys())))
    n_train_post_range = n_train_post_range[n_train_post_range >= min_n_train_post]

    test_data_dict = data_dict['test']
    features_test_post = test_data_dict['features']

    for n_train_post in n_train_post_range:
        # print("n", n_train_post)
        train_data_dict = data_dict['train'][n_train_post]

        features_post = train_data_dict['features']

        test_loss_array = []
        for i in range(len(train_data_dict['y'])):
            y_post = train_data_dict['y'][i]
            # print(features_post.shape, y_post.shape)
            w_post, loss = solve_ls(features_post, y_post, feature_weights)
            y_post_pred = features_post @ w_post

            y_test_post = test_data_dict['y'][i]
            y_test_post_pred = features_test_post @ w_post

            # Compute the regression loss
            test_loss = np.mean((y_test_post - y_test_post_pred)**2)
            test_loss_array.append(test_loss)

        test_loss_matrix.append(test_loss_array)

    test_loss_matrix = np.array(test_loss_matrix).T
    avg_test_loss = np.mean(test_loss_matrix, 0)

    top_10_loss = np.percentile(test_loss_matrix, 90, axis=0)
    bot_10_loss = np.percentile(test_loss_matrix, 10, axis=0)

    return n_train_post_range, avg_test_loss, top_10_loss, bot_10_loss


def test_zero_reg(data_dict):
    test_data_dict = data_dict['test']
    ys = test_data_dict['y']

    test_loss_array = []
    for i in range(len(ys)):
        y = ys[i]
        # Compute the regression loss
        test_loss = np.mean(y**2)
        test_loss_array.append(test_loss)

    avg_test_loss = np.mean(test_loss_array)
    top_10_loss = np.percentile(test_loss_array, 90)
    bot_10_loss = np.percentile(test_loss_array, 10)

    return avg_test_loss, top_10_loss, bot_10_loss


def meta_learning_reg_closed_form(params_dict):
    seed = params_dict["seed"]
    n_train_inner = params_dict["n_train_inner"]
    n_train_meta = params_dict["n_train_meta"]
    # n_train_post = params_dict["n_train_post"]
    n_test_post = params_dict["n_test_post"]
    x_type = params_dict["x_type"]
    d = params_dict["d"]
    phi_type = params_dict["phi_type"]
    k_idx = params_dict["k_idx"]
    optimizer_type = params_dict["optimizer_type"]
    stepsize_meta = params_dict["stepsize_meta"]
    num_inner_tasks = params_dict["num_inner_tasks"]
    num_tasks_test = params_dict["num_tasks_test"]
    num_stats = params_dict["num_stats"]
    num_iterations = params_dict["num_iterations"]
    noise_std = params_dict.get('noise_std', 0)
    num_n_train_post_range = params_dict['num_n_train_post_range']

    # Set seed:
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Parameters
    stats_every = num_iterations // num_stats
    init_n_train_post_range, init_avg_test_loss, init_top_10_loss, init_bot_10_loss = None, None, None, None

    # Initialize meta parameter: weights on the d features
    feature_weights_t = torch.tensor(np.ones(d), requires_grad=True)

    # Define meta parameter optimizer
    if optimizer_type == 'SGD':
        opt_meta = torch.optim.SGD([feature_weights_t], lr=stepsize_meta)
    elif optimizer_type == 'Adam':
        opt_meta = torch.optim.Adam([feature_weights_t], lr=stepsize_meta)
    else:
        raise ValueError

    # Meta training loop
    # Get post train and test data
    n_train_post_range = np.logspace(np.log10(1), np.log10(3 * d), num_n_train_post_range).astype('int')


    must_have_points = [n_train_inner, len(k_idx) - 1, len(k_idx), len(k_idx) + 1]

    for point in must_have_points:
        if point not in n_train_post_range:
            n_train_post_range = np.hstack([n_train_post_range, point])

    n_train_post_range = np.sort(n_train_post_range)
    n_train_post_range = np.unique(n_train_post_range)
    # print(n_train_post_range)
    data_dict = get_post_data_reg(x_type, phi_type, k_idx, num_tasks_test, d, n_train_post_range, n_test_post, noise_std)

    for i in range(num_iterations):
        opt_meta.zero_grad()

        # Get x and features
        x = generate_x(n_train_inner, x_type)
        features = featurize(x, phi_type=phi_type, d=d, normalize=True)
        weighted_features_t = to_tensor(features) * feature_weights_t

        # Loop over inner tasks
        for t in range(num_inner_tasks):
            # Get random coefficients
            k_val = np.random.uniform(-1, 1, size=len(k_idx))
            k_val /= np.linalg.norm(k_val)

            # Generate y
            y = generate_y(features, k_idx, k_val)
            y += np.random.normal(0, noise_std, y.shape)
            y_t = torch.tensor(y)

            # Get closed form solution for w_t as a function of feature_weights_t
            w_t = closed_form_ls(weighted_features_t, y_t)
            w_t = w_t * feature_weights_t  # Reweight the coefficients so that we can multiply with unweighted features to get prediction

            # Meta update
            meta_loss = meta_update_reg(w_t, feature_weights_t, n_train_meta, phi_type, d, k_idx, k_val, noise_std)
            meta_loss.backward(retain_graph=True)

        if i == 0:
            # n_train_post_range = np.logspace(0, np.log10(3 * d), 40).astype('int')
            print("-" * 70)
            print("Iteration: ", i)
            # #Oracle stats
            oracle_feature_weights = np.zeros(d)
            oracle_feature_weights[k_idx] = 1
            oracle_n_train_post_range, oracle_avg_test_loss, oracle_top_10_loss, oracle_bot_10_loss = test_oracle_reg(
                data_dict, feature_weights=oracle_feature_weights, x_type=x_type)

            # plt.plot(oracle_n_train_post_range, oracle_avg_test_loss)
            # plt.show()
            zero_avg_loss, zero_top_10_loss, zero_bot_10_loss = test_zero_reg(data_dict)

            feature_weights = to_numpy(feature_weights_t)
            init_n_train_post_range, init_avg_test_loss, init_top_10_loss, init_bot_10_loss = test_reg(data_dict, feature_weights)

            visualize_test_loss_reg(
                0, n_train_inner, init_n_train_post_range, init_avg_test_loss, init_top_10_loss, init_bot_10_loss,
                oracle_n_train_post_range=oracle_n_train_post_range, oracle_avg_test_loss=oracle_avg_test_loss,
                oracle_top_10_loss=oracle_top_10_loss, oracle_bot_10_loss=oracle_bot_10_loss, zero_avg_loss=zero_avg_loss,
                zero_top_10_loss=zero_top_10_loss, zero_bot_10_loss=zero_bot_10_loss, noise_std=noise_std)

            visualize_prediction_reg(data_dict, feature_weights, n_train_inner)

        #Stats
        if (i + 1) % stats_every == 0 or i == num_iterations - 1:
            print("-" * 70)
            print("Iteration: ", i + 1)
            feature_weights = to_numpy(feature_weights_t)

            n_train_post_range, avg_test_loss, top_10_loss, bot_10_loss = test_reg(data_dict, feature_weights)
            visualize_test_loss_reg(
                i, n_train_inner, n_train_post_range, avg_test_loss, top_10_loss, bot_10_loss, init_n_train_post_range, init_avg_test_loss,
                init_top_10_loss, init_bot_10_loss, oracle_n_train_post_range=oracle_n_train_post_range, oracle_avg_test_loss=oracle_avg_test_loss,
                oracle_top_10_loss=oracle_top_10_loss, oracle_bot_10_loss=oracle_bot_10_loss, zero_avg_loss=zero_avg_loss,
                zero_top_10_loss=zero_top_10_loss, zero_bot_10_loss=zero_bot_10_loss, noise_std=noise_std)
            visualize_prediction_reg(data_dict, feature_weights, n_train_inner)

        opt_meta.step()  # Finally update meta weights after loop through all tasks

    return to_numpy(feature_weights_t), data_dict


def meta_learning_reg_sgd(params_dict):
    seed = params_dict["seed"]
    n_train_inner = params_dict["n_train_inner"]
    n_train_meta = params_dict["n_train_meta"]
#     n_train_post = params_dict["n_train_post"]
    n_test_post = params_dict["n_test_post"]
    x_type = params_dict["x_type"]
    d = params_dict["d"]
    phi_type = params_dict["phi_type"]
    k_idx = params_dict["k_idx"]
    optimizer_type = params_dict["optimizer_type"]
    stepsize_meta = params_dict["stepsize_meta"]
    num_inner_tasks = params_dict["num_inner_tasks"]
    num_tasks_test = params_dict["num_tasks_test"]
    num_stats = params_dict["num_stats"]
    num_iterations = params_dict["num_iterations"]
    noise_std = params_dict.get('noise_std', 0)
    num_n_train_post_range = params_dict['num_n_train_post_range']

    stepsize_inner = params_dict["stepsize_inner"]
    num_gd_steps = params_dict['num_gd_steps']

    # Set seed:
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Parameters
    stats_every = num_iterations // num_stats
    init_n_train_post_range, init_avg_test_loss, init_top_10_loss, init_bot_10_loss = None, None, None, None

    # Define meta parameter optimizer
    dm = DummyModel(d)

    # Define meta parameter optimizer
    if optimizer_type == 'SGD':
        opt_meta = torch.optim.SGD([dm.feature_weights], lr=stepsize_meta)
    elif optimizer_type == 'Adam':
        opt_meta = torch.optim.Adam([dm.feature_weights], lr=stepsize_meta)
    else:
        raise ValueError

    # Meta training loop
    # Get post train and test data
    n_train_post_range = np.logspace(np.log10(1), np.log10(3 * d), num_n_train_post_range).astype('int')

    must_have_points = [n_train_inner, len(k_idx) - 1, len(k_idx), len(k_idx) + 1]

    for point in must_have_points:
        if point not in n_train_post_range:
            n_train_post_range = np.hstack([n_train_post_range, point])

    n_train_post_range = np.sort(n_train_post_range)
    n_train_post_range = np.unique(n_train_post_range)
    data_dict = get_post_data_reg(x_type, phi_type, k_idx, num_tasks_test, d, n_train_post_range, n_test_post, noise_std)

    for i in range(num_iterations):
        opt_meta.zero_grad()

        # Get x and features
        x = generate_x(n_train_inner, x_type)
        features = featurize(x, phi_type=phi_type, d=d, normalize=True)
        features_t = to_tensor(features)

        # Loop over inner tasks
        for t in range(num_inner_tasks):
            # Get random coefficients
            k_val = np.random.uniform(-1, 1, size=len(k_idx))
            k_val /= np.linalg.norm(k_val)

            dm.coeffs = torch.nn.Parameter(torch.zeros(d).double())

            # Generate y
            y = generate_y(features, k_idx, k_val)
            y += np.random.normal(0, noise_std, y.shape)
            y_t = torch.tensor(y)

            opt_inner = torch.optim.SGD([dm.coeffs], lr=stepsize_inner)
            with higher.innerloop_ctx(dm, opt_inner, copy_initial_weights=False, track_higher_grads=True) as (mod, opt):

                for j in range(num_gd_steps):
                    y_pred = mod(features_t)
                    loss = torch.mean((y_pred - y_t)**2)
#                     opt_inner.zero_grad()
#                     loss.backward()
                    opt.step(loss)

                # Meta update
                meta_loss = meta_update_reg(mod.coeffs * mod.feature_weights, mod.feature_weights, n_train_meta, phi_type, d, k_idx, k_val, noise_std)
                meta_loss.backward(retain_graph=True)

        if i == 0:
            # n_train_post_range = np.logspace(0, np.log10(3 * d), 40).astype('int')
            print("-" * 70)
            print("Iteration: ", i)
            # #Oracle stats
            oracle_feature_weights = np.zeros(d)
            oracle_feature_weights[k_idx] = 1
            oracle_n_train_post_range, oracle_avg_test_loss, oracle_top_10_loss, oracle_bot_10_loss = test_oracle_reg(
                data_dict, feature_weights=oracle_feature_weights, x_type=x_type)

            # plt.plot(oracle_n_train_post_range, oracle_avg_test_loss)
            # plt.show()
            zero_avg_loss, zero_top_10_loss, zero_bot_10_loss = test_zero_reg(data_dict)

            feature_weights = to_numpy(dm.feature_weights)
            init_n_train_post_range, init_avg_test_loss, init_top_10_loss, init_bot_10_loss = test_reg(data_dict, feature_weights)

            visualize_test_loss_reg(
                0, n_train_inner, init_n_train_post_range, init_avg_test_loss, init_top_10_loss, init_bot_10_loss,
                oracle_n_train_post_range=oracle_n_train_post_range, oracle_avg_test_loss=oracle_avg_test_loss,
                oracle_top_10_loss=oracle_top_10_loss, oracle_bot_10_loss=oracle_bot_10_loss, zero_avg_loss=zero_avg_loss,
                zero_top_10_loss=zero_top_10_loss, zero_bot_10_loss=zero_bot_10_loss, noise_std=noise_std)

            visualize_prediction_reg(data_dict, feature_weights, n_train_inner)

        # Stats
        if (i + 1) % stats_every == 0 or i == num_iterations - 1:
            print("-" * 70)
            print("Iteration: ", i + 1)
            feature_weights = to_numpy(dm.feature_weights)

            n_train_post_range, avg_test_loss, top_10_loss, bot_10_loss = test_reg(data_dict, feature_weights)
            visualize_test_loss_reg(
                i, n_train_inner, n_train_post_range, avg_test_loss, top_10_loss, bot_10_loss, init_n_train_post_range, init_avg_test_loss,
                init_top_10_loss, init_bot_10_loss, oracle_n_train_post_range=oracle_n_train_post_range, oracle_avg_test_loss=oracle_avg_test_loss,
                oracle_top_10_loss=oracle_top_10_loss, oracle_bot_10_loss=oracle_bot_10_loss, zero_avg_loss=zero_avg_loss,
                zero_top_10_loss=zero_top_10_loss, zero_bot_10_loss=zero_bot_10_loss, noise_std=noise_std)
            visualize_prediction_reg(data_dict, feature_weights, n_train_inner)

        opt_meta.step()  # Finally update meta weights after loop through all tasks
