import numpy as np
import torch
import higher
from utils import *
from utils_visualization import visualize_prediction_classification, visualize_test_loss_classification


# Available to students

####################################################################################################

def get_post_data_classification(x_type, phi_type, k_idx, num_tasks_test, d, n_train_post_range, n_test, noise_prob):
    k_val_test = np.random.uniform(-1, 1, size=(len(k_idx), num_tasks_test))
    k_val_test /= np.linalg.norm(k_val_test, axis=0)

    data_dict = {'train': {}, 'test': {}}

    # features_post_complete = featurize(x_train_post_complete, phi_type=phi_type,d=d, normalize = True)
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
        features_post = featurize(
            x_train_post, phi_type=phi_type, d=d, normalize=True)

        data_dict['train'][n_train_post]['x'] = x_train_post
        data_dict['train'][n_train_post]['features'] = features_post

        data_dict['train'][n_train_post]['y'] = []
        for i in range(num_tasks_test):
            k_val_post = k_val_test[:, i]
            y_post = generate_y(features_post, k_idx, k_val_post)

            y_post = add_label_noise(y_post, noise_prob)

            # y_post += np.random.normal(0, noise_std, y_post.shape)
            data_dict['train'][n_train_post]['y'].append(y_post)

    return data_dict

#Placeholder to be removed after copy pasting meta_learning_reg_sgd and renaming it to meta_learning_classification
def meta_learning_classification(params_dict):
    raise NotImplementedError