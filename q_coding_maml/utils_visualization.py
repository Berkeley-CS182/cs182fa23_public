import numpy as np
import matplotlib.pyplot as plt

from utils import *


def visualize_prediction_classification(data_dict, feature_weights, n_train_post):
    train_data_dict = data_dict['train'][n_train_post]
    test_data_dict  = data_dict['test']


    x_train_post = train_data_dict['x']
    features_post = train_data_dict['features']
    y_post = train_data_dict['y'][0]
    
    z_post = my_sign(y_post)
    w_post, loss = solve_logistic(features_post, z_post, feature_weights)

    w_post /= np.linalg.norm(w_post)
    y_post_pred = features_post@w_post
    z_post_pred = my_sign(y_post_pred)

    x_test_post = test_data_dict['x']
    features_test_post = test_data_dict['features']
    y_test_post = test_data_dict['y'][0]
    y_test_post_pred = features_test_post@w_post


    #print(y_test_post_pred.shape, y_test_post.shape)
    #print(np.mean((np.squeeze(my_sign(y_test_post))!= np.squeeze(my_sign(y_test_post_pred)))))
    # print(((my_sign(y_test_post)!= my_sign(y_test_post_pred)).astype('float')).shape)
    # print(((my_sign(y_test_post)!= my_sign(y_test_post_pred)).astype('float')))
    # plt.plot(my_sign(y_test_post) - my_sign(y_test_post_pred))
    # plt.show()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = [16,6])
    ax1.scatter(x_train_post, z_post, marker = 'o', s = 100, color = 'red', alpha = 0.5, label = 'Training points true')
    ax1.scatter(x_train_post, z_post_pred, marker='x', s = 100, color = 'green', alpha = 0.5, label = 'Training points predictions')
    ax1.plot(x_test_post, y_test_post_pred, '-', color = 'blue', alpha = 0.4,  label = 'Predicted function')
    
    z_test_post_pred = my_sign(y_test_post_pred)
    ax1.scatter(x_test_post, z_test_post_pred, s=10, color='green', label = 'Predicted Sign Labels', alpha = 0.8)
    z_test_post = my_sign(y_test_post)
    ax1.scatter(x_test_post, z_test_post, s = 10, color='orange', label = 'True Sign Labels', alpha = 0.8)
    
    ax1.plot(x_test_post, y_test_post, '-', color = 'brown', label = 'True function', alpha = 0.4)
    ax1.set_xlabel('x')
    ax1.set_title('n_train_post =' +str(n_train_post))
    ax1.legend()

    ax2.plot(np.abs(feature_weights), 'o-')
    ax2.set_title('Feature weights')
    ax2.set_xlabel('Feature #')
    ax2.set_ylabel('abs(feature_weight)')
#     ax2.set_yscale('log')
    plt.show()


def  visualize_test_loss_reg(iteration, n_train_inner, n_train_post_range, avg_test_loss, top_10_loss, bot_10_loss, 
        init_n_train_post_range=None,init_avg_test_loss=None, init_top_10_loss=None,init_bot_10_loss=None,\
            oracle_n_train_post_range=None,oracle_avg_test_loss=None, oracle_top_10_loss=None, oracle_bot_10_loss=None, zero_avg_loss = None, zero_top_10_loss = None, zero_bot_10_loss = None,  wrong_n_train_post_range=None, wrong_avg_test_loss=None, wrong_top_10_loss=None, wrong_bot_10_loss=None, noise_std = None):
        
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = [16,6])

   
    ax1.fill_between(n_train_post_range, bot_10_loss, top_10_loss, alpha = 0.5)  

    if iteration is not None:
        ax1.plot(n_train_post_range, avg_test_loss, 'o-', label = 'Iteration: ' + str(iteration))
    else:
        ax1.plot(n_train_post_range, avg_test_loss, 'o-', label = 'Meta-learned')
    ax1.set_yscale('log')
    ax1.set_ylabel('Test mse')
    ax1.set_xlabel('n_train_post')
    ax1.set_title('Test mse vs n_train_post')

    if init_avg_test_loss is not None:  
        ax1.fill_between(init_n_train_post_range, init_bot_10_loss, init_top_10_loss, alpha = 0.5, color = 'orange')
        ax1.plot(init_n_train_post_range, init_avg_test_loss, 'o-', c = 'orange', label = 'Init')    


    if oracle_avg_test_loss is not None:  
        ax1.fill_between(oracle_n_train_post_range,oracle_bot_10_loss, oracle_top_10_loss, alpha = 0.5, color = 'green')
        ax1.plot(oracle_n_train_post_range, oracle_avg_test_loss, 'o-', c = 'green', label = 'Oracle')    
    if wrong_avg_test_loss is not None:  
        ax1.fill_between(wrong_n_train_post_range,wrong_bot_10_loss, wrong_top_10_loss, alpha = 0.5, color = 'red')
        ax1.plot(wrong_n_train_post_range, wrong_avg_test_loss, 'o-', c = 'red', label = 'Wrong weights')    

    if zero_avg_loss is not None:
        ax1.fill_between(n_train_post_range,zero_bot_10_loss*np.ones(len(n_train_post_range)), zero_top_10_loss*np.ones(len(n_train_post_range)), alpha = 0.5, color = 'yellow')

        ax1.plot(n_train_post_range, np.ones(len(n_train_post_range))*zero_avg_loss, '-', c = 'yellow', label = 'Zero')    



    if noise_std is not None:
        ax1.plot(n_train_post_range, np.ones_like(n_train_post_range)*(noise_std**2), '--', c = 'black', label = 'Noise variance')

    ax1.legend()

    idx = np.where(n_train_post_range <= 4*n_train_inner)[0]
    cn_train_post_range = n_train_post_range[idx]
    cavg_test_loss = avg_test_loss[idx]    
    ctop_10_loss = top_10_loss[idx]
    cbot_10_loss = bot_10_loss[idx]


    ax2.fill_between(cn_train_post_range, cbot_10_loss, ctop_10_loss, alpha = 0.5)  
    if iteration is not None:
        ax2.plot(cn_train_post_range, cavg_test_loss, 'o-', label = 'Iteration: ' + str(iteration))
    else:
        ax2.plot(cn_train_post_range, cavg_test_loss, 'o-', label = 'Meta-learned')

    ax2.set_yscale('log')
    ax2.set_ylabel('Test mse')
    ax2.set_xlabel('n_train_post')
    ax2.set_title('Test mse vs n_train_post (zoomed)')

    if init_avg_test_loss is not None:  
        idx = np.where(init_n_train_post_range <= 4*n_train_inner)[0]
        cinit_n_train_post_range = init_n_train_post_range[idx]
        cinit_avg_test_loss = init_avg_test_loss[idx]    
        cinit_top_10_loss = init_top_10_loss[idx]
        cinit_bot_10_loss = init_bot_10_loss[idx]

        ax2.fill_between(cinit_n_train_post_range, cinit_bot_10_loss, cinit_top_10_loss, alpha = 0.5, color = 'orange')
        ax2.plot(cinit_n_train_post_range, cinit_avg_test_loss, 'o-', c = 'orange', label = 'Init')

    if oracle_avg_test_loss is not None:  
        idx = np.where(oracle_n_train_post_range <= 4*n_train_inner)[0]
        coracle_n_train_post_range = oracle_n_train_post_range[idx]
        coracle_avg_test_loss = oracle_avg_test_loss[idx]    
        coracle_top_10_loss = oracle_top_10_loss[idx]
        coracle_bot_10_loss = oracle_bot_10_loss[idx]
        ax2.fill_between(coracle_n_train_post_range,coracle_bot_10_loss, coracle_top_10_loss, alpha = 0.5, color = 'green')
        ax2.plot(coracle_n_train_post_range, coracle_avg_test_loss, 'o-', c = 'green', label = 'Oracle') 

    if wrong_avg_test_loss is not None:  
        idx = np.where(wrong_n_train_post_range <= 4*n_train_inner)[0]
        cwrong_n_train_post_range = wrong_n_train_post_range[idx]
        cwrong_avg_test_loss = wrong_avg_test_loss[idx]    
        cwrong_top_10_loss = wrong_top_10_loss[idx]
        cwrong_bot_10_loss = wrong_bot_10_loss[idx]
        ax2.fill_between(cwrong_n_train_post_range,cwrong_bot_10_loss, cwrong_top_10_loss, alpha = 0.5, color = 'red')
        ax2.plot(cwrong_n_train_post_range, cwrong_avg_test_loss, 'o-', c = 'red', label = 'Wrong weights')


    if zero_avg_loss is not None:
        ax2.fill_between(cn_train_post_range,zero_bot_10_loss*np.ones(len(cn_train_post_range)), zero_top_10_loss*np.ones(len(cn_train_post_range)), alpha = 0.5, color = 'yellow')

        ax2.plot(cn_train_post_range, np.ones(len(cn_train_post_range))*zero_avg_loss, '-', c = 'yellow', label = 'Zero')

    if noise_std is not None:
        ax2.plot(cn_train_post_range, np.ones_like(cn_train_post_range)*(noise_std**2), '--', label = 'Noise variance', c = 'black')
    ax2.legend()


    plt.show()


def visualize_prediction_reg(data_dict, feature_weights, n_train_post):
    train_data_dict = data_dict['train'][n_train_post]
    test_data_dict  = data_dict['test']


    x_train_post = train_data_dict['x']
    features_post = train_data_dict['features']
    y_post = train_data_dict['y'][0]
    
    w_post, loss = solve_ls(features_post, y_post, feature_weights)

    y_post_pred = features_post@w_post

    x_test_post = test_data_dict['x']
    features_test_post = test_data_dict['features']
    y_test_post = test_data_dict['y'][0]
    y_test_post_pred = features_test_post@w_post



    #For plotting purposes add x_train to x_test

    x_test_post = np.concatenate([x_train_post, x_test_post])
    y_test_post = np.concatenate([y_post, y_test_post])
    y_test_post_pred = np.concatenate([y_post_pred, y_test_post_pred])

    idx = np.argsort(x_test_post)
    x_test_post = x_test_post[idx]
    y_test_post = y_test_post[idx]
    y_test_post_pred = y_test_post_pred[idx]


    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = [16,6])
    ax1.scatter(x_train_post, y_post, marker = 'o', color = 'red', alpha = 0.5, label = 'Training points true')
    ax1.scatter(x_train_post, y_post_pred, marker='x', color = 'green', alpha = 0.5, label = 'Training points predictions')



    ax1.plot(x_test_post, y_test_post_pred, label = 'Predicted function')
    ax1.plot(x_test_post, y_test_post, label = 'True function')
    ax1.set_xlabel('x')
    ax1.set_title('n_train_post =' +str(n_train_post))
    ax1.legend()

    ax2.plot(np.abs(feature_weights), 'o-')
    ax2.set_title('Feature weights')
    ax2.set_xlabel('Feature #')
    ax2.set_ylabel('abs(feature_weight)')
#     ax2.set_yscale('log')
    plt.show()




def visualize_test_loss_classification(iteration, n_train_inner, n_train_post_range, avg_test_loss, top_10_loss, bot_10_loss, 
        init_n_train_post_range=None,init_avg_test_loss=None, init_top_10_loss=None, init_bot_10_loss=None,\
            oracle_n_train_post_range=None,oracle_avg_test_loss=None, oracle_top_10_loss=None, oracle_bot_10_loss=None, zero_avg_loss = None, zero_top_10_loss = None, zero_bot_10_loss = None, noise_prob = None):
        
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = [16,6])

   
    ax1.fill_between(n_train_post_range, bot_10_loss, top_10_loss, alpha = 0.5)  
    ax1.plot(n_train_post_range, avg_test_loss, 'o-', label = 'Iteration: ' + str(iteration))
    ax1.set_yscale('log')
    ax1.set_ylabel('Test classification error')
    ax1.set_xlabel('n_train_post')
    ax1.set_title('Test classification error vs n_train_post')

    if init_avg_test_loss is not None:  
        ax1.fill_between(init_n_train_post_range, init_bot_10_loss, init_top_10_loss, alpha = 0.5, color = 'orange')
        ax1.plot(init_n_train_post_range, init_avg_test_loss, 'o-', c = 'orange', label = 'Init')    



    if oracle_avg_test_loss is not None:  
        ax1.fill_between(oracle_n_train_post_range,oracle_bot_10_loss, oracle_top_10_loss, alpha = 0.5, color = 'green')
        ax1.plot(oracle_n_train_post_range, oracle_avg_test_loss, 'o-', c = 'green', label = 'Oracle')    
   

    if zero_avg_loss is not None:
        ax1.fill_between(n_train_post_range,zero_bot_10_loss*np.ones(len(n_train_post_range)), zero_top_10_loss*np.ones(len(n_train_post_range)), alpha = 0.5, color = 'yellow')

        ax1.plot(n_train_post_range, np.ones(len(n_train_post_range))*zero_avg_loss, '-', c = 'yellow', label = 'Zero')    

    if noise_prob is not None and noise_prob != 0:
        ax1.plot(n_train_post_range, np.ones_like(n_train_post_range)*(noise_prob), '--', c = 'black', label = 'Noise variance')

    ax1.legend()
    idx = np.where(n_train_post_range <= 2*n_train_inner)[0]
    cn_train_post_range = n_train_post_range[idx]
    cavg_test_loss = avg_test_loss[idx]    
    ctop_10_loss = top_10_loss[idx]
    cbot_10_loss = bot_10_loss[idx]


    ax2.fill_between(cn_train_post_range, cbot_10_loss, ctop_10_loss, alpha = 0.5)  
    ax2.plot(cn_train_post_range, cavg_test_loss, 'o-', label = 'Iteration: ' + str(iteration))

    ax2.set_yscale('log')
    ax2.set_ylabel('Test classification error')
    ax2.set_xlabel('n_train_post')
    ax2.set_title('Test mse vs n_train_post (zoomed)')

    if init_avg_test_loss is not None:  
        idx = np.where(init_n_train_post_range <= 2*n_train_inner)[0]
        cinit_n_train_post_range = init_n_train_post_range[idx]
        cinit_avg_test_loss = init_avg_test_loss[idx]    
        cinit_top_10_loss = init_top_10_loss[idx]
        cinit_bot_10_loss = init_bot_10_loss[idx]

        ax2.fill_between(cinit_n_train_post_range, cinit_bot_10_loss, cinit_top_10_loss, alpha = 0.5, color = 'orange')
        ax2.plot(cinit_n_train_post_range, cinit_avg_test_loss, 'o-', c = 'orange', label = 'Init')

    if oracle_avg_test_loss is not None:  
        idx = np.where(oracle_n_train_post_range <= 2*n_train_inner)[0]
        coracle_n_train_post_range = oracle_n_train_post_range[idx]
        coracle_avg_test_loss = oracle_avg_test_loss[idx]    
        coracle_top_10_loss = oracle_top_10_loss[idx]
        coracle_bot_10_loss = oracle_bot_10_loss[idx]
        ax2.fill_between(coracle_n_train_post_range,coracle_bot_10_loss, coracle_top_10_loss, alpha = 0.5, color = 'green')
        ax2.plot(coracle_n_train_post_range, coracle_avg_test_loss, 'o-', c = 'green', label = 'Oracle') 


    if zero_avg_loss is not None:
        ax2.fill_between(cn_train_post_range,zero_bot_10_loss*np.ones(len(cn_train_post_range)), zero_top_10_loss*np.ones(len(cn_train_post_range)), alpha = 0.5, color = 'yellow')

        ax2.plot(cn_train_post_range, np.ones(len(cn_train_post_range))*zero_avg_loss, '-', c = 'yellow', label = 'Zero')

    if noise_prob is not None and noise_prob != 0:
        ax2.plot(cn_train_post_range, np.ones_like(cn_train_post_range)*(noise_prob), '--', label = 'Noise probability', c = 'black')
    ax2.legend()


    plt.show()





