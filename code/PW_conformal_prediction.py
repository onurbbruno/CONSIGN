import numpy as np
import matplotlib.pyplot as plt
from helpers_cp import *
from pytictoc import TicToc
import torch
from config_cp import Config
from scipy.ndimage import uniform_filter

def PW_calibration(cal_smx,cal_labels,strategy,spatial='NO'):
    print('Pixel Wise Calibration:')
    n = Config.n #number of calibration images

    dim = cal_smx.shape[-1] #width height dimensions images
    mean_sample = np.mean(cal_smx,axis=1) #mean softmax prediction

    ################# Calibration phase #######################
    lam = 0.0 #initial lambda
    l = 0.01 #step size

    converged = Config.converged
    cal_already_checked = [] #calibrated images
    while not converged:
        for i in range(n):
            print_loading_bar(i, n)
            if i not in cal_already_checked:
                Y_cal = cal_labels[i,:] #gt
                X_cal = mean_sample[i,:] #current mean softmax

                if spatial=='NO':
                    X_flat = X_cal.reshape(Config.labels,-1)
                    sorted_idx = np.argsort(-X_flat, axis=0) # sort softmax
                    sorted_vals = np.take_along_axis(X_flat, sorted_idx, axis=0)
                    cumsum = np.cumsum(sorted_vals, axis=0)
                    if strategy == 'APS':
                        num_needed = (cumsum > lam).argmax(axis=0) + 1 # include labels until cumalitive softmax reaches lam
                    elif strategy == 'RAPS':
                        o_ranks = np.array([1+i for i in range(Config.labels)])
                        reg = Config.theta*(o_ranks-Config.k_reg)
                        reg[reg<0]=0
                        num_needed = (cumsum + reg[:,None] > lam).argmax(axis=0) + 1 # include labels until cumalitive softmax + regularization reach lam

                elif spatial=='SACP':
                    if strategy == 'APS':
                        # get new sorting
                        X_avg = uniform_filter(X_cal,size=Config.neigh_size,mode='reflect')
                        X_blend = (1. - Config.neigh_weight)*X_cal + Config.neigh_weight*X_avg
                        X_flat = X_blend.reshape(Config.labels,-1)
                        sorted_idx = np.argsort(-X_flat, axis=0)
                        #
                        sorted_vals = np.take_along_axis(X_flat, sorted_idx, axis=0)
                        cum_p_avg = np.cumsum(sorted_vals, axis=0)

                        ge_mask = (cum_p_avg > lam)
                        num_needed = (np.where(ge_mask.any(axis=0),ge_mask.argmax(axis=0) + 1, Config.labels)).reshape(-1)
                    else:
                        # get new sortings
                        X_avg = uniform_filter(X_cal,size=Config.neigh_size,mode='reflect')
                        X_blend = (1. - Config.neigh_weight)*X_cal + Config.neigh_weight*X_avg
                        X_flat = X_blend.reshape(Config.labels,-1)
                        sorted_idx = np.argsort(-X_flat, axis=0)
                        #
                        #For RAPS need to treat each sorting separately
                        order = np.argsort(-X_cal, axis=0) # sort softmax
                        order_rows = order.transpose(1,2,0).reshape(-1,Config.labels)
                        unique_orders, inv_flat = np.unique(order_rows,axis=0,return_inverse=True) # get all of the possible orders
                        inv2d = inv_flat.reshape(Config.dim,Config.dim)
                        n_perms = unique_orders.shape[0]
                        ranks = np.arange(1, Config.labels+1)
                        reg = np.clip(Config.theta * (ranks - Config.k_reg), 0, None)
                        final_cum_ordered = np.zeros((Config.labels,Config.dim,Config.dim),dtype=X_cal.dtype)
                        # iterate trough all orders and apply RAPS
                        for pid,p_row in enumerate(unique_orders):
                            p=p_row.astype(int).tolist()
                            X_p = X_cal[p,:,:] # sort softmax with order p
                            # RAPS for order p
                            cum_p = np.cumsum(X_p,axis=0)
                            cum_p += reg[:,None,None]
                            # average with neighborhood pixels
                            cum_p_neigh = uniform_filter(cum_p,size=Config.neigh_size,mode='reflect')
                            # weighted average
                            cum_p_blend = (1. - Config.neigh_weight)*cum_p + Config.neigh_weight*cum_p_neigh
                            mask = (inv2d==pid)
                            if not mask.any():
                                continue
                            # update only pixels with order p, e.g. pixels with softma ordered as {0,1,2,3}
                            final_cum_ordered[:,mask] = cum_p_blend[:,mask]

                        ge_mask = (final_cum_ordered > lam)
                        num_needed = (np.where(ge_mask.any(axis=0),ge_mask.argmax(axis=0) + 1, Config.labels)).reshape(-1)

                prediction_set = np.zeros_like(X_flat, dtype=bool)
                cols = np.arange(X_flat.shape[1])
                prediction_set[sorted_idx[:Config.labels, cols], cols] = np.less.outer(np.arange(Config.labels), num_needed).astype(bool)
                prediction_set = prediction_set.reshape(X_cal.shape)

                gt_is_in_set = prediction_set[Y_cal,np.arange(dim)[:,None],np.arange(dim)[None,:]] #{True,False}^{HxW}, matrix to check if each ground-truth label is in the prediction set
                score = cal_score_PW(gt_is_in_set,Y_cal,Config.beta,Config.diff_between_lab) #score as defined in the paper

                if score > Config.beta:
                    cal_already_checked.append(i)

        R_hat = 1 - len(cal_already_checked)/n
        if R_hat <= (Config.alpha - (Config.B-Config.alpha)/n):
            converged = True
        elif lam>=Config.lam_max:
            print('Lambda has reached lambda_max, consider relaxing the parameters alpha and beta.')
            exit()
        else:
            lam += l
            if lam >= 1:
                print(f"\nLambda PW: {1}\n")
                return 1-1e-9
    print(f"\nLambda PW: {lam:.4f}\n")
    return lam


def PW_sample(lam_PW,X,N_samples,strategy,spatial):
    ############################ RANDOM SAMPLES ###########################################
    """
    input: calibrated lambda, test image X, number of samples to sample
    output: coordinates of uncertain pixels dim = (n_unc_pixels),
            random sampled labels for uncertain pixels dim = (N_samples,n_unc_pixels),
            labels included in the preiction set for the whole segmentation dim = (N_labels, H*W)
    """
    if spatial=='NO':
        X_flat = X.reshape(Config.labels,-1)
        sorted_idx = np.argsort(-X_flat, axis=0)
        sorted_vals = np.take_along_axis(X_flat, sorted_idx, axis=0)
        cumsum = np.cumsum(sorted_vals, axis=0)

        if strategy == 'APS':
            num_needed = (cumsum > lam_PW).argmax(axis=0) + 1
        elif strategy == 'RAPS':
            o_ranks = np.array([1+i for i in range(Config.labels)])
            reg = Config.theta*(o_ranks-Config.k_reg)
            reg[reg<0]=0
            shifted_cumsum = np.vstack([np.zeros((1, cumsum.shape[-1])), cumsum[:-1, :]])
            num_needed = (cumsum  > lam_PW).argmax(axis=0) + 1

    elif spatial=='SACP':
        if strategy == 'APS':
            # get new sorting
            X_avg = uniform_filter(X,size=Config.neigh_size,mode='reflect')
            X_blend = (1. - Config.neigh_weight)*X + Config.neigh_weight*X_avg
            X_flat = X_blend.reshape(Config.labels,-1)
            sorted_idx = np.argsort(-X_flat, axis=0)
            #
            sorted_vals = np.take_along_axis(X_flat, sorted_idx, axis=0)
            cum_p_avg = np.cumsum(sorted_vals, axis=0)

            ge_mask = (cum_p_avg > lam_PW)
            num_needed = (np.where(ge_mask.any(axis=0),ge_mask.argmax(axis=0) + 1, Config.labels)).reshape(-1)
        else:
            # get new sortings
            X_avg = uniform_filter(X,size=Config.neigh_size,mode='reflect')
            X_blend = (1. - Config.neigh_weight)*X + Config.neigh_weight*X_avg
            X_flat = X_blend.reshape(Config.labels,-1)
            sorted_idx = np.argsort(-X_flat, axis=0)
            #
            #For RAPS need to treat each sorting separately
            order = np.argsort(-X, axis=0)
            order_rows = order.transpose(1,2,0).reshape(-1,Config.labels)
            unique_orders, inv_flat = np.unique(order_rows,axis=0,return_inverse=True)
            inv2d = inv_flat.reshape(Config.dim,Config.dim)
            n_perms = unique_orders.shape[0]
            ranks = np.arange(1, Config.labels+1)
            reg = np.clip(Config.theta * (ranks - Config.k_reg), 0, None)
            final_cum_ordered = np.zeros((Config.labels,Config.dim,Config.dim),dtype=X.dtype)
            for pid,p_row in enumerate(unique_orders):
                p=p_row.astype(int).tolist()
                X_p = X[p,:,:]
                cum_p = np.cumsum(X_p,axis=0)
                cum_p += reg[:,None,None]
                cum_p_neigh = uniform_filter(cum_p,size=Config.neigh_size,mode='reflect')
                cum_p_blend = (1. - Config.neigh_weight)*cum_p + Config.neigh_weight*cum_p_neigh
                mask = (inv2d==pid)
                if not mask.any():
                    continue
                final_cum_ordered[:,mask] = cum_p_blend[:,mask]

            ge_mask = (final_cum_ordered > lam_PW)
            num_needed = (np.where(ge_mask.any(axis=0),ge_mask.argmax(axis=0) + 1, Config.labels)).reshape(-1)


    thresh_prediction_set = np.zeros_like(X_flat, dtype=bool)
    cols = np.arange(X_flat.shape[1])
    thresh_prediction_set[sorted_idx[:Config.labels, cols], cols] = np.less.outer(np.arange(Config.labels), num_needed).astype(bool)
    thresh_prediction_set = thresh_prediction_set.reshape(X.shape)

    #count how many labels per classes
    prediction_set = np.sum(thresh_prediction_set,axis=0)
    #flatten label per classes
    vec_prediction_set = prediction_set.flatten()
    #flatten softmax
    vec_thresh_prediction_set = thresh_prediction_set.reshape(Config.labels,-1)

    #find the pixels which more than one possible label. i.e. uncertain pixels
    unc_pixels = (np.where(vec_prediction_set>1))[0]
    #extract classes of the uncertain pixels
    unc_pixels_smx = vec_thresh_prediction_set[:,unc_pixels]
    #sample possible valid label for each pixel
    unc_pixels_samples = uq_samples(unc_pixels_smx,N_samples)

    return unc_pixels, unc_pixels_samples, vec_thresh_prediction_set

def PW_metrics(lam_PW,X,Y,N_samples,strategy,spatial='NO'):
    """
    input: calibrated lambda, test image X, gt Y, number of samples to sample
    output: sampled empirical coverage, Chao, number of uncertain pixels, correlation
    """

    unc_pixels, unc_pixels_samples, vec_thresh_prediction_set = PW_sample(lam_PW,X,N_samples,strategy,spatial)

    ################################## Evaluate Proj  #####################################
    pred_samples = uq_samples(vec_thresh_prediction_set,N_samples)
    score = val_score(pred_samples.reshape(N_samples,Config.dim,Config.dim),Y)

    #evaluate SEC
    flag_over_delta = np.any(score>Config.beta)

    if unc_pixels.shape[0]>0:
        #evaluate chao estimator
        col,count = np.unique(unc_pixels_samples.T,axis=1,return_counts=True)
        chao = count.shape[0] + (col.T[count == 1].shape[0]*(col.T[count == 1].shape[0]-1))/(2*(col.T[count == 2].shape[0]+1))
    else:
        chao = 1.

    #evaluate correlation
    if N_samples <= 1000:
        if unc_pixels_samples.shape[1]>1:
            corr_mean = torch.mean(torch.nan_to_num(torch.corrcoef(torch.abs(torch.tensor(unc_pixels_samples))), nan=0.0))
        else:
            corr_mean = 1.
    else:
        corr_mean = 0.

    n_unc_pixels = unc_pixels.shape[0]

    return np.array([flag_over_delta,chao,corr_mean])

def PW_plot(lam_PW,X,strategy,spatial='NO'):

    _, _, vec_thresh_prediction_set = PW_sample(lam_PW,X,Config.n_samples_4_visualization,strategy,spatial)
    samples = []
    ############################ PLOT STUFF ###############################################
    pred_samples = uq_samples(vec_thresh_prediction_set,Config.n_samples_4_visualization)
    for i in range(Config.n_samples_4_visualization):

        samples.append(pred_samples[i].reshape(Config.dim,Config.dim))

    return np.array(samples)

#
