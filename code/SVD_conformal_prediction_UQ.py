import numpy as np
import matplotlib.pyplot as plt
from helpers_cp import *
import torch
from config_cp import Config
from scipy.stats.qmc import LatinHypercube
import torch.nn as nn
import torch.optim as optim
import os

import logging

def SVD_calibration(cal_smx,cal_labels):
    print('SVD Calibration:')
    n = Config.n
    ###########################################################
    a_bound,b_bound,U,Sig,mean_sample = SVD_preprocess(cal_smx)
    ###########################################################

    ################# Calibration phase #######################
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lam = torch.tensor(Config.lam).to(device)
    # bringing everything to tensor and device
    a_bound = torch.tensor(a_bound).to(device)#quantiles bounds
    b_bound = torch.tensor(b_bound).to(device)#quantiles bounds

    converged = Config.converged
    cal_already_checked = [] #list of images that satisfy coverage condition

    while not converged:
        for i in range(n):
            print_loading_bar(i, n)
            if i not in cal_already_checked:
                Y_cal = cal_labels[i,:] #gt
                mean_cal = torch.tensor(mean_sample[i,:]).to(device) #mean of the samples
                U_cal = torch.tensor(U[i,:,:Config.K_max]).double().to(device) #principal directions
                Sig_cal = torch.tensor(Sig[i,:Config.K_max]).to(device) #singular values

                ### bounds ######
                mid_point = (a_bound[i,:]+b_bound[i,:])/2
                length = Sig_cal*lam*(b_bound[i,:]-a_bound[i,:])
                l_b = mid_point - length/2 #A(X)
                up_b = mid_point + length/2 #B(X)
                ############

                ######## optimization ###################
                criterion = CoverageLoss().to(device)
                # sampling random starting points
                sampler = LatinHypercube(Config.K_max)
                random_coeff = torch.tensor(sampler.random(Config.init_trials*100)).to(device)
                random_coeff = random_coeff*(length) + l_b
                try:
                    random_losses = criterion(recon_batch(random_coeff,U_cal,mean_cal), torch.tensor(Y_cal)[None,:].to(device),True)
                except:
                    # if doesnt fit in memory
                    random_losses = []
                    for chunk in range(20):
                        random_losses.append(criterion(recon_batch(random_coeff[chunk*50:(chunk+1)*50,:],U_cal,mean_cal), torch.tensor(Y_cal)[None,:].to(device),True))
                    random_losses = torch.stack(random_losses).view(-1)

                #select most promising starting point
                min_losses = torch.argsort(random_losses)
                random_coeff = random_coeff[min_losses[:Config.init_trials]]

                for trial in range(Config.init_trials):
                    coeff = random_coeff[trial].requires_grad_(True)
                    optimizer = optim.Adam([coeff], lr=Config.lr)
                    early_count = 0
                    best_loss = float('inf')
                    patience = 50
                    for epoch in range(Config.epochs):
                        optimizer.zero_grad()
                        smx = recon(coeff,U_cal,mean_cal) # smx = mu + sum c_i*u_i
                        loss = criterion(smx[None,:], torch.tensor(Y_cal)[None,:].to(device))
                        if loss < 1-Config.beta:
                            break
                        loss.backward() #compute gradients
                        optimizer.step() #update
                        coeff.data.clamp_(min=l_b, max=up_b) #project

                        if loss.item() < best_loss:
                            best_loss = loss
                            early_count = 0
                        else:
                            early_count += 1
                        if early_count >= patience:
                            break

                    ############################################
                    sigma = recon(coeff,U_cal,mean_cal)
                    score = cal_score_SVD(sigma.detach().cpu().numpy(),Y_cal,Config.beta,Config.diff_between_lab)

                    if score > Config.beta:
                        cal_already_checked.append(i)
                        break

        R_hat = 1 - len(cal_already_checked)/n
        print(f"New risk = {R_hat:.4f}")
        if R_hat <= (Config.alpha - (Config.B-Config.alpha)/n):
            converged = True
        elif lam>=Config.lam_max:
            print('Lambda has reached lambda_max, consider relaxing the parameters alpha and beta.')
            exit()
        else:
            lam += Config.step_lam

    print(f"\nLambda SVD: {lam:.2f}\n")
    return lam.detach().cpu().numpy()
###########################################################

def SVD_preprocess(smx):
    """
    input: softmax scores, gts
    output: quantiles lower bound, quantiles upper bound, principal components u_k, singular values, mean sample
    """
    n = smx.shape[0]
    dim = smx.shape[-1]
    mean_sample = np.mean(smx,axis=1)
    distances = smx-mean_sample[:,None,:]
    reshaped_distances = np.transpose(distances.reshape(-1,smx.shape[1], (dim**2)*Config.labels),(0,2,1))
    reshaped_distances = torch.tensor(reshaped_distances)
    U, Sig, Vt = torch.linalg.svd(reshaped_distances, full_matrices=False)
    U = np.array(U)
    Sig = np.array(Sig)
    Vt = np.array(Vt)
    a_bound = np.zeros((n,Config.K_max))
    b_bound = np.zeros((n,Config.K_max))
    coefficients = Sig[..., None] * Vt

    for k in range(Config.K_max):
        inner_product = coefficients[:,k]
        a_bound[:,k] = np.quantile(inner_product,Config.alpha/2,method = 'linear', axis = 1)
        b_bound[:,k] = np.quantile(inner_product,1-(Config.alpha/2),method = 'linear', axis = 1)

    return a_bound,b_bound,U,Sig,mean_sample

def SVD_sample(lam_SVD,a_bound,b_bound,U,X,Sig,N_samples):
    """
    input: calibrated lambda, low quanitle boun, up quantile bound, principal component, test image X, singular values, number of samples to sample
    output: coordinates of uncertain pixels dim = (n_unc_pixels),
            random sampled labels for uncertain pixels dim = (N_samples,n_unc_pixels),
            random sampled labels for whole image dim = (N_samples, H*W)
    """
    ################################ SAMPLE ##################################
    #sampled coefficients c_k
    c = np.zeros((N_samples,Config.K_max))
    #sample all the coefficients
    for k in range(Config.K_max):
        l_b = (a_bound[k]+b_bound[k])/2 - Sig[k]*lam_SVD*(b_bound[k]-a_bound[k])/2
        c[:,k] = np.random.rand(N_samples)*(Sig[k]*lam_SVD*(b_bound[k]-a_bound[k])) + l_b

    if N_samples<=Config.chunks:
        sigma = P(U,c,Config.K_max,N_samples,X)
        #mean softmax + sum_k <u_k,nu>u_k
        # sigma = X[:,:,:,None] + proj # (labels,dim,dim,n_samples)
        pred_samples = (np.argmax(sigma,axis=0).reshape(-1,N_samples)).T

    else:
        pred_samples = []
        for j in range(N_samples//Config.chunks):
            #construct sampled segmentation of the form: mean_softmax + sum_k c_k*u_k
            pred_samples_tmp = []
            sigma = P(U,c[j*Config.chunks:(j+1)*Config.chunks],Config.K_max,Config.chunks,X)
            #mean softmax + sum_k c_ku_k
            # sigma = X[:,:,:,None] + proj # (labels,dim,dim,n_samples)

            pred_samples_tmp = (np.argmax(sigma,axis=0).reshape(-1,Config.chunks)).T
            pred_samples.append(pred_samples_tmp)
        pred_samples = np.array(pred_samples).reshape(N_samples,-1)


    #count how many pixels change values in the different N_samples, i.e. how many uncertain pixels
    unc_pixels = np.where(pred_samples.ptp(axis=0) > 0)[0]
    # select only uncertain pixels
    unc_pixels_samples = pred_samples[:,unc_pixels]
    return unc_pixels,unc_pixels_samples,pred_samples

def SVD_metrics(lam_SVD,a_bound,b_bound,U,X,Sig,Y,N_samples):
    """
    input: calibrated lambda, low quanitle boun, up quantile bound, principal component, test image X, singular values, gt Y, number of samples to sample
    output: sampled empirical coverage, Chao, number of uncertain pixels, correlation
    """

    unc_pixels,unc_pixels_samples,pred_samples = SVD_sample(lam_SVD,a_bound,b_bound,U,X,Sig,N_samples)

    ################################## Evaluate Proj  #####################################
    score = val_score(pred_samples.reshape(N_samples,Config.dim,Config.dim),Y)
    #evaluate SEC
    flag_over_delta = np.any(score>Config.beta)

    if unc_pixels.shape[0]>0:
        #evaluate chao estimator
        col,count = np.unique(unc_pixels_samples.T,axis=1,return_counts=True)
        chao = count.shape[0] + (col.T[count == 1].shape[0]*(col.T[count == 1].shape[0]-1))/(2*(col.T[count == 2].shape[0]+1))
    else:
        chao = 1

    #evalaute correalation
    if N_samples <= 1000:
        if unc_pixels_samples.shape[1]>1:
            corr_mean = torch.mean(torch.nan_to_num(torch.corrcoef(torch.abs(torch.tensor(unc_pixels_samples))), nan=0.0))
        else:
            corr_mean = 1
    else:
        corr_mean = 0

    n_unc_pixels = unc_pixels.shape[0]

    return np.array([flag_over_delta,chao,corr_mean])

def SVD_plot(lam_SVD,a_bound,b_bound,U,X,Sig):

    _,_,pred_samples = SVD_sample(lam_SVD,a_bound,b_bound,U,X,Sig,Config.n_samples_4_visualization)

    ############################ PLOT STUFF ##############################################
    samples = []
    for i in range(Config.n_samples_4_visualization):
        samples.append(pred_samples[i,:].reshape(Config.dim,Config.dim))

    return np.array(samples)
