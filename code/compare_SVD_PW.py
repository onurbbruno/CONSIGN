import numpy as np
from helpers_cp import *
from config_cp import Config
from PW_conformal_prediction import PW_calibration,PW_metrics
from SVD_conformal_prediction_UQ import SVD_calibration,SVD_preprocess,SVD_metrics
import os

print(f'\nStarting calibration for {Config.challenge} dataset')
# exctract softmax
smx, labels, imgs, c_splits, v_splits = extract_softmax()
# list of stats
cross_val_metrics_SVD = []
cross_val_metrics_PW = []
cross_val_metrics_SACP = []
lambdas = []

# for loop for different validation splits
for cal,(c_split,v_split) in enumerate(zip(c_splits,v_splits)):
    print(f'\n########## Calibration split {cal+1} #############\n')
    # calibration validation split
    cal_smx = smx[c_split,:]
    cal_labels = labels[c_split,:]
    cal_imgs = imgs[c_split,:]

    val_smx = smx[v_split,:]
    val_labels = labels[v_split,:]
    val_imgs = imgs[v_split,:]

    if not Config.converged:
        # compute calibration lambda
        lam_PW = PW_calibration(cal_smx,cal_labels,Config.pw_strategy)
        lam_SVD = SVD_calibration(cal_smx,cal_labels)
        lam_SACP = PW_calibration(cal_smx,cal_labels,Config.pw_strategy,'SACP')
    else:
        # load lambdas
        lams = extract_lambdas(f"metrics/{Config.challenge}_{Config.alpha}_{Config.beta}_{Config.K_max}.txt")
        lam_SVD = lams[:,0][cal]
        lam_PW = lams[:,1][cal]
        lam_SACP = lams[:,2][cal]

    lambdas.append(np.array([lam_SVD,lam_PW,lam_SACP]))

    # number test images
    n_test = val_smx.shape[0]

    # pre stuff for SVD and mean softmax for PW
    softmax_mean = np.mean(val_smx,axis=1)
    a_bound,b_bound,U,Sig,mean_sample = SVD_preprocess(val_smx)

    metrics_PW = []
    metrics_SVD = []
    metrics_SACP = []

    for N_samples in Config.n_samples_4_metrics:
        print(f'Evaluating metrics with {N_samples} samples')
        avg_metrics_PW = 0
        avg_metrics_SVD = 0
        avg_metrics_SACP = 0

        for i in range(n_test):
            X = softmax_mean[i,:] # current softmax
            Y = val_labels[i,:] # gt
            U_test = U[i,:] # principal vectors
            Sig_test = Sig[i,:] # singular values
            avg_metrics_SACP += PW_metrics(lam_SACP,X,Y,N_samples,Config.pw_strategy,'SACP') # metrics
            avg_metrics_PW += PW_metrics(lam_PW,X,Y,N_samples,Config.pw_strategy) # metrics
            avg_metrics_SVD += SVD_metrics(lam_SVD,a_bound[i,:],b_bound[i,:],U_test,mean_sample[i,:],Sig_test,Y,N_samples) # metrics

        metrics_PW.append(avg_metrics_PW/n_test)
        metrics_SVD.append(avg_metrics_SVD/n_test)
        metrics_SACP.append(avg_metrics_SACP/n_test)

    cross_val_metrics_PW.append(metrics_PW)
    cross_val_metrics_SVD.append(metrics_SVD)
    cross_val_metrics_SACP.append(metrics_SACP)

##########################################################
# average across validations splits
cross_val_metrics_SVD = np.array(cross_val_metrics_SVD)
cross_val_metrics_PW = np.array(cross_val_metrics_PW)
cross_val_metrics_SACP = np.array(cross_val_metrics_SACP)

mean_SVD = np.mean(cross_val_metrics_SVD,axis=0)
std_SVD = np.std(cross_val_metrics_SVD,axis=0)

mean_PW = np.mean(cross_val_metrics_PW,axis=0)
std_PW = np.std(cross_val_metrics_PW,axis=0)

mean_SACP = np.mean(cross_val_metrics_SACP,axis=0)
std_SACP = np.std(cross_val_metrics_SACP,axis=0)


# write results in .txt
os.makedirs('metrics', exist_ok=True)
titles = [f'Coverage',f'Chao estimator',f'Correlation']
with open(f"metrics/{Config.challenge}_{Config.alpha}_{Config.beta}_{Config.K_max}.txt", "w") as f:
    for i in range(mean_SVD.shape[1]):
        f.write(f"{titles[i]}:\n")
        print1 = ' '.join(f"{val:.3f}" for val in mean_SVD[:, i])
        print2 = ' '.join(f"{val:.3f}" for val in std_SVD[:, i])
        print3 = ' '.join(f"{val:.3f}" for val in mean_PW[:, i])
        print4 = ' '.join(f"{val:.3f}" for val in std_PW[:, i])
        print5 = ' '.join(f"{val:.3f}" for val in mean_SACP[:, i])
        print6 = ' '.join(f"{val:.3f}" for val in std_SACP[:, i])
        f.write(f"mean SVD = {print1}\n")
        f.write(f"std SVD = {print2}\n\n")
        f.write(f"mean PW = {print3}\n")
        f.write(f"std PW = {print4}\n\n")
        f.write(f"mean SACP = {print5}\n")
        f.write(f"std SACP = {print6}\n\n")
    for i, arr3 in enumerate(lambdas):
        f.write(f"# lambdas split {i+1}\n")
        np.savetxt(f, arr3[None, :], fmt="%.3f", header="lambda", comments='')
        f.write("\n")
