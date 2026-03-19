""" Config file containing all the hyeperparameters of
    the CNN and the GP
"""
import torch
from torch.cuda.amp import GradScaler
import numpy as np

class Config:
    """Hyeperparameters config file"""

    challenge = 'COCO_vehicles' #mscmr19 #MnM2 #COCO_animals #COCO_vehicles #LIDC

    # calibration hyeperparameters
    alpha = 0.3 #confidence
    beta = 0.7 #threshold accuracy
    lam = 0.01 #initial lambda for CONSIGN calibration
    step_lam = 0.01 #step lambda for CONSIGN calibration
    lam_max = 500 #maximum lambda after which algorithm stops
    K_max = 5 #number of principal components
    converged = False #if True only metrics will be evaluated (True need .txt file with calibrated lambdas)
    neigh_size = (1, 7, 7) #size of neighborhood to compute average scores
    neigh_weight = 0.3 #weight of neighborhood pixels
    ########
    diff_between_lab = True #differentiate labels in the loss
    B = 1 #maximum of the loss

    if challenge in ['mscmr19','MnM2']:
        labels = 4
        n=500 #calibration image
        dim = 128 #image dimension
    elif challenge in ['COCO_animals','COCO_vehicles']:
        labels = 8
        n = 275 #calibration image
        dim = 256 #image dimension
    elif challenge == 'LIDC':
        labels = 2
        n = 700 #calibration image
        dim = 128 #image dimension

    #######RAPS
    theta = 0.05
    k_reg = labels//2
    pw_strategy = 'APS' # or RAPS # The results for APS are \approx the results for RAPS
    ###############################

    #optimization algorithm for CONSIGN
    epochs = 10000
    init_trials = 10 #number of random initial points
    if challenge in ['MnM2','mscmr19','LIDC']:
        lr = 1
    else:
        lr = 10

    ## test
    n_samples_4_metrics = [10,50,100,500,1000,5000,10000] #samples to evaluate metrics
    n_samples_4_visualization = 2 #sample to plot
    #to not exceed GPU memory maximum capacity
    if challenge in ['MnM2','mscmr19','LIDC']:
        chunks = 10000
    else:
        chunks = 1000

    #alphas and beta for plotting metrics. The order is: MnM2,mscmr19,LIDC,COCO_animals,COCO_vehicles
    alphas = [0.1,0.05,0.2,0.25,0.3]
    betas = [0.9,0.8,0.8,0.8,0.7]
