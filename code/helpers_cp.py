import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.special import softmax
from config_cp import Config
import pickle
import torch
import torch.nn.functional as F
import torch.nn as nn

def print_loading_bar(iterations, total):
    progress = int(iterations / total * 100)
    bar_length = 20
    num_blocks = int(bar_length * progress / 100)
    bar = "[" + "=" * num_blocks + " " * (bar_length - num_blocks) + "]"
    print(f"\r{bar} {progress}%", end="", flush=True)

# reconstruction functions ##############################################################
def recon(c,U_cal,mean_cal):
    """
    input: coefficients c, principal vectors U, mean
    output: compute mean + sum_i u_i*c_i
    """
    proj = U_cal @ c
    return mean_cal + proj.view(Config.labels,Config.dim,Config.dim)


def recon_batch(c,U_cal,mean_cal):
    """
    input: coefficients c, principal vectors U, mean
    output: compute mean + sum_i u_i*c_i
    """
    proj = U_cal @ c.permute(1,0)
    return mean_cal[None,:] + proj.permute(1,0).view(-1,Config.labels,Config.dim,Config.dim)

def P(U,c,K_max,n_samples,X):
    """
    input: principal vectors U, coefficients c, number comopnents K, number of proj to compute, mean softmax X
    output: compute mean + sum_i u_i*c_i using gpu
    """

    U = torch.tensor(U, device='cuda', dtype=torch.float32)
    c = torch.tensor(c, device='cuda', dtype=torch.float32)

    proj_gpu = torch.zeros((Config.labels, Config.dim, Config.dim, n_samples), device='cuda', dtype=torch.float32)

    c = c.T
    for k in range(K_max):
        proj_gpu += c[k][None,None,None,:] * U[:,k].reshape(Config.labels,Config.dim,Config.dim)[:,:,:,None]

    proj = proj_gpu.cpu().numpy()
    sigma = X[:,:,:,None] + proj
    return sigma


##############################################
# score functions ##############################################################

def cal_score_SVD(sigma,Y,beta,diff_between_lab):
    """
    input: projection sigma, gt Y, threshold accuracy beta, binary or multiclass flag
    output: score
    """
    predictions = np.argmax(sigma,axis=0)
    if not diff_between_lab:
        return np.sum(predictions==Y)/Config.dim**2 > beta
    else:
        score = []
        labels = np.unique(Y)
        for l in labels:
            score.append(np.sum((predictions==Y)*(Y==l))/np.sum(Y==l))
        return np.mean(np.array(score),axis=0)

def cal_score_PW(gt_is_in_set,Y,beta,diff_between_lab):
    """
    input: boolean matrix gt_is_in_set, gt Y, threshold accuracy beta, binary or multiclass flag
    output: score
    """
    if not diff_between_lab:
        return np.sum(gt_is_in_set)/(dim**2)>Config.beta
    else:
        score = []
        labels = np.unique(Y)
        for l in labels:
            score.append(np.sum(gt_is_in_set*(Y==l))/(np.sum(Y==l)))
        return np.mean(score)

def val_score(predictions,Y):
    """
    input: predictions matrix, gt Y
    output: score
    """
    labels = np.unique(Y)
    score = []
    for l in labels:
        score.append(np.sum((predictions==Y[None,:,:])*(Y==l)[None,:,:],axis=(1,2))/np.sum(Y==l))
    return np.mean(np.array(score),axis=0)
#######################
# load function ##############################################################
def extract_softmax():
    """
    extract predictions of pre-trained models
    """
    # Download data
    print('Extracting softmax...\n')
    challenge = Config.challenge
    # load the softmax and the ground truth
    file_softmax = f'./softmax/{challenge}/smx_{challenge}.pkl'
    file_gt = f'./softmax/{challenge}/labels_{challenge}.pkl'
    file_img = f'./softmax/{challenge}/imgs.pkl'

    with open(file_softmax, 'rb') as file:
        smx_MC = np.array(pickle.load(file)) #[N_samples,N_labels,H,W]
    with open(file_gt, 'rb') as file:
        labels = np.array(pickle.load(file)) #[N_samples,H,W]
    with open(file_img, 'rb') as file:
        imgs = np.array(pickle.load(file)) #[N_samples,H,W]

    print("Done!\n")

    ## remove cardiac images with no heart foreground
    if Config.challenge in ['MnM2','mscmr19']:
        index_n0 = [i for i,lbl in enumerate(labels[:len(smx_MC)]) if lbl.max()>0]
        labels = labels[:len(smx_MC)][index_n0]
        imgs = imgs[:len(smx_MC)][index_n0]
        smx_MC = smx_MC[index_n0]

    n = Config.n #number of calibration images
    np.random.seed(111111) #for reproducibility
    cal_splits = []
    val_splits = []
    for i in range(5):
        cal_splits.append(np.random.choice(np.arange(smx_MC.shape[0]),n,replace=False))
        val_splits.append(np.array([v for v in range(smx_MC.shape[0]) if v not in cal_splits[i]]))

    return smx_MC, labels, imgs, cal_splits, val_splits

##############################################################
# sampling function ##############################################################

def uq_samples(smx,n_samples):
    """
    input: softmax smx, #samples to randomly sample
    output: random samples
    """
    samples = []
    if n_samples==0:
        n_samples=smx.shape[1]

    num_rows, num_cols = smx.shape
    sampled_indices = np.full((n_samples, num_cols), 0, dtype=int)  # preallocate with 0

    for i in range(num_cols):
        true_indices = np.where(smx[:, i])[0]  # get True indices
        if true_indices.size > 0:
            sampled_indices[:, i] = np.random.choice(true_indices, size=n_samples, replace=True)

    return sampled_indices

##############################################################
# optimization loss function ##############################################################
class CoverageLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(CoverageLoss, self).__init__()
        self.smooth = smooth

    def forward(self, preds, targets,parallel=False):
        """
        input: predictions preds, gt seg targets
        targets: smooth dice loss
        """
        num_classes = preds.shape[1]
        batch_size = preds.shape[0]
        # convert predictions to class labels (argmax over classes)
        targets_one_hot = torch.nn.functional.one_hot(targets.to(torch.long), num_classes).permute(0, 3, 1, 2).float()

        softargmax = torch.nn.functional.softmax(preds/0.0001, dim=1)
        intersection = (softargmax * targets_one_hot).sum(dim=(2, 3))  # Sum over spatial dimensions
        union = targets_one_hot.sum(dim=(2, 3))
        # compute the loss: Average of (1 - coverage) for each class
        valid_classes = targets_one_hot.sum(dim=(2, 3)) > 0

        coverage = (intersection + self.smooth) / (union + self.smooth)
        if parallel:
            coverage = coverage[valid_classes.repeat(coverage.shape[0],1)]
            return 1 - torch.mean(coverage.reshape(-1,int(torch.sum(valid_classes))),dim=1)
        else:
            coverage = coverage[valid_classes]
        loss = 1 - coverage.mean()


        return loss

##############################################################
############### helpers function for plots ##############################################################
def colorized(img,label_to_color):
    img = np.array(np.vectorize(lambda x: label_to_color.get(x, (1, 1, 1)))(img))
    return np.transpose(img[:3,:,:],(1,2,0))

def heart_visual(image,PW_pred,SVD_pred,SACP_pred,gt):
    label_names = {0: "Background", 1: "Left Ventricle", 2: "Myocardium",3: "Right Ventricle"}

    cmap = plt.cm.get_cmap("tab20", len(label_names))
    label_to_color = {label: cmap(idx) for idx, label in enumerate(label_names.keys())}
    patches = []
    for label, idx in label_names.items():
        if isinstance(label, int):  # ensure the label is an integer
            patch = mpatches.Patch(color=label_to_color[label], label=f"{label} {idx}")
            patches.append(patch)
        else:
            print(f"Warning: Label {label} is not an integer. Skipping.")

    fig, ax = plt.subplots(1,Config.n_samples_4_visualization*3+2,figsize=(30,8))
    image = (image - image.min()) / (image.max() - image.min())
    image = np.clip(image + 0.5, 0, 1)
    ax[0].imshow(image,cmap='gray')
    ax[1].imshow(colorized(gt,label_to_color),cmap=cmap, interpolation='none')

    for i in range(Config.n_samples_4_visualization):
        ax[i+2].imshow(colorized(SVD_pred[i],label_to_color),cmap=cmap, interpolation='none')
        ax[i+2+Config.n_samples_4_visualization].imshow(colorized(PW_pred[i],label_to_color),cmap=cmap, interpolation='none')
        ax[i+4+Config.n_samples_4_visualization].imshow(colorized(SACP_pred[i],label_to_color),cmap=cmap, interpolation='none')
    for i in range(Config.n_samples_4_visualization*3+2):
        ax[i].set_yticks([])
        ax[i].set_xticks([])

    fig.legend(handles=patches, loc="lower center", title="",fontsize=30,ncol=4)
    plt.tight_layout()
    plt.savefig(f'{Config.challenge}.png')
    plt.show()
    return 0

def COCO_visual(image,PW_pred,SVD_pred,SACP_pred,gt):
    if Config.challenge == 'COCO_animals':
        label_names = {0: "Background", 1: "Bird", 2: "Cat", 3: "Cow", 4: "Dog", 5: "Horse", 6: "Person", 7: "Sheep"}
    else:
        label_names = {0: "Background", 1: "Plane", 2: "Bycicle", 3: "Bus", 4: "Car", 5: "Motorbike", 6: "Person", 7: "Train"}

    cmap = plt.cm.get_cmap("tab20", len(label_names))
    label_to_color = {label: cmap(idx) for idx, label in enumerate(label_names.keys())}
    patches = []
    for label, idx in label_names.items():
        if isinstance(label, int):  # ensure the label is an integer
            patch = mpatches.Patch(color=label_to_color[label], label=f"{label} {idx}")
            patches.append(patch)
        else:
            print(f"Warning: Label {label} is not an integer. Skipping.")

    fig, ax = plt.subplots(1,Config.n_samples_4_visualization*3+2,figsize=(30,8))
    image = (image - image.min()) / (image.max() - image.min())
    ax[0].imshow(np.transpose(image,(1,2,0))[128:384,128:384,:])
    ax[1].imshow(colorized(gt,label_to_color),cmap=cmap, interpolation='none')

    for i in range(Config.n_samples_4_visualization):
        ax[i+2].imshow(colorized(SVD_pred[i],label_to_color),cmap=cmap, interpolation='none')
        ax[i+2+Config.n_samples_4_visualization].imshow(colorized(PW_pred[i],label_to_color),cmap=cmap, interpolation='none')
        ax[i+4+Config.n_samples_4_visualization].imshow(colorized(SACP_pred[i],label_to_color),cmap=cmap, interpolation='none')
    for i in range(Config.n_samples_4_visualization*3+2):
        ax[i].set_xticks([])
        ax[i].set_yticks([])

    plt.tight_layout()
    fig.legend(handles=patches, loc="lower center", title="",fontsize=30,ncol=8)
    plt.savefig(f'{Config.challenge}.png')
    plt.show()
    return 0

def LIDC_visual(image,PW_pred,SVD_pred,SACP_pred,gt):
    label_names = {0: "Background", 1: "Cancer"}

    cmap = plt.cm.get_cmap("tab20", len(label_names))
    label_to_color = {label: cmap(idx) for idx, label in enumerate(label_names.keys())}
    patches = []
    for label, idx in label_names.items():
        if isinstance(label, int):  # ensure the label is an integer
            patch = mpatches.Patch(color=label_to_color[label], label=f"{label} {idx}")
            patches.append(patch)
        else:
            print(f"Warning: Label {label} is not an integer. Skipping.")

    fig, ax = plt.subplots(1,Config.n_samples_4_visualization*3+2,figsize=(30,8))
    image = (image - image.min()) / (image.max() - image.min())
    ax[0].imshow(image,cmap='gray')
    ax[1].imshow(colorized(gt,label_to_color),cmap=cmap, interpolation='none')

    for i in range(Config.n_samples_4_visualization):
        ax[i+2].imshow(colorized(SVD_pred[i],label_to_color),cmap=cmap, interpolation='none')
        ax[i+2+Config.n_samples_4_visualization].imshow(colorized(PW_pred[i],label_to_color),cmap=cmap, interpolation='none')
        ax[i+4+Config.n_samples_4_visualization].imshow(colorized(SACP_pred[i],label_to_color),cmap=cmap, interpolation='none')
    for i in range(Config.n_samples_4_visualization*3+2):
        ax[i].set_xticks([])
        ax[i].set_yticks([])

    plt.tight_layout()
    fig.legend(handles=patches, loc="lower center", title="",fontsize=30,ncol=2)
    plt.savefig(f'{Config.challenge}.png')
    plt.show()
    return 0


##############################################################
#### read functions ##############################################################
def extract_lambdas(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        lambdas = []
        for i in [32, 36, 40, 44, 48]:
            line = lines[i].strip()
            parts = line.split()
            lambdas.append(np.array([float(parts[0]), float(parts[1]), float(parts[2])]))

        return np.array(lambdas)


def extract_metrics(file_path):
    cov = []
    chao = []
    corr = []
    with open(file_path, "r") as file:
        for idx, line in enumerate(file):
            if idx in [1,2,4,5,7,8]:
                numbers = [float(x) for x in line.replace(line.split('=')[0], '').replace('=', '').strip().split()]
                cov.append(numbers)
            elif idx in [11,12,14,15,17,18]:
                numbers = [float(x) for x in line.replace(line.split('=')[0], '').replace('=', '').strip().split()]
                chao.append(numbers)
            elif idx in [21,22,24,25,27,28]:
                numbers = [float(x) for x in line.replace(line.split('=')[0], '').replace('=', '').strip().split()]
                corr.append(numbers)
    return np.array(cov),np.array(chao),np.array(corr)
