import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from helpers_cp import *
from matplotlib.ticker import LogLocator
from matplotlib.ticker import LogFormatterSciNotation
sns.set_style("whitegrid")
colors = sns.color_palette("tab10", 4)

cov2 = []
corr2 = []
chao2 = []
cov5 = []
corr5 = []
chao5 = []

chal = ['MnM2','mscmr19','LIDC','COCO_animals','COCO_vehicles']
for challenge in [f'MnM2_{Config.alphas[0]}_{Config.betas[0]}',f'mscmr19_{Config.alphas[1]}_{Config.betas[1]}',f'LIDC_{Config.alphas[2]}_{Config.betas[2]}',f'COCO_animals_{Config.alphas[3]}_{Config.betas[3]}',f'COCO_vehicles_{Config.alphas[4]}_{Config.betas[4]}']:
    cov, chao, corr = extract_metrics(f"metrics/{challenge}_{2}.txt")
    cov2.append(cov)
    corr2.append(corr)
    chao2.append(chao)
    cov, chao, corr = extract_metrics(f"metrics/{challenge}_{5}.txt")
    cov5.append(cov)
    corr5.append(corr)
    chao5.append(chao)

cov2 = np.array(cov2)
cov5 = np.array(cov5)
chao2 = np.array(chao2)
chao5 = np.array(chao5)
corr2 = np.array(corr2)
corr5 = np.array(corr5)

################################################################################
################################################################################
################################################################################
################################################################################
fig, ax = plt.subplots(1,5,figsize=(21,8))
n_samples = np.array(Config.n_samples_4_metrics)
n_samples_cov = [10,5000,10000]
columns=[0,5,6]

cov_SVD2 = np.array(cov2[:,0,columns])
std_cov_SVD2 = np.array(cov2[:,1,columns])

cov_SVD5 = np.array(cov5[:,0,columns])
std_cov_SVD5 = np.array(cov5[:,1,columns])

cov_PW = np.array(cov2[:,2,columns])
std_cov_PW = np.array(cov2[:,3,columns])

cov_SACP = np.array(cov2[:,4,columns])
std_cov_SACP = np.array(cov2[:,5,columns])

# Bar width and positions
bar_width = 0.18
x_indices = np.arange(len(n_samples_cov))

for i in range(5):
    if i==0:
        ax[i].bar(x_indices - bar_width, cov_SVD2[i], yerr=std_cov_SVD2[i], width=bar_width, color=colors[0],label=r"$CONSIGN_2$", capsize=5)
        ax[i].bar(x_indices,            cov_SVD5[i], yerr=std_cov_SVD5[i], width=bar_width, color=colors[2],label=r"$CONSIGN_5$", capsize=5)
        ax[i].bar(x_indices + bar_width, cov_PW[i], yerr=std_cov_PW[i], width=bar_width, color=colors[1],label='PW (RAPS)', capsize=5)
        ax[i].bar(x_indices + 2*bar_width, cov_SACP[i], yerr=std_cov_SACP[i], width=bar_width, color=colors[3],label='SACP', capsize=5)
        ax[i].axhline(y=1-Config.alphas[i],label=r'1-$\alpha$', color='black', linestyle='--', linewidth=1)

    else:
        ax[i].bar(x_indices - bar_width, cov_SVD2[i], yerr=std_cov_SVD2[i], width=bar_width, color=colors[0], capsize=5)
        ax[i].bar(x_indices,            cov_SVD5[i], yerr=std_cov_SVD5[i], width=bar_width, color=colors[2], capsize=5)
        ax[i].bar(x_indices + bar_width, cov_PW[i], yerr=std_cov_PW[i], width=bar_width, color=colors[1], capsize=5)
        ax[i].bar(x_indices + 2*bar_width, cov_SACP[i], yerr=std_cov_SACP[i], width=bar_width, color=colors[3], capsize=5)
        ax[i].axhline(y=1-Config.alphas[i], color='black', linestyle='--', linewidth=1)

    # Customization
    ax[i].set_xticks(x_indices)
    ax[i].set_xticklabels([str(i) for i in n_samples_cov])
    ax[i].set_xlim(-0.5, len(n_samples_cov) - 0.5)
    ax[i].set_ylim(0.05, 1.02)
    # ax[i].set_xlabel('# samples', fontsize=40)
    ax[i].tick_params(axis='y', labelsize=40)
    ax[i].tick_params(axis='x', labelsize=25)
    ax[i].grid(True, axis='y', linestyle='--', alpha=0.7)
    ax[i].set_title(f'{chal[i]}\n'.replace('_',' ')+rf' $\alpha={Config.alphas[i]}$ $\beta={Config.betas[i]}$',fontsize=30)

    if i==0:
        # ax[i].legend(loc = 'upper center',fontsize=15,ncol=2)
        ax[i].set_ylabel(r'$sEC$', fontsize=40)
    else:
        ax[i].set_yticklabels([])

    if i ==2:
        ax[i].set_xlabel('# samples', fontsize=40)

fig.legend(loc='upper center', ncol=5, fontsize=25)
plt.tight_layout(rect=[0, 0, 1, 0.89])
plt.savefig(f'cov.pdf',bbox_inches='tight')
plt.show()

################################################################################
################################################################################
################################################################################
################################################################################
fig, ax = plt.subplots(1,5,figsize=(21,8))
colors = sns.color_palette("tab10", 4)
n_samples = np.array(Config.n_samples_4_metrics)

chao_SVD2 = np.array(chao2[:,0,:])
std_chao_SVD2 = np.array(chao2[:,1,:])

chao_SVD5 = np.array(chao5[:,0,:])
std_chao_SVD5 = np.array(chao5[:,1,:])

chao_PW = np.array(chao2[:,2,:])
std_chao_PW = np.array(chao2[:,3,:])

chao_SACP = np.array(chao2[:,4,:])
std_chao_SACP = np.array(chao2[:,5,:])

sns.set_style("whitegrid")
for i in range(5):
    if i==0:
        ax[i].plot(n_samples,chao_SVD2[i], label= r"$CONSIGN_2$", color=colors[0], marker='^', linestyle='-',linewidth=3,markersize=20,
         markeredgecolor="black", markeredgewidth=1.2)
        ax[i].fill_between(n_samples, chao_SVD2[i] - std_chao_SVD2[i], chao_SVD2[i] + std_chao_SVD2[i], color=colors[0], alpha=0.2)
        ax[i].plot(n_samples,chao_SVD5[i], label=r"$CONSIGN_5$", color=colors[2], marker='v', linestyle='-',linewidth=3,markersize=20,
         markeredgecolor="black", markeredgewidth=1.2)
        ax[i].fill_between(n_samples, chao_SVD5[i] - std_chao_SVD5[i], chao_SVD5[i] + std_chao_SVD5[i], color=colors[2], alpha=0.2)
        ax[i].plot(n_samples,chao_PW[i], label=f"PW (RAPS)", color=colors[1], marker='o', linestyle='-',linewidth=3,markersize=20,
         markeredgecolor="black", markeredgewidth=1.2)
        ax[i].fill_between(n_samples, chao_PW[i] - std_chao_PW[i], chao_PW[i] + std_chao_PW[i], color=colors[1], alpha=0.2)
        ax[i].plot(n_samples,chao_SACP[i], label=f"SACP", color=colors[3], marker='s', linestyle='-',linewidth=3,markersize=15,
         markeredgecolor="black", markeredgewidth=1.2)
        ax[i].fill_between(n_samples, chao_SACP[i] - std_chao_SACP[i], chao_SACP[i] + std_chao_SACP[i], color=colors[3], alpha=0.2)
    else:
        ax[i].plot(n_samples,chao_SVD2[i],  color=colors[0], marker='^', linestyle='-',linewidth=3,markersize=20,
         markeredgecolor="black", markeredgewidth=1.2)
        ax[i].fill_between(n_samples, chao_SVD2[i] - std_chao_SVD2[i], chao_SVD2[i] + std_chao_SVD2[i], color=colors[0], alpha=0.2)
        ax[i].plot(n_samples,chao_SVD5[i],  color=colors[2], marker='v', linestyle='-',linewidth=3,markersize=20,
         markeredgecolor="black", markeredgewidth=1.2)
        ax[i].fill_between(n_samples, chao_SVD5[i] - std_chao_SVD5[i], chao_SVD5[i] + std_chao_SVD5[i], color=colors[2], alpha=0.2)
        ax[i].plot(n_samples,chao_PW[i], color=colors[1], marker='o', linestyle='-',linewidth=3,markersize=20,
         markeredgecolor="black", markeredgewidth=1.2)
        ax[i].fill_between(n_samples, chao_PW[i] - std_chao_PW[i], chao_PW[i] + std_chao_PW[i], color=colors[1], alpha=0.2)
        ax[i].plot(n_samples,chao_SACP[i], color=colors[3], marker='s', linestyle='-',linewidth=3,markersize=15,
         markeredgecolor="black", markeredgewidth=1.2)
        ax[i].fill_between(n_samples, chao_SACP[i] - std_chao_SACP[i], chao_SACP[i] + std_chao_SACP[i], color=colors[3], alpha=0.2)

    # ax[i].set_yticks([10**2,10**5])

    ax[i].tick_params(axis='y', labelsize=22)
    ax[i].tick_params(axis='x', labelsize=30)
    ax[i].set_axisbelow(True)
    ax[i].minorticks_on()
    ax[i].grid(True, which='both', linestyle='--')
    ax[i].set_yscale("log")
    ax[i].set_title(f'{chal[i]}\n'.replace('_',' ')+rf' $\alpha={Config.alphas[i]}$ $\beta={Config.betas[i]}$',fontsize=30)
    if i == 0:
        # ax[i].legend(fontsize=20)
        ax[i].set_ylabel("Chao Estimator", fontsize=40)

    if i == 2:
        ax[i].set_xlabel("# samples", fontsize=40)
plt.tight_layout(rect=[0.05, 0, 1, 0.88],pad=0.1)

fig.legend(loc='upper center', ncol=4, fontsize=25)

plt.savefig(f'chao.pdf',bbox_inches='tight')
plt.show()


################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
fig, ax = plt.subplots(1,5,figsize=(28,7))
colors = sns.color_palette("tab10", 4)
n_samples = np.array(Config.n_samples_4_metrics)
n_samples_corr = [10,50,100,500,1000]
columns=[0,1,2,3,4]

corr_SVD2 = np.array(corr2[:,0,columns])
std_corr_SVD2 = np.array(corr2[:,1,columns])

corr_SVD5 = np.array(corr5[:,0,columns])
std_corr_SVD5 = np.array(corr5[:,1,columns])

corr_PW = np.array(corr2[:,2,columns])
std_corr_PW = np.array(corr2[:,3,columns])

corr_SACP = np.array(corr2[:,4,columns])
std_corr_SACP = np.array(corr2[:,5,columns])

sns.set_style("whitegrid")
for i in range(5):
    if i==0:
        ax[i].plot(n_samples_corr,corr_SVD2[i], label= r"$CONSIGN_2$", color=colors[0], marker='^', linestyle='-',linewidth=3,markersize=20,
         markeredgecolor="black", markeredgewidth=1.2)
        ax[i].fill_between(n_samples_corr, corr_SVD2[i] - std_corr_SVD2[i], corr_SVD2[i] + std_corr_SVD2[i], color=colors[0], alpha=0.2)
        ax[i].plot(n_samples_corr,corr_SVD5[i], label=r"$CONSIGN_5$", color=colors[2], marker='v', linestyle='-',linewidth=3,markersize=20,
         markeredgecolor="black", markeredgewidth=1.2)
        ax[i].fill_between(n_samples_corr, corr_SVD5[i] - std_corr_SVD5[i], corr_SVD5[i] + std_corr_SVD5[i], color=colors[2], alpha=0.2)
        ax[i].plot(n_samples_corr,corr_PW[i], label=f"PW (RAPS)", color=colors[1], marker='o', linestyle='-',linewidth=3,markersize=20,
         markeredgecolor="black", markeredgewidth=1.2)
        ax[i].fill_between(n_samples_corr, corr_PW[i] - std_corr_PW[i], corr_PW[i] + std_corr_PW[i], color=colors[1], alpha=0.2)
        ax[i].plot(n_samples_corr,corr_SACP[i], label=f"SACP", color=colors[3], marker='s', linestyle='-',linewidth=3,markersize=15,
         markeredgecolor="black", markeredgewidth=1.2)
        ax[i].fill_between(n_samples_corr, corr_SACP[i] - std_corr_SACP[i], corr_SACP[i] + std_corr_SACP[i], color=colors[3], alpha=0.2)
    else:
        ax[i].plot(n_samples_corr,corr_SVD2[i], color=colors[0], marker='^', linestyle='-',linewidth=3,markersize=20,
         markeredgecolor="black", markeredgewidth=1.2)
        ax[i].fill_between(n_samples_corr, corr_SVD2[i] - std_corr_SVD2[i], corr_SVD2[i] + std_corr_SVD2[i], color=colors[0], alpha=0.2)
        ax[i].plot(n_samples_corr,corr_SVD5[i], color=colors[2], marker='v', linestyle='-',linewidth=3,markersize=20,
         markeredgecolor="black", markeredgewidth=1.2)
        ax[i].fill_between(n_samples_corr, corr_SVD5[i] - std_corr_SVD5[i], corr_SVD5[i] + std_corr_SVD5[i], color=colors[2], alpha=0.2)
        ax[i].plot(n_samples_corr,corr_PW[i], color=colors[1], marker='o', linestyle='-',linewidth=3,markersize=20,
         markeredgecolor="black", markeredgewidth=1.2)
        ax[i].fill_between(n_samples_corr, corr_PW[i] - std_corr_PW[i], corr_PW[i] + std_corr_PW[i], color=colors[1], alpha=0.2)
        ax[i].plot(n_samples_corr,corr_SACP[i], color=colors[3], marker='s', linestyle='-',linewidth=3,markersize=15,
         markeredgecolor="black", markeredgewidth=1.2)
        ax[i].fill_between(n_samples_corr, corr_SACP[i] - std_corr_SACP[i], corr_SACP[i] + std_corr_SACP[i], color=colors[3], alpha=0.2)

    # ax[i].set_yticks([10**2,10**5])

    ax[i].tick_params(axis='y', labelsize=30)
    ax[i].tick_params(axis='x', labelsize=30)
    ax[i].set_axisbelow(True)
    ax[i].minorticks_on()
    ax[i].grid(True, which='both', linestyle='--')
    # ax[i].set_yscale("log")
    ax[i].set_title(f'{chal[i]}\n'.replace('_',' ')+rf' $\alpha={Config.alphas[i]}$ $\beta={Config.betas[i]}$',fontsize=30)
    if i == 0:
        # ax[i].legend(fontsize=20)
        ax[i].set_ylabel("Correlation", fontsize=40)

    if i == 2:
        ax[i].set_xlabel("# samples", fontsize=40)
plt.tight_layout(rect=[0, 0, 1, 0.87],pad=0.1)

fig.legend(loc='upper center', ncol=4, fontsize=25)

plt.savefig(f'corr.pdf',bbox_inches='tight')
plt.show()
