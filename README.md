# CONSIGN: Conformal Segmentation Informed by Spatial Groupings via Decomposition

## 1. General Introduction

This project implements a conformal prediction strategy for image segmentation, which takes into account spatial correlation within the image. The [Paper](https://arxiv.org/abs/2505.14113) has been accepted as poster at [ICLR2026](https://iclr.cc/).

---
## 2. Requirements

- Python 3.10.12

Install the required packages using:

```bash
pip install -r requirements.txt
```

---
## 3. Test Case

### Test Case
You can test the whole pipeline for the COCO datasets doing the following:
```
cd ./code python test.py --category [animals or vehicles]
```
The other arguments are:
```
--skip-download (if you have already the data) --skip-preprocess (if you have already done the pre-process) --device (cpu or cuda)
```
The code will run three main sections:
- Download the data, pre-process them, and run inference to obtain the softmax.
  Create a folder `/softmax/COCO_[animals/vehicles]/` with the data.
- Run the conformal prediction for CONSIGN and the two baselines (calibration and inference).
  **IMPORTANT:** check the options in `config_cp.py` and set them accordingly to your experiment.
- Plot some qualitative results.
---

## 4. Datasets

The following datasets have been used in the paper:
- **M&Ms2** – [website](https://www.ub.edu/mnms-2/)
- **MS-CMR19** – [website](https://zmiclab.github.io/zxh/0/mscmrseg19/)
- **LIDC** – [website](https://www.cancerimagingarchive.net/collection/lidc-idri/)
- **COCO** – [website](https://cocodataset.org/#home)
---

## 5. Pre-trained Models

Three different pre-trained model are used to generate the softmax scores
- **U-Net** – for the M&Ms2 and MS-CMR19 dataset
- [Probabilistic U-Net](https://github.com/stefanknegt/Probabilistic-Unet-Pytorch) – for the LIDC dataset
- [DeepLabV3+](https://github.com/VainF/DeepLabV3Plus-Pytorch) – for the COCO datasets

Use these model to generate different softmax predictions and store the softmax in the following way:
```
code/
├── softmax/
│   └── [name of the challenge]/
│       ├── smx_[name of the challenge].pkl #softmax
│       ├── labels_[name of the challenge].pkl #gt
│       └── imgs.pkl #images
│   
```
### Generating samples
In order to run the calibration you need to predict samples of softmax scores as described in the paper.
- The U-Net code can be found in the corresponding folder, use `python create_samples.py` to generate and save samples. Remember to download the cardiac datasets [M&Ms2](https://www.ub.edu/mnms-2/) and [MS-CMR19](https://zmiclab.github.io/zxh/0/mscmrseg19/). See the paper appendix for further details on the datasets split.
- For the LIDC experiment use the code provided in [Probabilistic U-Net](https://github.com/stefanknegt/Probabilistic-Unet-Pytorch), to generate the corresponding samples.
- For the COCO experiments use [DeepLabV3+](https://github.com/VainF/DeepLabV3Plus-Pytorch). In this case, each sample correspond to a different backbones. The backbones used in the paper are: DeepLabV3-MobileNet, DeepLabV3-ResNet50, DeepLabV3-ResNet101, DeepLabV3Plus-MobileNet, DeepLabV3Plus-ResNet50, DeepLabV3Plus-ResNet101. In the `/COCO_test` folder there is an automatic pipeline to produce those softmax.

---
## 6. Code and Instructions

### Calibration

Once you have obtained the softmax scores (and relative groud-truths and images) of the dataset you want to test, you can calibrate the prediction sets running  `python compare_SVD_PW.py`. Remember to set up the `config_cp.py`, modifying the hyeperparameters (alpha,beta,K,etc...) and dataset details. The `python compare_SVD_PW.py` command will output a `.txt` file with metrics and calibrated lambdas.

### Output

The calibrated lambdas and metrics are saved under:

```
code/
├── metrics/
│   └── [name of the challenge]_[alpha]_[beta]_[K].txt/
```
To obtain the plots shown in the paper, run `python plot_metrics.py`. In order to run the metric command you need `.txt` results for all the dataset used in the paper.
If you want to plot random predictions sampled from the calibrated prediction set, run `python visualize_pred.py`.

### Code overview
- [ ] `compare_SVD_PW.py` - main file, it runs the calibration and metric evaluation
- [ ] `PW_conformal_prediction.py` - contains calibration algorithm, sampling, metric evaluation for PW
- [ ] `SVD_conformal_prediction_UQ.py` - contains calibration algorithm, sampling, metric evaluation for CONSIGN
- [ ] `config_cp.py`- contains the hyper-parameters
- [ ] `helpers_cp.py` - contains useful functions
- [ ] `plot_metrics.py` - plot Chao estimator, SEC, correlation
- [ ] `visualize_pred.py` - plot samples
- [ ] `test.py` - test script for COCO datasets

---
## Acknowledgements
- [Probabilistic U-Net](https://github.com/stefanknegt/Probabilistic-Unet-Pytorch)
- [DeepLabV3+](https://github.com/VainF/DeepLabV3Plus-Pytorch)
- [Probabilistic U-Net original code](https://github.com/SimonKohl/probabilistic_unet)
- [Principal Uncertainty Quantification With Spatial Correlation for Image Restoration Problems](https://ieeexplore.ieee.org/abstract/document/10360418)
- [Uncertainty Quantification via Neural Posterior Principal Components](https://proceedings.neurips.cc/paper_files/paper/2023/hash/74fc5575632191d96881d8015f79dde3-Abstract-Conference.html)

---
## Citation
Please consider citing our paper if you use the code:
```
@article{viti2025consign,
  title={CONSIGN: Conformal Segmentation Informed by Spatial Groupings via Decomposition},
  author={Viti, Bruno and Karabelas, Elias and Holler, Martin},
  journal={arXiv preprint arXiv:2505.14113},
  year={2025}
}
```
