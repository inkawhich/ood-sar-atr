"""Binary Cross Entropy Refiner Network Experiment

Refiner net trains a logistic regressor to identify a single class. The
training data is highly negatively oversampled. This is done to increase the
model's ability to correctly identify OOD samples. Precision and sensitiviy are
the major ID performance metrics.

After training for N epochs, refiner samples are generated. The point of this
step is to generate "adversarial examples" that are close to the ID samples so
that the network will restrict the feature space used to predict positive
samples. This is done by clustering the samples in the penultimate layer's
feature space, randomly generating samples around that cluster, generating an
image from the feature space representation (following Madry group's priors
paper) and adding this image to the training set as a negative sample.

"""
## Imports
from __future__ import print_function
import numpy as np
import sys
import os
from collections import defaultdict
import random
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
# matplotlib.use('TkAgg')
%matplotlib inline

import torch
import torch.nn as nn
import torch.utils.data as utilsdata
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as tvdatasets

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.mixture import GaussianMixture

import seaborn as sns
import time
import pandas as pd

# Custom
import models
import create_split
import Dataset_fromPythonList as custom_dset
import helpers
import training_helpers
import ood_helpers
import calculate_log as callog

## Module Constants
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
DSIZE = 64

# DATASETS = ["ID", "holdout", "mnist", "random", "cifar10"]
DATASETS = ["ID", "holdout"]
# SEED = 1234567

# Normalization Constants for range [-1,+1]
MEAN = torch.tensor([0.5], dtype=torch.float32).view([1,1,1]).to(device)
STD  = torch.tensor([0.5], dtype=torch.float32).view([1,1,1]).to(device)

OODS = ['baseline', 'odin', 'odin_ipp', 'mahala', 'mahala_ipp']

## Experiment Constants
K = 0. # float(sys.argv[1])
num_holdout_classes = 1 # int(sys.argv[2])
dataset_root = "./SAMPLE_dataset/png_images/qpm"

## Hyperparameters
num_epochs = 60
batch_size = 128

learning_rate_decay_schedule = [20]
learning_rate = 0.001
gamma = 0.1
weight_decay = 0.001
dropout = 0.4

# stat_accuracy = {}
# stat_ood = {}
# save_ckpt = None # "ckpts/bce_ensemble_advoe_exp0/"

cls_method = 'softmax'


## Create dataset splits
submod=0
# ORIGINAL CLASS MAPPING:  {0: '2s1', 1: 'bmp2', 2: 'btr70', 3: 'm1', 4: 'm2', 5: 'm35', 6: 'm548', 7: 'm60', 8: 't72', 9: 'zsu23'}
holdout_classes=[9]
# splits = create_split.create_bce_dataset_splits(
#     dataset_root, K, holdout_classes, submod, advOE=True, eq_smpl=False)
# id_trainlist, id_testlist, ood_testlist, oe_trainlist = splits

splits = create_split.create_dataset_splits(
    dataset_root, K, holdout_classes, advOE=True)
id_trainlist, id_testlist, ood_testlist, bce_weights, oe_trainlist = splits
id_trainloader, id_testloader, ood_testloader, oe_trainloader = \
    create_split.get_data_loaders(
        id_trainlist, id_testlist, ood_testlist, oe_trainlist,
        batch_size=batch_size)

## Model Setup
net = models.resnet18(num_classes=9,
                      drop_prob=dropout,
                      cls_method=cls_method).to(device)

## Training N epochs
net, final_test_acc = training_helpers.train_model_advOE(
    net, num_epochs,
    id_trainloader, oe_trainloader,
    data_mean=MEAN,
    data_std=STD,
    lr=learning_rate,
    weight_decay=weight_decay,
    scheduler=learning_rate_decay_schedule,
    testloader=id_testloader,
    neg_hard_smpl_epochs=[40],
    neg_hard_smpl_factor=0.75,
)
net.eval()

## Training N epochs
# net, final_test_acc = training_helpers.train_model(
#     net, num_epochs, id_trainloader,
#     data_mean=MEAN,
#     data_std=STD,
#     lr=learning_rate,
#     weight_decay=weight_decay,
#     scheduler=learning_rate_decay_schedule,
#     testloader=id_testloader,
# )

## Test Hard Negative OOD Sampling

## Gather OOD Stats
MAHALA_LAYER_SET = [
    "resnet18_2",
    "resnet18_2_2",
    "resnet18_2_2_2",
    "resnet18_2_2_2_2",
]
hooked_activation_dict = {}; hook_handles = []
def get_activation(name):
    def hook(module, input, output):
        hooked_activation_dict[name]=output
    return hook

for l in MAHALA_LAYER_SET:
    hook_handles.append(ood_helpers.set_model_hook(l, net, get_activation(l), device))

with torch.no_grad():
    dd = torch.zeros(1,1,64,64).uniform_().to(device)
    net(dd)
    dd = None
for l in MAHALA_LAYER_SET:
    activation_shape = hooked_activation_dict[l].shape[1]
    print("Hooked Layer: {}; Shape: {}; Min: {}; Max: {}".format(
        l, hooked_activation_dict[l].shape,
        hooked_activation_dict[l].min(),
        hooked_activation_dict[l].max()))

layer_means, layer_precisions = ood_helpers.compute_empirical_means_and_precision(
    net, hooked_activation_dict, MAHALA_LAYER_SET,
    id_trainloader, 10-num_holdout_classes)


baseline_scores_dict = {}
odin_scores_dict = {}
odin_ipp_scores_dict = {}
mahala_scores_dict = {}
mahala_ipp_scores_dict = {}

for dset in DATASETS:
    print("Computing OOD scores for {}...".format(dset))
    currloader = None
    if dset == "ID":
        currloader = id_testloader
    elif dset == "holdout":
        currloader = ood_testloader
    else:
        currloader = ood_helpers.get_ood_dataloader(dset,DSIZE,bsize=128,shuf=False)

    # COMPUTE OOD SCORES FOR THIS DATASET
    base_scores,odin_scores,odin_ipp_scores = \
            ood_helpers.generate_ood_scores_ODIN_and_BASELINE(
                net, currloader, 1000., MEAN, STD, IPP=True,
                eps=0.01)
    mahalanobis_scores,mahalanobis_ipp_scores = \
            ood_helpers.generate_ood_scores_MAHALANOBIS(
                net, currloader, hooked_activation_dict,
                MAHALA_LAYER_SET, layer_means,
                layer_precisions, MEAN, STD, IPP=True,
                eps=0.01)

    # Save raw OOD scores into dictionaries
    baseline_scores_dict[dset] = base_scores
    odin_scores_dict[dset] = odin_scores
    odin_ipp_scores_dict[dset] = odin_ipp_scores
    mahala_scores_dict[dset] = mahalanobis_scores
    mahala_ipp_scores_dict[dset] = mahalanobis_ipp_scores

print("Computing OOD Statistics...")
for dd in range(1,len(DATASETS)):
    print("** DATASET: {} **".format(DATASETS[dd]))

    metric_results = callog.metric(
        np.array(baseline_scores_dict["ID"]),
        np.array(baseline_scores_dict[DATASETS[dd]]) )
    print("\tBaseline.          AUROC: {:.4f}. TNR@95TPR: {:.4f}. DetAcc: {:.4f}".format(
        metric_results['TMP']['AUROC'],
        metric_results['TMP']['TNR'],
        metric_results['TMP']['DTACC'],
    ))

    metric_results = callog.metric(
        np.array(odin_scores_dict["ID"]),
        np.array(odin_scores_dict[DATASETS[dd]]))
    print("\tODIN (T=1000).     AUROC: {:.4f}. TNR@95TPR: {:.4f}. DetAcc: {:.4f}".format(
        metric_results['TMP']['AUROC'],
        metric_results['TMP']['TNR'],
        metric_results['TMP']['DTACC'],
    ))

    metric_results = callog.metric(
        np.array(mahala_scores_dict["ID"]),
        np.array(mahala_scores_dict[DATASETS[dd]]) )
    print("\tMahalanobis.       AUROC: {:.4f}. TNR@95TPR: {:.4f}. DetAcc: {:.4f}".format(
        metric_results['TMP']['AUROC'],
        metric_results['TMP']['TNR'],
        metric_results['TMP']['DTACC'],
    ))

    metric_results = callog.metric(
        np.array(odin_ipp_scores_dict["ID"]),
        np.array(odin_ipp_scores_dict[DATASETS[dd]]) )
    print("\tODIN (T=1000) IPP. AUROC: {:.4f}. TNR@95TPR: {:.4f}. DetAcc: {:.4f}".format(
        metric_results['TMP']['AUROC'],
        metric_results['TMP']['TNR'],
        metric_results['TMP']['DTACC'],
    ))

    metric_results = callog.metric(
        np.array(mahala_ipp_scores_dict["ID"]),
        np.array(mahala_ipp_scores_dict[DATASETS[dd]]) )
    print("\tMahalanobis IPP.   AUROC: {:.4f}. TNR@95TPR: {:.4f}. DetAcc: {:.4f}".format(
        metric_results['TMP']['AUROC'],
        metric_results['TMP']['TNR'],
        metric_results['TMP']['DTACC'],
    ))

## Forward pass the input images to the penultimate layer
hooked_activation_dict = {}; hook_handles = []
MAHALA_LAYER_SET = [
    # "resnet18_2",
    # "resnet18_2_2",
    # "resnet18_2_2_2",
    "resnet18_2_2_2_2",
]
for l in MAHALA_LAYER_SET:
    hook_handles.append(ood_helpers.set_model_hook(l, net, get_activation(l), device))

with torch.no_grad():
    dd = torch.zeros(1,1,64,64).uniform_().to(device)
    net(dd)
    dd = None
for l in MAHALA_LAYER_SET:
    activation_shape = hooked_activation_dict[l].shape[1]
    print("Hooked Layer: {}; Shape: {}; Min: {}; Max: {}".format(
        l, hooked_activation_dict[l].shape,
        hooked_activation_dict[l].min(), 
        hooked_activation_dict[l].max()))

## Get list of hard examples
net.eval()
prune_factor=0.9
# DATA AUGMENTATION AND LOSS FXN CONFIGS
gaussian_std = 0.4
# gaussian_std = 0.3
# gaussian_std = 0.05
running_total = 0
all_losses = np.empty((0,))
all_pths = []
with torch.no_grad():
    for data, labels, pth in oe_trainloader:
        data = data.to(device)
        labels = labels.to(device)

        if(gaussian_std != 0):
            data += torch.randn_like(data)*gaussian_std
            data = torch.clamp(data, 0, 1)

        # Forward pass data through model. Normalize before forward pass
        outputs = net((data-MEAN)/STD)

        loss = training_helpers.OESoftmaxLoss(reduction=None)(outputs,labels)
        running_total += labels.size(0)
        all_losses = np.concatenate((all_losses, loss.cpu().numpy()), -1)
        all_pths.extend(list(pth))

    num_to_keep = round((1-prune_factor) * running_total)
    sorted_idxs = np.argsort(-all_losses)  # Sort descending
    pths_to_keep = [p for i, p in enumerate(all_pths)
                    if i in sorted_idxs[:num_to_keep]]


## Get model activations
net.eval()
all_activations=np.empty((0,activation_shape))
all_labels = np.empty((0,))
with torch.no_grad():
    oeiter = iter(oe_trainloader)
    for (data,labels,pth) in id_trainloader:
        oedata, oelabels, oepths = next(oeiter)
        # ID Data
        data = data.to(device)
        outputs = net((data-MEAN)/STD)
        activations = net.avgpool(hooked_activation_dict['resnet18_2_2_2_2'])
        activations = activations.squeeze().cpu().numpy()
        all_activations = np.concatenate((all_activations, activations),0)
        # all_labels = np.concatenate((all_labels, labels.numpy()), 0)
        all_labels = np.concatenate((all_labels, np.ones((labels.shape[0],))), 0)

        # OOD Data
        oedata = oedata.to(device)
        outputs = net((oedata-MEAN)/STD)
        activations = net.avgpool(hooked_activation_dict['resnet18_2_2_2_2'])
        activations = activations.squeeze().cpu().numpy()
        all_activations = np.concatenate((all_activations, activations),0)
        labs = [3 if p in pths_to_keep else -1 for p in oepths]

        all_labels = np.concatenate((all_labels, labs), 0)

    # ID Test data
    for (data, labels, _) in id_testloader:
        data = data.to(device)
        outputs = net((data-MEAN)/STD)
        activations = net.avgpool(hooked_activation_dict['resnet18_2_2_2_2'])
        activations = activations.squeeze().cpu().numpy()
        all_activations = np.concatenate((all_activations, activations),0)
        # all_labels = np.concatenate((all_labels, labels.numpy()+2), 0)
        all_labels = np.concatenate((all_labels, 2*np.ones((labels.shape[0],))), 0)

    # OOD Test data
    for (data, labels, _) in ood_testloader:
        data = data.to(device)
        outputs = net((data-MEAN)/STD)
        activations = net.avgpool(hooked_activation_dict['resnet18_2_2_2_2'])
        activations = activations.squeeze().cpu().numpy()
        all_activations = np.concatenate((all_activations, activations),0)
        # all_labels = np.concatenate((all_labels, -2.*np.ones((labels.shape[0],))), 0)
        all_labels = np.concatenate((all_labels, np.zeros((labels.shape[0],))), 0)

## Perform PCA to reduce dimensions
pca = PCA(n_components=50)
pca_result = pca.fit_transform(all_activations)

print('Explained variation per principal component: {}'.format(
    np.sum(pca.explained_variance_ratio_)))

## Perform t-SNE
time_start = time.time()
tsne = TSNE(n_components=2, verbose=0, perplexity=50, n_iter=1000)
tsne_pca_results = tsne.fit_transform(pca_result)
print('t-SNE done! Time elapsed: {} seconds'.format(
    time.time()-time_start))

## Plot t-SNE matplotlib
# fig = plt.figure()
# plt.scatter(
#     tsne_pca_results[:,0],
#     tsne_pca_results[:,1],
#     c=all_labels,
# )
# fig.show()

## Plot t-SNE
label_s = pd.Series(all_labels)
label_s_txt = label_s.map({-1.:"OE", 0.:"HLD", 1.:"ID Train", 2.:"ID Test",
                           3.:"Hard Negative"})
# label_s_txt = label_s.map({-2.: "HLD",-1.:"OE", 0.:"ID- Train", 1.:"ID+ Train",
#                            2.:"ID- Test", 3.:"ID+ Test"})
print(label_s.value_counts())
##
sns.scatterplot(
    x=tsne_pca_results[:,0],
    y=tsne_pca_results[:,1],
    hue=label_s_txt,
    alpha=0.5,
    palette=sns.color_palette('hls', 5),
    # palette=sns.color_palette('hls', 6),
)
plt.title("Shared FE Soft OE HLD={} t-SNE".format(holdout_classes[0]))
# plt.show()

## Perform PCA to reduce dimensions
pca = PCA(n_components=2)
pca_result = pca.fit_transform(all_activations)

print('Explained variation per principal component: {}'.format(
    np.sum(pca.explained_variance_ratio_)))

## Plot 2D PCA
# sns.scatterplot(
#     x=pca_result[:,0],
#     y=pca_result[:,1],
#     hue=label_s_txt,
#     alpha=0.5,
#     palette=sns.color_palette('hls', 4),
#     # palette=sns.color_palette('hls', 6),
# )
# plt.title("Shared FE Soft t-SNE")
# plt.show()

## Fit Guassian to t-SNE
gm = GaussianMixture(n_components=9, random_state=0).fit(
    tsne_pca_results[label_s==1])
    # tsne_pca_results[np.logical_or(label_s==-1, label_s==1)])

x = np.linspace(tsne_pca_results[:,0].min()-25, tsne_pca_results[:,0].max()+25)
y = np.linspace(tsne_pca_results[:,1].min()-25, tsne_pca_results[:,1].max()+25)
X, Y = np.meshgrid(x, y)
XX = np.array([X.ravel(), Y.ravel()]).T
Z = -gm.score_samples(XX)
Z = Z.reshape(X.shape)
# import pdb; pdb.set_trace()

CS = plt.contour(X, Y, Z, norm=LogNorm(vmin=1.0, vmax=1000.0),
                 levels=np.logspace(0, 2, 25))
CB = plt.colorbar(CS, shrink=0.8, extend='both')
# plt.scatter(X_train[:, 0], X_train[:, 1], .8)
plt.show()


## Generate restriction examples in feature space
## Create images from penultimate layer activations
## Visualize generated images
## Add to training set and loop

# if __name__ == '__main__':
#     main()
