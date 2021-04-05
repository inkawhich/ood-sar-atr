"""Smart outlier set generated from AdvOE trained network
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
import itertools

# Custom
import models
import create_split
import Dataset_fromPythonList as custom_dset
import helpers
import training_helpers
import ood_helpers
import calculate_log as callog

import pickle

## Module Constants
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
DSIZE = 64

# DATASETS = ["ID", "holdout", "mnist", "random", "cifar10"]
DATASETS = ["ID", "holdout"]
SEED = 1234567
random.seed(SEED)
torch.manual_seed(SEED)

# Normalization Constants for range [-1,+1]
MEAN = torch.tensor([0.5], dtype=torch.float32).view([1,1,1]).to(device)
STD  = torch.tensor([0.5], dtype=torch.float32).view([1,1,1]).to(device)

OODS = ['baseline', 'odin', 'odin_ipp', 'mahala', 'mahala_ipp']
REPEAT_ITERS=5

## Experiment Constants
K = 0. # float(sys.argv[1])
num_holdout_classes = 3 # int(sys.argv[2])
dataset_root = "./SAMPLE_dataset/png_images/qpm"

## Hyperparameters
num_epochs = 60
batch_size = 128
learning_rate_decay_schedule = [50]
learning_rate = 0.001
gamma = 0.1
weight_decay = 0.
dropout = 0.4

cls_method = 'softmax'

# save_ckpt = None # "ckpts/bce_ensemble_advoe_exp0/"
STAT_accuracy = {}
STAT_time = {}
STAT_ood = {}
classes = list(range(10))
# hld_groups = list(itertools.combinations(classes, 3))
# done_hld_groups = [(2,6,9), (6,7,8), (1,2,5), (0,6,7), (2,3,4), (3,5,6),
#                    (1,2,9), (1,6,9), (1,6,8), (4,6,9), (3,6,8), (6,8,9), (3,6,9),
#                    (1,2,6), (0,2,6), (1,7,8), (0,1,9), (1,4,9), (1,4,6), (1,8,9)]
hld_groups = [(2,6,9),]
for hld_id, hld in enumerate(hld_groups):
    STAT_accuracy[hld_id] = []
    STAT_time[hld_id] = []
    STAT_ood[hld_id] = {}
    for ood in OODS:
        STAT_ood[hld_id][ood] = {}
        for dd in DATASETS:
            STAT_ood[hld_id][ood][dd] = defaultdict(list)

## Train AdvOE Model as the oracle {{{
print("Training AdvOE Model...")
splits = create_split.create_dataset_splits(dataset_root, K, hld, advOE=True)
id_trainlist, id_testlist, ood_testlist, bce_weights, oe_trainlist = splits
id_trainloader, id_testloader, ood_testloader, oe_trainloader = \
    create_split.get_data_loaders(
        id_trainlist, id_testlist, ood_testlist, oe_trainlist,
        batch_size=batch_size)

# Model Setup
net = models.resnet18(num_classes=10-num_holdout_classes,
                      drop_prob=dropout,
                      cls_method=cls_method).to(device)

# Training N epochs
net, final_test_acc = training_helpers.train_model_advOE(
    net, num_epochs,
    id_trainloader, oe_trainloader,
    data_mean=MEAN,
    data_std=STD,
    lr=learning_rate,
    weight_decay=weight_decay,
    scheduler=learning_rate_decay_schedule,
    testloader=id_testloader,
    neg_hard_smpl_epochs=[],  # [30, 40, 50],
    neg_hard_smpl_factor=0.,
)

net.eval()
# }}}
## Gather AdvOE Oracle OOD Stats {{{
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
    print("\tBaseline.          AUROC: {:.4f}. TNR@95TPR: {:.4f}. AUPR OUT: {:.4f}".format(
        metric_results['TMP']['AUROC'],
        metric_results['TMP']['TNR'],
        metric_results['TMP']['AUOUT'],
    ))
    metric_results = callog.metric(
        np.array(odin_scores_dict["ID"]),
        np.array(odin_scores_dict[DATASETS[dd]]))
    print("\tODIN (T=1000).     AUROC: {:.4f}. TNR@95TPR: {:.4f}. AUPR OUT: {:.4f}".format(
        metric_results['TMP']['AUROC'],
        metric_results['TMP']['TNR'],
        metric_results['TMP']['AUOUT'],
    ))

    metric_results = callog.metric(
        np.array(mahala_scores_dict["ID"]),
        np.array(mahala_scores_dict[DATASETS[dd]]) )
    print("\tMahalanobis.       AUROC: {:.4f}. TNR@95TPR: {:.4f}. AUPR OUT: {:.4f}".format(
        metric_results['TMP']['AUROC'],
        metric_results['TMP']['TNR'],
        metric_results['TMP']['AUOUT'],
    ))

    metric_results = callog.metric(
        np.array(odin_ipp_scores_dict["ID"]),
        np.array(odin_ipp_scores_dict[DATASETS[dd]]) )
    print("\tODIN (T=1000) IPP. AUROC: {:.4f}. TNR@95TPR: {:.4f}. AUPR OUT: {:.4f}".format(
        metric_results['TMP']['AUROC'],
        metric_results['TMP']['TNR'],
        metric_results['TMP']['AUOUT'],
    ))

    metric_results = callog.metric(
        np.array(mahala_ipp_scores_dict["ID"]),
        np.array(mahala_ipp_scores_dict[DATASETS[dd]]) )
    print("\tMahalanobis IPP.   AUROC: {:.4f}. TNR@95TPR: {:.4f}. AUPR OUT: {:.4f}".format(
        metric_results['TMP']['AUROC'],
        metric_results['TMP']['TNR'],
        metric_results['TMP']['AUOUT'],
    ))
## }}}

## Get oe and id num images {{{
num_oeimages = 0
for batch_idx, (oedata, oelabels, _) in enumerate(oe_trainloader):
    num_oeimages += oedata.shape[0]
print(num_oeimages)

num_idimages = 0
for batch_idx, (iddata, idlabels, _) in enumerate(id_trainloader):
    num_idimages += iddata.shape[0]
print(num_idimages)
print(num_idimages/num_oeimages)
## }}}
## Create Smart Outlier set from oracle {{{
print(int((num_idimages/num_oeimages)* num_oeimages))
smart_oe_loader = training_helpers.hard_negative_mining_pass(
    net, oe_trainloader, prune_factor=1-(num_idimages/num_oeimages))
net.eval()
print("DONE")
## }}}
## Check smart outlier size {{{
num_smartoeimgs = 0
for batch_idx, (oedata, oelabels, _) in enumerate(smart_oe_loader):
    num_smartoeimgs += oedata.shape[0]
print(num_smartoeimgs)
## }}}

## Train with smart outliers {{{
print("Training AdvOE Model...")
# Model Setup
smartnet = models.resnet18(num_classes=10-num_holdout_classes,
                      drop_prob=dropout,
                      cls_method=cls_method).to(device)

# Training N epochs
smartnet, smartfinal_test_acc = training_helpers.train_model_advOE(
    smartnet, num_epochs,
    id_trainloader, smart_oe_loader,
    data_mean=MEAN,
    data_std=STD,
    lr=learning_rate,
    weight_decay=weight_decay,
    scheduler=learning_rate_decay_schedule,
    testloader=id_testloader,
    neg_hard_smpl_epochs=[],  # [30, 40, 50],
    neg_hard_smpl_factor=0.,
)

smartnet.eval()
##}}}

## Gather OOD Stats {{{
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
    hook_handles.append(ood_helpers.set_model_hook(l, smartnet, get_activation(l), device))

with torch.no_grad():
    dd = torch.zeros(1,1,64,64).uniform_().to(device)
    smartnet(dd)
    dd = None
for l in MAHALA_LAYER_SET:
    activation_shape = hooked_activation_dict[l].shape[1]
    print("Hooked Layer: {}; Shape: {}; Min: {}; Max: {}".format(
        l, hooked_activation_dict[l].shape,
        hooked_activation_dict[l].min(),
        hooked_activation_dict[l].max()))

layer_means, layer_precisions = ood_helpers.compute_empirical_means_and_precision(
    smartnet, hooked_activation_dict, MAHALA_LAYER_SET,
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
                smartnet, currloader, 1000., MEAN, STD, IPP=True,
                eps=0.01)
    mahalanobis_scores,mahalanobis_ipp_scores = \
            ood_helpers.generate_ood_scores_MAHALANOBIS(
                smartnet, currloader, hooked_activation_dict,
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
    print("\tBaseline.          AUROC: {:.4f}. TNR@95TPR: {:.4f}. AUPR OUT: {:.4f}".format(
        metric_results['TMP']['AUROC'],
        metric_results['TMP']['TNR'],
        metric_results['TMP']['AUOUT'],
    ))
    metric_results = callog.metric(
        np.array(odin_scores_dict["ID"]),
        np.array(odin_scores_dict[DATASETS[dd]]))
    print("\tODIN (T=1000).     AUROC: {:.4f}. TNR@95TPR: {:.4f}. AUPR OUT: {:.4f}".format(
        metric_results['TMP']['AUROC'],
        metric_results['TMP']['TNR'],
        metric_results['TMP']['AUOUT'],
    ))

    metric_results = callog.metric(
        np.array(mahala_scores_dict["ID"]),
        np.array(mahala_scores_dict[DATASETS[dd]]) )
    print("\tMahalanobis.       AUROC: {:.4f}. TNR@95TPR: {:.4f}. AUPR OUT: {:.4f}".format(
        metric_results['TMP']['AUROC'],
        metric_results['TMP']['TNR'],
        metric_results['TMP']['AUOUT'],
    ))

    metric_results = callog.metric(
        np.array(odin_ipp_scores_dict["ID"]),
        np.array(odin_ipp_scores_dict[DATASETS[dd]]) )
    print("\tODIN (T=1000) IPP. AUROC: {:.4f}. TNR@95TPR: {:.4f}. AUPR OUT: {:.4f}".format(
        metric_results['TMP']['AUROC'],
        metric_results['TMP']['TNR'],
        metric_results['TMP']['AUOUT'],
    ))

    metric_results = callog.metric(
        np.array(mahala_ipp_scores_dict["ID"]),
        np.array(mahala_ipp_scores_dict[DATASETS[dd]]) )
    print("\tMahalanobis IPP.   AUROC: {:.4f}. TNR@95TPR: {:.4f}. AUPR OUT: {:.4f}".format(
        metric_results['TMP']['AUROC'],
        metric_results['TMP']['TNR'],
        metric_results['TMP']['AUOUT'],
    ))
##}}}

## Create Smart Outlier set from oracle epochs*num id images {{{
print(int(((num_idimages*num_epochs)/num_oeimages)* num_oeimages))
smart_oe_loader_big = training_helpers.hard_negative_mining_pass(
    net, oe_trainloader, prune_factor=1-((num_idimages*num_epochs)/num_oeimages))
net.eval()
print("DONE")
## }}}
## Check smart outlier size {{{
num_smartoeimgs = 0
for batch_idx, (oedata, oelabels, _) in enumerate(smart_oe_loader_big):
    num_smartoeimgs += oedata.shape[0]
print(num_smartoeimgs)
## }}}

## Train with smart outliers {{{
print("Training AdvOE Model...")
# Model Setup
smartnet_bigOE = models.resnet18(num_classes=10-num_holdout_classes,
                      drop_prob=dropout,
                      cls_method=cls_method).to(device)

# Training N epochs
smartnet_bigOE, smartfinal_test_acc_bigOE = training_helpers.train_model_advOE(
    smartnet_bigOE, num_epochs,
    id_trainloader, smart_oe_loader_big,
    data_mean=MEAN,
    data_std=STD,
    lr=learning_rate,
    weight_decay=weight_decay,
    scheduler=learning_rate_decay_schedule,
    testloader=id_testloader,
    neg_hard_smpl_epochs=[],  # [30, 40, 50],
    neg_hard_smpl_factor=0.,
)

smartnet_bigOE.eval()
##}}}

## Gather OOD Stats {{{
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
    hook_handles.append(ood_helpers.set_model_hook(l, smartnet_bigOE, get_activation(l), device))

with torch.no_grad():
    dd = torch.zeros(1,1,64,64).uniform_().to(device)
    smartnet_bigOE(dd)
    dd = None
for l in MAHALA_LAYER_SET:
    activation_shape = hooked_activation_dict[l].shape[1]
    print("Hooked Layer: {}; Shape: {}; Min: {}; Max: {}".format(
        l, hooked_activation_dict[l].shape,
        hooked_activation_dict[l].min(),
        hooked_activation_dict[l].max()))

layer_means, layer_precisions = ood_helpers.compute_empirical_means_and_precision(
    smartnet_bigOE, hooked_activation_dict, MAHALA_LAYER_SET,
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
                smartnet_bigOE, currloader, 1000., MEAN, STD, IPP=True,
                eps=0.01)
    mahalanobis_scores,mahalanobis_ipp_scores = \
            ood_helpers.generate_ood_scores_MAHALANOBIS(
                smartnet_bigOE, currloader, hooked_activation_dict,
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
    print("\tBaseline.          AUROC: {:.4f}. TNR@95TPR: {:.4f}. AUPR OUT: {:.4f}".format(
        metric_results['TMP']['AUROC'],
        metric_results['TMP']['TNR'],
        metric_results['TMP']['AUOUT'],
    ))
    metric_results = callog.metric(
        np.array(odin_scores_dict["ID"]),
        np.array(odin_scores_dict[DATASETS[dd]]))
    print("\tODIN (T=1000).     AUROC: {:.4f}. TNR@95TPR: {:.4f}. AUPR OUT: {:.4f}".format(
        metric_results['TMP']['AUROC'],
        metric_results['TMP']['TNR'],
        metric_results['TMP']['AUOUT'],
    ))

    metric_results = callog.metric(
        np.array(mahala_scores_dict["ID"]),
        np.array(mahala_scores_dict[DATASETS[dd]]) )
    print("\tMahalanobis.       AUROC: {:.4f}. TNR@95TPR: {:.4f}. AUPR OUT: {:.4f}".format(
        metric_results['TMP']['AUROC'],
        metric_results['TMP']['TNR'],
        metric_results['TMP']['AUOUT'],
    ))

    metric_results = callog.metric(
        np.array(odin_ipp_scores_dict["ID"]),
        np.array(odin_ipp_scores_dict[DATASETS[dd]]) )
    print("\tODIN (T=1000) IPP. AUROC: {:.4f}. TNR@95TPR: {:.4f}. AUPR OUT: {:.4f}".format(
        metric_results['TMP']['AUROC'],
        metric_results['TMP']['TNR'],
        metric_results['TMP']['AUOUT'],
    ))

    metric_results = callog.metric(
        np.array(mahala_ipp_scores_dict["ID"]),
        np.array(mahala_ipp_scores_dict[DATASETS[dd]]) )
    print("\tMahalanobis IPP.   AUROC: {:.4f}. TNR@95TPR: {:.4f}. AUPR OUT: {:.4f}".format(
        metric_results['TMP']['AUROC'],
        metric_results['TMP']['TNR'],
        metric_results['TMP']['AUOUT'],
    ))
##}}}
# AdvOE Train
# Test Acc Final = .84697
# OE Acc Final = .67031
# ** DATASET: holdout **
# 	Baseline.          AUROC: 0.6357. TNR@95TPR: 0.0250. AUPR OUT: 0.3526
# 	ODIN (T=1000).     AUROC: 0.6498. TNR@95TPR: 0.0250. AUPR OUT: 0.3653
# 	Mahalanobis.       AUROC: 0.8000. TNR@95TPR: 0.3187. AUPR OUT: 0.6220
# 	ODIN (T=1000) IPP. AUROC: 0.6706. TNR@95TPR: 0.0250. AUPR OUT: 0.3875
# 	Mahalanobis IPP.   AUROC: 0.8117. TNR@95TPR: 0.3313. AUPR OUT: 0.6385

# Smart Outlier
# Test Acc Final = .82322
# OE Acc Final = .62413
# ** DATASET: holdout **
# 	Baseline.          AUROC: 0.5371. TNR@95TPR: 0.0188. AUPR OUT: 0.2997
# 	ODIN (T=1000).     AUROC: 0.5433. TNR@95TPR: 0.0188. AUPR OUT: 0.3045
# 	Mahalanobis.       AUROC: 0.8931. TNR@95TPR: 0.3187. AUPR OUT: 0.6878
# 	ODIN (T=1000) IPP. AUROC: 0.5547. TNR@95TPR: 0.0250. AUPR OUT: 0.3151
# 	Mahalanobis IPP.   AUROC: 0.9032. TNR@95TPR: 0.4563. AUPR OUT: 0.7219

# Smart Outlier Iter 2
# Test Acc Final = .78364
# OE Acc Final = .43182
# ** DATASET: holdout **
# 	Baseline.          AUROC: 0.6162. TNR@95TPR: 0.0875. AUPR OUT: 0.3564
# 	ODIN (T=1000).     AUROC: 0.6193. TNR@95TPR: 0.0875. AUPR OUT: 0.3607
# 	Mahalanobis.       AUROC: 0.8760. TNR@95TPR: 0.5563. AUPR OUT: 0.7842
# 	ODIN (T=1000) IPP. AUROC: 0.6443. TNR@95TPR: 0.0813. AUPR OUT: 0.3753
# 	Mahalanobis IPP.   AUROC: 0.8815. TNR@95TPR: 0.5687. AUPR OUT: 0.7967

# Smart Outlier Big
# Test Acc Final = .88127
# OE Acc Final = .16719
# ** DATASET: holdout **
# 	Baseline.          AUROC: 0.6487. TNR@95TPR: 0.0437. AUPR OUT: 0.3758
# 	ODIN (T=1000).     AUROC: 0.6536. TNR@95TPR: 0.0500. AUPR OUT: 0.3822
# 	Mahalanobis.       AUROC: 0.8443. TNR@95TPR: 0.3562. AUPR OUT: 0.6790
# 	ODIN (T=1000) IPP. AUROC: 0.6592. TNR@95TPR: 0.0563. AUPR OUT: 0.3943
# 	Mahalanobis IPP.   AUROC: 0.8548. TNR@95TPR: 0.4250. AUPR OUT: 0.7021

























## Create dataset splits
# ORIGINAL CLASS MAPPING:  {0: '2s1', 1: 'bmp2', 2: 'btr70', 3: 'm1', 4: 'm2', 5: 'm35', 6: 'm548', 7: 'm60', 8: 't72', 9: 'zsu23'}
# holdout_classes=[9]
for hld_id, hld in enumerate(hld_groups):
    for ITER in range(REPEAT_ITERS):
        # splits = create_split.create_bce_dataset_splits(
        #     dataset_root, K, holdout_classes, submod, advOE=True, eq_smpl=False)
        # id_trainlist, id_testlist, ood_testlist, oe_trainlist = splits
        starttime = time.time()
        print(40*'*')
        print("Starting Iter: {} / {} for K = {} and J = {}, Cls={}, Holdout ID={}".format(
            ITER, REPEAT_ITERS, K, num_holdout_classes, cls_method, hld_id))
        print(40*'*')

        splits = create_split.create_dataset_splits(
            dataset_root, K, hld, advOE=True)
        id_trainlist, id_testlist, ood_testlist, bce_weights, oe_trainlist = splits
        id_trainloader, id_testloader, ood_testloader, oe_trainloader = \
            create_split.get_data_loaders(
                id_trainlist, id_testlist, ood_testlist, oe_trainlist,
                batch_size=batch_size)

        ## Model Setup
        net = models.resnet18(num_classes=10-num_holdout_classes,
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
            neg_hard_smpl_epochs=[],  # [30, 40, 50],
            neg_hard_smpl_factor=0.,
        )

        net.eval()

        # ## Training N epochs
        # net, final_test_acc = training_helpers.train_model(
        #     net, num_epochs, id_trainloader,
        #     data_mean=MEAN,
        #     data_std=STD,
        #     lr=learning_rate,
        #     weight_decay=weight_decay,
        #     scheduler=learning_rate_decay_schedule,
        #     testloader=id_testloader,
        # )
        STAT_accuracy[hld_id].append(final_test_acc)

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
            print("\tBaseline.          AUROC: {:.4f}. TNR@95TPR: {:.4f}. AUPR OUT: {:.4f}".format(
                metric_results['TMP']['AUROC'],
                metric_results['TMP']['TNR'],
                metric_results['TMP']['AUOUT'],
            ))
            STAT_ood[hld_id]['baseline'][DATASETS[dd]]["auroc"].append(metric_results['TMP']['AUROC'])
            STAT_ood[hld_id]['baseline'][DATASETS[dd]]["tnr"].append(metric_results['TMP']['TNR'])
            STAT_ood[hld_id]['baseline'][DATASETS[dd]]["auout"].append(metric_results['TMP']['AUOUT'])

            metric_results = callog.metric(
                np.array(odin_scores_dict["ID"]),
                np.array(odin_scores_dict[DATASETS[dd]]))
            print("\tODIN (T=1000).     AUROC: {:.4f}. TNR@95TPR: {:.4f}. AUPR OUT: {:.4f}".format(
                metric_results['TMP']['AUROC'],
                metric_results['TMP']['TNR'],
                metric_results['TMP']['AUOUT'],
            ))
            STAT_ood[hld_id]['odin'][DATASETS[dd]]["auroc"].append(metric_results['TMP']['AUROC'])
            STAT_ood[hld_id]['odin'][DATASETS[dd]]["tnr"].append(metric_results['TMP']['TNR'])
            STAT_ood[hld_id]['odin'][DATASETS[dd]]["auout"].append(metric_results['TMP']['AUOUT'])

            metric_results = callog.metric(
                np.array(mahala_scores_dict["ID"]),
                np.array(mahala_scores_dict[DATASETS[dd]]) )
            print("\tMahalanobis.       AUROC: {:.4f}. TNR@95TPR: {:.4f}. AUPR OUT: {:.4f}".format(
                metric_results['TMP']['AUROC'],
                metric_results['TMP']['TNR'],
                metric_results['TMP']['AUOUT'],
            ))
            STAT_ood[hld_id]['mahala'][DATASETS[dd]]["auroc"].append(metric_results['TMP']['AUROC'])
            STAT_ood[hld_id]['mahala'][DATASETS[dd]]["tnr"].append(metric_results['TMP']['TNR'])
            STAT_ood[hld_id]['mahala'][DATASETS[dd]]["auout"].append(metric_results['TMP']['AUOUT'])

            metric_results = callog.metric(
                np.array(odin_ipp_scores_dict["ID"]),
                np.array(odin_ipp_scores_dict[DATASETS[dd]]) )
            print("\tODIN (T=1000) IPP. AUROC: {:.4f}. TNR@95TPR: {:.4f}. AUPR OUT: {:.4f}".format(
                metric_results['TMP']['AUROC'],
                metric_results['TMP']['TNR'],
                metric_results['TMP']['AUOUT'],
            ))
            STAT_ood[hld_id]['odin_ipp'][DATASETS[dd]]["auroc"].append(metric_results['TMP']['AUROC'])
            STAT_ood[hld_id]['odin_ipp'][DATASETS[dd]]["tnr"].append(metric_results['TMP']['TNR'])
            STAT_ood[hld_id]['odin_ipp'][DATASETS[dd]]["auout"].append(metric_results['TMP']['AUOUT'])

            metric_results = callog.metric(
                np.array(mahala_ipp_scores_dict["ID"]),
                np.array(mahala_ipp_scores_dict[DATASETS[dd]]) )
            print("\tMahalanobis IPP.   AUROC: {:.4f}. TNR@95TPR: {:.4f}. AUPR OUT: {:.4f}".format(
                metric_results['TMP']['AUROC'],
                metric_results['TMP']['TNR'],
                metric_results['TMP']['AUOUT'],
            ))
            STAT_ood[hld_id]['mahala_ipp'][DATASETS[dd]]["auroc"].append(metric_results['TMP']['AUROC'])
            STAT_ood[hld_id]['mahala_ipp'][DATASETS[dd]]["tnr"].append(metric_results['TMP']['TNR'])
            STAT_ood[hld_id]['mahala_ipp'][DATASETS[dd]]["auout"].append(metric_results['TMP']['AUOUT'])

            STAT_time[hld_id].append(time.time()-starttime)
    # Print Final Run Statistics {{{
    # Helper to print min/max/avg/std/len of values in a list
    def print_stats_of_list(prefix,dat):
        dat = np.array(dat)
        print("{} Min: {:.4f}; Max: {:.4f}; Avg: {:.4f}; Std: {:.4f}; Len: {}".format(
                prefix, dat.min(), dat.max(), dat.mean(), dat.std(), len(dat))
        )

    print("\n\n")
    print("Printing Final Accuracy + OOD Detection stats for {} Runs with K = {} and J = {}".format(
        REPEAT_ITERS, K, num_holdout_classes))

    print_stats_of_list("Accuracy: ", STAT_accuracy[hld_id])
    print_stats_of_list("Iter Train/Eval Time: ", STAT_time[hld_id])
    for dset in DATASETS:
        if dset == "ID":
            continue
        print("*"*70)
        print("* HLD ID: {} HLD CLASSES: {}".format(hld_id, hld))
        print("* Dataset: {}".format(dset))
        print("*"*70)
        print_stats_of_list("\tBaseline (AUROC):              ", STAT_ood[hld_id]['baseline'][dset]["auroc"])
        print_stats_of_list("\tBaseline (TNR@TPR95):          ", STAT_ood[hld_id]['baseline'][dset]["tnr"])
        print_stats_of_list("\tBaseline (AUPR OUT):             ", STAT_ood[hld_id]['baseline'][dset]["auout"])
        print()
        print_stats_of_list("\tODIN (T=1000) (AUROC):         ",STAT_ood[hld_id]['odin'][dset]["auroc"])
        print_stats_of_list("\tODIN (T=1000) (TNR@TPR95):     ",STAT_ood[hld_id]['odin'][dset]["tnr"])
        print_stats_of_list("\tODIN (T=1000) (AUPR OUT):        ",STAT_ood[hld_id]['odin'][dset]["auout"])
        print()
        print_stats_of_list("\tODIN (T=1000) IPP (AUROC):     ",STAT_ood[hld_id]['odin_ipp'][dset]["auroc"])
        print_stats_of_list("\tODIN (T=1000) IPP (TNR@TPR95): ",STAT_ood[hld_id]['odin_ipp'][dset]["tnr"])
        print_stats_of_list("\tODIN (T=1000) IPP (AUPR OUT):    ",STAT_ood[hld_id]['odin_ipp'][dset]["auout"])
        print()
        print_stats_of_list("\tMahalanobis (AUROC):           ",STAT_ood[hld_id]['mahala'][dset]["auroc"])
        print_stats_of_list("\tMahalanobis (TNR@TPR95):       ",STAT_ood[hld_id]['mahala'][dset]["tnr"])
        print_stats_of_list("\tMahalanobis (AUPR OUT):          ",STAT_ood[hld_id]['mahala'][dset]["auout"])
        print()
        print_stats_of_list("\tMahalanobis IPP (AUROC):       ",STAT_ood[hld_id]['mahala_ipp'][dset]["auroc"])
        print_stats_of_list("\tMahalanobis IPP (TNR@TPR95):   ",STAT_ood[hld_id]['mahala_ipp'][dset]["tnr"])
        print_stats_of_list("\tMahalanobis IPP (AUPR OUT):      ",STAT_ood[hld_id]['mahala_ipp'][dset]["auout"])
        print()

    with open('advOE_baseline_k0_j3_stats_long.pickle', 'wb') as f:
        pickle.dump({'acc':STAT_accuracy, 'ood': STAT_ood,
                     'holdout_groups': hld_groups, 'time': STAT_time}, f)

# Print Final Run Statistics {{{
# Helper to print min/max/avg/std/len of values in a list
def print_stats_of_list(prefix,dat):
    dat = np.array(dat)
    print("{} Min: {:.4f}; Max: {:.4f}; Avg: {:.4f}; Std: {:.4f}; Len: {}".format(
            prefix, dat.min(), dat.max(), dat.mean(), dat.std(), len(dat))
    )

print("\n\n")
print("Printing Final Accuracy + OOD Detection stats for {} Runs with K = {} and J = {}".format(
    REPEAT_ITERS, K, num_holdout_classes))


for hld_id, hld in enumerate(hld_groups):
    print()
    print_stats_of_list("Accuracy: ", STAT_accuracy[hld_id])
    print_stats_of_list("Iter Train/Eval Time: ", STAT_time[hld_id])
    for dset in DATASETS:
        if dset == "ID":
            continue
        print("*"*70)
        print("* HLD ID: {} HLD CLASSES: {}".format(hld_id, hld))
        print("* Dataset: {}".format(dset))
        print("*"*70)
        print_stats_of_list("\tBaseline (AUROC):              ", STAT_ood[hld_id]['baseline'][dset]["auroc"])
        print_stats_of_list("\tBaseline (TNR@TPR95):          ", STAT_ood[hld_id]['baseline'][dset]["tnr"])
        print_stats_of_list("\tBaseline (AUPR OUT):             ", STAT_ood[hld_id]['baseline'][dset]["auout"])
        print()
        print_stats_of_list("\tODIN (T=1000) (AUROC):         ",STAT_ood[hld_id]['odin'][dset]["auroc"])
        print_stats_of_list("\tODIN (T=1000) (TNR@TPR95):     ",STAT_ood[hld_id]['odin'][dset]["tnr"])
        print_stats_of_list("\tODIN (T=1000) (AUPR OUT):        ",STAT_ood[hld_id]['odin'][dset]["auout"])
        print()
        print_stats_of_list("\tODIN (T=1000) IPP (AUROC):     ",STAT_ood[hld_id]['odin_ipp'][dset]["auroc"])
        print_stats_of_list("\tODIN (T=1000) IPP (TNR@TPR95): ",STAT_ood[hld_id]['odin_ipp'][dset]["tnr"])
        print_stats_of_list("\tODIN (T=1000) IPP (AUPR OUT):    ",STAT_ood[hld_id]['odin_ipp'][dset]["auout"])
        print()
        print_stats_of_list("\tMahalanobis (AUROC):           ",STAT_ood[hld_id]['mahala'][dset]["auroc"])
        print_stats_of_list("\tMahalanobis (TNR@TPR95):       ",STAT_ood[hld_id]['mahala'][dset]["tnr"])
        print_stats_of_list("\tMahalanobis (AUPR OUT):          ",STAT_ood[hld_id]['mahala'][dset]["auout"])
        print()
        print_stats_of_list("\tMahalanobis IPP (AUROC):       ",STAT_ood[hld_id]['mahala_ipp'][dset]["auroc"])
        print_stats_of_list("\tMahalanobis IPP (TNR@TPR95):   ",STAT_ood[hld_id]['mahala_ipp'][dset]["tnr"])
        print_stats_of_list("\tMahalanobis IPP (AUPR OUT):      ",STAT_ood[hld_id]['mahala_ipp'][dset]["auout"])
        print()
# vim : foldmethod=marker : foldlevel=0 :
