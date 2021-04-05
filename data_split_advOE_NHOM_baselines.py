"""Baselines for dataset splits
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
hld_groups = [(2,6,9), (6,7,8), (1,2,5), (0,6,7), (2,3,4), (3,5,6)]
for hld_id, hld in enumerate(hld_groups):
    STAT_accuracy[hld_id] = []
    STAT_time[hld_id] = []
    STAT_ood[hld_id] = {}
    for ood in OODS:
        STAT_ood[hld_id][ood] = {}
        for dd in DATASETS:
            STAT_ood[hld_id][ood][dd] = defaultdict(list)


## Create dataset splits
# ORIGINAL CLASS MAPPING:  {0: '2s1', 1: 'bmp2', 2: 'btr70', 3: 'm1', 4: 'm2', 5: 'm35', 6: 'm548', 7: 'm60', 8: 't72', 9: 'zsu23'}
# holdout_classes=[9]
for hld_id, hld in enumerate(hld_groups):
    for ITER in range(REPEAT_ITERS):
        # splits = create_split.create_bce_dataset_splits(
        #     dataset_root, K, holdout_classes, submod, advOE=True, eq_smpl=False)
        # id_trainlist, id_testlist, ood_testlist, oe_trainlist = splits
        starttime=time.time()
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
            neg_hard_smpl_epochs=[30],  # [30, 40, 50],
            neg_hard_smpl_factor=0.9,
        )


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
        net.eval()
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
                # metric_results['TMP']['AUIN'],
                # metric_results['TMP']['AUOUT'],
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
    print_stats_of_list("Iter time: ", STAT_time[hld_id])
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

with open('advOE_NHOM_baseline_k0_j3_stats.pickle', 'wb') as f:
    pickle.dump({'acc':STAT_accuracy, 'ood': STAT_ood, 'holdout_groups': hld_groups, 'time': STAT_time }, f)

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
    print_stats_of_list("Iter time: ", STAT_time[hld_id])
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
