# NAI

# Main file for testing OOD detection performance of models following experiment 4.3 specs
#   from the SAMPLE paper.

from __future__ import print_function
import numpy as np
import sys
import os
from collections import defaultdict
import random
import torch
import torch.nn as nn
import torch.utils.data as utilsdata
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as tvdatasets
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pickle

# Custom
import models
import create_split
import Dataset_fromPythonList as custom_dset
import helpers
import training_helpers
import ood_helpers
import calculate_log as callog

# Constants
device = 'cuda' if torch.cuda.is_available() else 'cpu'
DSIZE = 64


# Inputs
# Percentage of measured data for training the sample classifier (range = [0, 1])
K = float(0) #float(sys.argv[1])
# Number of classes to hold out as OOD classes (range = [1, 8])
NUM_HOLDOUT_CLASSES = 1 #int(sys.argv[2])

# Path to SAMPLE dataset
dataset_root = "./SAMPLE_dataset/png_images/qpm"

REPEAT_ITERS = 1
#DATASETS = ["ID","holdout","mnist","random", "cifar10"]
DATASETS = ["ID","holdout"]
SEED = 1234567
random.seed(SEED)
torch.manual_seed(SEED)

# SAMPLE Classifier Learning Params
num_epochs = 60
batch_size = 128
learning_rate_decay_schedule = [50]
learning_rate = 0.001
gamma = 0.1
weight_decay = 0.
dropout = 0.4


# Normalization Constants for range [-1,+1]
MEAN = torch.tensor([0.5], dtype=torch.float32).view([1,1,1]).to(device)
STD  = torch.tensor([0.5], dtype=torch.float32).view([1,1,1]).to(device)
# Main Loop over classifers - for each of these loops we train a model from {{{
#    scratch on the sample data then test its OOD detection performance
# Dictionaries indexed first by experiment
CLS = ['softmax']
OODS = ['baseline', 'odin', 'odin_ipp', 'mahala', 'mahala_ipp']
STAT_accuracy = {}
STAT_ood = {}
SAVE_CKPT = "ckpts/soft_ensemble_advOE_exp0/"

for c in CLS:
    STAT_accuracy[c] = []
    STAT_ood[c] = {}
    for ood in OODS:
        STAT_ood[c][ood] = {}
        for submod in range(10-NUM_HOLDOUT_CLASSES):
            STAT_ood[c][ood][str(submod)] = {}
            for dd in DATASETS:
                STAT_ood[c][ood][str(submod)][dd] = defaultdict(list)


for cls_method in CLS:
    for holdout in range(1):
        holdout_classes = [holdout]
        for ITER in range(REPEAT_ITERS):
            print(40*'*')
            print("Starting Iter: {} / {} for K = {} and J = {}, Cls={}, Holdout={}".format(
                ITER,REPEAT_ITERS,K,NUM_HOLDOUT_CLASSES, cls_method, holdout))
            print(40*'*')
            for submod in range(10-NUM_HOLDOUT_CLASSES):
                print("Training submodel {} of {}".format(submod, 10-NUM_HOLDOUT_CLASSES))
                print(40*"-")

                # Create dataset splits
                # Create the measured/synthetic split training and test data
                splits = create_split.create_dataset_splits(
                    dataset_root, K, holdout_classes)
                ID_trainlist, ID_testlist, OOD_testlist, bce_weights, OE_trainlist = splits
                ID_trainloader, ID_testloader, OOD_testloader, OE_trainloader = \
                    create_split.get_data_loaders(
                        ID_trainlist, ID_testlist, OOD_testlist, OE_trainlist,
                        batch_size=batch_size)

                # Model Setup
                net = models.resnet18(num_classes=10-NUM_HOLDOUT_CLASSES,
                                      drop_prob=dropout,
                                      cls_method=cls_method).to(device)

                # Training Loop
                net, final_test_acc = training_helpers.train_model_advOE(
                    net, num_epochs, ID_trainloader, OE_trainloader,
                    data_mean=MEAN,
                    data_std=STD,
                    lr=learning_rate,
                    weight_decay=weight_decay,
                    scheduler=[50],
                    testloader=ID_testloader,
                    weights=bce_weights,
                )

                STAT_accuracy[cls_method].append(final_test_acc)

                net.eval()

                # Optional: Save model checkpoint and move to next iter
                helpers.save_checkpoint(
                    {'test_acc': final_test_acc,
                     'state_dict': net.state_dict()},
                    False,
                    "{}K{}_J{}_SEED{}_ITER{}_HLD{}_SubMod{}".format(
                        SAVE_CKPT, int(100*K), int(NUM_HOLDOUT_CLASSES), 
                        SEED, ITER, holdout, submod))

                # Test ID and OOD data {{{
                # Hook into Net for Mahalanobis layer
                MAHALA_LAYER_SET = [
                    "resnet18_2",
                    "resnet18_2_2",
                    "resnet18_2_2_2",
                    "resnet18_2_2_2_2",
                ]

                hooked_activation_dict={}; hook_handles = []
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
                    print("Hooked Layer: {}; Shape: {}; Min: {}; Max: {}".format(
                        l, hooked_activation_dict[l].shape,
                        hooked_activation_dict[l].min(), 
                        hooked_activation_dict[l].max()))

                ### Calculate means and covariance stats for Mahalanobis detector
                layer_means, layer_precisions = ood_helpers.compute_empirical_means_and_precision(
                    net, hooked_activation_dict, MAHALA_LAYER_SET, ID_trainloader, 10-NUM_HOLDOUT_CLASSES)

                ### Measure OOD scores with state-of-the-art methods

                baseline_scores_dict = {}
                odin_scores_dict = {}
                odin_ipp_scores_dict = {}
                mahala_scores_dict = {}
                mahala_ipp_scores_dict = {}

                for dset in DATASETS:
                    print("Computing OOD scores for {}...".format(dset))
                    currloader = None
                    if dset == "ID":
                        currloader = ID_testloader
                    elif dset == "holdout":
                        currloader = OOD_testloader
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

                # Compute all OOD statistics for this model over all the tested datasets {{{
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
                    STAT_ood[cls_method]['baseline'][str(submod)][DATASETS[dd]]["auroc"].append(metric_results['TMP']['AUROC'])
                    STAT_ood[cls_method]['baseline'][str(submod)][DATASETS[dd]]["tnr"].append(metric_results['TMP']['TNR'])
                    STAT_ood[cls_method]['baseline'][str(submod)][DATASETS[dd]]["dtacc"].append(metric_results['TMP']['DTACC'])

                    metric_results = callog.metric(
                        np.array(odin_scores_dict["ID"]),
                        np.array(odin_scores_dict[DATASETS[dd]]))
                    print("\tODIN (T=1000).     AUROC: {:.4f}. TNR@95TPR: {:.4f}. DetAcc: {:.4f}".format(
                        metric_results['TMP']['AUROC'],
                        metric_results['TMP']['TNR'],
                        metric_results['TMP']['DTACC'],
                    ))
                    STAT_ood[cls_method]['odin'][str(submod)][DATASETS[dd]]["auroc"].append(metric_results['TMP']['AUROC'])
                    STAT_ood[cls_method]['odin'][str(submod)][DATASETS[dd]]["tnr"].append(metric_results['TMP']['TNR'])
                    STAT_ood[cls_method]['odin'][str(submod)][DATASETS[dd]]["dtacc"].append(metric_results['TMP']['DTACC'])

                    metric_results = callog.metric(
                        np.array(mahala_scores_dict["ID"]),
                        np.array(mahala_scores_dict[DATASETS[dd]]) )
                    print("\tMahalanobis.       AUROC: {:.4f}. TNR@95TPR: {:.4f}. DetAcc: {:.4f}".format(
                        metric_results['TMP']['AUROC'],
                        metric_results['TMP']['TNR'],
                        metric_results['TMP']['DTACC'],
                    ))
                    STAT_ood[cls_method]['mahala'][str(submod)][DATASETS[dd]]["auroc"].append(metric_results['TMP']['AUROC'])
                    STAT_ood[cls_method]['mahala'][str(submod)][DATASETS[dd]]["tnr"].append(metric_results['TMP']['TNR'])
                    STAT_ood[cls_method]['mahala'][str(submod)][DATASETS[dd]]["dtacc"].append(metric_results['TMP']['DTACC'])

                    metric_results = callog.metric(
                        np.array(odin_ipp_scores_dict["ID"]),
                        np.array(odin_ipp_scores_dict[DATASETS[dd]]) )
                    print("\tODIN (T=1000) IPP. AUROC: {:.4f}. TNR@95TPR: {:.4f}. DetAcc: {:.4f}".format(
                        metric_results['TMP']['AUROC'],
                        metric_results['TMP']['TNR'],
                        metric_results['TMP']['DTACC'],
                    ))
                    STAT_ood[cls_method]['odin_ipp'][str(submod)][DATASETS[dd]]["auroc"].append(metric_results['TMP']['AUROC'])
                    STAT_ood[cls_method]['odin_ipp'][str(submod)][DATASETS[dd]]["tnr"].append(metric_results['TMP']['TNR'])
                    STAT_ood[cls_method]['odin_ipp'][str(submod)][DATASETS[dd]]["dtacc"].append(metric_results['TMP']['DTACC'])

                    metric_results = callog.metric(
                        np.array(mahala_ipp_scores_dict["ID"]),
                        np.array(mahala_ipp_scores_dict[DATASETS[dd]]) )
                    print("\tMahalanobis IPP.   AUROC: {:.4f}. TNR@95TPR: {:.4f}. DetAcc: {:.4f}".format(
                        metric_results['TMP']['AUROC'],
                        metric_results['TMP']['TNR'],
                        metric_results['TMP']['DTACC'],
                    ))
                    STAT_ood[cls_method]['mahala_ipp'][str(submod)][DATASETS[dd]]["auroc"].append(metric_results['TMP']['AUROC'])
                    STAT_ood[cls_method]['mahala_ipp'][str(submod)][DATASETS[dd]]["tnr"].append(metric_results['TMP']['TNR'])
                    STAT_ood[cls_method]['mahala_ipp'][str(submod)][DATASETS[dd]]["dtacc"].append(metric_results['TMP']['DTACC'])

with open(SAVE_CKPT+'experiment_stats.pickle', 'wb') as f:
    pickle.dump({'acc':STAT_accuracy, 'ood': STAT_ood}, f)


# # Print Final Run Statistics {{{
# # Helper to print min/max/avg/std/len of values in a list
# def print_stats_of_list(prefix,dat):
#     dat = np.array(dat)
#     print("{} Min: {:.4f}; Max: {:.4f}; Avg: {:.4f}; Std: {:.4f}; Len: {}".format(
#             prefix, dat.min(), dat.max(), dat.mean(), dat.std(), len(dat))
#     )

# print("\n\n")
# print("Printing Final Accuracy + OOD Detection stats for {} Runs with K = {} and J = {}".format(REPEAT_ITERS, K, NUM_HOLDOUT_CLASSES))
# print()
# print_stats_of_list("Accuracy: ",STAT_accuracy)
# print_stats_of_list("ILR Accuracy: ",STAT_ilr_accuracy)

# for dset in DATASETS:
#     if dset == "ID":
#         continue
#     print("*"*70)
#     print("* Dataset: {}".format(dset))
#     print("*"*70)
#     print_stats_of_list("\tBaseline (AUROC):              ",STAT_ood_baseline[dset]["auroc"])
#     print_stats_of_list("\tBaseline (TNR@TPR95):          ",STAT_ood_baseline[dset]["tnr"])
#     print_stats_of_list("\tBaseline (DETACC):             ",STAT_ood_baseline[dset]["dtacc"])
#     print()
#     print_stats_of_list("\tODIN (T=1000) (AUROC):         ",STAT_ood_odin[dset]["auroc"])
#     print_stats_of_list("\tODIN (T=1000) (TNR@TPR95):     ",STAT_ood_odin[dset]["tnr"])
#     print_stats_of_list("\tODIN (T=1000) (DETACC):        ",STAT_ood_odin[dset]["dtacc"])
#     print()
#     print_stats_of_list("\tODIN (T=1000) IPP (AUROC):     ",STAT_ood_odin_ipp[dset]["auroc"])
#     print_stats_of_list("\tODIN (T=1000) IPP (TNR@TPR95): ",STAT_ood_odin_ipp[dset]["tnr"])
#     print_stats_of_list("\tODIN (T=1000) IPP (DETACC):    ",STAT_ood_odin_ipp[dset]["dtacc"])
#     print()
#     print_stats_of_list("\tMahalanobis (AUROC):           ",STAT_ood_mahala[dset]["auroc"])
#     print_stats_of_list("\tMahalanobis (TNR@TPR95):       ",STAT_ood_mahala[dset]["tnr"])
#     print_stats_of_list("\tMahalanobis (DETACC):          ",STAT_ood_mahala[dset]["dtacc"])
#     print()
#     print_stats_of_list("\tMahalanobis IPP (AUROC):       ",STAT_ood_mahala_ipp[dset]["auroc"])
#     print_stats_of_list("\tMahalanobis IPP (TNR@TPR95):   ",STAT_ood_mahala_ipp[dset]["tnr"])
#     print_stats_of_list("\tMahalanobis IPP (DETACC):      ",STAT_ood_mahala_ipp[dset]["dtacc"])
#     print()
#     print("*"*70)
#     print("* ILR Dataset: {}".format(dset))
#     print("*"*70)
#     print_stats_of_list("\tBaseline (AUROC):              ",STAT_ilr_ood_baseline[dset]["auroc"])
#     print_stats_of_list("\tBaseline (TNR@TPR95):          ",STAT_ilr_ood_baseline[dset]["tnr"])
#     print_stats_of_list("\tBaseline (DETACC):             ",STAT_ilr_ood_baseline[dset]["dtacc"])
#     print()
#     print_stats_of_list("\tODIN (T=1000) (AUROC):         ",STAT_ilr_ood_odin[dset]["auroc"])
#     print_stats_of_list("\tODIN (T=1000) (TNR@TPR95):     ",STAT_ilr_ood_odin[dset]["tnr"])
#     print_stats_of_list("\tODIN (T=1000) (DETACC):        ",STAT_ilr_ood_odin[dset]["dtacc"])
#     print()
#     print_stats_of_list("\tODIN (T=1000) IPP (AUROC):     ",STAT_ilr_ood_odin_ipp[dset]["auroc"])
#     print_stats_of_list("\tODIN (T=1000) IPP (TNR@TPR95): ",STAT_ilr_ood_odin_ipp[dset]["tnr"])
#     print_stats_of_list("\tODIN (T=1000) IPP (DETACC):    ",STAT_ilr_ood_odin_ipp[dset]["dtacc"])
#     print()
#     print_stats_of_list("\tMahalanobis (AUROC):           ",STAT_ilr_ood_mahala[dset]["auroc"])
#     print_stats_of_list("\tMahalanobis (TNR@TPR95):       ",STAT_ilr_ood_mahala[dset]["tnr"])
#     print_stats_of_list("\tMahalanobis (DETACC):          ",STAT_ilr_ood_mahala[dset]["dtacc"])
#     print()
#     print_stats_of_list("\tMahalanobis IPP (AUROC):       ",STAT_ilr_ood_mahala_ipp[dset]["auroc"])
#     print_stats_of_list("\tMahalanobis IPP (TNR@TPR95):   ",STAT_ilr_ood_mahala_ipp[dset]["tnr"])
#     print_stats_of_list("\tMahalanobis IPP (DETACC):      ",STAT_ilr_ood_mahala_ipp[dset]["dtacc"])
#     print()
