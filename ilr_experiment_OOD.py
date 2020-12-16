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


# Inputs {{{
# Percentage of measured data for training the sample classifier (range = [0, 1])
K = float(sys.argv[1])
# Number of classes to hold out as OOD classes (range = [1, 8])
NUM_HOLDOUT_CLASSES = int(sys.argv[2])

# Path to SAMPLE dataset
dataset_root = "./SAMPLE_dataset/png_images/qpm"

REPEAT_ITERS = 5
#DATASETS = ["ID","holdout","mnist","random", "cifar10"]
DATASETS = ["ID","holdout"]
SEED = 1234567

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
# }}}
# Main Loop over classifers - for each of these loops we train a model from {{{
#    scratch on the sample data then test its OOD detection performance 
# Initialize stat keepers. {{{
STAT_accuracy = []
STAT_ood_baseline = {}
STAT_ood_odin = {}
STAT_ood_odin_ipp = {}
STAT_ood_mahala = {}
STAT_ood_mahala_ipp = {}
STAT_ilr_accuracy = []
STAT_ilr_ood_baseline = {}
STAT_ilr_ood_odin = {}
STAT_ilr_ood_odin_ipp = {}
STAT_ilr_ood_mahala = {}
STAT_ilr_ood_mahala_ipp = {}

for dd in range(1,len(DATASETS)):
    STAT_ood_baseline[DATASETS[dd]] = defaultdict(list)
    STAT_ood_odin[DATASETS[dd]] = defaultdict(list)
    STAT_ood_odin_ipp[DATASETS[dd]] = defaultdict(list)
    STAT_ilr_ood_baseline[DATASETS[dd]] = defaultdict(list)
    STAT_ilr_ood_odin[DATASETS[dd]] = defaultdict(list)
    STAT_ilr_ood_odin_ipp[DATASETS[dd]] = defaultdict(list)
    STAT_ood_mahala[DATASETS[dd]] = defaultdict(list)
    STAT_ood_mahala_ipp[DATASETS[dd]] = defaultdict(list)
    STAT_ilr_ood_mahala[DATASETS[dd]] = defaultdict(list)
    STAT_ilr_ood_mahala_ipp[DATASETS[dd]] = defaultdict(list)
# }}}

random.seed(SEED)
torch.manual_seed(SEED)

for cls_method in ['ilr', 'softmax']:
    for ITER in range(REPEAT_ITERS):

        print("**********************************************************")
        print("Starting Iter: {} / {} for K = {} and J = {}".format(ITER,REPEAT_ITERS,K,NUM_HOLDOUT_CLASSES))
        print("**********************************************************")

        # Model Setup {{{
        # Load Model
        #net = models.sample_model(num_classes=10-NUM_HOLDOUT_CLASSES, drop_prob=dropout).to(device)
        ilr = (cls_method == 'ilr')
        net = models.resnet18(num_classes=10-NUM_HOLDOUT_CLASSES, drop_prob=dropout, ilr=ilr).to(device)

        # Optimizer
        optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
        learning_rate_table = helpers.create_learning_rate_table(learning_rate,learning_rate_decay_schedule,gamma,num_epochs)
        # }}}
        # Create dataset splits {{{

        # Create the measured/synthetic split training and test data
        full_train_list,full_test_list = create_split.create_mixed_dataset_exp41(dataset_root, K)
        clsmap = helpers.get_class_mapping_from_dataset_list(full_train_list)
        print("ORIGINAL CLASS MAPPING: ",clsmap)

        # Pick which of the 10 classes we should holdout
        holdout_classes = list(range(NUM_HOLDOUT_CLASSES))
        #holdout_classes = random.sample(list(range(10)),NUM_HOLDOUT_CLASSES)
        #holdout_classes = [ITER%10] # BE CAREFUL!
        print("HOLDOUT CLASSES: ",holdout_classes)
        remaining_classes = [x for x in list(range(10)) if x not in holdout_classes]
        print("Remaining Classes: ",remaining_classes)

        # Remove the holdout class data from the training dataset and reassign labels
        ID_trainlist = []
        for elem in full_train_list:
            if elem[1] in holdout_classes:
                continue
            else:
                ID_trainlist.append([elem[0], remaining_classes.index(elem[1])])
        # Split the test dataset into ID and OOD data and reassign labels
        ID_testlist = []; OOD_testlist = []
        for elem in full_test_list:
            if elem[1] in holdout_classes:
                OOD_testlist.append([elem[0],0])
            else:
                ID_testlist.append([elem[0], remaining_classes.index(elem[1])])

        print("# ID Train: ",len(ID_trainlist))
        print("# ID Test:  ",len(ID_testlist))
        print("# OOD Test: ",len(OOD_testlist))

        clsmap = helpers.get_class_mapping_from_dataset_list(ID_trainlist)
        print("NEW TRAINING CLASS MAPPING: ",clsmap)
        clsmap = helpers.get_class_mapping_from_dataset_list(ID_testlist)
        print("NEW TESTING CLASS MAPPING:  ",clsmap)

        # Construct datasets and dataloaders
        data_transform = transforms.Compose([transforms.Grayscale(),transforms.CenterCrop(DSIZE),transforms.ToTensor()])
        ID_trainloader = utilsdata.DataLoader(
            custom_dset.Dataset_fromPythonList(ID_trainlist, transform=data_transform),
            batch_size=batch_size,shuffle=True,num_workers=2,timeout=1000,
        )
        ID_testloader = utilsdata.DataLoader(
            custom_dset.Dataset_fromPythonList(ID_testlist, transform=data_transform),
            batch_size=batch_size,shuffle=False,num_workers=2,timeout=1000,
        )
        OOD_testloader = utilsdata.DataLoader(
            custom_dset.Dataset_fromPythonList(OOD_testlist, transform=data_transform),
            batch_size=batch_size,shuffle=False,num_workers=2,timeout=1000,
        )
        # }}}
        # Training Loop {{{
        final_test_acc = 0.
        for epoch in range(num_epochs):

            # Decay learning rate according to decay schedule 
            helpers.adjust_learning_rate(optimizer, epoch, learning_rate_table)
            print("Starting Epoch {}/{}. lr = {}".format(epoch,num_epochs,learning_rate_table[epoch]))
            # Train
            train_acc, train_loss, percent_real = training_helpers.train_model(net, optimizer, ID_trainloader, 10-NUM_HOLDOUT_CLASSES)
            print("[{}] Epoch [ {} / {} ]; lr: {} TrainAccuracy: {:.5f} TrainLoss: {:.5f} %-Real: {}".format(ITER,epoch,num_epochs,learning_rate_table[epoch],train_acc,train_loss,percent_real))
            # Test
            test_acc,test_loss = helpers.test_model(net,device,ID_testloader,MEAN,STD)
            print("\t[{}] Epoch [ {} / {} ]; TestAccuracy: {:.5f} TestLoss: {:.5f}".format(ITER,epoch,num_epochs,test_acc,test_loss))
            final_test_acc = test_acc
        if cls_method == 'ilr':
            STAT_ilr_accuracy.append(final_test_acc)
        else:
            STAT_accuracy.append(final_test_acc)
        net.eval()


        # Optional: Save model checkpoint and move to next iter
        #helpers.save_checkpoint({'test_acc': final_test_acc,'state_dict': net.state_dict()}, False, "{}_K{}_J{}_SEED{}_ITER{}".format(SAVE_CKPT,int(100*K), int(NUM_HOLDOUT_CLASSES),SEED,ITER))
        # }}}
        # Test ID and OOD data {{{

        ### Hook into Net for Mahalanobis layer

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
            print("Hooked Layer: {}; Shape: {}; Min: {}; Max: {}".format(l,hooked_activation_dict[l].shape,hooked_activation_dict[l].min(),hooked_activation_dict[l].max()))

        ### Calculate means and covariance stats for Mahalanobis detector

        layer_means, layer_precisions = ood_helpers.compute_empirical_means_and_precision(net, hooked_activation_dict, MAHALA_LAYER_SET, ID_trainloader, 10-NUM_HOLDOUT_CLASSES)

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

            ##### COMPUTE OOD SCORES FOR THIS DATASET
            base_scores,odin_scores,odin_ipp_scores = ood_helpers.generate_ood_scores_ODIN_and_BASELINE(net, currloader, 1000., MEAN, STD, IPP=True, eps=0.01)
            mahalanobis_scores,mahalanobis_ipp_scores = ood_helpers.generate_ood_scores_MAHALANOBIS(net, currloader, hooked_activation_dict, MAHALA_LAYER_SET, layer_means, layer_precisions, MEAN, STD, IPP=True, eps=0.01)
            #####

            # Save raw OOD scores into dictionaries
            baseline_scores_dict[dset] = base_scores
            odin_scores_dict[dset] = odin_scores
            odin_ipp_scores_dict[dset] = odin_ipp_scores
            mahala_scores_dict[dset] = mahalanobis_scores
            mahala_ipp_scores_dict[dset] = mahalanobis_ipp_scores

        # }}}
        # Compute all OOD statistics for this model over all the tested datasets {{{
        print("Computing OOD Statistics...")
        for dd in range(1,len(DATASETS)):
            print("** DATASET: {} **".format(DATASETS[dd]))

            metric_results = callog.metric( np.array(baseline_scores_dict["ID"]) , np.array(baseline_scores_dict[DATASETS[dd]]) )
            print("\tBaseline.          AUROC: {:.4f}. TNR@95TPR: {:.4f}. DetAcc: {:.4f}".format(
                metric_results['TMP']['AUROC'],
                metric_results['TMP']['TNR'],
                metric_results['TMP']['DTACC'],
            ))
            if cls_method == 'ilr':
                STAT_ilr_ood_baseline[DATASETS[dd]]["auroc"].append(metric_results['TMP']['AUROC'])
                STAT_ilr_ood_baseline[DATASETS[dd]]["tnr"].append(metric_results['TMP']['TNR'])
                STAT_ilr_ood_baseline[DATASETS[dd]]["dtacc"].append(metric_results['TMP']['DTACC'])
            else:
                STAT_ood_baseline[DATASETS[dd]]["auroc"].append(metric_results['TMP']['AUROC'])
                STAT_ood_baseline[DATASETS[dd]]["tnr"].append(metric_results['TMP']['TNR'])
                STAT_ood_baseline[DATASETS[dd]]["dtacc"].append(metric_results['TMP']['DTACC'])

            metric_results = callog.metric( np.array(odin_scores_dict["ID"]) , np.array(odin_scores_dict[DATASETS[dd]]) )
            print("\tODIN (T=1000).     AUROC: {:.4f}. TNR@95TPR: {:.4f}. DetAcc: {:.4f}".format(
                metric_results['TMP']['AUROC'],
                metric_results['TMP']['TNR'],
                metric_results['TMP']['DTACC'],
            ))
            if cls_method == 'ilr':
                STAT_ilr_ood_odin[DATASETS[dd]]["auroc"].append( metric_results['TMP']['AUROC'] )
                STAT_ilr_ood_odin[DATASETS[dd]]["tnr"].append(   metric_results['TMP']['TNR'] )
                STAT_ilr_ood_odin[DATASETS[dd]]["dtacc"].append( metric_results['TMP']['DTACC'] )
            else:
                STAT_ood_odin[DATASETS[dd]]["auroc"].append( metric_results['TMP']['AUROC'] )
                STAT_ood_odin[DATASETS[dd]]["tnr"].append(   metric_results['TMP']['TNR'] )
                STAT_ood_odin[DATASETS[dd]]["dtacc"].append( metric_results['TMP']['DTACC'] )

            metric_results = callog.metric( np.array(mahala_scores_dict["ID"]) , np.array(mahala_scores_dict[DATASETS[dd]]) )
            print("\tMahalanobis.       AUROC: {:.4f}. TNR@95TPR: {:.4f}. DetAcc: {:.4f}".format(
                metric_results['TMP']['AUROC'],
                metric_results['TMP']['TNR'],
                metric_results['TMP']['DTACC'],
            ))
            if cls_method == 'ilr':
                STAT_ilr_ood_mahala[DATASETS[dd]]["auroc"].append( metric_results['TMP']['AUROC'] )
                STAT_ilr_ood_mahala[DATASETS[dd]]["tnr"].append(   metric_results['TMP']['TNR'] )
                STAT_ilr_ood_mahala[DATASETS[dd]]["dtacc"].append( metric_results['TMP']['DTACC'] )
            else:
                STAT_ood_mahala[DATASETS[dd]]["auroc"].append( metric_results['TMP']['AUROC'] )
                STAT_ood_mahala[DATASETS[dd]]["tnr"].append(   metric_results['TMP']['TNR'] )
                STAT_ood_mahala[DATASETS[dd]]["dtacc"].append( metric_results['TMP']['DTACC'] )

            metric_results = callog.metric( np.array(odin_ipp_scores_dict["ID"]) , np.array(odin_ipp_scores_dict[DATASETS[dd]]) )
            print("\tODIN (T=1000) IPP. AUROC: {:.4f}. TNR@95TPR: {:.4f}. DetAcc: {:.4f}".format(
                metric_results['TMP']['AUROC'],
                metric_results['TMP']['TNR'],
                metric_results['TMP']['DTACC'],
            ))
            if cls_method == 'ilr':
                STAT_ilr_ood_odin_ipp[DATASETS[dd]]["auroc"].append( metric_results['TMP']['AUROC'] )
                STAT_ilr_ood_odin_ipp[DATASETS[dd]]["tnr"].append(   metric_results['TMP']['TNR'] )
                STAT_ilr_ood_odin_ipp[DATASETS[dd]]["dtacc"].append( metric_results['TMP']['DTACC'] )
            else:
                STAT_ood_odin_ipp[DATASETS[dd]]["auroc"].append( metric_results['TMP']['AUROC'] )
                STAT_ood_odin_ipp[DATASETS[dd]]["tnr"].append(   metric_results['TMP']['TNR'] )
                STAT_ood_odin_ipp[DATASETS[dd]]["dtacc"].append( metric_results['TMP']['DTACC'] )

            metric_results = callog.metric( np.array(mahala_ipp_scores_dict["ID"]) , np.array(mahala_ipp_scores_dict[DATASETS[dd]]) )
            print("\tMahalanobis IPP.   AUROC: {:.4f}. TNR@95TPR: {:.4f}. DetAcc: {:.4f}".format(
                metric_results['TMP']['AUROC'],
                metric_results['TMP']['TNR'],
                metric_results['TMP']['DTACC'],
            ))
            if cls_method == 'ilr':
                STAT_ilr_ood_mahala_ipp[DATASETS[dd]]["auroc"].append( metric_results['TMP']['AUROC'] )
                STAT_ilr_ood_mahala_ipp[DATASETS[dd]]["tnr"].append(   metric_results['TMP']['TNR'] )
                STAT_ilr_ood_mahala_ipp[DATASETS[dd]]["dtacc"].append( metric_results['TMP']['DTACC'] )
            else:
                STAT_ood_mahala_ipp[DATASETS[dd]]["auroc"].append( metric_results['TMP']['AUROC'] )
                STAT_ood_mahala_ipp[DATASETS[dd]]["tnr"].append(   metric_results['TMP']['TNR'] )
                STAT_ood_mahala_ipp[DATASETS[dd]]["dtacc"].append( metric_results['TMP']['DTACC'] )
        # }}}
# Print Final Run Statistics {{{
# Helper to print min/max/avg/std/len of values in a list
def print_stats_of_list(prefix,dat):
    dat = np.array(dat)
    print("{} Min: {:.4f}; Max: {:.4f}; Avg: {:.4f}; Std: {:.4f}; Len: {}".format(
            prefix, dat.min(), dat.max(), dat.mean(), dat.std(), len(dat))
    )

print("\n\n")
print("Printing Final Accuracy + OOD Detection stats for {} Runs with K = {} and J = {}".format(REPEAT_ITERS, K, NUM_HOLDOUT_CLASSES))
print()
print_stats_of_list("Accuracy: ",STAT_accuracy)
print_stats_of_list("ILR Accuracy: ",STAT_ilr_accuracy)

for dset in DATASETS:
    if dset == "ID":
        continue
    print("*"*70)
    print("* Dataset: {}".format(dset))
    print("*"*70)
    print_stats_of_list("\tBaseline (AUROC):              ",STAT_ood_baseline[dset]["auroc"])
    print_stats_of_list("\tBaseline (TNR@TPR95):          ",STAT_ood_baseline[dset]["tnr"])
    print_stats_of_list("\tBaseline (DETACC):             ",STAT_ood_baseline[dset]["dtacc"])
    print()
    print_stats_of_list("\tODIN (T=1000) (AUROC):         ",STAT_ood_odin[dset]["auroc"])
    print_stats_of_list("\tODIN (T=1000) (TNR@TPR95):     ",STAT_ood_odin[dset]["tnr"])
    print_stats_of_list("\tODIN (T=1000) (DETACC):        ",STAT_ood_odin[dset]["dtacc"])
    print()
    print_stats_of_list("\tODIN (T=1000) IPP (AUROC):     ",STAT_ood_odin_ipp[dset]["auroc"])
    print_stats_of_list("\tODIN (T=1000) IPP (TNR@TPR95): ",STAT_ood_odin_ipp[dset]["tnr"])
    print_stats_of_list("\tODIN (T=1000) IPP (DETACC):    ",STAT_ood_odin_ipp[dset]["dtacc"])
    print()
    print_stats_of_list("\tMahalanobis (AUROC):           ",STAT_ood_mahala[dset]["auroc"])
    print_stats_of_list("\tMahalanobis (TNR@TPR95):       ",STAT_ood_mahala[dset]["tnr"])
    print_stats_of_list("\tMahalanobis (DETACC):          ",STAT_ood_mahala[dset]["dtacc"])
    print()
    print_stats_of_list("\tMahalanobis IPP (AUROC):       ",STAT_ood_mahala_ipp[dset]["auroc"])
    print_stats_of_list("\tMahalanobis IPP (TNR@TPR95):   ",STAT_ood_mahala_ipp[dset]["tnr"])
    print_stats_of_list("\tMahalanobis IPP (DETACC):      ",STAT_ood_mahala_ipp[dset]["dtacc"])
    print()
    print("*"*70)
    print("* ILR Dataset: {}".format(dset))
    print("*"*70)
    print_stats_of_list("\tBaseline (AUROC):              ",STAT_ilr_ood_baseline[dset]["auroc"])
    print_stats_of_list("\tBaseline (TNR@TPR95):          ",STAT_ilr_ood_baseline[dset]["tnr"])
    print_stats_of_list("\tBaseline (DETACC):             ",STAT_ilr_ood_baseline[dset]["dtacc"])
    print()
    print_stats_of_list("\tODIN (T=1000) (AUROC):         ",STAT_ilr_ood_odin[dset]["auroc"])
    print_stats_of_list("\tODIN (T=1000) (TNR@TPR95):     ",STAT_ilr_ood_odin[dset]["tnr"])
    print_stats_of_list("\tODIN (T=1000) (DETACC):        ",STAT_ilr_ood_odin[dset]["dtacc"])
    print()
    print_stats_of_list("\tODIN (T=1000) IPP (AUROC):     ",STAT_ilr_ood_odin_ipp[dset]["auroc"])
    print_stats_of_list("\tODIN (T=1000) IPP (TNR@TPR95): ",STAT_ilr_ood_odin_ipp[dset]["tnr"])
    print_stats_of_list("\tODIN (T=1000) IPP (DETACC):    ",STAT_ilr_ood_odin_ipp[dset]["dtacc"])
    print()
    print_stats_of_list("\tMahalanobis (AUROC):           ",STAT_ilr_ood_mahala[dset]["auroc"])
    print_stats_of_list("\tMahalanobis (TNR@TPR95):       ",STAT_ilr_ood_mahala[dset]["tnr"])
    print_stats_of_list("\tMahalanobis (DETACC):          ",STAT_ilr_ood_mahala[dset]["dtacc"])
    print()
    print_stats_of_list("\tMahalanobis IPP (AUROC):       ",STAT_ilr_ood_mahala_ipp[dset]["auroc"])
    print_stats_of_list("\tMahalanobis IPP (TNR@TPR95):   ",STAT_ilr_ood_mahala_ipp[dset]["tnr"])
    print_stats_of_list("\tMahalanobis IPP (DETACC):      ",STAT_ilr_ood_mahala_ipp[dset]["dtacc"])
    print()
# }}}
# }}}
