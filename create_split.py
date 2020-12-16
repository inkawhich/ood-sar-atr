# NAI

# This file creates a train/test split of the SAMPLE dataset according to the notation
#   in the SAMPLE paper

import glob
import os
import random

import torch
import torch.utils.data as utilsdata
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

# Custom
import helpers
import Dataset_fromPythonList as custom_dset

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def create_mixed_dataset_exp41(root, k):

    CLASSES = ["2s1", "bmp2", "btr70", "m1", "m2", "m35", "m548", "m60", "t72", "zsu23"]

    # Lists of tuples that make up the datasets ("/pth/to/f.png", class#)
    dataset_list_train = []
    dataset_list_test  = []

    # Create the splits for each of the classes individually
    for cls in CLASSES:
        all_measured = glob.glob("{}/{}/{}/*.png".format(root,"real",cls))
        Nmj = len(all_measured)
        test_data = []; train_data = []
        for fname in all_measured:
            if "elevDeg_017" in fname:
                test_data.append([fname,CLASSES.index(cls)])
            else:
                train_data.append([fname,CLASSES.index(cls)])
        Smj = len(test_data)
        Tmj = round(k*(Nmj - Smj))  # How many "real" samples to use for this class
        Tsj = (Nmj - Smj) - Tmj     # How many "synth" samples to use for this class
        assert((Nmj-Smj)==len(train_data))
        # For each measured sample we dont use replace it with its synthetic version
        synth_inds = random.sample( list(range(len(train_data))), Tsj)
        for ind in synth_inds:
            train_data[ind][0] = train_data[ind][0].replace("/real/","/synth/")
            train_data[ind][0] = train_data[ind][0].replace("_real_","_synth_")
            assert(os.path.isfile(train_data[ind][0]))
        dataset_list_train.extend(train_data)
        dataset_list_test.extend(test_data)
        print("Class: {}\tNmj: {} \tSmj: {}\tTmj: {}\tTsj: {}".format(cls,Nmj,Smj,Tmj,Tsj))
    print("len( train ): ",len(dataset_list_train))
    print("len( test ):  ",len(dataset_list_test))

    return dataset_list_train, dataset_list_test


def create_OE_dataset(root):
    # Lists of tuples that make up the datasets ("/pth/to/f.png", class#)
    dataset_list = []

    # Create the splits for each of the classes individually
    all_imgs = glob.glob("{}/**/*.jpg".format(root))
    Nmj = len(all_imgs)
    for fname in all_imgs:
        dataset_list.append([fname,-1])
    assert(len(dataset_list)==len(all_imgs))
    print("len( AdvOE ): ",len(dataset_list))
    return dataset_list

def create_dataset_splits(dataset_root, K, holdout_classes, advOE=False):
        full_train_list, full_test_list = create_mixed_dataset_exp41(dataset_root, K)
        clsmap = helpers.get_class_mapping_from_dataset_list(full_train_list)
        print("ORIGINAL CLASS MAPPING: ",clsmap)
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

        class_counts = torch.zeros((10-len(holdout_classes)),
                                   dtype=torch.float32).to(device)
        for i in ID_trainlist:
            class_counts[i[1]] += 1
            bce_weights = (class_counts.sum()-class_counts)/class_counts

        if advOE:
            full_OE_trainlist = create_OE_dataset('./ships-big/')
            OE_inds = random.sample(list(range(len(full_OE_trainlist))),
                                    len(ID_trainlist))
            OE_trainlist = [full_OE_trainlist[i] for i in OE_inds]
            return ID_trainlist, ID_testlist, OOD_testlist, bce_weights, OE_trainlist
        return ID_trainlist, ID_testlist, OOD_testlist, bce_weights


def create_bce_dataset_splits(dataset_root, K, holdout_classes, bce_class,
                              advOE=False):
        full_train_list, full_test_list = create_mixed_dataset_exp41(dataset_root, K)
        clsmap = helpers.get_class_mapping_from_dataset_list(full_train_list)
        print("ORIGINAL CLASS MAPPING: ",clsmap)
        print("HOLDOUT CLASSES: ",holdout_classes)
        remaining_classes = [x for x in list(range(10)) if x not in holdout_classes]
        print("Remaining Classes: ",remaining_classes)
        print("BCE Class: ", bce_class)

        # Remove the holdout class data from the training dataset and reassign labels
        ID_trainlist = []
        ID_trainlist_neg = []
        for elem in full_train_list:
            if elem[1] in holdout_classes:
                continue
            elif remaining_classes.index(elem[1]) == bce_class:
                ID_trainlist.append([elem[0], 1])
            else:
                ID_trainlist_neg.append([elem[0], 0])

        # Sample from  the negatives to balance train list
        num_pos_train = len(ID_trainlist)
        neg_inds = random.sample(list(range(len(ID_trainlist_neg))),
                                 min(len(ID_trainlist), len(ID_trainlist_neg)))
        num_neg_train = len(neg_inds)
        ID_trainlist.extend([ID_trainlist_neg[i] for i in neg_inds])

        # Split the test dataset into ID and OOD data and reassign labels
        ID_testlist = []; ID_testlist_neg = []; OOD_testlist = []
        for elem in full_test_list:
            if elem[1] in holdout_classes:
                OOD_testlist.append([elem[0],0])
            elif remaining_classes.index(elem[1]) == bce_class:
                ID_testlist.append([elem[0], 1])
            else:
                ID_testlist_neg.append([elem[0], 0])

        # Sample from  the negatives to balance test list
        num_pos_test = len(ID_testlist)
        neg_inds = random.sample(list(range(len(ID_testlist_neg))),
                                 min(len(ID_testlist), len(ID_testlist_neg)))
        num_neg_test = len(neg_inds)
        ID_testlist.extend([ID_testlist_neg[i] for i in neg_inds])


        print("# ID Train: ",len(ID_trainlist))
        print("# ID Train Pos: ", num_pos_train)
        print("# ID Train Neg: ", num_neg_train)
        print("# ID Test:  ",len(ID_testlist))
        print("# ID Test Pos: ", num_pos_test)
        print("# ID Test Neg: ", num_neg_test)
        print("# OOD Test: ",len(OOD_testlist))

        # clsmap = helpers.get_class_mapping_from_dataset_list(ID_trainlist)
        # print("NEW TRAINING CLASS MAPPING: ",clsmap)
        # clsmap = helpers.get_class_mapping_from_dataset_list(ID_testlist)
        # print("NEW TESTING CLASS MAPPING:  ",clsmap)
        if advOE:
            full_OE_trainlist = create_OE_dataset('ships-big/')
            OE_inds = random.sample(list(range(len(full_OE_trainlist))),
                                    len(ID_trainlist))
            OE_trainlist = [[full_OE_trainlist[i][0], 0] for i in OE_inds]
            return ID_trainlist, ID_testlist, OOD_testlist, OE_trainlist
        return ID_trainlist, ID_testlist, OOD_testlist


def get_data_loaders(ID_trainlist, ID_testlist, OOD_testlist, OE_trainlist=None,
                     dsize=64, batch_size=128):
    # Construct datasets and dataloaders
    data_transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.CenterCrop(dsize),
        transforms.ToTensor()])

    ID_trainloader = utilsdata.DataLoader(
        custom_dset.Dataset_fromPythonList(ID_trainlist, transform=data_transform),
        batch_size=batch_size, shuffle=True, num_workers=2, timeout=1000,
    )
    ID_testloader = utilsdata.DataLoader(
        custom_dset.Dataset_fromPythonList(ID_testlist, transform=data_transform),
        batch_size=batch_size, shuffle=False, num_workers=2, timeout=1000,
    )
    OOD_testloader = utilsdata.DataLoader(
        custom_dset.Dataset_fromPythonList(OOD_testlist, transform=data_transform),
        batch_size=batch_size, shuffle=False, num_workers=2, timeout=1000,
    )
    if OE_trainlist is not None:
        data_transform_oe = transforms.Compose([
            transforms.Resize(dsize),
            transforms.Grayscale(),
            transforms.CenterCrop(dsize),
            transforms.ToTensor()])
        OE_trainloader = utilsdata.DataLoader(
            custom_dset.Dataset_fromPythonList(OE_trainlist, transform=data_transform_oe),
            batch_size=batch_size, shuffle=True, num_workers=2, timeout=1000,
        )
        return ID_trainloader, ID_testloader, OOD_testloader, OE_trainloader
    return ID_trainloader, ID_testloader, OOD_testloader
