# NAI

# This file contains helper functions specifically for training a DNN on in-distribution
#    data from some training dataloader. 


from __future__ import print_function
#import numpy as np
#import sys
#import os
#import random
import torch
import torch.nn as nn
#import torch.utils.data as utilsdata
import torch.nn.functional as F
#import torchvision
#import torchvision.transforms as transforms
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
#import scipy.stats as st
from torch.autograd import Variable
#import torch.nn.functional as F
from torch.autograd.gradcheck import zero_gradients

import helpers

device = 'cuda' if torch.cuda.is_available() else 'cpu'


###############################################################################
#### LABEL SMOOTHING STUFF
# Transform the true "Long" labels to softlabels. The confidence of the gt class is 
#  1-smoothing, and the rest of the probability (i.e. smoothing) is uniformly distributed
#  across the non-gt classes. Note, this is slightly different than standard smoothing
#### MIXUP STUFF
def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


###############################################################################
#### ADVERSARIAL TRAINING STUFF

# Gradient - Forward pass data through model and return gradient w.r.t. data
# model - pytorch model to be used for forward pass
# device - device the model is running on
# data -  NCHW tensor of range [0.,1.]
# lbl - label to calculate loss against (usually GT label or target label for targeted attacks)
# returns gradient of loss w.r.t data
# Note: It is important this is treated as an atomic operation because of the
#       normalization. We carry the data around unnormalized, so in this fxn we
#       normalize, forward pass, then unnorm the gradients before returning. Any
#       fxn that uses this method should handle the data in [0,1] range
def gradient_wrt_data(model,device,data,lbl):
    # Manually Normalize
    mean = torch.tensor([0.5], dtype=torch.float32).view([1,1,1]).to(device)
    std = torch.tensor([0.5], dtype=torch.float32).view([1,1,1]).to(device)
    dat = (data-mean)/std
    # Forward pass through the model
    dat.requires_grad = True
    out = model(dat)
    # Calculate loss
    loss = F.cross_entropy(out,lbl)
    # zero all old gradients in the model
    model.zero_grad()
    # Back prop the loss to calculate gradients
    loss.backward()
    # Extract gradient of loss w.r.t data
    data_grad = dat.grad.data
    # Unnorm gradients back into [0,1] space
    #   As shown in foolbox/models/base.py
    grad = data_grad / std
    return grad.data.detach()

def gradient_wrt_data_OE(model,device,data,lbl):
    # Manually Normalize
    mean = torch.tensor([0.5], dtype=torch.float32).view([1,1,1]).to(device)
    std = torch.tensor([0.5], dtype=torch.float32).view([1,1,1]).to(device)
    dat = (data-mean)/std
    # Forward pass through the model
    dat.requires_grad = True
    out = model(dat)
    # Calculate loss
    loss = F.log_softmax(out, 1)
    loss *= -(1./out.size(1))
    loss = loss.sum(1)
    # zero all old gradients in the model
    model.zero_grad()
    # Back prop the loss to calculate gradients
    loss.backward()
    # Extract gradient of loss w.r.t data
    data_grad = dat.grad.data
    # Unnorm gradients back into [0,1] space
    #   As shown in foolbox/models/base.py
    grad = data_grad / std
    return grad.data.detach()


# Projected Gradient Descent Attack (PGD) with random start
def PGD_Linf_attack(model, device, dat, lbl, eps, alpha, iters, withOE=False):
    x_nat = dat.clone().detach()
    # Randomly perturb within small eps-norm ball
    x_adv = dat.clone().detach() + torch.FloatTensor(dat.shape).uniform_(-eps, eps).to(device)
    x_adv = torch.clamp(x_adv,0.,1.) # respect image bounds
    # Iteratively Perturb data
    for i in range(iters):
        zero_gradients(x_adv)
        model.zero_grad()
        # Calculate gradient w.r.t. data
        grad=None
        if withOE:
            grad = gradient_wrt_data_OE(model,device,x_adv.clone().detach(),lbl.clone().detach())
        else:
            grad = gradient_wrt_data(model,device,x_adv.clone().detach(),lbl)
        # Perturb by the small amount a
        x_adv = x_adv + alpha*grad.sign()
        # Clip the perturbations w.r.t. the original data so we still satisfy l_infinity
        x_adv = torch.max(torch.min(x_adv, x_nat+eps), x_nat-eps)
        # Make sure we are still in bounds
        x_adv = torch.clamp(x_adv, 0., 1.)
    return x_adv.data.clone().detach()



class BCELoss:
    """Binary Cross Entropy criteria for use with ILR classifiers"""
    def __init__(self, weights=None):
        self.weights = weights

    def to_one_hot(self, inp, num_classes):
        if inp.max() <= 1:
            out = inp.view(-1,1).float()
        else:
            out = torch.zeros((inp.size()[0], num_classes), dtype=float,
                              requires_grad=False, device=device)
            out[torch.arange(inp.size(0)), inp] = 1
        return out

    def __call__(self, outputs, labels):
        num_classes = outputs.shape[1]
        lbls = self.to_one_hot(labels, num_classes)
        loss = F.binary_cross_entropy_with_logits(outputs, lbls,
                                                  pos_weight=self.weights)
        return loss


class L2Loss:
    """Custom l2 loss"""
    def __init__(self, ):
        pass

    def __call__(self, outputs, labels):
        assert (outputs.shape[0] == labels.shape[0])
        y_mat = torch.zeros_like(outputs)
        y_inds = torch.arange(labels.shape[0])
        y_mat[y_inds, labels] = 1.
        diff = outputs-y_mat
        loss = torch.norm(diff, p=2, dim=1).mean()
        return loss


class OESoftmaxLoss:
    """Binary Cross Entropy criteria for use with ILR classifiers"""
    def __init__(self,):
        pass

    def __call__(self, outputs, labels):
        num_classes = outputs.size(1)
        loss = F.log_softmax(outputs,1)
        loss *= -(1./num_classes)
        loss = loss.sum(1)
        return loss.sum()


###############################################################################
#### ONE EPOCH OF CLASSIFIER MODEL TRAINING
def train_model(net, epochs, trainloader,
                checkpoint_prefix=None,
                data_mean=torch.tensor([0.5], dtype=torch.float32).view([1,1,1]).to(device),
                data_std=torch.tensor([0.5], dtype=torch.float32).view([1,1,1]).to(device),
                lr=0.001,
                weight_decay=0.,
                optimizer=None,
                scheduler=None,
                testloader=None,
                weights=None,
               ):
    """Training helper function

    Parameters
    ----------
    net : nn.Module
    epochs : int
    trainloader : torch.utils.data.DataLoader
    data_mean : float
    data_std : float
    lr : float
        initial learning rate
    weight_decay : float
    optimizer : None, torch.optim.Optimizer
        default: Adam
    scheduler : None, torch.optim.lr_scheduler._LRScheduler, list, tuple
        default: MultiStepLR w/ 1 step at 50% completion and gamma = 0.1
        A list of epoch nums can be passed in and a default gamma value of 0.1
        will be applied at each step
        If a tuple is passed, the first element will be a list of epochs and
        the second is the gamma value
    testloaders : None, torch.utils.data.DataLoader
    weights : torch.FloatTensor
        Values to weight the loss by for each class
    """

    if optimizer is None:
        optimizer = torch.optim.Adam(
            net.parameters(), lr, weight_decay=weight_decay)

    if scheduler is None:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, gamma=0.1, milestones=[int(epochs*0.5)])
    elif scheduler is list:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, gamma=0.1, milestones=scheduler)
    elif scheduler is tuple:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, gamma=scheduler[1], milestones=scheduler[0])

    net.train()

    try:
        if net.cls_method == 'softmax':
            criterion = nn.CrossEntropyLoss()
        elif net.cls_method == 'ilr':
            criterion = BCELoss(weights=weights)
        elif net.cls_method == 'l2':
            criterion = L2Loss()
        else:
            criterion = nn.CrossEntropyLoss()
    except AttributeError:
        criterion = nn.CrossEntropyLoss()


    # DATA AUGMENTATION AND LOSS FXN CONFIGS
    gaussian_std = 0.4
    # gaussian_std = 0.3
    # gaussian_std = 0.05

    #LBLSMOOTHING_PARAM = 0.1 # Only for label smoothing
    #MIXUP_ALPHA = 0.1  # Only for mixup
    #AT_EPS = 2./255.; AT_ALPHA = 0.5/255.; AT_ITERS = 7
    #AT_EPS = 4./255.; AT_ALPHA = 1./255. ; AT_ITERS = 7
    # AT_EPS = 8./255.; AT_ALPHA = 2./255. ; AT_ITERS = 7

    best_acc = 0.

    for epoch in range(epochs):

        running_correct = 0.
        running_total = 0.
        running_loss_sum = 0.
        running_real_cnt = 0.


        for batch_idx,(data,labels,pth) in enumerate(trainloader):
            data = data.to(device); labels = labels.to(device)

            # MIXUP
            # mixed_data, targets_a, targets_b, lam = mixup_data(
            #     data, labels, MIXUP_ALPHA, use_cuda=True)
            # mixed_data, targets_a, targets_b = map(
            #     Variable, (mixed_data, targets_a, targets_b))

            if(gaussian_std != 0):
                data += torch.randn_like(data)*gaussian_std;
                data = torch.clamp(data, 0, 1);
                #mixed_data += torch.randn_like(mixed_data)*gaussian_std;
                #mixed_data = torch.clamp(mixed_data, 0, 1);

            # ADVERSARIALLY PERTURB DATA
            # data = PGD_Linf_attack(net, device, data.clone().detach(), labels,
                                       # eps=AT_EPS, alpha=AT_ALPHA, iters=AT_ITERS)

            # Plot some training samples
            #plt.figure(figsize=(10,3))
            #plt.subplot(1,6,1);plt.imshow(data[0].cpu().numpy().squeeze(),vmin=0,vmax=1);plt.title(pth[0].split("/")[-1].split("_")[:2])
            #plt.subplot(1,6,2);plt.imshow(data[1].cpu().numpy().squeeze(),vmin=0,vmax=1);plt.title(pth[1].split("/")[-1].split("_")[:2])
            #plt.subplot(1,6,3);plt.imshow(data[2].cpu().numpy().squeeze(),vmin=0,vmax=1);plt.title(pth[2].split("/")[-1].split("_")[:2])
            #plt.subplot(1,6,4);plt.imshow(unshifted[0].cpu().numpy().squeeze(),vmin=0,vmax=1);plt.title(pth[0].split("/")[-1].split("_")[:2])
            #plt.subplot(1,6,5);plt.imshow(unshifted[1].cpu().numpy().squeeze(),vmin=0,vmax=1);plt.title(pth[1].split("/")[-1].split("_")[:2])
            #plt.subplot(1,6,6);plt.imshow(unshifted[2].cpu().numpy().squeeze(),vmin=0,vmax=1);plt.title(pth[2].split("/")[-1].split("_")[:2])
            #plt.show()
            #exit()

            # MIXUP
            #outputs = net((mixed_data-data_mean)/data_std)
            #loss = mixup_criterion(nn.CrossEntropyLoss(), outputs, targets_a, targets_b, lam)

            # Forward pass data through model. Normalize before forward pass
            outputs = net((data-data_mean)/data_std)
            # VANILLA CROSS-ENTROPY
            loss = criterion(outputs, labels)

            # LABEL SMOOTHING LOSS
            #sl = smooth_one_hot(labels,num_classes,smoothing=LBLSMOOTHING_PARAM)
            #loss =  xent_with_soft_targets(outputs, sl)

            # COSINE LOSS
            #one_hots = smooth_one_hot(labels,10,smoothing=0.)
            #loss = (1. - (one_hots * F.normalize(outputs,p=2,dim=1)).sum(1)).mean()

            # Calculate gradient and update parameters
            optimizer.zero_grad()
            net.zero_grad()
            loss.backward()
            #torch.nn.utils.clip_grad_norm(net.parameters(), max_norm=10., norm_type=2)  # COSINE LOSS
            optimizer.step()

            # Measure accuracy and loss for this batch
            try:
                preds = net.predict(outputs)
            except:
                _, preds= outputs.max(1)
            running_total += labels.size(0)
            running_correct += preds.eq(labels).sum().item()
            #running_correct += (lam * preds.eq(targets_a.data).cpu().sum().float() + (1 - lam) * preds.eq(targets_b.data).cpu().sum().float()) # MIXUP
            running_loss_sum += loss.item()

            # Compute measured/synthetic split for the batch
            for tp in pth:
                if "/real/" in tp:
                    running_real_cnt += 1.

        train_acc = running_correct/running_total
        train_loss =  running_loss_sum/len(trainloader)
        percent_real = running_real_cnt/running_total

        print("Epoch [ {} / {} ]; lr: {} TrainAccuracy: {:.5f} TrainLoss: {:.5f} %-Real: {}".format(
            epoch,epochs,optimizer.param_groups[0]['lr'],
            train_acc,train_loss,percent_real))

        # Test
        if testloader is not None:
            test_acc,test_loss = helpers.test_model(
                net,device,testloader,data_mean,data_std, criterion=criterion)
            print("\tEpoch [ {} / {} ]; TestAccuracy: {:.5f} TestLoss: {:.5f}".format(
                epoch,epochs,test_acc,test_loss))
            final_test_acc = test_acc

            if (checkpoint_prefix is not None) and (test_acc>best_acc):
                best_acc = test_acc
                print("Saving...")
                torch.save(
                    {'test_acc':test_acc, 'state_dict': net.state_dict()},
                    checkpoint_prefix+'_best_checkpoint.pth.tar')

    return net, final_test_acc


def train_model_advOE(net, epochs, trainloader, OE_trainloader,
                checkpoint_prefix=None,
                data_mean=torch.tensor([0.5], dtype=torch.float32).view([1,1,1]).to(device),
                data_std=torch.tensor([0.5], dtype=torch.float32).view([1,1,1]).to(device),
                lr=0.001,
                weight_decay=0.,
                optimizer=None,
                scheduler=None,
                testloader=None,
                weights=None,
               ):
    """Training helper function

    Parameters
    ----------
    net : nn.Module
    epochs : int
    trainloader : torch.utils.data.DataLoader
    data_mean : float
    data_std : float
    lr : float
        initial learning rate
    weight_decay : float
    optimizer : None, torch.optim.Optimizer
        default: Adam
    scheduler : None, torch.optim.lr_scheduler._LRScheduler, list, tuple
        default: MultiStepLR w/ 1 step at 50% completion and gamma = 0.1
        A list of epoch nums can be passed in and a default gamma value of 0.1
        will be applied at each step
        If a tuple is passed, the first element will be a list of epochs and
        the second is the gamma value
    testloaders : None, torch.utils.data.DataLoader
    weights : torch.FloatTensor
        Values to weight the loss by for each class
    """

    if optimizer is None:
        optimizer = torch.optim.Adam(
            net.parameters(), lr, weight_decay=weight_decay)

    if scheduler is None:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, gamma=0.1, milestones=[int(epochs*0.5)])
    elif scheduler is list:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, gamma=0.1, milestones=scheduler)
    elif scheduler is tuple:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, gamma=scheduler[1], milestones=scheduler[0])

    net.train()

    try:
        if net.cls_method == 'softmax':
            criterion = nn.CrossEntropyLoss()
        elif net.cls_method == 'ilr':
            criterion = BCELoss(weights=weights)
        elif net.cls_method == 'l2':
            criterion = L2Loss()
        else:
            criterion = nn.CrossEntropyLoss()
    except AttributeError:
        criterion = nn.CrossEntropyLoss()


    # DATA AUGMENTATION AND LOSS FXN CONFIGS
    gaussian_std = 0.4
    # gaussian_std = 0.3
    # gaussian_std = 0.05

    #LBLSMOOTHING_PARAM = 0.1 # Only for label smoothing
    #MIXUP_ALPHA = 0.1  # Only for mixup
    AT_EPS = 2./255.; AT_ALPHA = 0.5/255.; AT_ITERS = 7
    #AT_EPS = 4./255.; AT_ALPHA = 1./255. ; AT_ITERS = 7
    # AT_EPS = 8./255.; AT_ALPHA = 2./255. ; AT_ITERS = 7

    best_acc = 0.

    for epoch in range(epochs):

        running_correct = 0.
        running_total = 0.
        running_loss_sum = 0.
        running_real_cnt = 0.


        for batch_idx,((data,labels,pth), (oedata,oelabels,oepth)) in enumerate(
                zip(trainloader, OE_trainloader)):
            data = data.to(device); labels = labels.to(device)
            oedata = oedata.to(device); oelabels = oelabels.to(device)

            # MIXUP
            # mixed_data, targets_a, targets_b, lam = mixup_data(
            #     data, labels, MIXUP_ALPHA, use_cuda=True)
            # mixed_data, targets_a, targets_b = map(
            #     Variable, (mixed_data, targets_a, targets_b))

            if(gaussian_std != 0):
                data += torch.randn_like(data)*gaussian_std;
                data = torch.clamp(data, 0, 1);
                oedata += torch.randn_like(oedata)*gaussian_std;
                oedata = torch.clamp(oedata, 0, 1);
                #mixed_data += torch.randn_like(mixed_data)*gaussian_std;
                #mixed_data = torch.clamp(mixed_data, 0, 1);

            # ADVERSARIALLY PERTURB DATA
            data = PGD_Linf_attack(net, device, data.clone().detach(), labels,
                                       eps=AT_EPS, alpha=AT_ALPHA, iters=AT_ITERS)
            if net.cls_method == 'ilr':
                oedata = PGD_Linf_attack(net, device, oedata.clone().detach(), oelabels,
                                         eps=AT_EPS, alpha=AT_ALPHA, iters=AT_ITERS)
            elif net.cls_method == 'softmax':
                oedata = PGD_Linf_attack(net, device, oedata.clone().detach(),
                                         eps=AT_EPS, alpha=AT_ALPHA,
                                         iters=AT_ITERS, withOE=True)
            else:
                raise ValueError

            # Plot some training samples
            #plt.figure(figsize=(10,3))
            #plt.subplot(1,6,1);plt.imshow(data[0].cpu().numpy().squeeze(),vmin=0,vmax=1);plt.title(pth[0].split("/")[-1].split("_")[:2])
            #plt.subplot(1,6,2);plt.imshow(data[1].cpu().numpy().squeeze(),vmin=0,vmax=1);plt.title(pth[1].split("/")[-1].split("_")[:2])
            #plt.subplot(1,6,3);plt.imshow(data[2].cpu().numpy().squeeze(),vmin=0,vmax=1);plt.title(pth[2].split("/")[-1].split("_")[:2])
            #plt.subplot(1,6,4);plt.imshow(unshifted[0].cpu().numpy().squeeze(),vmin=0,vmax=1);plt.title(pth[0].split("/")[-1].split("_")[:2])
            #plt.subplot(1,6,5);plt.imshow(unshifted[1].cpu().numpy().squeeze(),vmin=0,vmax=1);plt.title(pth[1].split("/")[-1].split("_")[:2])
            #plt.subplot(1,6,6);plt.imshow(unshifted[2].cpu().numpy().squeeze(),vmin=0,vmax=1);plt.title(pth[2].split("/")[-1].split("_")[:2])
            #plt.show()
            #exit()

            # MIXUP
            #outputs = net((mixed_data-data_mean)/data_std)
            #loss = mixup_criterion(nn.CrossEntropyLoss(), outputs, targets_a, targets_b, lam)

            # Forward pass data through model. Normalize before forward pass
            outputs = net((data-data_mean)/data_std)
            oeoutputs = net((oedata-data_mean)/data_std)
            # VANILLA CROSS-ENTROPY
            loss = criterion(outputs, labels)
            if net.cls_method == 'ilr':
                loss += criterion(oeoutputs, oelabels)
            elif net.cls_method == 'softmax':
                loss += OESoftmaxLoss()(oeoutputs, oelabels)

            else:
                raise ValueError

            # LABEL SMOOTHING LOSS
            #sl = smooth_one_hot(labels,num_classes,smoothing=LBLSMOOTHING_PARAM)
            #loss =  xent_with_soft_targets(outputs, sl)

            # COSINE LOSS
            #one_hots = smooth_one_hot(labels,10,smoothing=0.)
            #loss = (1. - (one_hots * F.normalize(outputs,p=2,dim=1)).sum(1)).mean()

            # Calculate gradient and update parameters
            optimizer.zero_grad()
            net.zero_grad()
            loss.backward()
            #torch.nn.utils.clip_grad_norm(net.parameters(), max_norm=10., norm_type=2)  # COSINE LOSS
            optimizer.step()

            # Measure accuracy and loss for this batch
            try:
                preds = net.predict(outputs)
            except:
                _, preds= outputs.max(1)
            running_total += labels.size(0)
            running_correct += preds.eq(labels).sum().item()
            #running_correct += (lam * preds.eq(targets_a.data).cpu().sum().float() + (1 - lam) * preds.eq(targets_b.data).cpu().sum().float()) # MIXUP
            running_loss_sum += loss.item()

            # Compute measured/synthetic split for the batch
            for tp in pth:
                if "/real/" in tp:
                    running_real_cnt += 1.

        train_acc = running_correct/running_total
        train_loss =  running_loss_sum/len(trainloader)
        percent_real = running_real_cnt/running_total

        print("Epoch [ {} / {} ]; lr: {} TrainAccuracy: {:.5f} TrainLoss: {:.5f} %-Real: {}".format(
            epoch,epochs,optimizer.param_groups[0]['lr'],
            train_acc,train_loss,percent_real))

        # Test
        if testloader is not None:
            test_acc,test_loss = helpers.test_model(
                net,device,testloader,data_mean,data_std, criterion=criterion)
            print("\tEpoch [ {} / {} ]; TestAccuracy: {:.5f} TestLoss: {:.5f}".format(
                epoch,epochs,test_acc,test_loss))
            final_test_acc = test_acc

            if (checkpoint_prefix is not None) and (test_acc>best_acc):
                best_acc = test_acc
                print("Saving...")
                torch.save(
                    {'test_acc':test_acc, 'state_dict': net.state_dict()},
                    checkpoint_prefix+'_best_checkpoint.pth.tar')

    return net, final_test_acc


def gather_l2_metrics(net,
                      ID_trainloader,
                      ID_testloader,
                      OOD_testloader,
                      num_classes,
                      data_mean, data_std,
                      **kwargs):

    net.eval()
    stats = {
        'idtrain': {'correct':0,
                    'loss':0.,
                    'total':0,
                    'norms':num_classes*[torch.empty((0,))],
                    'preds':num_classes*[torch.empty((0,))]
                   },
        'idtest': {'correct':0,
                   'loss':0.,
                   'total':0,
                   'norms':num_classes*[torch.empty((0,))],
                   'preds':num_classes*[torch.empty((0,))]
                  },
        'oodtest': {'correct':0,
                    'loss':0.,
                    'total':0,
                    'norms':num_classes*[torch.empty((0,))],
                    'preds':num_classes*[torch.empty((0,))]
                   },
    }

    criterion = L2Loss()
    all_norms = torch.empty((0,))
    with torch.no_grad():
        for loader, name in zip([ID_trainloader, ID_testloader, OOD_testloader],
                                ['idtrain', 'idtest', 'oodtest']):
            for batch_idx, (inputs, targets, pth) in enumerate(loader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net((inputs-data_mean)/data_std)
                loss = criterion(outputs, targets)

                stats[name]['loss'] += loss.item()
                predicted, norms = net.predict(outputs, return_norms=True)
                norms = norms.cpu()
                stats[name]['total'] += targets.size(0)
                stats[name]['correct'] += predicted.eq(targets).sum().item()
                for i in range(num_classes):
                    stats[name]['norms'][i] = torch.cat(
                        (stats[name]['norms'][i], norms[targets==i]), dim=0)
                    stats[name]['preds'][i] = torch.cat(
                        (stats[name]['preds'][i], predicted[targets==i].cpu()), dim=0)
            stats[name]['num_steps'] = len(loader)
            stats[name]['accuracy'] = (stats[name]['correct'] /
                                       stats[name]['total'])
    if 'f' in kwargs:
        for name in ['idtrain', 'idtest', 'oodtest']:
            kwargs["f"].write("%s Loss=%.4f, Test accuracy=%.4f\n" % (name,
                stats[name]['loss'] / stats[name]['num_steps'],
                stats[name]['accuracy']))
    else:
        for name in ['idtrain', 'idtest', 'oodtest']:
            print("%s Loss=%.4f, Accuracy=%.4f\n" % (name,
                stats[name]['loss'] / stats[name]['num_steps'],
                stats[name]['accuracy']))
    return stats
