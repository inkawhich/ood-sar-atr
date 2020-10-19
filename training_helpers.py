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

device = 'cuda' if torch.cuda.is_available() else 'cpu'


############################################################################################################
#### LABEL SMOOTHING STUFF
# Transform the true "Long" labels to softlabels. The confidence of the gt class is 
#  1-smoothing, and the rest of the probability (i.e. smoothing) is uniformly distributed
#  across the non-gt classes. Note, this is slightly different than standard smoothing
#  notation.  
def smooth_one_hot(true_labels: torch.Tensor, classes: int, smoothing=0.0):
	"""
	if smoothing == 0, it's one-hot method
	if 0 < smoothing < 1, it's smooth method
	"""
	assert 0 <= smoothing < 1
	confidence = 1.0 - smoothing
	label_shape = torch.Size((true_labels.size(0), classes))
	with torch.no_grad():
		true_dist = torch.empty(size=label_shape, device=true_labels.device)
		true_dist.fill_(smoothing / (classes - 1))
		true_dist.scatter_(1, true_labels.data.unsqueeze(1), confidence)
	return true_dist.float()
def xent_with_soft_targets(logit_preds, targets):
	logsmax = F.log_softmax(logit_preds, dim=1)
	batch_loss = targets * logsmax
	batch_loss =  -1*batch_loss.sum(dim=1)
	return batch_loss.mean()
def xent_with_soft_targets_noreduce(logit_preds, targets):
	logsmax = F.log_softmax(logit_preds, dim=1)
	batch_loss = targets * logsmax
	batch_loss =  -1*batch_loss.sum(dim=1)
	return batch_loss


############################################################################################################
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


############################################################################################################
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



############################################################################################################
#### ONE EPOCH OF CLASSIFIER MODEL TRAINING
def train_model(net, optimizer, trainloader, num_classes):

	net.train()

	### DATA AUGMENTATION AND LOSS FXN CONFIGS
	gaussian_std = 0.4
	#gaussian_std = 0.3
	#gaussian_std = 0.05

	#LBLSMOOTHING_PARAM = 0.1 # Only for label smoothing
	#MIXUP_ALPHA = 0.1  # Only for mixup
	#AT_EPS = 2./255.; AT_ALPHA = 0.5/255.; AT_ITERS = 7
	#AT_EPS = 4./255.; AT_ALPHA = 1./255. ; AT_ITERS = 7
	#AT_EPS = 8./255.; AT_ALPHA = 2./255. ; AT_ITERS = 7

	# Normalization Constants for range [-1,+1]
	MEAN = torch.tensor([0.5], dtype=torch.float32).view([1,1,1]).to(device)
	STD  = torch.tensor([0.5], dtype=torch.float32).view([1,1,1]).to(device)

	running_correct = 0.
	running_total = 0.
	running_loss_sum = 0.
	running_real_cnt = 0.

	for batch_idx,(data,labels,pth) in enumerate(trainloader):
		data = data.to(device); labels = labels.to(device)

		# MIXUP
		#mixed_data, targets_a, targets_b, lam = mixup_data(data, labels, MIXUP_ALPHA, use_cuda=True)
		#mixed_data, targets_a, targets_b      = map(Variable, (mixed_data, targets_a, targets_b))

		if(gaussian_std != 0):
			data += torch.randn_like(data)*gaussian_std;
			data = torch.clamp(data, 0, 1);
			#mixed_data += torch.randn_like(mixed_data)*gaussian_std;
			#mixed_data = torch.clamp(mixed_data, 0, 1);

		# ADVERSARIALLY PERTURB DATA
		#data = PGD_Linf_attack(net, device, data.clone().detach(), labels, eps=AT_EPS, alpha=AT_ALPHA, iters=AT_ITERS)

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
		#outputs = net((mixed_data-MEAN)/STD)
		#loss = mixup_criterion(nn.CrossEntropyLoss(), outputs, targets_a, targets_b, lam)	

		# Forward pass data through model. Normalize before forward pass
		outputs = net((data-MEAN)/STD)

		# VANILLA CROSS-ENTROPY
		loss = F.cross_entropy(outputs, labels);
			
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
		#torch.nn.utils.clip_grad_norm(net.parameters(), max_norm=10., norm_type=2)	 # COSINE LOSS
		optimizer.step()

		# Measure accuracy and loss for this batch
		_,preds = outputs.max(1)
		running_total += labels.size(0)
		running_correct += preds.eq(labels).sum().item()
		#running_correct += (lam * preds.eq(targets_a.data).cpu().sum().float() + (1 - lam) * preds.eq(targets_b.data).cpu().sum().float()) # MIXUP
		running_loss_sum += loss.item()		

		# Compute measured/synthetic split for the batch
		for tp in pth:
			if "/real/" in tp:
				running_real_cnt += 1.

	# Return train_acc, train_loss, %measured_traindata
	return running_correct/running_total, running_loss_sum/len(trainloader), running_real_cnt/running_total




