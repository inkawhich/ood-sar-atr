# NAI

# Helper file for very general functions like saving checkpoints and running a test pass

import numpy as np
import random
import os
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
#from torch.autograd.gradcheck import zero_gradients


# Save model checkpoints
def save_checkpoint(state, is_best, checkpoint_prefix):
	filepath = checkpoint_prefix+"_checkpoint.pth.tar"
	torch.save(state, filepath)
	if is_best:
		print("New best file! Saving.")
		shutil.copyfile(filepath, checkpoint_prefix+'_checkpoint_best.pth.tar')

def create_learning_rate_table(initial_lr, decay_schedule, gamma, epochs):
	lr_table = np.zeros((epochs))
	prev_lr = initial_lr
	for i in range(epochs):
		if i in decay_schedule:
			prev_lr *= gamma
		lr_table[i] = prev_lr
	return lr_table

def adjust_learning_rate(optimizer, curr_epoch, learning_rate_table):
	for param_group in optimizer.param_groups:
		param_group['lr'] = learning_rate_table[curr_epoch]

# Getting the class labels in cls#:clsName form
def get_class_mapping_from_dataset_list(dset_list):
	class_map = {}
	for i in dset_list:
		class_name = i[0].split("/")[-2]
		class_number = i[1]
		if class_number in class_map.keys():
			assert(class_map[class_number] == class_name)
		else:
			class_map[class_number] = class_name
	return class_map

# Test the input model on data from the loader. Used in training script
def test_model(net,device,loader,mean,std):

	net.eval()
	
	# Stat keepers
	running_clean_correct = 0.
	running_clean_loss = 0.
	running_total = 0.

	with torch.no_grad():

		for batch_idx,(data,labels,_) in enumerate(loader):
			data = data.to(device); labels = labels.to(device)
			clean_outputs = net((data-mean)/std)
			clean_loss = F.cross_entropy(clean_outputs, labels)
			_,clean_preds = clean_outputs.max(1)
			running_clean_correct += clean_preds.eq(labels).sum().item()
			running_clean_loss += clean_loss.item()
			running_total += labels.size(0)
	
		clean_acc = running_clean_correct/running_total
		clean_loss = running_clean_loss/len(loader)
	
	net.train()
	return clean_acc,clean_loss






