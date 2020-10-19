# NAI

# Adapted from "https://github.com/pokaxpoka/deep_Mahalanobis_detector/blob/master/calculate_log.py"
#   to input numpy arrays directly instead of file paths

## Measure the detection performance - Kibok Lee
from __future__ import print_function
#import torch
#from torch.autograd import Variable
#import torch.nn as nn
#import torch.nn.functional as F
import numpy as np
#import torch.optim as optim
#import torchvision
#import torchvision.transforms as transforms
import numpy as np
import time
from scipy import misc

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

#def get_curve(dir_name, stypes = ['Baseline', 'Gaussian_LDA']):
def get_curve(known,novel):
	tp, fp = dict(), dict()
	tnr_at_tpr95 = dict()
	for stype in ['TMP']:
		#known = np.loadtxt('{}/confidence_{}_In.txt'.format(dir_name, stype), delimiter='\n')
		#novel = np.loadtxt('{}/confidence_{}_Out.txt'.format(dir_name, stype), delimiter='\n')
		## NAI
		#print(known.shape)
		#print(novel.shape)
		#exit("HA")
		known.sort()
		novel.sort()
		end = np.max([np.max(known), np.max(novel)])
		start = np.min([np.min(known),np.min(novel)])
		num_k = known.shape[0]
		num_n = novel.shape[0]
		tp[stype] = -np.ones([num_k+num_n+1], dtype=int)
		fp[stype] = -np.ones([num_k+num_n+1], dtype=int)
		tp[stype][0], fp[stype][0] = num_k, num_n
		k, n = 0, 0
		for l in range(num_k+num_n):
			if k == num_k:
				tp[stype][l+1:] = tp[stype][l]
				fp[stype][l+1:] = np.arange(fp[stype][l]-1, -1, -1)
				break
			elif n == num_n:
				tp[stype][l+1:] = np.arange(tp[stype][l]-1, -1, -1)
				fp[stype][l+1:] = fp[stype][l]
				break
			else:
				if novel[n] < known[k]:
					n += 1
					tp[stype][l+1] = tp[stype][l]
					fp[stype][l+1] = fp[stype][l] - 1
				else:
					k += 1
					tp[stype][l+1] = tp[stype][l] - 1
					fp[stype][l+1] = fp[stype][l]
		tpr95_pos = np.abs(tp[stype] / num_k - .95).argmin()
		tnr_at_tpr95[stype] = 1. - fp[stype][tpr95_pos] / num_n
		return tp, fp, tnr_at_tpr95

#def metric(dir_name, stypes = ['Bas', 'Gau'], verbose=False):
def metric(in_scores, out_scores):
	#tp, fp, tnr_at_tpr95 = get_curve(dir_name, stypes)
	tp, fp, tnr_at_tpr95 = get_curve(in_scores,out_scores)
	results = dict()
	mtypes = ['TNR', 'AUROC', 'DTACC', 'AUIN', 'AUOUT']
	for stype in ['TMP']:
		results[stype] = dict()
		# TNR
		mtype = 'TNR'
		results[stype][mtype] = tnr_at_tpr95[stype]
		# AUROC
		mtype = 'AUROC'
		tpr = np.concatenate([[1.], tp[stype]/tp[stype][0], [0.]])
		fpr = np.concatenate([[1.], fp[stype]/fp[stype][0], [0.]])
		results[stype][mtype] = -np.trapz(1.-fpr, tpr)
		# DTACC
		mtype = 'DTACC'
		results[stype][mtype] = .5 * (tp[stype]/tp[stype][0] + 1.-fp[stype]/fp[stype][0]).max()
		# AUIN
		mtype = 'AUIN'
		denom = tp[stype]+fp[stype]
		denom[denom == 0.] = -1.
		pin_ind = np.concatenate([[True], denom > 0., [True]])
		pin = np.concatenate([[.5], tp[stype]/denom, [0.]])
		results[stype][mtype] = -np.trapz(pin[pin_ind], tpr[pin_ind])
		# AUOUT
		mtype = 'AUOUT'
		denom = tp[stype][0]-tp[stype]+fp[stype][0]-fp[stype]
		denom[denom == 0.] = -1.
		pout_ind = np.concatenate([[True], denom > 0., [True]])
		pout = np.concatenate([[0.], (fp[stype][0]-fp[stype])/denom, [.5]])
		results[stype][mtype] = np.trapz(pout[pout_ind], 1.-fpr[pout_ind])
	return results





