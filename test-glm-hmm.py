#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 11:19:36 2022

@author: jthompsz
"""

import sys


from collections import OrderedDict
import warnings
from copy import deepcopy
import glob
import os
import re
import json
import csv
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.patches as patches

from tqdm import tqdm

import numpy as np
import pandas as pd

import nibabel as nib
import nilearn as ni
from nilearn.input_data import NiftiLabelsMasker
from nilearn.input_data import NiftiMasker
from nilearn import image
from nilearn import plotting

from hmmlearn import hmm
from sklearn.decomposition import PCA
from scipy.stats import zscore

import autograd.numpy as np
import autograd.numpy.random as npr
npr.seed(0)

import ssm
from ssm.util import find_permutation
from ssm.plots import gradient_cmap, white_to_color_cmap
color_names = [
    "windows blue",
    "red",
    "amber",
    "faded green",
    "dusty purple",
    "orange"
    ]

colors = sns.xkcd_palette(color_names)
cmap = gradient_cmap(colors)

base_dir = '/mnt/EE9A47C59A478953/data/FNL'
out_dir = '/mnt/EE9A47C59A478953/data/FNL/output'
func_data = '/ds003521/derivatives/denoised/smoothed'


os.chdir(f'{base_dir}')

# load in target ROI
target_roi = image.load_img(f'{base_dir}/ROIs/vmpfc-chang.nii.gz')
plotting.plot_roi(target_roi)

# load source ROIs
atlas = image.load_img(f'{base_dir}/ROIs/source_rois.nii.gz')

# load in some data
sdata = image.load_img(f'{base_dir}/{func_data}/sub-sid000216_task-movie_run-1_space-MNI152NLin2009cAsym_desc-preproc_trim_smooth6_denoised_bold.nii.gz')

# extract target data from roi
target_masker = NiftiMasker(mask_img=target_roi, standardize=False)
target_time_series = target_masker.fit_transform(sdata)

target_data = pd.DataFrame(target_time_series)


# extract source data from rois
masker = NiftiLabelsMasker(labels_img=atlas, standardize=False)
source_time_series = masker.fit_transform(sdata)

plt.figure(figsize=(10,3))
plt.plot(source_time_series)
plt.show()

#####                HMM as in Chang et al

# Reduce Data Dimensionality
target_var = 0.9
pca = PCA(n_components=target_var)
reduced = pca.fit_transform(zscore(target_data))

# Run HMM
k=4
m1 = hmm.GaussianHMM(n_components=4, covariance_type="diag", algorithm='map', n_iter=500)
m1.fit(reduced)

# Get HMM Weights
w = pd.DataFrame(pca.inverse_transform(m1.means_))
w.round(decimals=4).to_csv(os.path.join(out_dir, f'HMMWeights.vmpfc.k4.csv'))
    
# Write out HMM Covariance
for i,x in enumerate(m1.covars_):
    pd.DataFrame(x).to_csv(os.path.join(out_dir, f'HMMCovariates.vmpfc.k4.csv'))
    
# Write out Transition matrix
transmat = pd.DataFrame(m1.transmat_)
transmat.to_csv(os.path.join(out_dir,f'HMMTransitionMatrix.vmpfc.k4.csv'))
    
# Write out predicted states
pred = {}
p = m1.decode(reduced, algorithm='viterbi')
pred['Viterbi'] = p[1]
pred['MAP'] = m1.decode(reduced, algorithm='map')[1]
pred_prob = m1.predict_proba(reduced)
for i in range(k):
        pred[f'Probability_{i}'] = pred_prob[:,i]
        proj = np.dot(reduced, m1.means_.T)
        for i in range(k):
            pred[f'Projected_{i}'] = proj[:,i]
            pred = pd.DataFrame(pred)
            pred['ModelFit'] = m1.score(reduced)
            #pred['Subject'] = sub
            #pred['Study'] = study
            pred['PCA_Components'] = reduced.shape[1]
            pred.to_csv(os.path.join(out_dir,f'HMMPredictedStates.vmpfc.k{k}.csv'))

#plt.figure(figsize=(10,3))
#plt.plot(pred[['Probability_0','Probability_1','Probability_2','Probability_3' ]])
#plt.show()

# read in HMM data

p = pd.read_csv(os.path.join(out_dir, f'HMMPredictedStates.vmpfc.k{k}.csv'), index_col=0)

#####                HMM as in SSM

N_iters = 500

## testing the constrained transitions class
hmm = ssm.HMM(k, 70, observations="diagonal_gaussian")

hmm_lls = hmm.fit(reduced, method="em", num_iters=N_iters, init_method="kmeans")

hmm_z = hmm.most_likely_states(reduced)

ssm = pd.DataFrame()
ssm['zPredicted States'] = hmm_z

ssm.to_csv(os.path.join(out_dir,f'SSM-HMMPredictedStates.vmpfc.k{k}.csv'))

# align states
#viterbi 2 = 1
#viterbi 0 = 3
#viterbi 3 = 0
#viterbi 1 = 2




fig = plt.figure(figsize=(8, 4))
plt.subplot(211)
plt.imshow(hmm_z[None,:], aspect="auto", cmap=cmap, vmin=0, vmax=len(colors)-1)
plt.ylabel("SSM - $z_{\\mathrm{inferred}}$")
plt.yticks([])
plt.xlabel("time")

plt.subplot(212)
plt.imshow([p['Viterbi']], aspect="auto", cmap=cmap, vmin=0, vmax=len(colors)-1)
plt.ylabel("HMM - Viterbi")
plt.yticks([])
plt.xlabel("time")

fig.savefig('most_likely_states.png', dpi=300)

