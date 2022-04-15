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
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import cm

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
from scipy import stats
from scipy.stats import zscore, pearsonr, spearmanr
from scipy.spatial.distance import hamming
import math

import pickle

from fnl_tools.stats import hmm_bic
from nltools.mask import expand_mask, collapse_mask
from nltools.stats import (fisher_r_to_z,
                           correlation_permutation)

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
func_data = '/data/train'
test_data = '/data/test'

os.chdir(f'{base_dir}')

# load in target ROI
target_roi = image.load_img(f'{base_dir}/ROIs/vmpfc-chang.nii.gz')
plotting.plot_roi(target_roi)

# load source ROIs
atlas = image.load_img(f'{base_dir}/ROIs/source_target_rois.nii.gz')
plotting.plot_roi(atlas)


#####               Group-based Analyses - Extract data


# Make a file list of data files
file_list = glob.glob(f'{base_dir}{func_data}/sub-*nii.gz')

# make ROIs
target_masker = NiftiMasker(mask_img=target_roi, standardize=False)
masker = NiftiLabelsMasker(labels_img=atlas, standardize=False)

target_data = {}
source_data = {}
target_group_data = pd.DataFrame()
source_group_data = pd.DataFrame()
for f in tqdm(file_list):  
    
    sub = f.partition("sub-")[2].rpartition('_task')[0]
    # load in data
    sdata = image.load_img(f)

    # extract target data from roi
    #target_time_series = target_masker.fit_transform(sdata)
    #target_data = pd.DataFrame(zscore(target_time_series))
    #target_data['Subject'] = sub
    #target_group_data = target_group_data.append(target_data)
    
    # extract source data from rois
    source_time_series = masker.fit_transform(sdata)
    source_data = pd.DataFrame(zscore(source_time_series))
    source_data['Subject'] = sub
    source_group_data = source_group_data.append(source_data)

source_group_data.columns = ['Amygdala', 'NAcc', 'Hippocampus', 'DLPFC', 'DMPFC', 'pInsula', 'TPJ', 'vmpfc','Subject']      
source_group_data.to_csv(os.path.join(out_dir, f'train_sources_zscoredata.csv'))
target_group_data.to_csv(os.path.join(out_dir, f'vmpfc_rawdata.csv'))

###              Load in and extract Test data 04.06.22

# Make a file list of data files
file_list = glob.glob(f'{base_dir}{test_data}/sub-*nii.gz')

# make ROIs
target_masker = NiftiMasker(mask_img=target_roi, standardize=False)
masker = NiftiLabelsMasker(labels_img=atlas, standardize=False)

target_data = {}
source_data = {}
target_group_data = pd.DataFrame()
source_group_data = pd.DataFrame()
for f in tqdm(file_list):  
    
    sub = f.partition("sub-")[2].rpartition('_task')[0]
    # load in data
    sdata = image.load_img(f)

    # extract target data from roi
    target_time_series = target_masker.fit_transform(sdata)
    target_data = pd.DataFrame(zscore(target_time_series))
    target_data['Subject'] = sub
    target_group_data = target_group_data.append(target_data)
    
    # extract source data from rois
    source_time_series = masker.fit_transform(sdata)
    source_data = pd.DataFrame(zscore(source_time_series))
    source_data['Subject'] = sub
    source_group_data = source_group_data.append(source_data)

source_group_data.columns = ['Amygdala', 'NAcc', 'Hippocampus', 'DLPFC', 'DMPFC', 'pInsula', 'TPJ', 'vmpfc','Subject']      
source_group_data.to_csv(os.path.join(out_dir, f'test_sources_zscoredata.csv'))

target_group_data.to_csv(os.path.join(out_dir, f'test_vmpfc_rawdata.csv'))

# Train PCA on training data, apply to test data and save
#training = pd.read_csv(os.path.join(out_dir, f'vmpfc_rawdata.csv'), index_col=0)
#test = pd.read_csv(os.path.join(out_dir, f'test_vmpfc_rawdata.csv'), index_col=0)

# Reduce Data Dimensionality, train on training and apply to test
#target_var = 0.9
#pca = PCA(n_components=target_var)

#training_fit = pca.fit(training.drop(columns='Subject'))
#training_comps = pca.transform(training.drop(columns='Subject'))
#test_comps = pca.transform(test.drop(columns='Subject'))
        
#X = pd.DataFrame(training_comps)
#X['Subject'] = training['Subject'].values
#X.to_csv(os.path.join(out_dir, f'train_vmpfc_PCdata.csv'))

#X = pd.DataFrame(test_comps)
#X['Subject'] = test['Subject'].values
#X.to_csv(os.path.join(out_dir, f'test_vmpfc_PCdata.csv'))

# Train PCA on combined training and test data - tbh I think the above method
# would be better, especially with multifold CV. But it yeilds weird results,
# I think bc the PCs from train don't quite match the component structure of
# test, and I ran out of time trying to understand it fully.

training = pd.read_csv(os.path.join(out_dir, f'vmpfc_rawdata.csv'), index_col=0)
test = pd.read_csv(os.path.join(out_dir, f'test_vmpfc_rawdata.csv'), index_col=0)
all_data = pd.DataFrame()
all_data = all_data.append(training)
all_data = all_data.append(test)

# Reduce Data Dimensionality
target_var = 0.9
pca = PCA(n_components=target_var)

pca_fit = pca.fit_transform(all_data.drop(columns='Subject'))

        
X = pd.DataFrame(pca_fit[0:training.shape[0],:])
X['Subject'] = training['Subject'].values
X.to_csv(os.path.join(out_dir, f'train_vmpfc_PCAdata.csv'))

X = pd.DataFrame(pca_fit[training.shape[0]:,:])
X['Subject'] = test['Subject'].values
X.to_csv(os.path.join(out_dir, f'test_vmpfc_PCAdata.csv'))

#####               Group-based Input-driven HMM as in SSM

reduced = pd.read_csv(os.path.join(out_dir, f'train_vmpfc_PCAdata.csv'), index_col=0)
source = pd.read_csv(os.path.join(out_dir, f'train_sources_zscoredata.csv'),index_col=0)
#source.columns = ['Index', 'Amygdala', 'NAcc', 'Hippocampus', 'DLPFC', 'DMPFC', 'pInsula', 'TPJ', 'Subject']

# Let's check the correlations between inputs to the HMM
pearsoncorr = source.drop(columns=['Subject']).corr(method='pearson')
with sns.plotting_context(context='paper', font_scale=1.5):    
    fig = plt.figure(figsize=(10, 8))
    sns.heatmap(pearsoncorr, 
            xticklabels=pearsoncorr.columns,
            yticklabels=pearsoncorr.columns,
            cmap='RdBu_r',
            annot=True,
            linewidth=0.5)
    plt.tight_layout()
    plt.show()
    fig.savefig('Inputs-Correlation4.15.22.png', dpi=150)

####             Vanilla HMM for group

# Chang et al (2021) originally found that 4 states was best model, let's check
N_iters = 500
mle_hmmk = {}
bic_hmmk = {}
for k in tqdm(range(2, 16)):
    hmmk = []
    hmmk = ssm.HMM(k, reduced.drop(columns='Subject').shape[1], observations="diagonal_gaussian")
    hmmk_fit = hmmk.fit(reduced.drop(columns='Subject').values, method="em", num_iters=N_iters)
    mle_hmmk[k] = hmmk.log_likelihood(reduced.drop(columns='Subject').values)
    bic_hmmk[k] = hmm_bic(LL=mle_hmmk[k], n_states=k, n_features=reduced.drop(columns='Subject').shape[1])

model_fit = pd.DataFrame(list(bic_hmmk.items()))
model_fit = model_fit.rename(columns={0: "k", 1: "BIC"})
model_fit.to_csv(os.path.join(out_dir, f'Vanilla-HMM-ModelFitk2-15.csv'))

with sns.plotting_context(context='paper', font_scale=2.5):
    fig = plt.figure(figsize=(6,5))
    sns.lineplot(data=model_fit, x='k', y='BIC', linewidth=3)
    plt.ylabel('Model Fit (BIC)', fontsize=18)
    plt.xlabel('k', fontsize=18)
    plt.axhline(bic_hmmk[3], color='red', linestyle='--') # red dashed line at k=4
    plt.tight_layout()
    plt.show()
    fig.savefig('Model Fit-BIC-VanillaHMM4.15.22.png', dpi=150)

# With this split half training data, k=3 looks better than k = 4, so will use for the rest of the analyses
    
# Vanilla HMM for Training Group
N_iters = 500
k = 3
hmm = ssm.HMM(k, reduced.drop(columns='Subject').shape[1], observations="diagonal_gaussian")

# Fit
hmm_fit = hmm.fit(reduced.drop(columns='Subject').values, method="em", num_iters=N_iters, init_method="kmeans")

# Most likely (Viterbi) states
hmm_z = hmm.most_likely_states(reduced.drop(columns='Subject').values)

# Log Likelihood - when comparing with the input-driven hmm, I'm not 100% on calculation of BIC. Let's use -LL instead, even though it is not ideal.
mle_hmm = hmm.log_likelihood(reduced.drop(columns='Subject').values)

# Save copy of the hmm
filehandler = open(f'{out_dir}/Vanilla-HMM-k3.obj', 'wb')
pickle.dump(hmm, filehandler)

####             Input-HMM for Training Group - AMYG and DMPFC

N_iters = 500

# Amygdala
k = 3
inputhmm1 = ssm.HMM(k, reduced.drop(columns='Subject').shape[1], source[['Amygdala']].shape[1], observations="diagonal_gaussian", transitions="inputdriven")
hmm_amyg = inputhmm1.fit(reduced.drop(columns='Subject').values, inputs=source[['Amygdala']].values, method="em", num_iters=N_iters)
mle_amyg = inputhmm1.log_likelihood(reduced.drop(columns='Subject').values, source[['Amygdala']].values)

# Save copy of the hmm
filehandler = open(f'{out_dir}/Input-HMM-Amygdala-k3.obj', 'wb')
pickle.dump(inputhmm1, filehandler)

#filein = open(f'{out_dir}/Input-HMM-Amygdala-k3.obj', 'rb')
#hmmglm1 = pickle.load(filein)

# DMPFC
k = 3
inputhmm2 = ssm.HMM(k, reduced.drop(columns='Subject').shape[1], source[['DMPFC']].shape[1], observations="diagonal_gaussian", transitions="inputdriven")
hmm_dmpfc = inputhmm2.fit(reduced.drop(columns='Subject').values, inputs=source[['DMPFC']].values, method="em", num_iters=N_iters)
mle_dmpfc = inputhmm2.log_likelihood(reduced.drop(columns='Subject').values, source[['DMPFC']].values)

# Save copy of the hmm
filehandler = open(f'{out_dir}/Input-HMM-DMPFC-k3.obj', 'wb')
pickle.dump(inputhmm2, filehandler)
    
# Plot -LL for HMM and Input-HMM models
with sns.plotting_context(context='paper', font_scale=2):    
    fig = plt.figure(figsize=(8, 6))
    loglikelihood_vals = [mle_hmm, mle_amyg, mle_dmpfc]
    for z, occ in enumerate(loglikelihood_vals):
        plt.bar(z, occ, width = 0.8, color = colors[z])
    plt.ylim([-6023650, -6023550])
    plt.xticks([0,1,2], ['hmm', 'amyg', 'dmpfc'], fontsize = 15)
    plt.xlabel('Models (training data)', fontsize = 20)
    plt.ylabel('loglikelihood', fontsize=20)
    plt.tight_layout()
    plt.show()
    fig.savefig('ModelFit-TrainData-MLE-k3-4.15.22.png', dpi=150)

# Model fit for in-sample data looks best for the dmpfc input-HMM

####     Cross-Validation using train and test data
reduced_test = pd.read_csv(os.path.join(out_dir, f'test_vmpfc_PCAdata.csv'), index_col=0)
source_test = pd.read_csv(os.path.join(out_dir, f'test_sources_zscoredata.csv'), index_col=0)

# Cross-Validated Log Likelihood for the different models - use the model trained on training
# data, test on test data
mle_hmm_cv = hmm.log_likelihood(reduced_test.drop(columns='Subject').values)

mle_amyg_cv = inputhmm1.log_likelihood(reduced_test.drop(columns='Subject').values, source_test[['Amygdala']].values)

mle_dmpfc_cv = inputhmm2.log_likelihood(reduced_test.drop(columns='Subject').values, source_test[['DMPFC']].values)

# Plot -LL for HMM and Input-HMM models
with sns.plotting_context(context='paper', font_scale=2):    
    fig = plt.figure(figsize=(8, 6))
    loglikelihood_vals = [mle_hmm_cv, mle_amyg_cv, mle_dmpfc_cv]
    for z, occ in enumerate(loglikelihood_vals):
        plt.bar(z, occ, width = 0.8, color = colors[z])
    plt.ylim([-6069350, -6069200])
    plt.xticks([0,1,2], ['hmm', 'amyg', 'dmpfc'], fontsize = 15)
    plt.xlabel('Models (test data)', fontsize = 20)
    plt.ylabel('loglikelihood', fontsize=20)
    plt.tight_layout()
    plt.show()
    fig.savefig('ModelFit-TestData-MLE-k3-4.15.22.png', dpi=150)

# Model fit for out-of-sample data looks best for the dmpfc input-HMM!

####################################################
filein = open(f'{out_dir}/Vanilla-HMM-k3.obj', 'rb')
hmm1 = pickle.load(filein)
hmm_z = hmm1.most_likely_states(reduced.drop(columns='Subject').values)

filein = open(f'{out_dir}/Input-HMM-Amygdala-k3.obj', 'rb')
hmm2 = pickle.load(filein)
hmm2_z = hmm2.most_likely_states(reduced.drop(columns='Subject').values,input=source[['Amygdala']].values)
hmm2.permute(find_permutation(hmm_z, hmm2_z))
hmm2_states = hmm2.most_likely_states(reduced.drop(columns='Subject').values,input=source[['Amygdala']].values)
    
filein = open(f'{out_dir}/Input-HMM-DMPFC-k3.obj', 'rb')
hmm3= pickle.load(filein)
hmm3_z = hmm3.most_likely_states(reduced.drop(columns='Subject').values,input=source[['DMPFC']].values)
hmm3.permute(find_permutation(hmm_z, hmm3_z))
hmm3_states = hmm3.most_likely_states(reduced.drop(columns='Subject').values,input=source[['DMPFC']].values)
 
## Individual Input HMMs on test data
N_iters = 500
k = 3
Amygdala = source_test[['Amygdala', 'Subject']]
DMPFC = source_test[['DMPFC', 'Subject']]

for subj in tqdm(reduced_test['Subject'].unique()):
    hmm1ss = ssm.HMM(k, reduced_test[reduced_test['Subject']==subj].drop(columns='Subject').shape[1], observations="diagonal_gaussian")
    tmp = hmm1ss.fit(reduced_test[reduced_test['Subject']==subj].drop(columns='Subject').values, method="em", num_iters=N_iters)
    filehandler = open(f'{out_dir}/{subj}-Vanilla-HMM-k3.obj', 'wb')
    pickle.dump(hmm1ss, filehandler)
    
    inputhmm1ss = ssm.HMM(k, reduced_test[reduced_test['Subject']==subj].drop(columns='Subject').shape[1], Amygdala[Amygdala['Subject']==subj].drop(columns='Subject').shape[1], observations="diagonal_gaussian", transitions="inputdriven")
    tmp = inputhmm1ss.fit(reduced_test[reduced_test['Subject']==subj].drop(columns='Subject').values, inputs=Amygdala[Amygdala['Subject']==subj].drop(columns='Subject').values, method="em", num_iters=N_iters)
    filehandler = open(f'{out_dir}/{subj}-Input-HMM-Amygdala-k3.obj', 'wb')
    pickle.dump(inputhmm1ss, filehandler)
    
    inputhmm2ss = ssm.HMM(k, reduced_test[reduced_test['Subject']==subj].drop(columns='Subject').shape[1], DMPFC[DMPFC['Subject']==subj].drop(columns='Subject').shape[1], observations="diagonal_gaussian", transitions="inputdriven")
    tmp = inputhmm2ss.fit(reduced_test[reduced_test['Subject']==subj].drop(columns='Subject').values, inputs=DMPFC[DMPFC['Subject']==subj].drop(columns='Subject').values, method="em", num_iters=N_iters)
    filehandler = open(f'{out_dir}/{subj}-Input-HMM-DMPFC-k3.obj', 'wb')
    pickle.dump(inputhmm2ss, filehandler)


# Load in subject model, align test Vanilla HMM with subject from Vanilla HMM
# from train data, get most
# likely states, probability of states, transition matrix, transition weights.
# Use Vanilla HMM from each subject to align Input HMMs
test_subs = reduced_test['Subject'].unique()
train_subs = reduced['Subject'].unique()



states_Out_m1 = pd.DataFrame()
transmat_Out_m1 = pd.DataFrame()
states_Out_m2 = pd.DataFrame()
transmat_Out_m2 = pd.DataFrame()
transweight_Out_m2 = pd.DataFrame()
states_Out_m3 = pd.DataFrame()
transmat_Out_m3 = pd.DataFrame()
transweight_Out_m3 = pd.DataFrame()

for subj in tqdm(test_subs):
    
    # Vanilla HMM
    States = pd.DataFrame()
    tmp_z = []
    filein = open(f'{out_dir}/{subj}-Vanilla-HMM-k3.obj', 'rb')
    hmm1ss = pickle.load(filein)
    # Align with subject from Group HMM from test data
    tmp_z = hmm1ss.most_likely_states(reduced_test[reduced_test['Subject']==subj].drop(columns='Subject').values)
    hmm1ss.permute(find_permutation(hmm_z[0:1364], tmp_z))
    filehandler = open(f'{out_dir}/{subj}-ALIGNED-Vanilla-HMM-k3.obj', 'wb')
    pickle.dump(hmm1ss, filehandler)
    # Most likely states (using Viterbi)
    States['Viterbi'] = hmm1ss.most_likely_states(reduced_test[reduced_test['Subject']==subj].drop(columns='Subject').values)
    # Probability of each state
    posterior_probs = hmm1ss.expected_states(reduced_test[reduced_test['Subject']==subj].drop(columns='Subject').values)
    for i in range(k):
        States[f'Probability_{i}'] = posterior_probs[0][:,i]

    States['Subject'] = subj
    states_Out_m1 = states_Out_m1.append(States)

    # Transition matrix
    trans_mat = np.exp(hmm1ss.transitions.log_Ps)
    T = pd.DataFrame(trans_mat)
    T['Subject'] = subj
    transmat_Out_m1 = transmat_Out_m1.append(T)
       

    # Input HMM Amygdala
    States = pd.DataFrame()
    tmp_z = []
    filein = open(f'{out_dir}/{subj}-Input-HMM-Amygdala-k3.obj', 'rb')
    inputhmm1ss = pickle.load(filein)
    # Align with self from Vanilla HMM from test data
    self_z = hmm1ss.most_likely_states(reduced_test[reduced_test['Subject']==subj].drop(columns='Subject').values)
    tmp_z = inputhmm1ss.most_likely_states(reduced_test[reduced_test['Subject']==subj].drop(columns='Subject').values, Amygdala[Amygdala['Subject']==subj].drop(columns='Subject').values)
    inputhmm1ss.permute(find_permutation(self_z, tmp_z))
    filehandler = open(f'{out_dir}/{subj}-ALIGNED-Input-HMM-Amygdala-k3.obj', 'wb')
    pickle.dump(inputhmm1ss, filehandler)
    # Most likely states (using Viterbi)
    States['Viterbi'] = inputhmm1ss.most_likely_states(reduced_test[reduced_test['Subject']==subj].drop(columns='Subject').values, Amygdala[Amygdala['Subject']==subj].drop(columns='Subject').values)
    # Probability of each state
    posterior_probs = inputhmm1ss.expected_states(reduced_test[reduced_test['Subject']==subj].drop(columns='Subject').values, Amygdala[Amygdala['Subject']==subj].drop(columns='Subject').values)
    for i in range(k):
        States[f'Probability_{i}'] = posterior_probs[0][:,i]

    States['Subject'] = subj
    states_Out_m2 = states_Out_m2.append(States)

    # Transition matrix
    trans_mat = np.exp(inputhmm1ss.transitions.log_Ps)
    T = pd.DataFrame(trans_mat)
    T['Subject'] = subj
    transmat_Out_m2 = transmat_Out_m2.append(T)
    
    # Transition Input Weight
    Ws = pd.DataFrame(inputhmm1ss.transitions.Ws)
    Ws['Subject'] = subj
    transweight_Out_m2 = transweight_Out_m2.append(Ws)
    


    # Input HMM DMPFC

    States = pd.DataFrame()
    tmp_z = []
    filein = open(f'{out_dir}/{subj}-Input-HMM-DMPFC-k3.obj', 'rb')
    inputhmm2ss = pickle.load(filein)
    # Align with self from Vanilla HMM from test data
    self_z = hmm1ss.most_likely_states(reduced_test[reduced_test['Subject']==subj].drop(columns='Subject').values)
    tmp_z = inputhmm2ss.most_likely_states(reduced_test[reduced_test['Subject']==subj].drop(columns='Subject').values, DMPFC[DMPFC['Subject']==subj].drop(columns='Subject').values)
    inputhmm2ss.permute(find_permutation(self_z, tmp_z))
    filehandler = open(f'{out_dir}/{subj}-ALIGNED-Input-HMM-DMPFC-k3.obj', 'wb')
    pickle.dump(inputhmm2ss, filehandler)
    # Most likely states (using Viterbi)
    States['Viterbi'] = inputhmm2ss.most_likely_states(reduced_test[reduced_test['Subject']==subj].drop(columns='Subject').values, DMPFC[DMPFC['Subject']==subj].drop(columns='Subject').values)
    # Probability of each state
    posterior_probs = inputhmm2ss.expected_states(reduced_test[reduced_test['Subject']==subj].drop(columns='Subject').values, DMPFC[DMPFC['Subject']==subj].drop(columns='Subject').values)
    for i in range(k):
        States[f'Probability_{i}'] = posterior_probs[0][:,i]

    States['Subject'] = subj
    states_Out_m3 = states_Out_m3.append(States)

    # Transition matrix
    trans_mat = np.exp(inputhmm2ss.transitions.log_Ps)
    T = pd.DataFrame(trans_mat)
    T['Subject'] = subj
    transmat_Out_m3 = transmat_Out_m3.append(T)
    
    # Transition Input Weight
    Ws = pd.DataFrame(inputhmm2ss.transitions.Ws)
    Ws['Subject'] = subj
    transweight_Out_m3 = transweight_Out_m3.append(Ws)
    
states_Out_m1.to_csv(os.path.join(out_dir, f'Test-ALIGNED-Vanilla-HMM-PredictedStates-k3.csv'))
transmat_Out_m1.to_csv(os.path.join(out_dir, f'Test-ALIGNED-Vanilla-HMM-TransitionMatrix-k3.csv')) 

states_Out_m2.to_csv(os.path.join(out_dir, f'Test-ALIGNED-Input-HMM-Amygdala-PredictedStates-k3.csv'))
transmat_Out_m2.to_csv(os.path.join(out_dir, f'Test-ALIGNED-Input-HMM-Amygdala-TransitionMatrix-k3.csv'))  
transweight_Out_m2.to_csv(os.path.join(out_dir, f'Test-ALIGNED-Input-HMM-Amygdala-TransitionWeights-k3.csv'))
  
states_Out_m3.to_csv(os.path.join(out_dir, f'Test-ALIGNED-Input-HMM-DMPFC-PredictedStates-k3.csv'))
transmat_Out_m3.to_csv(os.path.join(out_dir, f'Test-ALIGNED-Input-HMM-DMPFC-TransitionMatrix-k3.csv'))  
transweight_Out_m3.to_csv(os.path.join(out_dir, f'Test-ALIGNED-Input-HMM-DMPFC-TransitionWeights-k3.csv'))

### Average the Transition Matrices, Weights and plot

# load in matrices
hmm1ss_transmat = pd.read_csv(os.path.join(out_dir, f'Test-ALIGNED-Vanilla-HMM-TransitionMatrix-k3.csv'), index_col=0)
inputhmm1ss_transmat = pd.read_csv(os.path.join(out_dir, f'Test-ALIGNED-Input-HMM-Amygdala-TransitionMatrix-k3.csv'), index_col=0)
inputhmm2ss_transmat = pd.read_csv(os.path.join(out_dir, f'Test-ALIGNED-Input-HMM-DMPFC-TransitionMatrix-k3.csv'), index_col=0)

# Mean
hmm1ss_transmat_mean = hmm1ss_transmat.groupby(hmm1ss_transmat.index).mean()
inputhmm1ss_transmat_mean = inputhmm1ss_transmat.groupby(inputhmm1ss_transmat.index).mean()
inputhmm2ss_transmat_mean = inputhmm2ss_transmat.groupby(inputhmm2ss_transmat.index).mean()

# Tstat and pval  
inputhmm1ss_hmm1ss = inputhmm1ss_transmat.drop(columns='Subject') - hmm1ss_transmat.drop(columns='Subject')
inputhmm1ss_tstat = inputhmm1ss_hmm1ss.groupby(inputhmm1ss_hmm1ss.index).mean()/inputhmm1ss_hmm1ss.groupby(inputhmm1ss_hmm1ss.index).sem()
inputhmm1ss_pval = stats.t.sf(abs(inputhmm1ss_tstat), df=16)

inputhmm2ss_hmm1ss = inputhmm2ss_transmat.drop(columns='Subject') - hmm1ss_transmat.drop(columns='Subject')
inputhmm2ss_tstat = inputhmm2ss_hmm1ss.groupby(inputhmm2ss_hmm1ss.index).mean()/inputhmm2ss_hmm1ss.groupby(inputhmm2ss_hmm1ss.index).sem()
inputhmm2ss_pval = stats.t.sf(abs(inputhmm2ss_tstat), df=16)

    
# Plotting Transition Matrices

num_states = k 
    
fig = plt.figure(figsize=(9, 4), dpi=80, facecolor='w', edgecolor='k')
plt.subplot(1, 3, 1)
plt.imshow(hmm1ss_transmat_mean, vmin=-1, vmax=1, cmap='bone')
for i in range(hmm1ss_transmat_mean.shape[0]):
    for j in range(hmm1ss_transmat_mean.shape[1]):
        text = plt.text(j, i, str(np.around(hmm1ss_transmat_mean.values[i, j], decimals=2)), ha="center", va="center",
                        color="k", fontsize=12)
plt.xlim(-0.5, num_states - 0.5)
plt.xticks(range(0, num_states), ('1', '2', '3'), fontsize=10)
plt.yticks(range(0, num_states), ('1', '2', '3'), fontsize=10)
plt.ylim(k - 0.5, -0.5)
plt.ylabel("state t", fontsize = 15)
plt.xlabel("state t+1", fontsize = 15)
plt.title("HMM", fontsize = 15)   

plt.subplot(1, 3, 2)
plt.imshow(inputhmm1ss_transmat_mean, vmin=-1, vmax=1, cmap='bone')
for i in range(inputhmm1ss_transmat_mean.shape[0]):
    for j in range(inputhmm1ss_transmat_mean.shape[1]):
        text = plt.text(j, i, str(np.around(inputhmm1ss_transmat_mean.values[i, j], decimals=2)), ha="center", va="center",
                        color="k", fontsize=12)
plt.xlim(-0.5, num_states - 0.5)
plt.xticks(range(0, k), ('1', '2', '3'), fontsize=10)
plt.yticks(range(0, k), ('1', '2', '3'), fontsize=10)
plt.ylim(k - 0.5, -0.5)
plt.ylabel("state t", fontsize = 15)
plt.xlabel("state t+1", fontsize = 15)
plt.title("Input HMM Amygdala", fontsize = 15) 

plt.subplot(1, 3, 3)
plt.imshow(inputhmm2ss_transmat_mean, vmin=-1, vmax=1, cmap='bone')
for i in range(inputhmm2ss_transmat_mean.shape[0]):
    for j in range(inputhmm2ss_transmat_mean.shape[1]):
        text = plt.text(j, i, str(np.around(inputhmm2ss_transmat_mean.values[i, j], decimals=2)), ha="center", va="center",
                        color="k", fontsize=12)
plt.xlim(-0.5, num_states - 0.5)
plt.xticks(range(0, k), ('1', '2', '3'), fontsize=10)
plt.yticks(range(0, k), ('1', '2', '3'), fontsize=10)
plt.ylim(k - 0.5, -0.5)
plt.ylabel("state t", fontsize = 15)
plt.xlabel("state t+1", fontsize = 15)
plt.title("Input HMM DMPFC", fontsize = 15) 
plt.subplots_adjust(0, 0, 1, 1)
plt.tight_layout()
plt.show()  
fig.savefig('MeanTransitionMatrices-k3-4.15.22.png', dpi=150)
    
# Plot p-vals of change in transition matrices. Shade of cell is -log10(p)
fig = plt.figure(figsize=(7, 4), dpi=80, facecolor='w', edgecolor='k')    
plt.subplot(1, 2, 1)
plt.imshow(-np.log10(inputhmm1ss_pval), vmin=0, vmax=5, cmap='bone')
for i in range(inputhmm1ss_pval.shape[0]):
    for j in range(inputhmm1ss_pval.shape[1]):
        text = plt.text(j, i, str(np.around(inputhmm1ss_pval[i, j], decimals=4)), ha="center", va="center",
                        color="k", fontsize=12)
plt.xlim(-0.5, num_states - 0.5)
plt.xticks(range(0, k), ('1', '2', '3'), fontsize=10)
plt.yticks(range(0, k), ('1', '2', '3'), fontsize=10)
plt.ylim(k - 0.5, -0.5)
plt.ylabel("state t", fontsize = 15)
plt.xlabel("state t+1", fontsize = 15)
plt.title("Input HMM Amygdala pval", fontsize = 15) 

plt.subplot(1, 2, 2)
plt.imshow(-np.log10(inputhmm2ss_pval), vmin=0, vmax=5, cmap='bone')
for i in range(inputhmm2ss_pval.shape[0]):
    for j in range(inputhmm2ss_pval.shape[1]):
        text = plt.text(j, i, str(np.around(inputhmm2ss_pval[i, j], decimals=4)), ha="center", va="center",
                        color="k", fontsize=12)
plt.xlim(-0.5, num_states - 0.5)
plt.xticks(range(0, k), ('1', '2', '3'), fontsize=10)
plt.yticks(range(0, k), ('1', '2', '3'), fontsize=10)
plt.ylim(k - 0.5, -0.5)
plt.ylabel("state t", fontsize = 15)
plt.xlabel("state t+1", fontsize = 15)
plt.title("Input HMM DMPFC pval", fontsize = 15) 
plt.subplots_adjust(0, 0, 1, 1)
plt.tight_layout()
plt.show()     
fig.savefig('DiffFromVanillaHMM-pvals-k3-4.15.22.png', dpi=150)
    
# Plotting Transition Weights
# load in weights
inputhmm1ss_Ws = pd.read_csv(os.path.join(out_dir, f'Test-ALIGNED-Input-HMM-Amygdala-TransitionWeights-k3.csv'), index_col=0)
inputhmm2ss_Ws = pd.read_csv(os.path.join(out_dir, f'Test-ALIGNED-Input-HMM-DMPFC-TransitionWeights-k3.csv'), index_col=0)

# Means
inputhmm1ss_Ws_mean = inputhmm1ss_Ws.groupby(inputhmm1ss_Ws.index).mean()
inputhmm2ss_Ws_mean = inputhmm2ss_Ws.groupby(inputhmm2ss_Ws.index).mean()

fig = plt.figure(figsize=(5, 4), dpi=100, facecolor='w', edgecolor='k')

plt.plot(range(k), inputhmm1ss_Ws_mean, color=colors[2], marker='o',
         lw=2.5, markeredgewidth=2.5, markerfacecolor='white', markersize=12, linestyle = '-',
         label="Amygdala")
plt.plot(range(k), inputhmm2ss_Ws_mean, color=colors[3], marker='o',
         lw=2.5, markeredgewidth=2.5, markerfacecolor='white', markersize=12, linestyle = '-',
         label="DMPFC")
plt.yticks(fontsize=10)
plt.ylabel("Mean input weight", fontsize=15)
plt.ylim([-.4, .4])
plt.yticks([-.4, -.2, 0, .2, .4], fontsize=12)
plt.xlabel("State", fontsize=15)
plt.xticks([0, 1,2], ['1', '2', '3'], fontsize=12)
plt.axhline(y=0, color="k", alpha=0.5, ls="--")
plt.legend()
plt.title("Weight recovery", fontsize=15)
plt.tight_layout()
fig.savefig('InputHMM-Weights-k3-4.15.22.png', dpi=150)
   
### Duration of each state   

# Vanilla HMM
all_sequence_counts = []
for subj in tqdm(test_subs):
    filein = open(f'{out_dir}/{subj}-Aligned-Vanilla-HMM-k3.obj', 'rb')
    hmm1ss = pickle.load(filein)
    inferred_state_list, inferred_durations = ssm.util.rle(hmm1ss.most_likely_states(reduced_test[reduced_test['Subject']==subj].drop(columns='Subject').values))
    sequence_count = pd.DataFrame()
    sequence_count['State'] = inferred_state_list
    sequence_count['Count'] = inferred_durations
    sequence_count['Subject'] = subj
    sequence_count['ROI'] = 'vmpfc'
    all_sequence_counts.append(sequence_count)
all_sequence_counts = pd.concat(all_sequence_counts, axis=0)
all_sequence_counts.to_csv(os.path.join(out_dir, f'Test-ALIGNED-Vanilla-HMM-StateDurations-k3.csv'))

# Input HMM Amygdala
all_sequence_counts = []
for subj in tqdm(test_subs):
    filein = open(f'{out_dir}/{subj}-Aligned-Input-HMM-Amygdala-k3.obj', 'rb')
    inputhmm1ss = pickle.load(filein)
    inferred_state_list, inferred_durations = ssm.util.rle(inputhmm1ss.most_likely_states(reduced_test[reduced_test['Subject']==subj].drop(columns='Subject').values, Amygdala[Amygdala['Subject']==subj].drop(columns='Subject').values))
    sequence_count = pd.DataFrame()
    sequence_count['State'] = inferred_state_list
    sequence_count['Count'] = inferred_durations
    sequence_count['Subject'] = subj
    sequence_count['ROI'] = 'vmpfc'
    all_sequence_counts.append(sequence_count)
all_sequence_counts = pd.concat(all_sequence_counts, axis=0)
all_sequence_counts.to_csv(os.path.join(out_dir, f'Test-ALIGNED-Input-HMM-Amygdala-StateDurations-k3.csv'))

# Input HMM DMPFC
all_sequence_counts = []
for subj in tqdm(test_subs):
    filein = open(f'{out_dir}/{subj}-Aligned-Input-HMM-DMPFC-k3.obj', 'rb')
    inputhmm1ss = pickle.load(filein)
    inferred_state_list, inferred_durations = ssm.util.rle(inputhmm1ss.most_likely_states(reduced_test[reduced_test['Subject']==subj].drop(columns='Subject').values, DMPFC[DMPFC['Subject']==subj].drop(columns='Subject').values))
    sequence_count = pd.DataFrame()
    sequence_count['State'] = inferred_state_list
    sequence_count['Count'] = inferred_durations
    sequence_count['Subject'] = subj
    sequence_count['ROI'] = 'vmpfc'
    all_sequence_counts.append(sequence_count)
all_sequence_counts = pd.concat(all_sequence_counts, axis=0)
all_sequence_counts.to_csv(os.path.join(out_dir, f'Test-ALIGNED-Input-HMM-DMPFC-StateDurations-k3.csv'))

# Plot state durations
all_sequence_counts = []
hmm1ss_seq = pd.read_csv(os.path.join(out_dir, f'Test-ALIGNED-Vanilla-HMM-StateDurations-k3.csv'), index_col=0)
hmm1ss_seq['Model'] = 'Vanilla HMM'
all_sequence_counts.append(hmm1ss_seq)
inputhmm1ss_seq = pd.read_csv(os.path.join(out_dir, f'Test-ALIGNED-Input-HMM-Amygdala-StateDurations-k3.csv'), index_col=0)
inputhmm1ss_seq['Model'] = 'Input HMM Amygdala'
all_sequence_counts.append(inputhmm1ss_seq)
inputhmm2ss_seq = pd.read_csv(os.path.join(out_dir, f'Test-ALIGNED-Input-HMM-DMPFC-StateDurations-k3.csv'), index_col=0)
inputhmm2ss_seq['Model'] = 'Input HMM DMPFC'
all_sequence_counts.append(inputhmm2ss_seq)
all_sequence_counts = pd.concat(all_sequence_counts, axis=0) 

max_length = 25
fig = plt.figure(figsize=(15, 5), dpi=100, facecolor='w', edgecolor='k')
with sns.plotting_context(context='paper', font_scale=2):
    f,a = plt.subplots(nrows=1, ncols=3, figsize=(15, 5), sharex=True, sharey=False)
    for i,model in enumerate(['Vanilla HMM','Input HMM Amygdala', 'Input HMM DMPFC']):
        tmp_data = all_sequence_counts.query('Model==@model')
        for state in tmp_data['State'].unique():
            sns.distplot(tmp_data.query('State==@state')['Count'], hist=False, kde_kws={"shade": True}, ax=a[i])
        a[i].set_xlim(0, max_length)
        a[i].set_ylim(0, 0.25)
        a[i].legend(['State1','State2','State3'])
        a[i].set_xlabel('State Duration (TR)')
        a[i].set_ylabel('Frequency')
        a[i].set_title(f'{model}')
plt.tight_layout()
f.savefig('MeanStateDurations-k3-4.15.22.png', dpi=150)

##### Plot example participants states and inputs
test_subs = reduced_test['Subject'].unique()


hmm1ss_states = pd.read_csv(os.path.join(out_dir, f'Test-ALIGNED-Vanilla-HMM-PredictedStates-k3.csv'))
inputhmm1ss_states = pd.read_csv(os.path.join(out_dir, f'Test-ALIGNED-Input-HMM-Amygdala-PredictedStates-k3.csv'))
inputhmm2ss_states = pd.read_csv(os.path.join(out_dir, f'Test-ALIGNED-Input-HMM-DMPFC-PredictedStates-k3.csv'))

# Example subject #1
example_subs = test_subs[np.random.randint(0, 17,1)].tolist()
fig = plt.figure(figsize=(12, 10), dpi=100, facecolor='w', edgecolor='k')
with sns.plotting_context(context='paper', font_scale=2):

    plt.subplot(511)
    tmp_data = hmm1ss_states[hmm1ss_states['Subject']==example_subs[0]]['Viterbi'].values
    plt.imshow(tmp_data[None,:], aspect="auto", cmap=cmap, vmin=0, vmax=len(colors)-1)
    plt.ylabel("Vanilla HMM", fontsize=12)
    plt.yticks([])
    plt.xticks([])
    plt.title(f"Most Likely States - {example_subs[0]}", fontsize=15)
    
    plt.subplot(512)
    tmp_data = source_test[source_test['Subject']==example_subs[0]]['Amygdala'].values
    plt.plot(tmp_data, color='black', linewidth=2)
    plt.xticks([])
    plt.yticks([])
    plt.xlim(0, tmp_data.shape[0])
    plt.ylabel("Amygdala\nInput", fontsize=12)

    plt.subplot(513)
    tmp_data = inputhmm1ss_states[inputhmm1ss_states['Subject']==example_subs[0]]['Viterbi'].values
    plt.imshow(tmp_data[None,:], aspect="auto", cmap=cmap, vmin=0, vmax=len(colors)-1)
    plt.ylabel("Input HMM\nAmygdala", fontsize=12)
    plt.xticks([])
    plt.yticks([])
    
    plt.subplot(514)
    tmp_data = source_test[source_test['Subject']==example_subs[0]]['DMPFC'].values
    plt.plot(tmp_data, color='black', linewidth=2)
    plt.xticks([])
    plt.yticks([])
    plt.xlim(0, tmp_data.shape[0])
    plt.ylabel("DMPFC\nInput", fontsize=12)
    
    plt.subplot(515)
    tmp_data = inputhmm2ss_states[inputhmm2ss_states['Subject']==example_subs[0]]['Viterbi'].values
    plt.imshow(tmp_data[None,:], aspect="auto", cmap=cmap, vmin=0, vmax=len(colors)-1)
    plt.ylabel("Input HMM\nDMPFC", fontsize=12)
    plt.yticks([])
    plt.xlabel("time", fontsize=12)

    plt.tight_layout()
    example_subs[0]
fig.savefig(f'LikelyStates-{example_subs[0]}.png', dpi=150)



