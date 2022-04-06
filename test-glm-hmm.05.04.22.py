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
matplotlib.use('PS') 
import matplotlib.pyplot as plt
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
from scipy.stats import zscore, pearsonr, spearmanr
from scipy.spatial.distance import hamming

from fnl_tools.stats import hmm_bic

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
func_data = '/data/test'


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
plt.plot(zscore(source_time_series[:,0]))
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

ssm_out = pd.DataFrame()
ssm_out['zPredicted States'] = hmm_z

ssm.to_csv(os.path.join(out_dir,f'SSM-HMMPredictedStates.vmpfc.k{k}.csv'))

# align states for comparison - this is a dumb way to do it

df = pd.DataFrame()
df['Viterbi'] = p['Viterbi']
df['Viterbi'] = df['Viterbi'].replace(0, 'D')
df['Viterbi'] = df['Viterbi'].replace(1, 'C')
df['Viterbi'] = df['Viterbi'].replace(2, 'B')
df['Viterbi'] = df['Viterbi'].replace(3, 'A')

df['Viterbi'] = df['Viterbi'].replace('D',3)
df['Viterbi'] = df['Viterbi'].replace('C',2)
df['Viterbi'] = df['Viterbi'].replace('B',1)
df['Viterbi'] = df['Viterbi'].replace('A',0)


fig = plt.figure(figsize=(8, 6))
plt.subplot(311)
plt.imshow(hmm_z[None,:], aspect="auto", cmap=cmap, vmin=0, vmax=len(colors)-1)
plt.ylabel("SSM - $z_{\\mathrm{inferred}}$")
plt.yticks([])
plt.xlabel("time")

plt.subplot(312)
plt.imshow([df['Viterbi']], aspect="auto", cmap=cmap, vmin=0, vmax=len(colors)-1)
plt.ylabel("HMM - Viterbi aligned")
plt.yticks([])
plt.xlabel("time")


# unaligned
plt.subplot(313)
plt.imshow([p['Viterbi']], aspect="auto", cmap=cmap, vmin=0, vmax=len(colors)-1)
plt.ylabel("HMM - Viterbi")
plt.yticks([])
plt.xlabel("time")

r = pearsonr(hmm_z,df['Viterbi'] )
dist = hamming(hmm_z,df['Viterbi'] )


fig.savefig('most_likely_states - aligned.png', dpi=300)

#####               Input-driven HMM as in SSM
N_iters = 500

hmm2 = ssm.HMM(k, 70, 1, observations="diagonal_gaussian", transitions="inputdriven")

# Fit
a = (zscore(source_time_series[:,0]))
a = a.reshape(len(a),1)
hmm_lps = hmm2.fit(reduced, inputs=a, method="em", num_iters=N_iters)

hmm2_z = hmm2.most_likely_states(reduced)

fig = plt.figure(figsize=(8, 4))
plt.subplot(211)
plt.imshow(hmm_z[None,:], aspect="auto", cmap=cmap, vmin=0, vmax=len(colors)-1)
plt.ylabel("SSM - $z_{\\mathrm{inferred}}$")
plt.yticks([])
plt.xlabel("time")

plt.subplot(212)
plt.imshow(hmm2_z[None,:], aspect="auto", cmap=cmap, vmin=0, vmax=len(colors)-1)
plt.ylabel("input driven SSM - $z_{\\mathrm{inferred}}$")
plt.yticks([])
plt.xlabel("time")

fig.savefig('most_likely_states - w input driven.png', dpi=300)

# Transition Matrices

learned_transition_mat = hmm.transitions.transition_matrix
fig = plt.figure(figsize=(8, 4))
plt.subplot(121)
im = plt.imshow(learned_transition_mat, cmap='bone', clim=(0.0, 0.2))
plt.title("Learned Transition Matrix")

learned_transition_mat2 = hmm2.transitions.transition_matrix
plt.subplot(122)
im = plt.imshow(learned_transition_mat2, cmap='bone', clim=(0.0, 0.2))
plt.title("Learned Transition Matrix")

cbar_ax = fig.add_axes([0.95, 0.15, 0.05, 0.7])
fig.colorbar(im, cax=cbar_ax)
plt.show()


# log likelihood - BIC or WAIC would be better
mle_lls = hmm.log_likelihood(reduced)
mle_lps = hmm2.log_likelihood(reduced, a)

bic_lls = hmm_bic(LL=mle_lls, n_states=4, n_features=70)
bic_lps = hmm_bic(LL=mle_lps, n_states=4, n_features=71)

fig = plt.figure(figsize=(2, 2.5), dpi=80, facecolor='w', edgecolor='k')
loglikelihood_vals = [bic_lls, bic_lps]
colors = ['Red', 'Purple']
for z, occ in enumerate(loglikelihood_vals):
    plt.bar(z, occ, width = 0.8, color = colors[z])
plt.ylim([547000, 547405])
plt.xticks([0, 1], ['hmm', 'input hmm'], fontsize = 10)
plt.xlabel('model', fontsize = 15)
plt.ylabel('loglikelihood', fontsize=15)


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
for f in file_list:  
    
    sub = f.partition("sub-")[2].rpartition('_task')[0]
    # load in data
    sdata = image.load_img(f)

    # extract target data from roi
    target_time_series = target_masker.fit_transform(sdata)
    target_data = pd.DataFrame(target_time_series)
    target_data['Subject'] = sub
    target_group_data = target_group_data.append(target_data)
    
    # extract source data from rois
    source_time_series = masker.fit_transform(sdata)
    source_data = pd.DataFrame(zscore(source_time_series))
    source_data['Subject'] = sub
    source_group_data = source_group_data.append(source_data)
    
source_group_data.to_csv(os.path.join(out_dir, f'sources_zscoredata.csv'))

target_group_data.to_csv(os.path.join(out_dir, f'vmpfc_rawdata.csv'))
# Reduce Data Dimensionality
target_var = 0.9
pca = PCA(n_components=target_var)
        
X = pd.DataFrame(pca.fit_transform(target_group_data.drop(columns='Subject')))
X['Subject'] = target_group_data['Subject'].values
X.to_csv(os.path.join(out_dir, f'vmpfc_PCdata.csv'))

#####               Group-based Input-driven HMM as in SSM

reduced = pd.read_csv(os.path.join(out_dir, f'vmpfc_PCdata.csv'), index_col=0)
source = pd.read_csv(os.path.join(out_dir, f'sources_zscoredata.csv'))
source.columns = ['Index', 'Amygdala', 'NAcc', 'Hippocampus', 'DLPFC', 'DMPFC', 'pInsula', 'TPJ', 'Subject']

# Let's check the correlations between our inputs to the HMM
pearsoncorr = source.drop(columns=['Index','Subject']).corr(method='pearson')
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
    fig.savefig('Inputs - Correlation.png', dpi=300)

####             Vanilla HMM for group
N_iters = 500
k = 4 # Chang et al (2021) originally found that 4 states was best solution
hmm = ssm.HMM(k, reduced.drop(columns='Subject').shape[1], observations="diagonal_gaussian")

# Fit
hmm_lls = hmm.fit(reduced.drop(columns='Subject').values, method="em", num_iters=N_iters, init_method="kmeans")

# Most likely (Viterbi) states
hmm_z = hmm.most_likely_states(reduced.drop(columns='Subject').values)

# Log Likelihood
mle_hmm = hmm.log_likelihood(reduced.drop(columns='Subject').values)
bic_hmm = hmm_bic(LL=mle_hmm, n_states=4, n_features=83)

#ssm_out = pd.DataFrame()
#ssm_out['zPredicted States'] = hmm_z

#ssm.to_csv(os.path.join(out_dir,f'SSM-HMMPredictedStates.vmpfc.k{k}.csv'))


####             GLM-HMM for group - AMYG and NAcc

# First, lets make sure we know which k to use
N_iters = 500
mle_amygnacc = {}
bic_amygnacc = {}
for k in tqdm(range(1, 11)):

    hmmglm = ssm.HMM(k, reduced.drop(columns='Subject').shape[1], source[['Amygdala', 'NAcc']].shape[1], observations="diagonal_gaussian", transitions="inputdriven")
    hmm_amygnacc = hmmglm.fit(reduced.drop(columns='Subject').values, inputs=source[['Amygdala', 'NAcc']].values, method="em", num_iters=N_iters)
    mle_amygnacc[k] = hmmglm.log_likelihood(reduced.drop(columns='Subject').values, source[['Amygdala', 'NAcc']].values)
    bic_amygnacc[k] = hmm_bic(LL=mle_amygnacc[k], n_states=k, n_features=reduced.drop(columns='Subject').shape[1] + source[['Amygdala', 'NAcc']].shape[1])

model_fit = pd.DataFrame(list(bic_amygnacc.items()))
model_fit = model_fit.rename(columns={0: "k", 1: "BIC"})
with sns.plotting_context(context='paper', font_scale=2.5):
    fig = plt.figure(figsize=(6,5))
    sns.lineplot(data=model_fit, x='k', y='BIC', linewidth=3)
    plt.ylabel('Model Fit (BIC)', fontsize=18)
    plt.xlabel('k', fontsize=18)
    plt.axhline(bic_hmm, color='red', linestyle='--')
    plt.tight_layout()
    plt.show()
    fig.savefig('Model Fit - BIC - AmygdalaNAcc.png', dpi=300)
    
# Looks like k = 4 is the winner, although it is *just* a bit higher than no input. 
model_fit.to_csv(os.path.join(out_dir, f'GLM-HMM-AmygNacc-ModelFit.csv'))  

#I wonder how Amygdala or NAcc do by themselves

# Amygdala
k = 4
hmmglm = ssm.HMM(k, reduced.drop(columns='Subject').shape[1], source[['Amygdala']].shape[1], observations="diagonal_gaussian", transitions="inputdriven")
hmm_amyg = hmmglm.fit(reduced.drop(columns='Subject').values, inputs=source[['Amygdala']].values, method="em", num_iters=N_iters)
mle_amyg = hmmglm.log_likelihood(reduced.drop(columns='Subject').values, source[['Amygdala']].values)
bic_amyg = hmm_bic(LL=mle_amyg, n_states=k, n_features=reduced.drop(columns='Subject').shape[1] + source[['Amygdala']].shape[1])

with sns.plotting_context(context='paper', font_scale=2.5):
    fig = plt.figure(figsize=(6,5))
    sns.lineplot(data=model_fit, x='k', y='BIC', linewidth=3)
    plt.ylabel('Model Fit (BIC)', fontsize=18)
    plt.xlabel('k', fontsize=18)
    plt.axhline(bic_hmm, color='red', linestyle='--')
    plt.axhline(bic_amyg, color='green', linestyle='--')
    plt.tight_layout()
    plt.show()
    fig.savefig('Model Fit - BIC - AmygdalaNAcc plys Amyg.png', dpi=300)
    
# NAcc
k = 4
hmmglm = ssm.HMM(k, reduced.drop(columns='Subject').shape[1], source[['NAcc']].shape[1], observations="diagonal_gaussian", transitions="inputdriven")
hmm_nacc = hmmglm.fit(reduced.drop(columns='Subject').values, inputs=source[['NAcc']].values, method="em", num_iters=N_iters)
mle_nacc = hmmglm.log_likelihood(reduced.drop(columns='Subject').values, source[['NAcc']].values)
bic_nacc = hmm_bic(LL=mle_nacc, n_states=k, n_features=reduced.drop(columns='Subject').shape[1] + source[['NAcc']].shape[1])

with sns.plotting_context(context='paper', font_scale=2.5):
    fig = plt.figure(figsize=(6,5))
    sns.lineplot(data=model_fit, x='k', y='BIC', linewidth=3)
    plt.ylabel('Model Fit (BIC)', fontsize=18)
    plt.xlabel('k', fontsize=18)
    plt.axhline(bic_hmm, color='red', linestyle='--')
    plt.axhline(bic_amyg, color='green', linestyle='--')
    plt.axhline(bic_nacc, color='magenta', linestyle='--')
    plt.tight_layout()
    plt.show()
    fig.savefig('Model Fit - BIC - AmygdalaNAcc plus Amyg and NAcc.png', dpi=300)

bic_amygnacc = model_fit['BIC'].loc[model_fit['k'] == 4]
  
with sns.plotting_context(context='paper', font_scale=2):    
    fig = plt.figure(figsize=(6, 6))
    loglikelihood_vals = [bic_hmm, bic_amygnacc, bic_amyg, bic_nacc]
    colors = ['Red', 'Blue', 'Green', 'Purple']
    for z, occ in enumerate(loglikelihood_vals):
        plt.bar(z, occ, width = 0.8, color = colors[z])
    plt.ylim([13250000, 13290000])
    plt.xticks([0, 1,2,3], ['hmm', 'amyg+nacc', 'amyg', 'nacc'], fontsize = 12)
    plt.xlabel('model', fontsize = 15)
    plt.ylabel('BIC', fontsize=15)
    plt.tight_layout()
    plt.show()
    fig.savefig('Model Fit - BIC - All models k = 4.png', dpi=300)
    
# collate and save the BICs
bics = np.append(bic_amyg, bic_nacc)
bics = np.append(bics, bic_amygnacc.values)
bics = np.append(bics, bic_hmm)
bicspd = pd.DataFrame(bics, columns = ['BICS'])
bicspd['models'] = ['Amygdala', 'NAcc', 'Amygdala+Nacc', 'Vanilla', 'Emotions']



# Emotion Ratings

emotions = pd.read_csv(os.path.join(f'{base_dir}/data/ratings/', f'subjectivityTimeCourse.csv'))
emotions = emotions.iloc[::2,:]
file_list = glob.glob(f'{base_dir}{func_data}/sub-*nii.gz')
emotions_long = pd.concat([emotions]*len(file_list), ignore_index=True)
plt.figure(figsize=(10,3))
plt.plot(emotions_long[['pc1', 'pc2']])

k = 4
hmmglm = ssm.HMM(k, reduced.drop(columns='Subject').shape[1], emotions_long[['pc1', 'pc2']].shape[1], observations="diagonal_gaussian", transitions="inputdriven")
hmm_emotions = hmmglm.fit(reduced.drop(columns='Subject').values, inputs=emotions_long[['pc1', 'pc2']].values, method="em", num_iters=N_iters)
mle_emotions = hmmglm.log_likelihood(reduced.drop(columns='Subject').values, emotions_long[['pc1', 'pc2']].values)
bic_emotions = hmm_bic(LL=mle_emotions, n_states=k, n_features=reduced.drop(columns='Subject').shape[1] + emotions_long[['pc1', 'pc2']].shape[1])

bics = np.append(bics, bic_hmm)
bicspd = pd.DataFrame(bics, columns = ['BICS'])
bicspd['models'] = ['Amygdala', 'NAcc', 'Amygdala+Nacc', 'Vanilla', 'Emotions']
bicspd.to_csv(os.path.join(out_dir, f'GLM-HMM-All-ModelFit.csv')) 

x = source
x[['pc1', 'pc2']] = emotions_long[['pc1', 'pc2']].values
pearsoncorr = x.drop(columns=['Index','Subject']).corr(method='pearson')
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
    # Most likely (Viterbi) states
    #hmmglm_z = hmmglm.most_likely_states(reduced.drop(columns='Subject').values)
    
# Maybe orthogonalize Amygdala and NAcc?

def gs(X):
    Q, R = np.linalg.qr(X)
    return Q

m = source[['Amygdala', 'NAcc']].values
N = gs(m)
plt.figure(figsize=(10,3))
plt.plot(N)
r = pearsonr(m[:,0], m[:,1])
rN = pearsonr(N[:,0], N[:,1])
ortho_source = pd.DataFrame(N)
ortho_source.columns = ['Amygdala', 'NAcc_orth']
ortho_source[['pc1','pc2']] = emotions_long[['pc1', 'pc2']].values

k = 4
hmmglm = ssm.HMM(k, reduced.drop(columns='Subject').shape[1], ortho_source[['Amygdala', 'NAcc_orth']].shape[1], observations="diagonal_gaussian", transitions="inputdriven")
hmm_amygnacco = hmmglm.fit(reduced.drop(columns='Subject').values, inputs=ortho_source[['Amygdala', 'NAcc_orth']].values, method="em", num_iters=N_iters)
mle_amygnacco  = hmmglm.log_likelihood(reduced.drop(columns='Subject').values, ortho_source[['Amygdala', 'NAcc_orth']].values)
bic_amygnacco  = hmm_bic(LL=mle_amygnacco , n_states=k, n_features=reduced.drop(columns='Subject').shape[1] + ortho_source[['Amygdala', 'NAcc_orth']].shape[1])

bics = np.append(bic_amyg, bic_nacc)
bics = np.append(bics, bic_amygnacc.values)
bics = np.append(bics, bic_hmm)
bics = np.append(bics, bic_emotions)
bics = np.append(bics, bic_amygnacco)
bicspd = pd.DataFrame(bics, columns = ['BICS'])
bicspd['models'] = ['Amygdala', 'NAcc', 'Amygdala+Nacc', 'Vanilla', 'Emotions', 'Amyg+Nacc_ortho+Emotions']

with sns.plotting_context(context='paper', font_scale=2):    
    fig = plt.figure(figsize=(8, 6))
    loglikelihood_vals = [bic_hmm, bic_amygnacc, bic_amyg, bic_nacc, bic_emotions, bic_amygnacco]
    colors = ['Red', 'Blue', 'Green', 'Magenta', 'Cyan', 'Orange']
    for z, occ in enumerate(loglikelihood_vals):
        plt.bar(z, occ, width = 0.8, color = colors[z])
    plt.ylim([13250000, 13290000])
    plt.xticks([0,1,2,3,4,5], ['hmm', 'amyg+nacc', 'amyg', 'nacc', 'emotions', 'a+n_o+e'], fontsize = 12)
    plt.xlabel('model', fontsize = 15)
    plt.ylabel('BIC', fontsize=15)
    plt.tight_layout()
    plt.show()
    fig.savefig('Model Fit - BIC - All 5 models k = 4.png', dpi=300)


