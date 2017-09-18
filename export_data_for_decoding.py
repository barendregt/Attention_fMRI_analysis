from __future__ import division
import os,sys,glob

from joblib import Parallel, delayed
import multiprocessing

import numpy as np
import scipy.linalg as la
import scipy.stats as stats
from scipy.signal import fftconvolve, resample
import scipy.io as sio
import nibabel as nib
import pickle
import tables

from Staircase import ThreeUpOneDownStaircase
from tools import two_gamma as hrf
from tools import add_subplot_axes

import ColorTools as ct

from sklearn.linear_model import RidgeCV

import matplotlib.pyplot as plt
import seaborn as sn

sn.set(style='ticks')

from IPython import embed

subid = 'sub-n001'
task = 'fullfield'#'location'#'fullfield' # 'ocinterleave'
rois = ['V1','V4']#,'V2','MT','BA3a','BA44','BA45']

locations = [[-1.5, -1.5], [-1.5, 1.5], [1.5, -1.5], [1.5, 1.5]]

# ROIs = ['V1','V4']
TR = 0.945

US_FACTOR = 10
DS_FACTOR = 10

print 'Running %s'%(subid) 

# Setup directories
data_dir = '/home/shared/2017/visual/OriColorMapper/preproc/'
nifti_dir = os.path.join(data_dir, subid, 'psc/')
deriv_dir = os.path.join(data_dir, subid, 'deriv/')
ROI_dir = os.path.join(data_dir, subid, 'masks/dc/')
pickle_dir = os.path.join(data_dir, subid, 'beh/')
fig_dir = os.path.join(data_dir, subid, 'figures/')

if not os.path.isdir(deriv_dir):
	os.makedirs(deriv_dir)

# Locate nifti files
nifti_files = glob.glob('%s*_task-%s_*.nii.gz'%(nifti_dir, task))
trialinfo_files = glob.glob('%s*_task-%s_*_trialinfo.pickle'%(pickle_dir, task))
params_files = glob.glob('%s*_task-%s_*_params.pickle'%(pickle_dir, task))

nifti_files.sort()
trialinfo_files.sort()
params_files.sort()


embed()
# Load fMRI data if not previously saved
mri_data = {}

for ROI in rois:
	# Get all cortex data and task orders
	lh_mask = np.array(nib.load(os.path.join(ROI_dir,'lh.%s_vol_dil.nii.gz'%ROI)).get_data(), dtype = bool)
	rh_mask = np.array(nib.load(os.path.join(ROI_dir,'rh.%s_vol_dil.nii.gz'%ROI)).get_data(), dtype = bool)

	mri_data[ROI] = np.array([np.vstack([nib.load(nf).get_data()[lh_mask,:], nib.load(nf).get_data()[rh_mask,:]]) for nf in nifti_files])


# Load trial data
task_data = {'trial_order': [],
			 'trial_stimuli': [],
			 'trial_params': []}

for ti,par in zip(trialinfo_files, params_files):
	[trial_array, trial_indices, trial_params, per_trial_parameters, per_trial_phase_durations, staircase] = pickle.load(open(ti,'rb'))
	task_data['trial_order'].append(trial_params[:,0])
	task_data['trial_params'].append(trial_params)
	task_data['trial_stimuli'].append(trial_array)


embed()