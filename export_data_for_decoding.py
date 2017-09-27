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

from convert_nii_to_h5 import nii_2_hdf5 as n2h5


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



orientations = np.linspace(90, 270, 8+1)[:-1]

# Compute evenly-spaced steps in (L)ab-space

color_theta = (np.pi*2)/8
color_angle = color_theta * np.arange(0, 8,dtype=float)
color_radius = 75

color_a = color_radius * np.cos(color_angle)
color_b = color_radius * np.sin(color_angle)

colors = [(55, a, b) for a,b in zip(color_a, color_b)]			 

#stimulus_positions = standard_parameters['stimulus_positions']

full_fact_stimulus_specs = []


full_fact_stimulus_specs = np.array([[[o,c[0],c[1],c[2]] for o in orientations] for c in colors]).reshape((64,4))
# dbstop()





embed()
# Load fMRI data if not previously saved
mri_data = {}

for runii, fname in enumerate(nifti_files):
	for ROI in rois:
		# Get all cortex data and task orders
		lh_mask = np.array(nib.load(os.path.join(ROI_dir,'lh.%s_vol_dil.nii.gz'%ROI)).get_data(), dtype = bool)
		rh_mask = np.array(nib.load(os.path.join(ROI_dir,'rh.%s_vol_dil.nii.gz'%ROI)).get_data(), dtype = bool)

		# mri_data[ROI] = np.array([np.vstack([nib.load(nf).get_data()[lh_mask,:], nib.load(nf).get_data()[rh_mask,:]]) for nf in nifti_files])

		old_img = nib.load(nifti_files[runii])
		new_img = nib.Nifti1Image(old_img.get_data()[lh_mask+rh_mask,:], old_img.affine)
		new_img.to_filename(os.path.join(deriv_dir,'%s-%s-%i.nii.gz'%(subid,ROI,runii)))


	# Load trial data
	task_data = {'trial_order': [],
				 'trial_stimuli': [],
				 'trial_params': []}


	# for ti,par in zip(trialinfo_files, params_files):
	[trial_array, trial_indices, trial_params, per_trial_parameters, per_trial_phase_durations, staircase] = pickle.load(open(trialinfo_files[runii],'rb'))

	trial_times = np.vstack(per_trial_phase_durations)[:,0] + np.arange(len(per_trial_phase_durations))*TR
	
	stimulus_times_w_codes = np.vstack([trial_times[trial_params[:,0]<64], trial_params[trial_params[:,0]<64,0]]).T

	stimulus_times_w_codes_and_specs = np.hstack([stimulus_times_w_codes, full_fact_stimulus_specs[np.array(stimulus_times_w_codes[:,1],dtype=int)]])

	np.savetxt(os.path.join(deriv_dir,'%s-%i.tsv'%(subid,runii)), stimulus_times_w_codes_and_specs, delimiter = '\t', header = 'Stim onset \t Stim code \t Orientation \t L \t a \t b')
