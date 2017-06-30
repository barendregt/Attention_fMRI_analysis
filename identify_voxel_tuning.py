# First attempt at fmri analysis in python :-)

import os,sys,glob

import numpy as np
import scipy.stats as stats
from scipy.signal import fftconvolve
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

subs = ['sub-n001','sub-n003','sub-n005']
task = 'fullfield' # 'ocinterleave'
rois = ['V1']#,'V2','MT','BA3a','BA44','BA45']


# mapper location order (from params):
# (T=top,B=bottom,L=left,R=right)
# TR-BR-BL-TL
# exp_location_order = [3, 0, 2, 1]


def bootstrap(data, num_samples, statistic, alpha):
    """Returns bootstrap estimate of 100.0*(1-alpha) CI for statistic."""
    n = len(data)
    idx = np.random.randint(0, n, (num_samples, n))
    samples = data[idx]
    stat = np.sort(statistic(samples, 1))
    return (stat[int((alpha/2.0)*num_samples)],
            stat[int((1-alpha/2.0)*num_samples)])

for subid in subs:
	print 'Running %s'%(subid) 

	# Setup directories
	data_dir = '/home/shared/2017/visual/OriColorMapper/preproc/'
	#data_dir = '/home/barendregt/Projects/Attention/'
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

	

	# Load fMRI data if not previously saved
	if not os.path.isfile(os.path.join(deriv_dir,'roi_data.mat')):

		mri_data = {}

		for ROI in rois:
			# Get all cortex data and task orders
			lh_mask = np.array(nib.load(os.path.join(ROI_dir,'lh.%s_vol_dil.nii.gz'%ROI)).get_data(), dtype = bool)
			rh_mask = np.array(nib.load(os.path.join(ROI_dir,'rh.%s_vol_dil.nii.gz'%ROI)).get_data(), dtype = bool)

			mri_data[ROI] = np.array([np.vstack([nib.load(nf).get_data()[lh_mask,:], nib.load(nf).get_data()[rh_mask,:]]) for nf in nifti_files])

		
		sio.savemat(file_name=os.path.join(deriv_dir,'roi_data.mat'), mdict=mri_data)
	else:
		mri_data = sio.loadmat(os.path.join(deriv_dir,'roi_data.mat'))


	task_data = {'trial_order': [],
				 'trial_stimuli': []}

	for ti,par in zip(trialinfo_files, params_files):
		[trial_array, trial_indices, trial_params, per_trial_parameters, per_trial_phase_durations, staircase] = pickle.load(open(ti,'rb'))

		task_data['trial_order'].append(trial_params[:,0])
		task_data['trial_stimuli'].append(trial_array)

	# embed()