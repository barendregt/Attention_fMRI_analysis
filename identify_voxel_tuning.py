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
task = 'location'#'fullfield' # 'ocinterleave'
rois = ['V1']#,'V2','MT','BA3a','BA44','BA45']

ROI = 'V1'
TR = 0.945


fit_per_run = False
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

		# for ROI in rois:
		# Get all cortex data and task orders
		lh_mask = np.array(nib.load(os.path.join(ROI_dir,'lh.%s_vol_dil.nii.gz'%ROI)).get_data(), dtype = bool)
		rh_mask = np.array(nib.load(os.path.join(ROI_dir,'rh.%s_vol_dil.nii.gz'%ROI)).get_data(), dtype = bool)

		mri_data[ROI] = np.array([np.vstack([nib.load(nf).get_data()[lh_mask,:], nib.load(nf).get_data()[rh_mask,:]]) for nf in nifti_files])

		
		sio.savemat(file_name=os.path.join(deriv_dir,'roi_data.mat'), mdict=mri_data)
	else:
		mri_data = sio.loadmat(os.path.join(deriv_dir,'roi_data.mat'))


	# Load trial data
	task_data = {'trial_order': [],
				 'trial_stimuli': []}

	for ti,par in zip(trialinfo_files, params_files):
		[trial_array, trial_indices, trial_params, per_trial_parameters, per_trial_phase_durations, staircase] = pickle.load(open(ti,'rb'))
		task_data['trial_order'].append(trial_params[:,0])
		task_data['trial_stimuli'].append(trial_array)


	if fit_per_run:
		nan_list = []
		# include_list = range(mri_data[ROI].shape[1])
		for run_ii in range(mri_data[ROI].shape[0]):
			this_run_data = (mri_data[ROI][run_ii,:,:] - mri_data[ROI][run_ii,:,:].mean(axis=1)[:,np.newaxis]) / mri_data[ROI][run_ii,:,:].std(axis=1)[:,np.newaxis]

			if len(np.where(np.sum(np.isnan(this_run_data),1))[0])>0:
				nan_list.append(np.squeeze(np.where(np.sum(np.isnan(this_run_data),1))[0][0]))


		# Fit GLM over runs


		embed()

	

		betas = np.zeros((mri_data[ROI].shape[0],mri_data[ROI].shape[1], 65))
		alphas = np.zeros((mri_data[ROI].shape[0],mri_data[ROI].shape[1], 65))

		for run_ii in range(mri_data[ROI].shape[0]):
			this_run_data = (mri_data[ROI][run_ii,:,:] - mri_data[ROI][run_ii,:,:].mean(axis=1)[:,np.newaxis]) / mri_data[ROI][run_ii,:,:].std(axis=1)[:,np.newaxis]

			this_run_order = task_data['trial_order'][run_ii]

			stimulus_order = np.zeros((len(this_run_order)))
			stimulus_order[this_run_order < 64] = this_run_order[this_run_order<64] + 1

			design_matrix = np.vstack([np.array(stimulus_order==stimulus,dtype=int) for stimulus in range(1,65)]).T

			design_matrix = np.hstack([np.ones((design_matrix.shape[0],1)), fftconvolve(design_matrix, hrf(np.arange(0,30,TR)[:,np.newaxis]))[:this_run_data.shape[1],:]])

			for vii in range(this_run_data.shape[0]):
				if np.sum(np.isnan(this_run_data[vii,:]))==0:
					mdl = RidgeCV(alphas=[0.0001,0.1,1,10,100,1000])
					mdl.fit(design_matrix, this_run_data[vii,:])
					betas[run_ii,vii,:] = mdl.coef_
					alphas[run_ii,vii,:] = mdl.alpha_

	else:

		embed()

		concat_mri_data = np.hstack([(x-x.mean(axis=1)[:,np.newaxis])/x.std(axis=1)[:,np.newaxis] for x in mri_data[ROI]])
		concat_trial_order = np.hstack(task_data['trial_order'])

		betas = np.zeros((concat_mri_data.shape[0], 65))
		alphas = np.zeros((mri_data[ROI].shape[0],mri_data[ROI].shape[1], 65))		

		# embed()

	# embed()