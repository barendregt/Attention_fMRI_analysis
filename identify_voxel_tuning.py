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

subs = ['sub-n001','sub-n003','sub-n005']
task = 'fullfield'#'location'#'fullfield' # 'ocinterleave'
rois = ['V1']#,'V2','MT','BA3a','BA44','BA45']

locations = [[-1.5, -1.5], [-1.5, 1.5], [1.5, -1.5], [1.5, 1.5]]

ROI = 'V1'
TR = 0.945

US_FACTOR = 10
DS_FACTOR = 10

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

# def identify_pref_locs(subid):
for subid in subs:
	print 'Running %s'%(subid) 

	all_betas = {}
	all_tvals = {}
	all_rs 	  = {}

	# Setup directories
	#data_dir = '/home/shared/2017/visual/OriColorMapper/preproc/'
	data_dir = '/home/barendregt/Project/OriColorMapper/fmri_data/'
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
	if not os.path.isfile(os.path.join(deriv_dir,'%s-roi_data.mat'%task)):

		mri_data = {}

		# for ROI in rois:
		# Get all cortex data and task orders
		lh_mask = np.array(nib.load(os.path.join(ROI_dir,'lh.%s_vol_dil.nii.gz'%ROI)).get_data(), dtype = bool)
		rh_mask = np.array(nib.load(os.path.join(ROI_dir,'rh.%s_vol_dil.nii.gz'%ROI)).get_data(), dtype = bool)

		mri_data[ROI] = np.array([np.vstack([nib.load(nf).get_data()[lh_mask,:], nib.load(nf).get_data()[rh_mask,:]]) for nf in nifti_files])

		
		sio.savemat(file_name=os.path.join(deriv_dir,'%s-roi_data.mat'%task), mdict=mri_data)
	else:
		mri_data = sio.loadmat(os.path.join(deriv_dir,'%s-roi_data.mat'%task))


	# Load trial data
	task_data = {'trial_order': [],
				 'trial_stimuli': [],
				 'trial_params': []}

	for ti,par in zip(trialinfo_files, params_files):
		[trial_array, trial_indices, trial_params, per_trial_parameters, per_trial_phase_durations, staircase] = pickle.load(open(ti,'rb'))
		task_data['trial_order'].append(trial_params[:,0])
		task_data['trial_params'].append(trial_params)
		task_data['trial_stimuli'].append(trial_array)


	if fit_per_run:
		# nan_list = []
		# # include_list = range(mri_data[ROI].shape[1])
		# for run_ii in range(mri_data[ROI].shape[0]):
		# 	this_run_data = (mri_data[ROI][run_ii,:,:] - mri_data[ROI][run_ii,:,:].mean(axis=1)[:,np.newaxis]) / mri_data[ROI][run_ii,:,:].std(axis=1)[:,np.newaxis]

		# 	if len(np.where(np.sum(np.isnan(this_run_data),1))[0])>0:
		# 		nan_list.append(np.squeeze(np.where(np.sum(np.isnan(this_run_data),1))[0][0]))


		# Fit GLM over runs


		# embed()

	

		betas = np.zeros((mri_data[ROI].shape[0],mri_data[ROI].shape[1], 5))
		alphas = np.zeros((mri_data[ROI].shape[0],mri_data[ROI].shape[1], 5))
		location_tvalues = np.zeros((mri_data[ROI].shape[0],mri_data[ROI].shape[1], 4))

		for run_ii in range(mri_data[ROI].shape[0]):
			this_run_data = mri_data[ROI][run_ii,:,:]#(mri_data[ROI][run_ii,:,:] - mri_data[ROI][run_ii,:,:].mean(axis=1)[:,np.newaxis]) / mri_data[ROI][run_ii,:,:].std(axis=1)[:,np.newaxis]

			this_run_order = task_data['trial_params'][run_ii][:,[1,2]]

			tmp_locations = np.zeros((this_run_data.shape[1],2))
			tmp_locations = this_run_order

			locations = [[-1.5, -1.5], [-1.5, 1.5], [1.5, -1.5], [1.5, 1.5]]

			design_matrix = np.vstack([np.array((tmp_locations[:,0]==a) * (tmp_locations[:,1]==b), dtype=int) for a,b in locations]).T

			design_matrix = np.hstack([np.ones((design_matrix.shape[0],1)), fftconvolve(design_matrix, hrf(np.arange(0,30,TR)[:,np.newaxis]))[:this_run_data.shape[1],:]])

			for vii in range(this_run_data.shape[0]):
				if np.sum(np.isnan(this_run_data[vii,:]))==0:
					mdl = RidgeCV(alphas=[0.0001,0.1,1,10,100,1000],fit_intercept=False,normalize=False)
					mdl.fit(design_matrix, this_run_data[vii,:])
					betas[run_ii,vii,:] = mdl.coef_
					alphas[run_ii,vii,:] = mdl.alpha_

			location_contrast = np.eye(4) + (np.eye(4)-1)/3

			df = this_run_data.shape[1] - betas.shape[2]
			location_tvalues[run_ii,:,:] = betas[run_ii,:,1:].dot(location_contrast) / (((design_matrix.dot(betas[run_ii,:,:].T)-this_run_data.T)**2).sum(axis=0) / df)[:,np.newaxis]


	else:

		# embed()

		concat_mri_data = np.hstack([(x-x.mean(axis=1)[:,np.newaxis])/x.std(axis=1)[:,np.newaxis] for x in mri_data[ROI]])#np.hstack(mri_data[ROI])#
		concat_trial_order = np.hstack(task_data['trial_order'])

		# trial_locations = np.vstack(task_data['trial_order'])[:,[1,2]]
		# tmp_locations = np.zeros((concat_mri_data.shape[1],2))
		# tmp_locations[::2,:] = trial_locations



		design_matrix = np.vstack([np.array(concat_trial_order==stim, dtype=int) for stim in range(64)]).T

		# resample signals to 1s resolution 
		resampled_mri_data = resample(concat_mri_data, int(concat_mri_data.shape[1]/TR), axis=1)
		resampled_dm       = resample(design_matrix, int(concat_mri_data.shape[1]/TR), axis=0)
		
		resampled_dm = np.hstack([np.ones((resampled_dm.shape[0],1)), fftconvolve(resampled_dm, hrf(np.arange(0,30,1/US_FACTOR)[::US_FACTOR,np.newaxis]))[:resampled_mri_data.shape[1],:]])


		# # resample back to 1s
		# ds_mri_data = resample(resampled_mri_data, int(resampled_mri_data.shape[1]/DS_FACTOR), axis=1)
		# ds_dm       = resample(resampled_dm, int(resampled_mri_data.shape[1]/DS_FACTOR), axis=0)

		betas = np.zeros((resampled_dm.shape[1],resampled_mri_data.shape[0]))
		alphas = np.zeros((resampled_dm.shape[1],resampled_mri_data.shape[0]))	

		for vii in range(resampled_mri_data.shape[0]):
			if np.sum(np.isnan(resampled_mri_data[vii,:]))==0:
				mdl = RidgeCV(alphas=[1.0,10.0,100.0,1000.0])
				mdl.fit(resampled_dm, resampled_mri_data[vii,:])

				betas[:,vii] = mdl.coef_
				alphas[:,vii] = mdl.alpha_

		# concat_mri_data = concat_mri_data[np.sum(np.isnan(concat_mri_data), axis=1)==0,:]

		location_rs = 1 - (((resampled_dm.dot(betas)-resampled_mri_data.T)**2).sum(axis=0) / (resampled_mri_data**2).sum(axis=1))

		# location_contrast = np.eye(4) + (np.eye(4)-1)/3

		# # location_contrast = [[0.5, -0.5],
		# # 					 [0.5, -0.5],
		# # 					 [-0.5,0.5],
		# # 					 [-0.5,0.5]]

		# df = resampled_mri_data.shape[0] - betas.shape[0]
		# location_tvalues = betas[1:,:].T.dot(location_contrast) / (((resampled_dm.dot(betas)-resampled_mri_data.T)**2).sum(axis=0) / df)[:,np.newaxis]

		# # location_pvalues = stats.t.sf(np.abs(location_tvalues), df)*2

		# location_count = [np.sum(np.argmax(location_tvalues, axis=1)==loc) for loc in np.unique(np.argmax(location_tvalues, axis=1))]

		# print location_count

		all_betas[ROI] = betas
		# all_tvals[ROI] = location_tvalues
		all_rs[ROI] = location_rs

	# Save stuff
	sio.savemat(file_name=os.path.join(deriv_dir,'%s-feature_betas.mat'%task), mdict=all_betas)
	# sio.savemat(file_name=os.path.join(deriv_dir,'%s-tvals.mat'%task), mdict=all_tvals)
	sio.savemat(file_name=os.path.join(deriv_dir,'%s-feature_rsquareds.mat'%task), mdict=all_rs)