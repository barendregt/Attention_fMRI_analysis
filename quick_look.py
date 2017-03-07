# First attempt at fmri analysis in python :-)

import os,sys,glob

import numpy as np
import scipy.stats as stats
import nibabel as nb
import pickle

from Staircase import ThreeUpOneDownStaircase
from functions import double_gamma as hrf

from sklearn import linear_model

import matplotlib.pyplot as plt
import seaborn as sn


from IPython import embed

subs = ['sub-001']#,'sub-002']
task = 'mapper' # 'ocinterleave'
rois = ['V1','V2','MT']

for subid in subs:
	print 'Running %s'%(subid) 

	# Setup directories
	data_dir = '/home/shared/2017/visual/Attention/me/'
	func_dir = os.path.join(data_dir, subid, 'psc/')
	par_dir = os.path.join(data_dir, subid, 'behaviour/')
	roi_dir = os.path.join(data_dir, subid, 'roi/')
	fig_dir = os.path.join(data_dir, subid, 'figures/tuning/')

	# Organize MRI and par files (also sanity check: if these don't match there is a problem!)
	mri_files = glob.glob(func_dir + '*' + task + '*.nii.gz')
	par_files = glob.glob(par_dir + '*' + task + '*.pickle')

	mri_files.sort()
	par_files.sort()

	sub_files = zip(mri_files, par_files)

	for roi in rois:

		try:
			os.makedirs(fig_dir + roi +'/')
		except:
			pass

		print 'ROI: %s'%roi
		roi_mask = (nb.load(roi_dir + 'lh.' + roi + '_vol.nii.gz').get_data()==1) + (nb.load(roi_dir + 'rh.' + roi + '_vol.nii.gz').get_data()==1)

		mri_data =[]
		all_trial_order = []

		print 'Collecting fMRI and task data'
		for mf, bf in sub_files:

			mri_data.append(nb.load(mf).get_data()[roi_mask])

			trial_params, trial_order, staircase = pickle.load(open(bf,'rb'))

			all_trial_order.append(trial_order)

		mri_data = np.hstack(mri_data)
		trial_order = np.vstack(all_trial_order)

		# Run GLM to find stimulus betas
		location_order = np.array(trial_order < 64,dtype=int)
		location_predictors = np.zeros((3*location_order.shape[0],4))
		location_predictors[::3] = location_order

		print 'Running GLM for locations'

		locationX = np.hstack([np.ones((np.shape(location_predictors)[0],1)), np.vstack([np.convolve(lp, hrf(np.arange(0,20)),'same') for lp in location_predictors.T]).T])

		location_betas = np.linalg.pinv(locationX).dot(mri_data.T)

		location_mask = [np.argmax(location_betas[1:,:],axis=0)==l for l in np.unique(np.argmax(location_betas[1:,:],axis=0))]

		# stimulus_order = trial_np.array(trial_order < 64,dtype=int)
		stimulus_predictors = np.zeros((3*trial_order.shape[0],4))
		stimulus_predictors[::3] = trial_order+1

		stimulus_betas = np.zeros((64,mri_data.shape[0]))
		stimulus_rsquared = np.zeros((1,mri_data.shape[0]))
		print 'Running GLM for stimuli'

		embed()
		for loci,mask in enumerate(location_mask):
			this_location_stimuli = np.hstack([np.ones((np.shape(stimulus_predictors)[0],1)), np.vstack([np.convolve(np.array(stimulus_predictors[:,loci]==stim,dtype=int), hrf(np.arange(0,20)),'same') for stim in range(1,65)]).T])

			stimulus_betas[:,mask] = np.linalg.pinv(this_location_stimuli).dot(mri_data[mask,:].T)[1:,:]
			stimulus_rsquared[:,mask] = 1 - (((stimulus_betas[:,mask].T.dot(this_location_stimuli[:,1:].T)-mri_data[mask,:])**2).sum() / (mri_data[mask,:]**2).sum())


		all_betas = np.zeros((256+1, mri_data.shape[0]))
		all_stimulus_predictors = np.zeros((mri_data.shape[1], 256))
		all_stimulus_predictors[::3] = np.vstack([[np.convolve(np.array(trial_column+1 == stim, dtype=int), hrf(np.arange(0,30)),'same') for stim in np.arange(1,65)] for trial_column in trial_order.T]).T
		# pred_indices = np.vstack([(tii, stim) for stim in np.arange(1,65)] for tii in np.arange(trial_order.shape[1])])
		all_predictors = np.hstack([np.ones((mri_data.shape[1],1)), all_stimulus_predictors])


		all_betas = np.linalg.pinv(all_predictors).dot(mri_data.T).T
		all_rsquared = 1 - (((all_betas.dot(all_predictors.T)-mri_data)**2).sum(axis=1) / (mri_data**2).sum(axis=1))

		location_contrast = np.vstack([np.zeros((1,4)), np.repeat(np.eye(4), 64, axis=0)])

		df = mri_data.shape[1] - location_contrast.shape[0]
		location_tvalues = all_betas.dot(location_contrast) / (((all_betas.dot(all_predictors.T)-mri_data)**2).sum(axis=1) / df)[:,np.newaxis]
		#location_pvalues = 

		location_mask = np.array(location_contrast[:,np.argmax(location_tvalues,axis=1)],dtype=bool)

		voxel_num = 993

		plt.figure(figsize=(8,8))
		plt.imshow(np.reshape(all_betas[voxel_num,location_mask[:,voxel_num]], (8,8)).T, interpolation='nearest')
		plt.savefig('test_voxel_max.png')

		voxel_num = 1211

		plt.figure(figsize=(8,8))
		plt.imshow(np.reshape(all_betas[voxel_num,location_mask[:,voxel_num]], (8,8)).T, interpolation='nearest')
		plt.savefig('test_voxel_min.png')		

		print 'Creating tuning images'
		# Make pretty pictures
		for vii in range(stimulus_betas.shape[1]):

			if stimulus_rsquared[vii] > 0.7:

				embed()

				plt.figure(figsize=(8,8))

				plt.imshow(np.reshape(stimulus_betas[:,vii],(8,8)).T, interpolation="nearest")

				#plt.savefig(fig_dir + 'voxel_%i.pdf'%vii)
				plt.savefig(fig_dir + '%s/%s-%s_%i.png'%(roi,subid,roi,vii))

				plt.close()

		embed()
