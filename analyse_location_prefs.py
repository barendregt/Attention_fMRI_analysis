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
task = 'location'#'fullfield' # 'ocinterleave'
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

embed()

# def identify_pref_locs(subid):
all_tvals = []
all_rs = []
all_lc = []

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


	# Load regression results
	# location_betas = sio.loadmat(os.path.join(deriv_dir,'%s-location_betas.mat'%task))[ROI]
	# location_r_squared = sio.loadmat(os.path.join(deriv_dir,'%s-location_rsquareds.mat'%task))[ROI]
	# location_tvals = sio.loadmat(os.path.join(deriv_dir,'%s-location_tvals.mat'%task))[ROI]

	# all_tvals.append(location_tvals)
	# all_rs.append(location_r_squared)	


	# Get location pref distribution
	location_count = np.array([np.sum(np.argmax(location_tvals, axis=1)==loc) for loc in np.unique(np.argmax(location_tvals, axis=1))]) / location_tvals.shape[0]
	all_lc.append(location_count)

	# Tuning prefs
	feature_betas = sio.loadmat(os.path.join(deriv_dir,'%s-feature_betas.mat'%task))[ROI]
	feature_r_squared = sio.loadmat(os.path.join(deriv_dir,'%s-feature_rsquareds.mat'%task))[ROI]

	ori_pref = np.zeros((feature_betas.shape[1]))
	col_pref = np.zeros((feature_betas.shape[1]))

	for vii in feature_betas.shape[1]:

		rs_betas = np.reshape(feature_betas[1:,vii],[8,8])

		ori_pref[vii] = np.argmax(rs_betas, axis=1)
		col_pref[vii] = np.argmax(rs_betas, axis=0)




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


	template_beta_mat = np.reshape(np.arange(64),[8,8])

	col_beta_mat = np.repeat(np.arange(8)[:,np.newaxis],8,axis=1)
	ori_beta_mat = np.repeat(np.arange(8)[:,np.newaxis],8,axis=1).T

	for vii in mri_data.shape[0]:

		concat_mri_data = np.hstack([(x-x.mean(axis=1)[:,np.newaxis])/x.std(axis=1)[:,np.newaxis] for x in mri_data[ROI]])#np.hstack(mri_data[ROI])#
		concat_trial_order = np.hstack(task_data['trial_order'])

		trial_locations = np.vstack(task_data['trial_order'])[:,[1,2]]

		betas = np.zeros((resampled_dm.shape[1],resampled_mri_data.shape[0]))
		alphas = np.zeros((resampled_dm.shape[1],resampled_mri_data.shape[0]))	

		for vii in range(resampled_mri_data.shape[0]):

			design_matrix = np.vstack([np.array((trial_locations[:,0]==a) * (trial_locations[:,1]==b) * (concat_trial_order==stim), dtype=int) for stim in range(64)]).T

			# resample signals to 1s resolution 
			resampled_mri_data = resample(concat_mri_data, int(concat_mri_data.shape[1]/TR), axis=1)
			resampled_dm       = resample(design_matrix, int(concat_mri_data.shape[1]/TR), axis=0)
			
			resampled_dm = np.hstack([np.ones((resampled_dm.shape[0],1)), fftconvolve(resampled_dm, hrf(np.arange(0,30,1/US_FACTOR)[::US_FACTOR,np.newaxis]))[:resampled_mri_data.shape[1],:]])
		
			if np.sum(np.isnan(resampled_mri_data[vii,:]))==0:
				mdl = RidgeCV(alphas=[1.0,10.0,100.0,1000.0])
				mdl.fit(resampled_dm, resampled_mri_data[vii,:])

				betas[:,vii] = mdl.coef_
				alphas[:,vii] = mdl.alpha_





plt.figure()
m_lc = np.mean(all_lc,axis=0)
s_lc = np.std(all_lc,axis=0)/len(subs)
plt.bar([0.1,1.1,2.1,3.1],m_lc)
plt.errorbar([0.5,1.5,2.5,3.5],m_lc,s_lc,fmt='.',color='k')
plt.axis([0,4,0,.5])
plt.ylabel('Proportion voxels/location')
sn.despine()
# plt.set('xticks',[.5,1.5,2.5,3.5])
plt.savefig('location_count.pdf')
plt.close()