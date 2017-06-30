# First attempt at fmri analysis in python :-)

import os,sys,glob

import numpy as np
import scipy.stats as stats
from scipy.signal import fftconvolve
import nibabel as nb
import pickle
import tables

# from Staircase import ThreeUpOneDownStaircase
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
exp_location_order = [3, 0, 2, 1]


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
	ROI_dir = os.path.join(data_dir, subid, 'masks/dc/')
	pickle_dir = os.path.join(data_dir, subid, 'beh/')
	fig_dir = os.path.join(data_dir, subid, 'figures/')



	# Locate nifti files
	nifti_files = glob.glob('%s*_task-%s_*.nii.gz'%(nifti_dir, task))
	trialinfo_files = glob.glob('%s*_task-%s_*_trialinfo.pickle'%(nifti_dir, task))
	params_files = glob.glob('%s*_task-%s_*_params.pickle'%(nifti_dir, task))

	mri_data = {}

	for ROI in rois:
		# Get all cortex data and task orders
		lh_mask = np.array(nib.load(os.path.join(ROI_dir,'lh.',ROI,'_vol_dil.nii.gz')).get_data(), dtype = bool)
		lh_mask = np.array(nib.load(os.path.join(ROI_dir,'rh.',ROI,'_vol_dil.nii.gz')).get_data(), dtype = bool)

		mri_data[ROI] = np.array([np.vstack(np.nib.load(nf).get_data()[lh,:], np.nib.load(nf).get_data()[rh,:]) for nf in nifti_files])

	
