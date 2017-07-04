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
for subid in subs:
	print 'Running %s'%(subid) 

	all_betas = {}
	all_tvals = {}
	all_rs 	  = {}

	# Setup directories
	data_dir = '/home/shared/2017/visual/OriColorMapper/preproc/'
	#data_dir = '/home/barendregt/Projects/Attention/'
	nifti_dir = os.path.join(data_dir, subid, 'psc/')
	deriv_dir = os.path.join(data_dir, subid, 'deriv/')
	ROI_dir = os.path.join(data_dir, subid, 'masks/dc/')
	pickle_dir = os.path.join(data_dir, subid, 'beh/')
	fig_dir = os.path.join(data_dir, subid, 'figures/')


	# Load regression results
	betas = sio.loadmat(os.path.join(deriv_dir,'%s-betas.mat'%task))
	r_squared = sio.loadmat(os.path.join(deriv_dir,'%s-rsquareds.mat'%task))
	tvals = sio.loadmat(os.path.join(deriv_dir,'%s-tvals.mat'%task))

