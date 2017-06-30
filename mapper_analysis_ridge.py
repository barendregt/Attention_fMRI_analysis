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

	for ROI in rois:
		# Get all cortex data and task orders
		num_runs=3
		all_mri_data = []
		all_trial_order = []
		for r in range(1,num_runs+1):
			lh_cortex = data_file.get_node(where = '/mapper/mri/lh.%s'%ROI, name = subid+'_task-'+task+'_acq-multiband_run-'+str(r)+'_bold_brain_B0_volreg_sg_psc').read()
			rh_cortex = data_file.get_node(where = '/mapper/mri/rh.%s'%ROI, name = subid+'_task-'+task+'_acq-multiband_run-'+str(r)+'_bold_brain_B0_volreg_sg_psc').read()

			all_mri_data.append(np.vstack([lh_cortex, rh_cortex]))

			all_trial_order.append(data_file.get_node(where = '/mapper/beh/r' + str(r-1), name = 'trials').read())

		
		all_mri_data = np.array(all_mri_data)[:,:,:np.array(all_trial_order).shape[0]*np.array(all_trial_order).shape[1]]
		mri_data = np.hstack(all_mri_data)

		trial_order = np.vstack(all_trial_order)


		# # Run GLM to find stimulus betas
		# location_order = np.array(trial_order < 64,dtype=int)
		# location_predictors = np.zeros((3*location_order.shape[0],4))
		# location_predictors[6::3] = location_order[:-2]
		# # location_predictors[location_predictors.sum(axis=1)>1,:] = [0,0,0,0]

		# print 'Running GLM for locations'

		# locationX = np.hstack([np.ones((np.shape(location_predictors)[0],1)), fftconvolve(location_predictors, hrf(np.arange(0,24))[:,np.newaxis],'full')[:mri_data.shape[1],:]])

		# location_betas = np.linalg.lstsq(locationX, mri_data.T)[0]
		# location_rsquared = 1 - (((locationX.dot(location_betas)-mri_data.T)**2).sum(axis=0) / (mri_data**2).sum(axis=1))

		# # location_model = RidgeCV(alphas = np.arange(0.001,1.0,100),fit_intercept = True, normalize = True)
		# # location_model.fit(locationX, mri_data.T)
		# # location_betas = location_model.coef_

		# location_contrast = np.hstack([np.zeros((5,1)), np.vstack([np.zeros((1,4)), np.eye(4)])])

		# df = mri_data.shape[1] - location_betas.shape[0]
		# location_tvalues = location_betas.T.dot(location_contrast) / (((locationX.dot(location_betas)-mri_data.T)**2).sum(axis=0) / df)[:,np.newaxis]

		# location_pvalues = stats.t.sf(np.abs(location_tvalues), df)*2

		all_betas = np.zeros((256+1, mri_data.shape[0]))



		all_stimulus_predictors = np.zeros((mri_data.shape[1], 256))
		all_stimulus_predictors[::3] = np.vstack([[np.array(trial_column+1 == stim, dtype=int) for stim in np.arange(1,65)] for trial_column in trial_order.T]).T


		pred_indices = np.vstack([(tii, stim) for stim in np.arange(1,65)] for tii in np.arange(trial_order.shape[1]))
		
		all_predictors = np.hstack([np.ones((mri_data.shape[1],1)), fftconvolve(all_stimulus_predictors, hrf(np.arange(0,30)*0.945)[:,np.newaxis],'full')[:mri_data.shape[1],:]])


		all_stuff_model = RidgeCV(alphas = np.linspace(1.0,100,100),fit_intercept = False, normalize = True)
		# all_stuff_model = RidgeCV(alphas = [8],fit_intercept = False, normalize = True)
		all_stuff_model.fit(all_predictors, mri_data.T)
		all_stuff_betas = all_stuff_model.coef_

		print('fitted alpha: %f'%all_stuff_model.alpha_)

		all_rsquared = 1 - (((all_predictors.dot(all_stuff_betas.T)-mri_data.T)**2).sum(axis=0) / (mri_data**2).sum(axis=1))

		print('beta min-max: %.2f - %.2f'%(all_stuff_betas[:,1:].min(), all_stuff_betas[:,1:].max()))
		print('r^2 min-max: %.2f - %.2f'%(all_rsquared.min(), all_rsquared.max()))

		location_contrast = np.vstack([np.zeros((1,4)), np.repeat(np.eye(4), 64, axis=0)])

		df = mri_data.shape[1] - location_contrast.shape[0]
		location_tvalues = all_stuff_betas.dot(location_contrast) / (((all_stuff_betas.dot(all_predictors.T)-mri_data)**2).sum(axis=1) / df)[:,np.newaxis]	
		#location_pvalues = stats.t.sf(np.abs(location_tvalues), df)*


		embed()


		# Convert all to beta-matrix
		# all_stuff_betas[all_stuff_betas < 0] = 0
		oc_beta_matrix = np.dstack([np.reshape(all_stuff_betas.T[1+(location_tvalues.argmax(axis=1)[voxelii]*64):65+(location_tvalues.argmax(axis=1)[voxelii]*64),voxelii],(8,8))[:,[0,1,2,3,4,5,6,7,0]][[0,1,2,3,4,5,6,7,0],:] for voxelii in range(all_stuff_betas.shape[0])])
		
		voxel_ori_pref = oc_beta_matrix.max(axis=0).argmax(axis=0)
		voxel_col_pref = oc_beta_matrix.max(axis=1).argmax(axis=0)

		valid_voxel = all_rsquared > 0.0



		shift_ori_betas = np.vstack([np.roll(oc_beta_matrix[:,:,vii].max(axis=0),4-voxel_ori_pref[vii]) for vii in range(oc_beta_matrix.shape[2])])
		shift_col_betas = np.vstack([np.roll(oc_beta_matrix[:,:,vii].max(axis=1),4-voxel_col_pref[vii]) for vii in range(oc_beta_matrix.shape[2])])


		fig = plt.figure(figsize=(10,10))

		# shift_ori_95_ci = np.hstack([np.vstack(bootstrap(shift_ori_betas[:,col], 10000, np.mean, 0.05)) for col in range(shift_ori_betas.shape[1])])
		# shift_col_95_ci = np.hstack([np.vstack(bootstrap(shift_col_betas[:,col], 10000, np.mean, 0.05)) for col in range(shift_col_betas.shape[1])])

		shift_ori_68_ci = np.hstack([np.vstack(bootstrap(shift_ori_betas[:,col], 10000, np.mean, 0.32)) for col in range(shift_ori_betas.shape[1])])
		shift_col_68_ci = np.hstack([np.vstack(bootstrap(shift_col_betas[:,col], 10000, np.mean, 0.32)) for col in range(shift_col_betas.shape[1])])


		fig.add_subplot(1,2,1)
		errorbar(np.arange(9),np.median(shift_ori_betas,axis=0), shift_ori_68_ci, fmt='o')

		fig.add_subplot(1,2,2)
		errorbar(np.arange(9),np.median(shift_col_betas,axis=0), shift_col_68_ci, fmt='o')


		fig = plt.figure(figsize=(10,10))

		fig.suptitle('Orientation tuning')

		for ori_ii in range(8):
			fig.add_subplot(2,4,ori_ii+1)

			mdata = np.multiply(oc_beta_matrix[:,:,(voxel_ori_pref==ori_ii) * valid_voxel].max(axis=0),all_rsquared[(voxel_ori_pref==ori_ii) * valid_voxel]).mean(axis=1)
			sdata = np.multiply(oc_beta_matrix[:,:,(voxel_ori_pref==ori_ii) * valid_voxel].max(axis=0),all_rsquared[(voxel_ori_pref==ori_ii) * valid_voxel]).std(axis=1) / np.sqrt(np.sum(valid_voxel))

			# plot(oc_beta_matrix[:,:,voxel_ori_pref==ori_ii].max(axis=0), color = 'b', alpha = 0.01)
			plt.fill_between(range(9), mdata-sdata,mdata+sdata, color='b',alpha=0.3)
			plt.plot(mdata, color = 'k',lw=1.5)


		fig.savefig(os.path.join(fig_dir, '%s_orientation_tuning_max-profile_weighted.pdf'%(ROI)))

		plt.close('all')
		fig = plt.figure(figsize=(10,10))

		fig.suptitle('Color tuning')

		for col_ii in range(8):
			fig.add_subplot(2,4,col_ii+1)

			mdata = np.multiply(oc_beta_matrix[:,:,(voxel_col_pref==col_ii) * valid_voxel].max(axis=1),all_rsquared[(voxel_col_pref==col_ii) * valid_voxel]).mean(axis=1)
			sdata = np.multiply(oc_beta_matrix[:,:,(voxel_col_pref==col_ii) * valid_voxel].max(axis=1),all_rsquared[(voxel_col_pref==col_ii) * valid_voxel]).std(axis=1) / np.sqrt(np.sum(valid_voxel))

			# plot(oc_beta_matrix[:,:,voxel_ori_pref==ori_ii].max(axis=0), color = 'b', alpha = 0.01)
			plt.fill_between(range(9), mdata-sdata,mdata+sdata, color='b',alpha=0.3)
			plt.plot(mdata, color = 'k',lw=1.5)


		fig.savefig(os.path.join(fig_dir, '%s_color_tuning_max-profile_weighted.pdf'%(ROI)))
		plt.close('all')
		# embed()

		# best_voxels = all_rsquared.argsort()[-8:]#location_tvalues.max(axis=1).argsort()[-5:]

		# all_stuff_betas[all_stuff_betas < 0] = 0

		# color_theta = (np.pi*2)/8
		# color_angle = color_theta * np.arange(0,8,dtype=float)
		# color_radius = 75

		# color_a = color_radius * np.cos(color_angle)
		# color_b = color_radius * np.sin(color_angle)

		# ax_colors = [(ct.lab2rgb((55, a, b))/255).tolist() for a,b in zip(color_a, color_b)]		
		# ax_colors.append(ax_colors[0])

		# # embed()
		# fig = plt.figure(figsize=(10,10))    

		# for subii,voxelii in enumerate(best_voxels):

		# 	#fig,axes = plt.subplots(nrows=2, ncols=2)
		# 	#for location in exp_location_order:
		# 	location = location_tvalues.argmax(axis=1)[voxelii]
		# 	mainax = fig.add_subplot(2,4,subii+1)
		# 	mainax.set_title('Voxel #%i - r^2 = %.2f'%(voxelii,all_rsquared[voxelii]))
		# 	# if location == :
		# 		# mainax.set_title('*')

		# 	# rect_main = [rect_scatter[0] + np.array(i%2==0, dtype=int)*0.25, rect_scatter[1] + np.array(i%2==1,dtype=int)*0.25, rect_scatter[2], rect_scatter[3]]
		# 	subax_x = add_subplot_axes(mainax, [0.0, 0.9, 1.0, 0.05])
		# 	subax_y = add_subplot_axes(mainax, [1.0, 0.01, 0.05, 0.75])


		# 	embed()
		# 	beta_image = np.reshape(all_stuff_betas.T[1+(location*64):65+(location*64),voxelii],(8,8))
		# 	beta_image = beta_image[:,[0,1,2,3,4,5,6,7,0]][[0,1,2,3,4,5,6,7,0],:]

		# 	# mainax = plt.axes(rect_main)
		# 	im = mainax.imshow(beta_image,clim=(0,np.abs(all_stuff_betas[voxelii,1:]).max()), cmap='viridis')
			

		# 	mainax.set_xlim([-0.5,8.5])
		# 	mainax.set_ylim([8.5,-0.5])

		# 	# mainax.set_axis_off()

		# 	# if subii in [4,5,6,7]:
		# 	mainax.set_xlabel('Orientation')
		# 	mainax.set_xticks(np.arange(9)[::2], minor=False)
		# 	mainax.set_xticklabels(np.linspace(90,270,9)[::2], minor=False)
		# # else:
		# 		# mainax.set_xticks([])

		# 	if subii in [0,4]:
		# 		mainax.set_ylabel('Color')
		# 		mainax.set_yticks(np.arange(9), minor=False)
		# 		[t.set_color(ax_colors[ii]) for ii,t in enumerate(mainax.yaxis.get_ticklabels())]
		# 	else:
		# 		mainax.set_yticks([])
		# 		# mainax.tick_params(axis='y', colors=colors)

		# 	#subax_x.bar(np.arange(9)-0.25,np.max(abs(beta_image), axis=0), width=0.5)
		# 	#subax_y.barh(np.arange(9)-0.25, np.max(abs(beta_image), axis=1), height=0.5)
		# 	subax_x.plot(np.arange(9),np.max(beta_image, axis=0),color='k',alpha=0.75)
		# 	subax_y.plot(np.max(beta_image, axis=1),np.arange(9),color='k',alpha=0.75)

		# 	subax_x.fill_between(np.arange(9), np.zeros(9), np.max(beta_image, axis=0), alpha=0.5)
		# 	subax_y.fill_betweenx(np.arange(9), np.zeros(9), np.max(beta_image, axis=1), alpha=0.5)
		# 	# subax_y.fill_between(np.max(beta_image, axis=1), np.zeros(9), np.arange(9), alpha=0.5)

		# 	subax_x.set_xlim(mainax.get_xlim())
		# 	subax_y.set_ylim(mainax.get_ylim())		

		# 	# subax_x.set(xticks=[], yticks=[])
		# 	# subax_y.set(xticks=[], yticks=[])
		# 	subax_x.set_frame_on(False)
		# 	subax_x.set_axis_off()
		# 	subax_y.set_frame_on(False)
		# 	subax_y.set_axis_off()
		# 	# ax_y.xticks()

		# 	sn.despine()

		# 	# fig.colorbar(im, ax=axes.ravel().tolist())
		# fig.savefig(os.path.join(fig_dir, '%s_best_voxels.pdf'%(ROI)))
		# plt.close(fig)
	# embed()
		# location_contrast = np.vstack([np.zeros((1,4)), np.repeat(np.eye(4), 64, axis=0)])

		# df = mri_data.shape[1] - location_contrast.shape[0]
		# location_tvalues = all_betas.dot(location_contrast) / (((all_betas.dot(all_predictors.T)-mri_data)**2).sum(axis=1) / df)[:,np.newaxis]
		# #location_pvalues = 

		# location_mask = np.array(location_contrast[:,np.argmax(location_tvalues,axis=1)],dtype=bool)

		# voxel_num = 993

		# plt.figure(figsize=(8,8))
		# plt.imshow(np.reshape(all_betas[voxel_num,location_mask[:,voxel_num]], (8,8)).T, interpolation='nearest')
		# plt.savefig('test_voxel_max.png')

		# voxel_num = 1211

		# plt.figure(figsize=(8,8))
		# plt.imshow(np.reshape(all_betas[voxel_num,location_mask[:,voxel_num]], (8,8)).T, interpolation='nearest')
		# plt.savefig('test_voxel_min.png')		

		# print 'Creating tuning images'
		# # Make pretty pictures
		# for vii in range(stimulus_betas.shape[1]):

		# 	if stimulus_rsquared[vii] > 0.7:

		# 		embed()

		# 		plt.figure(figsize=(8,8))

		# 		plt.imshow(np.reshape(stimulus_betas[:,vii],(8,8)).T, interpolation="nearest")

		# 		#plt.savefig(fig_dir + 'voxel_%i.pdf'%vii)
		# 		plt.savefig(fig_dir + '%s/%s-%s_%i.png'%(roi,subid,roi,vii))

		# 		plt.close()

		# embed()
