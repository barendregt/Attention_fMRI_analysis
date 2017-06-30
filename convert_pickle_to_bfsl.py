import os,glob
import cPickle as pickle
from Staircase import ThreeUpOneDownStaircase
import numpy as np
from IPython import embed

subs = ['sub-001','sub-002']
tasks = ['mapper', 'ocinterleave']

TR = 0.945

for subid in subs:
	for task in tasks:
		print 'Running %s, %s'%(subid,task) 

		# Setup directories
		data_dir = '/home/barendregt/disks/Aeneas_Shared/2017/visual/Attention/me/'

		out_dir = os.path.join('/home/barendregt/Projects/Attention/',subid,'fsl/')

		# func_dir = os.path.join(data_dir, subid, 'psc/')
		par_dir = os.path.join(data_dir, subid, 'behaviour/')
		# roi_dir = os.path.join(data_dir, subid, 'roi/')
		# h5_dir = os.path.join(data_dir, subid, 'h5/')

		try:
			os.makedirs(out_dir)
		except:
			pass

		# Organize MRI and par files (also sanity check: if these don't match there is a problem!)
		# mri_files = glob.glob(func_dir + '*' + task + '*.nii.gz')

		# mri_files.sort()

		par_files = glob.glob(par_dir + '*' + task + '*.pickle')

		par_files.sort()

		trial_params_files = []
		trial_order_files = []

		for bf in par_files:

			trial_params, trial_order, staircase = pickle.load(open(bf,'rb'))

			

			for to in range(trial_order.shape[1]):
				location_array = np.vstack([np.arange(0,trial_order.shape[0]*3,3)*TR, np.ones((trial_order.shape[0]))*TR, np.array(trial_order[:,to]<64, dtype=int).T]).T
				
				np.savetxt(os.path.join(out_dir,subid+'_r'+str(par_files.index(bf))+'_l'+str(to)+'.bfsl'), location_array, delimiter = '\t', fmt=['%.1f','%.3f','%i'])

				tmp = np.zeros((330))
				tmp[:trial_order.shape[0]*3:3] = np.array(trial_order[:,to]<64, dtype=int)
				
				np.savetxt(os.path.join(out_dir,subid+'_r'+str(par_files.index(bf))+'_l'+str(to)+'_single.bfsl'), tmp, fmt='%i')


			# trial_params_files.append(trial_params)
			# trial_order_files.append(trial_order)