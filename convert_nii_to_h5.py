from hedfpy.utils import mask_nii_2_hdf5

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
