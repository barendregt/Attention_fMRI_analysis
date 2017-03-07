import os, glob

def mask_nii_2_hdf5(in_files, mask_files, hdf5_file, folder_alias):
    """masks data in in_files with masks in mask_files,
    to be stored in an hdf5 file

    Takes a list of 3D or 4D fMRI nifti-files and masks the
    data with all masks in the list of nifti-files mask_files.
    These files are assumed to represent the same space, i.e.
    that of the functional acquisitions. 
    These are saved in hdf5_file, in the folder folder_alias.

    Parameters
    ----------
    in_files : list
        list of absolute path to functional nifti-files.
        all nifti files are assumed to have the same ndim
    mask_files : list
        list of absolute path to mask nifti-files.
        mask_files are assumed to be 3D
    hdf5_file : str
    	absolute path to hdf5 file.
   	folder_alias : str
   		name of the to-be-created folder in the hdf5 file.

    Returns
    -------
    hdf5_file : str
        absolute path to hdf5 file.
    """

    import nibabel as nib
    import os.path as op
    import numpy as np
    import tables

    success = True

    mask_data = [np.array(nib.load(mf).get_data(), dtype = bool) for mf in mask_files]
    nifti_data = [nib.load(nf).get_data() for nf in in_files]

    mask_names = [op.split(mf)[-1].split('_vol.nii.gz')[0] for mf in mask_files]
    nifti_names = [op.split(nf)[-1].split('.nii.gz')[0] for nf in in_files]

    h5file = tables.open_file(hdf5_file, mode = "a", title = hdf5_file)
    # get or make group for alias folder
    try:
        folder_alias_run_group = h5file.get_node("/", name = folder_alias, classname='Group')
    except tables.NoSuchNodeError:
        print('Adding group ' + folder_alias + ' to this file')
        folder_alias_run_group = h5file.create_group("/", folder_alias, folder_alias)

    for (roi, roi_name) in zip(mask_data, mask_names):
        # get or make group for alias/roi
        try:
            run_group = h5file.get_node(where = "/" + folder_alias, name = roi_name, classname='Group')
        except tables.NoSuchNodeError:
            print('Adding group ' + folder_alias + '_' + roi_name + ' to this file')
            run_group = h5file.create_group("/" + folder_alias, roi_name, folder_alias + '_' + roi_name)

        h5file.create_array(run_group, roi_name, roi, roi_name + ' mask file for reconstituting nii data from masked data')

        for (nii_d, nii_name) in zip(nifti_data, nifti_names):
            print('roi: %s, nifti: %s'%(roi_name, nii_name))
            n_dims = len(nii_d.shape)
            if n_dims == 3:
                these_roi_data = nii_d[roi]
            elif n_dims == 4:   # timeseries data, last dimension is time.
                these_roi_data = nii_d[roi,:]
            else:
                print("n_dims in data {nifti} do not fit with mask".format(nii_name))
                success = False

            h5file.create_array(run_group, nii_name, these_roi_data, roi_name + ' data from ' + nii_name)

    h5file.close()

    return hdf5_file




subs = ['sub-001','sub-002']
tasks = ['mapper', 'ocinterleave']

for subid in subs:
	for task in tasks:
		print 'Running %s'%(subid) 

		# Setup directories
		data_dir = '/home/shared/2017/visual/Attention/me/'
		func_dir = os.path.join(data_dir, subid, 'psc/')

		roi_dir = os.path.join(data_dir, subid, 'roi/')

		h5_dir = os.path.join(data_dir, subid, 'h5/')

		os.mkdirs(os.path.join(h5_dir))

		# Organize MRI and par files (also sanity check: if these don't match there is a problem!)
		mri_files = glob.glob(func_dir + '*' + task + '*.nii.gz')

		mri_files.sort()

		roi_masks = glob.glob(roi_dir + '*_vol.nii.gz')

		mask_nii_2_hdf5(mri_files, roi_masks, h5_dir + subid + '.h5', task) 

