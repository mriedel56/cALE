"""
Perform complementary ALE network workflow.
"""

import os
import numpy as np
import pandas as pd
import os.path as op
import nibabel as nib
from shutil import copyfile
from nimare.meta.cbma import ALE
from nimare.dataset import Dataset
import nipype.interfaces.fsl as fsl
from nimare.correct import FWECorrector
from neurosynth.base.dataset import download
from nimare.io import convert_neurosynth_to_dataset
from nilearn.datasets import load_mni152_brain_mask
from nimare.workflows.ale import ale_sleuth_workflow
from nipype.workflows.fmri.fsl.preprocess import create_susan_smooth


def macm_workflow(ns_data_dir, output_dir, prefix, mask_fn):
    # download neurosynth dataset if necessary
    dataset_file = op.join(ns_data_dir, 'neurosynth_dataset.pkl.gz')

    if not op.isfile(dataset_file):
        if not op.isdir(ns_data_dir):
            os.mkdir(ns_data_dir)
        download(ns_data_dir, unpack=True)
        ###############################################################################
        # Convert Neurosynth database to NiMARE dataset file
        # --------------------------------------------------
        dset = convert_neurosynth_to_dataset(
            op.join(ns_data_dir, 'database.txt'),
            op.join(ns_data_dir, 'features.txt'))
        dset.save(dataset_file)

    dset = Dataset.load(dataset_file)
    mask_ids = dset.get_studies_by_mask(mask_fn)
    maskdset = dset.slice(mask_ids)
    nonmask_ids = sorted(list(set(dset.ids) - set(mask_ids)))
    nonmaskdset = dset.slice(nonmask_ids)

    ale = ALE(kernel__fwhm=15)
    ale.fit(maskdset)

    corr = FWECorrector(method='permutation', n_iters=10, n_cores=-1, voxel_thresh=0.001)
    cres = corr.transform(ale.results)
    cres.save_maps(output_dir=output_dir, prefix=prefix)


def rs_workflow(rs_data_dir):
    #get participants
    ppt_df = pandas.read_csv(rs_data_dir, 'participants.tsv', sep='/t')
    for ppt in ppt_df['participant_id']:
        nii_files = os.listdir(rs_data_dir, ppt, 'func')
        for nii_fn in nii_files:
            #check to see if smoothed data exists
            if not op.isfile(rs_data_dir, 'derivatives', 'smoothed', ppt + nii_fn.split('.')[0] + '.feat', 'filtered_func_data.nii.gz'):
                #preprocessing workflow
                print('smooth data')
                std_img = fsl.maths.StdImage()
                std_img.in_file = op.join(rs_data_dir, ppt, 'func', nii_fn)
                std_img.dimension = "T"
                std_img.out_file = op.join(rs_data_dir, ppt, 'func', nii_fn.split('.')[0] + "_mask.nii.gz")
                std_img.run()

                bin_img = fsl.maths.UnaryMaths()
                bin_img.in_file = op.join(rs_data_dir, ppt, 'func', nii_fn.split('.')[0] + "_mask.nii.gz")
                bin_img.operation = "bin"
                bin_img.out_file = op.join(rs_data_dir, ppt, 'func', nii_fn.split('.')[0] + "_mask.nii.gz")
                bin_img.run()

                smooth = create_susan_smooth()
                smooth.inputs.inputnode.in_files = op.join(rs_data_dir, ppt, 'func', nii_fn)
                smooth.inputs.inputnode.fwhm = 4
                smooth.inputs.inputnode.mask_file = op.join(rs_data_dir, ppt, 'func', nii_fn.split('.')[0] + "_mask.nii.gz")
                smooth.run()

        else:
            #run analysis
            print('running roi analysis')


def cale_workflow(input_file, mask_file=None, output_dir=None, prefix=None, ns_data_dir=None, macm_data_dir=None, rs_data_dir=None):

    if mask_file == None:
        mask = load_mni152_brain_mask()

    if output_dir == None:
        output_dir = "."

    if prefix == None:
        prefix = op.basename(input_file).split('.')[0] + "_"

    if ns_data_dir == None:
        ns_data_dir = '.'

    if macm_data_dir == None:
        macm_data_dir = '.'

    if rs_data_dir == None:
        rs_data_dir = '.'


    """
    Import input file.
    If a text file or spreadsheet, run ALE workflow.
    Otherwise, start cALE workflow.
    """

    file_ext = '.'.join(op.basename(input_file).split('.')[1:])
    sheet_ext = ['txt', 'csv', 'tsv']
    img_ext = ['nii', 'nii.gz']

    if file_ext in sheet_ext:
        """
        Run ALE workflow first.
        """
        ale_sleuth_workflow(input_file, sleuth_file2=None, output_dir=output_dir,
                                prefix=prefix, n_iters=10000, v_thr=0.001,
                                fwhm=None, n_cores=-1)
        img_file = op.join(output_dir, prefix + "_ale.nii.gz")
    elif file_ext in img_ext:
        img_file = input_file
    else:
        print('Spreadsheets must be of type .txt, .csv, or .tsv. '
              'Image files must be of type .nii or .nii.gz.')


    """
    Identify cluster peaks.
    Generate spherical ROIs around cluster peaks.
    """

    # use NiPype's FSL wrapper for "cluster" to generate peaks
    cl = fsl.model.Cluster()
    cl.inputs.in_file = img_file
    cl.inputs.threshold = np.finfo(float).eps
    cl.inputs.connectivity = 26
    cl.inputs.no_table = True
    cl.inputs.peak_distance = 15
    cl.inputs.use_mm = True
    cl.inputs.out_localmax_txt_file = op.join(output_dir, prefix + "peaks-mm.txt")
    cl.run()

    # now do it again, but we want voxel indices, not mm indices
    cl.inputs.use_mm = False
    cl.inputs.out_localmax_txt_file = op.join(output_dir, prefix + "peaks-vox.txt")
    cl.run()

    #Use pandas to merge coordinate information
    df_mm = pd.read_csv(op.join(output_dir, prefix + "peaks-mm.txt"), sep = "\t", index_col = False)
    df_mm = df_mm.drop(['Cluster Index', df_mm.columns[5]], axis=1)
    df_mm.columns = df_mm.columns.str.replace('x','x-mm')
    df_mm.columns = df_mm.columns.str.replace('y','y-mm')
    df_mm.columns = df_mm.columns.str.replace('z','z-mm')
    df_vox = pd.read_csv(op.join(output_dir, prefix + "peaks-vox.txt"), sep = "\t", index_col = False)
    df_vox = df_vox.drop(['Cluster Index', df_vox.columns[5]], axis=1)
    df_final = df_mm.set_index('Value').join(df_vox.set_index('Value'))
    df_final.reset_index(drop=True, inplace=True)

    #Make spheres for each coordinate
    mask_img = mask.get_data()
    for i, row in df_final.iterrows():
        tmp_mask_img = mask_img * 0
        tmp_mask_img[row['x'], row['y'], row['z']] = 1
        tmp_roi_img = nib.Nifti1Image(tmp_mask_img, mask.affine, mask.header)
        tmp_roi_fn = op.join(output_dir, prefix + 'original-rois_{x}_{y}_{z}.nii.gz'.format(x=row['x-mm'], y=row['y-mm'], z=row['z-mm']))
        nib.save(tmp_roi_img, tmp_roi_fn)

        di = fsl.maths.DilateImage(in_file = tmp_roi_fn,
                                   operation = "mean",
                                   kernel_shape = "sphere",
                                   kernel_size = 6,
                                   out_file = tmp_roi_fn)
        di.run()


        """
        Connectivity Profiles.
        Generate MACMs using Neurosynth.
        Generate Resting-State connectivity maps using HCP data
        Generate Consensus connectivity profiles
        """
        roi_prefix = '{x}_{y}_{z}'.format(x=row['x-mm'], y=row['y-mm'], z=row['z-mm'])
        #MACMs
        # See if file already exists in MACM directory
        macm_fn = op.join(macm_data_dir, roi_prefix + '_logp_level-cluster_corr-FWE_method-permutation.nii.gz')
        if not op.isfile(macm_fn):
            macm_workflow(ns_data_dir, macm_data_dir, roi_prefix, tmp_roi_fn)

        copyfile(macm_fn, output_dir)

        #Resting-State
        rs_fn = op.join(rs_data_dir, 'derivatives', roi_prefix + '.gfeat', 'cope1.feat', 'thresh_zstat1.nii.gz')
        if not op.isfile(rs_fn):
            rs_workflow()

        copyfile(rs_fn, output_dir)
