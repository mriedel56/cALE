"""
Perform complementary ALE network workflow.
"""

import os
import numpy as np
import pandas as pd
import os.path as op
import nibabel as nib
from shutil import copyfile
from .peaks import get_peaks
from .macm import macm_workflow
import nipype.pipeline.engine as pe
from nimare.workflows.ale import ale_sleuth_workflow
from nipype.workflows.fmri.fsl.preprocess import create_susan_smooth


def cale_workflow(input_file, mask=None, output_dir=None, prefix=None, ns_data_dir=None, macm_data_dir=None, rs_data_dir=None, work_dir=None):

    if output_dir == None:
        output_dir = "."

    if prefix == None:
        prefix = op.basename(input_file).split('.')[0] + "_"

    if ns_data_dir == None:
        ns_data_dir = '.'

    if roi_data_dir == None:
        roi_data_dir = '.'

    if macm_data_dir == None:
        macm_data_dir = '.'

    if rs_data_dir == None:
        rs_data_dir = '.'

    if work_dir == None:
        work_dir = op.join('/scratch', prefix)
        if op.isdir(work_dir):
            shutil.rmtree(work_dir)
        os.makedirs(work_dir)


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
        img_file = op.join(output_dir, prefix + "_logp_level-cluster_corr-FWE_method-permutation.nii.gz")
    elif file_ext in img_ext:
        img_file = input_file
    else:
        print('Spreadsheets must be of type .txt, .csv, or .tsv. '
              'Image files must be of type .nii or .nii.gz.')


    """
    Identify cluster peaks.
    Generate spherical ROIs around cluster peaks.
    """
    og_roi_dir = op.join(output_dir, 'original', 'rois')
    if op.isdir(og_roi_dir):
        shutil.rmtree(og_roi_dir)
    os.makedirs(og_roi_dir)

    peaks_df = get_peaks(img_file, work_dir)

    #Make spheres for each coordinate
    for i, row in peaks_df.iterrows():
        roi_prefix = '{x}_{y}_{z}'.format(x=row['x-mm'], y=row['y-mm'], z=row['z-mm'])

        # See if file already exists in ROI directory
        roi_fn = op.join(roi_data_dir, roi_prefix + '.nii.gz')
        if not op.isfile(roi_fn):
            make_sphere(row['x-mm'], row['y-mm'], row['z-mm'], roi_data_dir)

        copyfile(roi_fn, og_roi_dr)
        """
        Connectivity Profiles.
        Generate MACMs using Neurosynth.
        Generate Resting-State connectivity maps using HCP data
        Generate Consensus connectivity profiles
        """
        #MACMs
        og_macm_dir = op.join(output_dir, 'original', 'macm')
        if op.isdir(og_macm_dir):
            shutil.rmtree(og_macm_dir)
        os.makedirs(og_macm_dir)

        # See if file already exists in MACM directory
        macm_fn = op.join(macm_data_dir, roi_prefix + '_logp_level-cluster_corr-FWE_method-permutation.nii.gz')
        if not op.isfile(macm_fn):
            macm_workflow(ns_data_dir, macm_data_dir, roi_prefix, tmp_roi_fn)

        copyfile(macm_fn, og_macm_dir)

        #Resting-State
        rs_fn = op.join(rs_data_dir, 'derivatives', roi_prefix + '.gfeat', 'cope1.feat', 'thresh_zstat1.nii.gz')
        if not op.isfile(rs_fn):
            rs_workflow(rs_data_dir, roi_prefix, tmp_roi_fn)

        copyfile(rs_fn, output_dir)
