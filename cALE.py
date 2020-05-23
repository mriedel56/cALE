"""
Perform complementary ALE network workflow.
"""

import os
import numpy as np
import os.path as op
import shutil
from peaks import get_peaks
from nimare.workflows.ale import ale_sleuth_workflow
from connectivity import connectivity_workflow


def cale_workflow(input_file, mask=None, output_dir=None, prefix=None, roi_data_dir=None, ns_data_dir=None, macm_data_dir=None, rsfc_data_dir=None, con_data_dir=None, work_dir=None):

    if output_dir == None:
        output_dir = "."

    if prefix == None:
        prefix = op.basename(input_file).split('.')[0] + "_"

    if ns_data_dir == None:
        ns_data_dir = '.'

    if con_data_dir == None:
        con_data_dir = '.'

    if roi_data_dir == None:
        roi_data_dir = '.'
    if not op.isdir(roi_data_dir):
        os.makedirs(roi_data_dir)

    if macm_data_dir == None:
        macm_data_dir = '.'
    if not op.isdir(macm_data_dir):
        os.makedirs(macm_data_dir)

    if rsfc_data_dir == None:
        rsfc_data_dir = '.'
    if not op.isdir(rsfc_data_dir):
        os.makedirs(rsfc_data_dir)

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

    og_ale_dir = op.join(output_dir, 'original', 'ale')
    if op.isdir(og_ale_dir):
        shutil.rmtree(og_ale_dir)
    os.makedirs(og_ale_dir)

    if file_ext in sheet_ext:
        """
        Run ALE workflow first.
        """
        ale_sleuth_workflow(input_file, sleuth_file2=None, output_dir=og_ale_dir,
                                prefix=prefix, n_iters=10000, v_thr=0.001,
                                fwhm=None, n_cores=-1)
        img_file = op.join(og_ale_dir, prefix + "_logp_level-cluster_corr-FWE_method-permutation.nii.gz")
    elif file_ext in img_ext:
        shutil.copy(input_file, og_ale_dir)
        img_file = op.join(og_ale_dir, op.basename(input_file))
    else:
        print('Spreadsheets must be of type .txt, .csv, or .tsv. '
              'Image files must be of type .nii or .nii.gz.')


    """
    Identify cluster peaks.
    Generate spherical ROIs around cluster peaks.
    """
    peaks_df = get_peaks(img_file, og_ale_dir)

    og_roi_dir = op.join(output_dir, 'original', 'rois')
    if op.isdir(og_roi_dir):
        shutil.rmtree(og_roi_dir)
    os.makedirs(og_roi_dir)

    #run connectivity workflow for each set of coordinates in the dataframe
    for i, row in peaks_df.iterrows():
        connectivity_workflow(row, output_dir, roi_data_dir, macm_data_dir, rs_data_dir, con_data_dir)

    com_ale_dir = op.join(output_dir, 'complementary', 'ale')
    if op.isdir(com_ale_dir):
        shutil.rmtree(com_ale_dir)
    os.makedirs(com_ale_dir)

    #sum consensus connectivity maps
    image_sum(og_con_dir com_ale_dir)

    cale_fn = glob(op.join(com_ale_dir, 'cALE_thresh-*.nii.gz'))
    #identify cluster peaks in cALE image
    com_peaks_df = get_peaks(cale_fn, work_dir)
