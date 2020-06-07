"""
Perform complementary ALE network workflow.
"""

import os
import numpy as np
import os.path as op
import shutil
import numpy as np
from glob import glob
from peaks import get_peaks
from nipype.interfaces.fsl.utils import ImageMaths
from nipype.interfaces.fsl.utils import Merge
from nimare.workflows.ale import ale_sleuth_workflow
from connectivity import connectivity_workflow
from complementary import cale
from clustering import clustering_workflow


def cale_workflow(input_file, output_dir=None, prefix=None, data_dir=None, work_dir=None):

    if prefix == None:
        prefix = op.basename(input_file).split('.')[0]

    if output_dir == None:
        output_dir = op.join(op.abspath(input_file), prefix)
    if op.isdir(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    if work_dir == None:
        work_dir = op.join(output_dir, 'workdir')
    if op.isdir(work_dir):
        shutil.rmtree(work_dir)
    os.makedirs(work_dir)

    if data_dir == None:
        data_dir = work_dir = op.join(output_dir, 'data')
    if not op.isdir(data_dir):
        os.makedirs(data_dir)

    roi_data_dir = op.join(data_dir, 'roi')
    if not op.isdir(roi_data_dir):
        os.makedirs(roi_data_dir)

    macm_data_dir = op.join(data_dir, 'macm')
    if not op.isdir(macm_data_dir):
        os.makedirs(macm_data_dir)

    rsfc_data_dir = op.join(data_dir, 'rsfc')
    if not op.isdir(rsfc_data_dir):
        os.makedirs(rsfc_data_dir)

    con_data_dir = op.join(data_dir, 'consensus')
    if not op.isdir(con_data_dir):
        os.makedirs(con_data_dir)

    ns_data_dir = op.join(data_dir, 'neurosynth_data')
    if not op.isdir(ns_data_dir):
        os.makedirs(ns_data_dir)


    """
    Import input file.
    If a text file or spreadsheet, run ALE workflow.
    Otherwise, start cALE workflow.
    """

    file_ext = '.'.join(op.basename(input_file).split('.')[1:])
    sheet_ext = ['txt', 'csv', 'tsv']
    img_ext = ['nii', 'nii.gz']

    og_ale_dir = op.join(output_dir, 'original', 'ale')
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
    """
    peaks_df = get_peaks(img_file, og_ale_dir)

    og_roi_dir = op.join(output_dir, 'original', 'rois')
    os.makedirs(og_roi_dir)

    #run connectivity workflow for each set of coordinates in the dataframe
    for i, row in peaks_df.iterrows():

      roi_prefix = '{x}_{y}_{z}'.format(x=row['x'], y=row['y'], z=row['z'])

      # See if file already exists in ROI directory
      roi_fn = op.join(roi_data_dir, roi_prefix + '.nii.gz')
      if not op.isfile(roi_fn):
          make_sphere(row['x'], row['y'], row['z'], roi_data_dir)

      shutil.copy(roi_fn, og_roi_dir)

      connectivity_workflow(roi_fn, op.join(output_dir, 'original'), data_dir, ['macm', 'rsfc'])

    com_ale_dir = op.join(output_dir, 'complementary', 'ale')
    os.makedirs(com_ale_dir)

    #sum consensus connectivity maps
    cale(og_con_dir, com_ale_dir)

    cale_fn = glob(op.join(com_ale_dir, 'cALE_thresh-*.nii.gz'))
    #identify cluster peaks in cALE image
    com_peaks_df = get_peaks(cale_fn, work_dir)

    com_roi_dir = op.join(output_dir, 'complementary', 'rois')
    os.makedirs(com_roi_dir)

    #run connectivity workflow for each set of coordinates in the dataframe
    for i, row in com_peaks_df.iterrows():

        roi_prefix = '{x}_{y}_{z}'.format(x=row['x'], y=row['y'], z=row['z'])

        # See if file already exists in ROI directory
        roi_fn = op.join(roi_data_dir, roi_prefix + '.nii.gz')
        if not op.isfile(roi_fn):
            make_sphere(row['x'], row['y'], row['z'], roi_data_dir)

        shutil.copy(roi_fn, com_roi_dir)

        connectivity_workflow(roi_fn, op.join(output_dir, 'complementary'), data_dir, ['macm', 'rsfc'])

    for nclust in range(2,9,1):

        for tmp_conn in ['macm', 'rsfc'']:
            tmp_dir = op.join(output_dir, 'complementary', tmp_conn)
            fnames_unthresh = sorted(glob(op.join(tmp_dir, '*thresh-none.nii.gz'))
            fnames_thresh = sorted(glob(op.join(tmp_dir, '*thresh-*.nii.gz'))

            model = clustering_workflow(fnames_unthresh, nclust, 'hierarchical')

            for tmp_n in range(0,nclust,1):

                os.makedirs(op.join(tmp_dir, 'clustering', str(nclust)))

                #calculate mean of unthresholded images for each cluster
                tmp_fnames_unthresh = fnames_unthresh[model.labels_ == tmp_n]

                merger = Merge()
                merger.inputs.in_files = tmp_fnames_unthresh
                merger.inputs.dimension = 't'
                merger.inputs.merged_file = op.join(in_dir, 'clustering', str(nclust), '{0}-avg.nii.gz'.format(tmp_n))
                merger.run()

                meanimg = ImageMaths()
                meanimg.inputs.in_file = op.join(in_dir, 'clustering', str(nclust), '{0}-avg.nii.gz'.format(tmp_n))
                meanimg.inputs.op_string = '-Tmean'
                meanimg.inputs.out_file = op.join(in_dir, 'clustering', str(nclust), '{0}-avg.nii.gz'.format(tmp_n))

                #calculate minimum of thresholded images for each cluster
                tmp_fnames_thresh = fnames_thresh[model.labels_ == tmp_n]

                merger = Merge()
                merger.inputs.in_files = tmp_fnames_thresh
                merger.inputs.dimension = 't'
                merger.inputs.merged_file = op.join(in_dir, 'clustering', str(nclust), '{0}-min.nii.gz'.format(tmp_n))
                merger.run()

                meanimg = ImageMaths()
                meanimg.inputs.in_file = op.join(in_dir, 'clustering', str(nclust), '{0}-min.nii.gz'.format(tmp_n))
                meanimg.inputs.op_string = '-Tmean'
                meanimg.inputs.out_file = op.join(in_dir, 'clustering', str(nclust), '{0}-min.nii.gz'.format(tmp_n))
