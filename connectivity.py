import os
import os.path as op
import shutil
from roi import make_sphere
from macm import macm_workflow
from rs import rs_workflow
from consensus import con_workflow


def connectivity_workflow(row, output_dir, roi_data_dir, macm_data_dir, rs_data_dir, con_data_dir):

  roi_prefix = '{x}_{y}_{z}'.format(x=row['x'], y=row['y'], z=row['z'])

  # See if file already exists in ROI directory
  roi_fn = op.join(roi_data_dir, roi_prefix + '.nii.gz')
  if not op.isfile(roi_fn):
      make_sphere(row['x'], row['y'], row['z'], roi_data_dir)

  shutil.copy(roi_fn, og_roi_dir)
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
      macm_workflow(ns_data_dir, macm_data_dir, roi_prefix, roi_fn)

  shutil.copy(macm_fn, og_macm_dir)
  exit()

  #Resting-State
  og_rsfc_dir = op.join(output_dir, 'original', 'rsfc')
  if op.isdir(og_rsfc_dir):
      shutil.rmtree(og_rsfc_dir)
  os.makedirs(og_rsfc_dir)

  rs_fn = op.join(rs_data_dir, 'derivatives', roi_prefix + '.gfeat', 'cope1.feat', 'thresh_zstat1.nii.gz')
  if not op.isfile(rs_fn):
      rs_workflow(rs_data_dir, roi_prefix, tmp_roi_fn, work_dir)

  shutil.copy(rs_fn, og_rsfc_dir)

  #Make binary consensus connectivity profiles
  og_con_dir = op.join(output_dir, 'original', 'consensus')
  if op.isdir(og_con_dir):
      shutil.rmtree(og_con_dir)
  os.makedirs(og_con_dir)

  con_fn = op.join(con_data_dir, roi_prefix + '.nii.gz')
  if not op.isfile(con_fn):
      con_workflow(macm_fn, rs_fn, roi_prefix, con_data_dir)

  shutil.copy(con_fn, og_con_dir)
