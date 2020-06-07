import os
import os.path as op
import shutil
from roi import make_sphere
from macm import macm_workflow
from rs import rs_workflow
from consensus import con_workflow


def connectivity_workflow(roi_fn, output_dir, data_dir, mod_list):

  """
  Connectivity Profiles.
  Generate MACMs using Neurosynth.
  Generate Resting-State connectivity maps using HCP data
  Generate Consensus connectivity profiles
  """
  roi_prefix = op.basename(roi_fn).split('.')[0]
  
  for modality in sorted(mod_list):
      mod_output_dir = op.join(output_dir, modality)
      if op.isdir(mod_output_dir):
          shutil.rmtree(mod_output_dir)
      os.makedirs(mod_output_dir)

      if modality == 'macm':
          #MACMs
          # See if file already exists in MACM directory
          macm_data_dir = op.join(data_dir, 'macm')
          ns_data_dir = op.join(data_dir, 'neurosynth_data')
          macm_fn = op.join(macm_data_dir, op.basename(roi_fn).split('.')[0] + '_logp_level-cluster_corr-FWE_method-permutation.nii.gz')
          if not op.isfile(macm_fn):
              macm_workflow(ns_data_dir, macm_data_dir, roi_fn)

          shutil.copy(macm_fn, mod_output_dir)
          exit()

      if modality == 'rsfc':
          #Resting-State
          rs_data_dir = op.join(data_dir, 'rsfc')
          rs_fn = op.join(rs_data_dir, 'derivatives', roi_prefix + '.gfeat', 'cope1.feat', 'thresh_zstat1.nii.gz')
          if not op.isfile(rs_fn):
              rs_workflow(rs_data_dir, roi_prefix, tmp_roi_fn, work_dir)

          shutil.copy(rs_fn, mod_output_dir)

  if sorted(mod_list) == ['macm', 'rsfc']:
      #Make binary consensus connectivity profiles
      mod_output_dir = op.join(output_dir, 'consensus')
      if op.isdir(mod_output_dir):
          shutil.rmtree(mod_output_dir)
      os.makedirs(mod_output_dir)

      con_data_dir = op.join(data_dir, 'consensus')
      con_fn = op.join(con_data_dir, roi_prefix + '.nii.gz')
      if not op.isfile(con_fn):
          con_workflow(macm_fn, rs_fn, roi_prefix, con_data_dir)

      shutil.copy(con_fn, mod_output_dir)
