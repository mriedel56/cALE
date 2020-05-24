import os
import os.path as op
import glob
import numpy as np
from nipype.interfaces.fsl import Merge
from nipype.interfaces.fsl.maths import MeanImage
from nipype.interfaces.fsl.maths import MultiImageMaths
from nipype.interfaces.fsl.maths import Threshold


def cale(input_dir, output_dir):

  fns = glob(op.join(input_dir, '*.nii.gz'))

  merger = Merge()
  merger.inputs.in_files = fns
  merger.inputs.dimension = 't'
  merger.inputs.merged_file = op.join(output_dir, 'cALE.nii.gz')

  meanimg = MeanImage()
  meanimg.inputs.in_file = op.join(output_dir, 'cALE.nii.gz')
  meanimg.inputs.dimensions = 'T'
  meanimg.inputs.out_file = op.join(output_dir, 'cALE.nii.gz')

  maths = MultiImageMaths()
  maths.inputs.in_file = op.join(output_dir, 'cALE.nii.gz')
  maths.inputs.op_string = '-mul {0}'.format(len(fns))
  maths.inputs.out_file = op.join(output_dir, 'cALE.nii.gz')

  thresh = Threshold()
  thresh.inputs.in_file = op.join(output_dir, 'cALE.nii.gz')
  thresh.inputs.thresh = np.floor(len(fns)/2)
  thresh.inputs.direction = 'below'
  thresh.inputs.out_file = op.join(output_dir, 'cALE_thresh-{0}.nii.gz'.format(np.floor(len(fns)/2)))
