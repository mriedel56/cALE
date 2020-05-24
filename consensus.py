import os
import os.path as op
from nipype.interfaces.fsl.maths import Merge
from nipype.interfaces.fsl.maths import MultiImageMaths


def con_workflow(macm_fn, rsfc_fn, prefix, output_dir):

    merger = Merge()
    merger.inputs.in_files = [macm_fn, rsfc_fn]
    merger.inputs.dimension = 't'
    merger.inputs.merged_file = op.join(output_dir, '{0}.nii.gz'.format(prefix))

    maths = MultImageMaths()
    maths.inputs.in_file = op.join(output_dir, '{0}.nii.gz'.format(prefix))
    maths.inputs.op_string = "-bin -Tmin"
    maths.inputs.out_file = op.join(output_dir, '{0}.nii.gz'.format(prefix))
