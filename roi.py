import os
import os.path as op
import nibabel as nib
import nipype.interfaces.fsl as fsl
from nilearn.datasets import load_mni152_brain_mask

def make_sphere(x, y, z, roi_prefix, output_dir):

    mask = load_mni152_brain_mask()
    mask_img = mask.get_data()

    tmp_mask_img = mask_img * 0
    tmp_mask_img[x,y,z] = 1
    tmp_roi_img = nib.Nifti1Image(tmp_mask_img, mask.affine, mask.header)
    tmp_roi_fn = op.join(output_dir, roi_prefix + '.nii.gz')
    nib.save(tmp_roi_img, tmp_roi_fn)

    di = fsl.maths.DilateImage(in_file = tmp_roi_fn,
                               operation = "mean",
                               kernel_shape = "sphere",
                               kernel_size = 6,
                               out_file = tmp_roi_fn)
    di.run()
