import os
import os.path as op
from glob import glob
from nipype.interfaces.fsl.Info import standard_image
import nibabel as nib
import numpy as np
from sklearn.cluster import AgglomerativeClustering


def clustering_workflow(in_dir, nclust, method):
    mni_img = nib.load(standard_image('MNI152_T1_2mm_brain_mask.nii.gz'))
    mni_img_data = mni_img.get_fdata()

    fns = glob(in_dir)
    for i, tmp_fn in enumerate(fns):
        tmp_img = nib.load(tmp_fn)
        tmp_img_data = tmp_img.get_fdata()

        if i == 0:
            data_array = tmp_img_data[mni_img_data > 0]
        else:
            data_array = np.vstack((data_array, tmp_img_data[mni_img_data > 0]))

    if method == 'hierarchical':
        model = AgglomerativeClustering(n_clusters=nclust, linkage="ward", affinity="euclidean")
        model.fit(data_array)

        for tmp_n in range(0,nclust,1):
            tmp_array = data_array[model.labels_ == tmp_n]

            tmp_array_avg = np.mean(tmp_array, axis=0)
            tmp_avg_img = mni_img_data * 0
            tmp_avg_img[mni_img_data > 0] = tmp_array_avg
            tmp_avg = nib.Nifti1Image(tmp_avg_img, mni_img.affine, mni_img.header)
            tmp_avg_fn = op.join(in_dir, 'clustering', str(nclust), '{0}-avg.nii.gz'.format(tmp_n))
            nib.save(tmp_avg, tmp_avg_fn)

            tmp_array_min = np.amin(tmp_array, axis=0)
            tmp_min_img = mni_img_data * 0
            tmp_min_img[mni_img_data > 0] = tmp_array_min
            tmp_min = nib.Nifti1Image(tmp_min_img, mni_img.affine, mni_img.header)
            tmp_min_fn = op.join(in_dir, 'clustering', str(nclust), '{0}-min.nii.gz'.format(tmp_n))
            nib.save(tmp_min, tmp_min_fn)

    elif method == 'kmeans':
        print('do kmeans model')

    else:
        print('Not an option!')
