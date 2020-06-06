import os
import os.path as op
from nipype.interfaces.fsl.Info import standard_image
import nibabel as nib
import numpy as np
from sklearn.cluster import AgglomerativeClustering


def clustering_workflow(fnames, nclust, method):
    mni_img = nib.load(standard_image('MNI152_T1_2mm_brain_mask.nii.gz'))
    mni_img_data = mni_img.get_fdata()

    for i, tmp_fn in enumerate(fnames):
        tmp_img = nib.load(tmp_fn)
        tmp_img_data = tmp_img.get_fdata()

        if i == 0:
            data_array = tmp_img_data[mni_img_data > 0]
        else:
            data_array = np.vstack((data_array, tmp_img_data[mni_img_data > 0]))

    if method == 'hierarchical':
        model = AgglomerativeClustering(n_clusters=nclust, linkage="ward", affinity="euclidean")
        model.fit(data_array)

        return model

    elif method == 'kmeans':
        print('kmeans model not implemented yet')

    else:
        print('Not an option!')
