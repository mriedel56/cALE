import os
import shutil
import pandas as pd
import os.path as op
import numpy as np
import nipype.interfaces.fsl as fsl

def get_peaks(img_file, output_dir):

    out_fn = op.join(output_dir, '{0}_peaks.txt'.format(op.basename(img_file).split('.')[0]))
    # use NiPype's FSL wrapper for "cluster" to generate peaks
    cl = fsl.model.Cluster()
    cl.inputs.in_file = img_file
    cl.inputs.threshold = np.finfo(float).eps
    cl.inputs.connectivity = 26
    cl.inputs.no_table = True
    cl.inputs.peak_distance = 15
    cl.inputs.use_mm = True
    cl.inputs.out_localmax_txt_file = out_fn
    cl.run()

    #Use pandas to clean up file
    df_mm = pd.read_csv(out_fn), sep = "\t", index_col = False)
    df_mm = df_mm.drop(['Cluster Index', 'Value', df_mm.columns[5]], axis=1)

    df_mm.to_csv(out_fn, sep=' ', index=False)
