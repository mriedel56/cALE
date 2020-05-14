import os
import shutil
import pandas as pd
import os.path as op
import nipype.interfaces.fsl as fsl

def get_peaks(img_file, work_dir):

    # use NiPype's FSL wrapper for "cluster" to generate peaks
    cl = fsl.model.Cluster()
    cl.inputs.in_file = img_file
    cl.inputs.threshold = np.finfo(float).eps
    cl.inputs.connectivity = 26
    cl.inputs.no_table = True
    cl.inputs.peak_distance = 15
    cl.inputs.use_mm = True
    cl.inputs.out_localmax_txt_file = op.join(work_dir, "peaks-mm.txt")
    cl.run()

    # now do it again, but we want voxel indices, not mm indices
    cl.inputs.use_mm = False
    cl.inputs.out_localmax_txt_file = op.join(work_dir, "peaks-vox.txt")
    cl.run()

    #Use pandas to merge coordinate information
    df_mm = pd.read_csv(op.join(work_dir, "peaks-mm.txt"), sep = "\t", index_col = False)
    df_mm = df_mm.drop(['Cluster Index', df_mm.columns[5]], axis=1)
    df_mm.columns = df_mm.columns.str.replace('x','x-mm')
    df_mm.columns = df_mm.columns.str.replace('y','y-mm')
    df_mm.columns = df_mm.columns.str.replace('z','z-mm')
    df_vox = pd.read_csv(op.join(work_dir, "peaks-vox.txt"), sep = "\t", index_col = False)
    df_vox = df_vox.drop(['Cluster Index', df_vox.columns[5]], axis=1)
    df_final = df_mm.set_index('Value').join(df_vox.set_index('Value'))
    df_final.reset_index(drop=True, inplace=True)

    return df_final
