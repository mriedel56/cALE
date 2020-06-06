import os
import shutil
import pandas as pd
import os.path as op
import nipype.interfaces.fsl as fsl
from atlasreader import atlasreader

def get_peaks(img_file, output_dir):

    stats = fsl.utils.ImageStats()
    stats.inputs.in_file = img_file
    stats.inputs.op_string = '-r'
    min = stats.run()
    minval = min.outputs.out_stat[0]

    out_fn = op.join(output_dir, '{0}_clusterinfo.tsv'.format(op.basename(img_file).split('.')[0]))
    # use NiPype's FSL wrapper for "cluster" to generate peaks
    cl = fsl.model.Cluster()
    cl.inputs.in_file = img_file
    cl.inputs.threshold = minval
    cl.inputs.connectivity = 26
    cl.inputs.no_table = True
    cl.inputs.peak_distance = 15
    cl.inputs.use_mm = True
    cl.inputs.out_localmax_txt_file = out_fn
    cl.run()

    #Use pandas to clean up file
    df_mm = pd.read_csv(out_fn, sep = "\t", index_col = False)
    df_mm = df_mm.drop([df_mm.columns[5]], axis=1)
    for i, row in df_mm.iterrows():
        tmplabel = atlasreader.read_atlas_peak('aal', [row['x'], row['y'], row['z']])
        if i == 0:
            if tmplabel.split('_')[-1] in ['L', 'R']:
                hemis = [tmplabel.split('_')[-1]]
                labels = [' '.join(tmplabel.split('_')[:-1])]
            else:
                hemis = ['']
                labels = [' '.join(tmplabel.split('_'))]
        else:
            if tmplabel.split('_')[-1] in ['L', 'R']:
                hemis.append(tmplabel.split('_')[-1])
                labels.append(' '.join(tmplabel.split('_')[:-1]))
            else:
                hemis.append('')
                labels.append(' '.join(tmplabel.split('_')))

    df_mm['Hemisphere'] = hemis
    df_mm['Label'] = labels
    df_mm.to_csv(out_fn, sep='\t', index=False)

    return df_mm
