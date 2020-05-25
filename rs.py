from __future__ import division

from nipype.interfaces import fsl as fsl          # fsl
from nipype.interfaces import utility as util     # utility
from nipype.pipeline import engine as pe          # pypeline engine
import nipype.interfaces.io as nio
import os
import os.path as op
import shutil
from glob import glob
import numpy as np


def pickfirst(files):
    if isinstance(files, list):
        return files[0]
    else:
        return files


def rs_preprocess(in_file, fwhm, work_dir):

    from nipype.workflows.fmri.fsl.preprocess import create_featreg_preproc

    rs_preproc_workflow = pe.Workflow(name="rs_preproc_workflow")
    rs_preproc_workflow.base_dir = work_dir

    featproc = create_featreg_preproc(name="featproc", highpass=False, whichvol='first')
    featproc.inputs.inputspec.func = in_file
    featproc.inputs.inputspec.fwhm = fwhm

    #remove motion correction nodes
    moco = featproc.get_node('realign')
    moplot = featproc.get_node('plot_motion')
    featproc.remove_nodes([moco, moplot])

    #remove connections dependent on motion correction
    featproc.disconnect(featproc.get_node('img2float'), 'out_file', featproc.get_node('motion_correct'), 'in_file')
    featproc.disconnect(featproc.get_node('extract_ref'), 'roi_file', featproc.get_node('motion_correct'), 'ref_file')
    featproc.disconnect(featproc.get_node('motion_correct'), ('mean_img', pickfirst), featproc.get_node('outputnode'), 'reference')
    featproc.disconnect(featproc.get_node('motion_correct'), 'par_file', featproc.get_node('outputnode'), 'motion_parameters')
    featproc.disconnect(featproc.get_node('motion_correct'), 'out_file', featproc.get_node('outputnode'), 'realigned_files')
    featproc.disconnect(featproc.get_node('motion_correct'), 'par_file', featproc.get_node('plot_motion'), 'in_file')
    featproc.disconnect(featproc.get_node('plot_motion'), 'out_file', featproc.get_node('outputnode'), 'motion_plots')
    featproc.disconnect(featproc.get_node('motion_correct'), ('out_file', pickfirst), featproc.get_node('meanfunc'), 'in_file')
    featproc.disconnect(featproc.get_node('motion_correct'), 'out_file', featproc.get_node('maskfunc'), 'in_file')
    featproc.disconnect(featproc.get_node('motion_correct'), 'out_file', featproc.get_node('medianval'), 'in_file')
    featproc.disconnect(featproc.get_node('motion_correct'), 'out_file', featproc.get_node('maskfunc2'), 'in_file')

    #add connections to fill in where motion correction files would have been entered
    featproc.connect(featproc.get_node('img2float'), ('out_file', pickfirst), featproc.get_node('meanfunc'), 'in_file')
    featproc.connect(featproc.get_node('img2float'), ('out_file', pickfirst), featproc.get_node('maskfunc'), 'in_file')
    featproc.connect(featproc.get_node('img2float'), ('out_file', pickfirst), featproc.get_node('medianval'), 'in_file')
    featproc.connect(featproc.get_node('img2float'), ('out_file', pickfirst), featproc.get_node('maskfunc2'), 'in_file')

    datasink = pe.Node(nio.DataSink(), name='sinker')
    datasink.inputs.base_directory = work_dir

    rs_preproc_workflow.connect(featproc, 'meanscale.out_file', datasink, 'gms')
    rs_preproc_workflow.connect(featproc, 'dilatemask.out_file', datasink, 'mask')

    rs_preproc_workflow.run()


def rs_firstlevel(name="rsworkflow", outputdir=None):

    import nipype.algorithms.modelgen as model  # model generation
    from niflow.nipype1.workflows.fmri.fsl import create_modelfit_workflow


    level1_workflow = pe.Workflow(name='level1flow')

    modelfit = create_modelfit_workflow()
    modelfit.get_node('modelestimate').inputs.smooth_autocorr = False
    modelfit.get_node('modelestimate').inputs.autocorr_noestimate = True
    modelfit.get_node('modelestimate').inputs.mask_size = 0
    modelspec = pe.Node(model.SpecifyModel(), name="modelspec")

    inputnode = pe.Node(interface=util.IdentityInterface(fields=['func',
                                                                 'subjectinfo']),
                        name='inputspec')

    """
    This node will write out image files in output directory
    """
    datasink = pe.Node(nio.DataSink(), name='sinker')
    datasink.inputs.base_directory = outputdir

    level1_workflow.connect(
        [(inputnode, modelspec, [('func', 'functional_runs')]),
         (inputnode, modelspec, [('subjectinfo', 'subject_info')]),
         (modelspec, modelfit, [('session_info', 'inputspec.session_info')]),
         (inputnode, modelfit, [('func', 'inputspec.functional_data')])])


    cont1 = ['corr', 'T', ['mean', 'roi'], [0,1]]
    contrasts = [cont1]

    modelspec.inputs.input_units = 'secs'
    modelspec.inputs.time_repetition = 0.72
    modelspec.inputs.high_pass_filter_cutoff = 0

    modelfit.inputs.inputspec.interscan_interval = 0.72
    modelfit.inputs.inputspec.bases = {'none': {'none': None}}
    modelfit.inputs.inputspec.contrasts = contrasts
    modelfit.inputs.inputspec.model_serial_correlations = False
    modelfit.inputs.inputspec.film_threshold = 1000

    return level1_workflow

#def rs_secondlevel():

#def rs_grouplevel():


def rs_workflow(rs_data_dir, roi_prefix, tmp_roi_fn, work_dir):

    from nipype.interfaces.base import Bunch

    os.makedirs(op.join(work_dir, 'rsfc'))

    #get participants
    ppt_df = pandas.read_csv(rs_data_dir, 'participants.tsv', sep='/t')
    for ppt in ppt_df['participant_id']:
        nii_files = os.listdir(op.join(rs_data_dir, ppt, 'func'))
        for nii_fn in nii_files:

            #check to see if smoothed data exists
            smooth_fn = op.join(rs_data_dir, 'derivatives', 'smoothed', ppt, nii_fn.split('.')[0], '{0}_smooth.nii.gz'.format(nii_fn.split('.')[0]))
            if not op.isfile(smooth_fn):
                nii_work_dir = op.join(work_dir, 'rsfc', nii_fn.split('.')[0])

                smooth_flow = rs_preprocess(op.join(rs_data_dir, ppt, 'func', nii_fn), 4, nii_work_dir)

                gms_fn = glob(op.join(nii_work_dir, 'gms', '_meanscale0', '*_gms.nii.gz'))
                shutil.copyfile(gms_fn, smooth_fn)
                mask = op.join(rs_data_dir, 'derivatives', 'smoothed', ppt, nii_fn.split('.')[0], '{0}_mask.nii.gz'.format(nii_fn.split('.')[0]))
                mask_fn = glob(op.join(nii_work_dir, 'mask', '_dilatemask0', '*_dil.nii.gz'))
                shutil.copyfile(mask_fn, smooth_fn)

            else:
                #run analysis
                roi_out_dir = op.join(rs_data_dir, 'derivatives', roi_prefix, ppt, nii_fn.split('.')[0])

                mask = op.join(rs_data_dir, 'derivatives', 'smoothed', ppt, nii_fn.split('.')[0], '{0}_mask.nii.gz'.format(nii_fn.split('.')[0]))
                meants = fsl.utils.ImageMeants()
                meants.in_file = smooth_fn
                meants.mask = tmp_roi_fn
                meants.out_file = op.join(roi_out_dir, '{0}_{1}.txt'.format(nii_fn.split('.')[0], op.basename(tmp_roi_fn).split('.')[0]))
                meants.run()

                roi_ts = np.atleast_2d(np.loadtxt(op.join(roi_out_dir, '{0}.{1}.txt'.format(op.basename(tmp_roi_fn).split('.')[0], nii_fn.split('.')[0]))))
                subject_info = Bunch(conditions=['mean'], onsets=[list(np.arange(0,0.72*len(roi_ts[0]),0.72))], durations=[[0.72]], amplitudes=[np.ones(len(roi_ts[0]))], regressor_names=['roi'], regressors=[roi_ts[0]])

                firstlevel = rs_firstlevel(name="firstlevel", outputdir=roi_out_dir)
                firstlevel.inputs.inputspec.func = smooth_fn
                firstlevel.inputs.inputspec.subjectinfo = subject_info

                firstlevel.run()


        if len(nii_files)>1:
            rs_secondlevel()

    rs_grouplevel()
