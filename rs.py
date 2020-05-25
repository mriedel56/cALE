from __future__ import division

         # fsl
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


def rs_preprocess(in_file, fwhm, work_dir, output_dir):

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

    #copy data to directory
    gms_fn = glob(op.join(work_dir, 'gms', '_meanscale0', '*_gms.nii.gz'))
    mask_fn = glob(op.join(work_dir, 'mask', '_dilatemask0', '*_dil.nii.gz'))
    gms_fn2 = op.join(output_dir, '{0}_smooth.nii.gz'.format(op.basename(in_file).split('.')[0]))
    mask_fn2 = op.join(output_dir, '{0}_mask.nii.gz'.format(op.basename(in_file).split('.')[0]))

    shutil.copyfile(gms_fn, gms_fn2)
    shutil.copyfile(mask_fn, mask_fn2)

    shutil.rmtree(work_dir)


def rs_firstlevel(unsmooth_fn, smooth_fn, roi_mask, output_dir, work_dir):

    import nipype.algorithms.modelgen as model  # model generation
    from niflow.nipype1.workflows.fmri.fsl import create_modelfit_workflow
    from nipype.interfaces import fsl as fsl
    from nipype.interfaces.base import Bunch

    meants = fsl.utils.ImageMeants()
    meants.inputs.in_file = unsmooth_fn
    meants.inputs.mask = roi_mask
    meants.inputs.out_file = op.join(work_dir, '{0}_{1}.txt'.format(unsmooth_fn.split('.')[0], op.basename(roi_mask).split('.')[0]))
    meants.cmdline
    meants.run()

    roi_ts = np.atleast_2d(np.loadtxt(op.join(work_dir, '{0}_{1}.txt'.format(unsmooth_fn.split('.')[0], op.basename(roi_mask).split('.')[0]))))
    subject_info = Bunch(conditions=['mean'], onsets=[list(np.arange(0,0.72*len(roi_ts[0]),0.72))], durations=[[0.72]], amplitudes=[np.ones(len(roi_ts[0]))], regressor_names=['roi'], regressors=[roi_ts[0]])

    level1_workflow = pe.Workflow(name='level1flow')

    inputnode = pe.Node(interface=util.IdentityInterface(fields=['func',
                                                                 'subjectinfo']),
                        name='inputspec')

    modelspec = pe.Node(model.SpecifyModel(), name="modelspec")
    modelspec.inputs.input_units = 'secs'
    modelspec.inputs.time_repetition = 0.72
    modelspec.inputs.high_pass_filter_cutoff = 0

    modelfit = create_modelfit_workflow()
    modelfit.get_node('modelestimate').inputs.smooth_autocorr = False
    modelfit.get_node('modelestimate').inputs.autocorr_noestimate = True
    modelfit.get_node('modelestimate').inputs.mask_size = 0
    modelfit.inputs.inputspec.interscan_interval = 0.72
    modelfit.inputs.inputspec.bases = {'none': {'none': None}}
    modelfit.inputs.inputspec.model_serial_correlations = False
    modelfit.inputs.inputspec.film_threshold = 1000
    contrasts = [['corr', 'T', ['mean', 'roi'], [0,1]]]
    modelfit.inputs.inputspec.contrasts = contrasts

    """
    This node will write out image files in output directory
    """
    datasink = pe.Node(nio.DataSink(), name='sinker')
    datasink.inputs.base_directory = work_dir

    level1_workflow.connect(
        [(inputnode, modelspec, [('func', 'functional_runs')]),
         (inputnode, modelspec, [('subjectinfo', 'subject_info')]),
         (modelspec, modelfit, [('session_info', 'inputspec.session_info')]),
         (inputnode, modelfit, [('func', 'inputspec.functional_data')]),
         (modelfit, datasink, [('outputspec.copes','copes'), ('outputspec.varcopes','varcopes'), ('outputspec.dof_file','dof_file'), ('outputspec.zfiles','zfiles')])])

    level1_workflow.inputs.inputspec.func = smooth_fn
    level1_workflow.inputs.inputspec.subjectinfo = subject_info
    level1_workflow.base_dir = work_dir

    level1_workflow.run()

    #copy data to directory
    shutil.rmtree(op.join(work_dir, 'level1flow'))
    files_to_copy = glob(op.join(work_dir, '*', '_modelestimate0', '*'))
    for tmp_fn in files_to_copy:
        shutil.copy(tmp_fn, output_dir)

    shutil.rmtree(work_dir)

#def rs_secondlevel():

#def rs_grouplevel():


def rs_workflow(rs_data_dir, roi_prefix, roi_mask, work_dir):

    from nipype.interfaces.base import Bunch

    os.makedirs(op.join(work_dir, 'rsfc'))

    #get participants
    ppt_df = pandas.read_csv(rs_data_dir, 'participants.tsv', sep='/t')
    for ppt in ppt_df['participant_id']:
        nii_files = os.listdir(op.join(rs_data_dir, ppt, 'func'))
        for nii_fn in nii_files:

            #check to see if smoothed data exists
            unsmooth_fn = op.join(rs_data_dir, ppt, 'func', nii_fn)
            smooth_fn = op.join(rs_data_dir, 'derivatives', 'smoothed', ppt, '{0}_smooth.nii.gz'.format(nii_fn.split('.')[0]))

            if not op.isfile(smooth_fn):

                output_dir = op.join(rs_data_dir, 'derivatives', 'smoothed', ppt)
                if not op.isdir(output_dir):
                    os.makedirs(output_dir)
                nii_work_dir = op.join(work_dir, 'rsfc', nii_fn.split('.')[0])
                rs_preprocess(unsmooth_fn, 4, nii_work_dir, output_dir)

            else:
                #run analysis
                output_dir = op.join(rs_data_dir, 'derivatives', roi_prefix, ppt, nii_fn.split('.')[0])
                nii_work_dir = op.join(work_dir, 'rsfc', nii_fn.split('.')[0])
                rs_firstlevel(unsmooth_fn, smooth_fn, roi_mask, output_dir, nii_work_dir)

        if len(nii_files)>1:
            rs_secondlevel()

    rs_grouplevel()
