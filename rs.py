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


def getthreshop(thresh):
    return ['-thr %.10f -Tmin -bin' % (0.1 * val[1]) for val in thresh]


def pickfirst(files):
    if isinstance(files, list):
        return files[0]
    else:
        return files


def pickmiddle(files):
    from nibabel import load
    import numpy as np
    middlevol = []
    for f in files:
        middlevol.append(int(np.ceil(load(f).shape[3] / 2)))
    return middlevol


def pickvol(filenames, fileidx, which):
    from nibabel import load
    import numpy as np
    if which.lower() == 'first':
        idx = 0
    elif which.lower() == 'middle':
        idx = int(np.ceil(load(filenames[fileidx]).shape[3] / 2))
    elif which.lower() == 'last':
        idx = load(filenames[fileidx]).shape[3] - 1
    else:
        raise Exception('unknown value for volume selection : %s' % which)
    return idx


def getbtthresh(medianvals):
    return [0.75 * val for val in medianvals]


def chooseindex(fwhm):
    if fwhm < 1:
        return [0]
    else:
        return [1]


def getmeanscale(medianvals):
    return ['-mul %.10f' % (10000. / val) for val in medianvals]


def getusans(x):
    return [[tuple([val[0], 0.75 * val[1]])] for val in x]

tolist = lambda x: [x]
highpass_operand = lambda x: '-bptf %.10f -1' % x


def rs_preprocess(name='featpreproc', highpass=True, realign=True, whichvol='middle', outputdir=None):
    """Create a FEAT preprocessing workflow with registration to one volume of the first run
    Parameters
    ----------
    ::
        name : name of workflow (default: featpreproc)
        highpass : boolean (default: True)
        realign : boolean (default: True)
        whichvol : which volume of the first run to register to ('first', 'middle', 'last', 'mean')
    Inputs::
        inputspec.func : functional runs (filename or list of filenames)
        inputspec.fwhm : fwhm for smoothing with SUSAN
        inputspec.highpass : HWHM in TRs (if created with highpass=True)
    Outputs::
        outputspec.reference : volume to which runs are realigned
        outputspec.motion_parameters : motion correction parameters
        outputspec.realigned_files : motion corrected files
        outputspec.motion_plots : plots of motion correction parameters
        outputspec.mask : mask file used to mask the brain
        outputspec.smoothed_files : smoothed functional data
        outputspec.highpassed_files : highpassed functional data (if highpass=True)
        outputspec.mean : mean file
    Example
    -------
    >>> preproc = rs_preprocess()
    >>> preproc.inputs.inputspec.func = ['f3.nii', 'f5.nii']
    >>> preproc.inputs.inputspec.fwhm = 5
    >>> preproc.inputs.inputspec.highpass = 128./(2*2.5)
    >>> preproc.base_dir = '/tmp'
    >>> preproc.run() # doctest: +SKIP
    >>> preproc = create_featreg_preproc(highpass=False, whichvol='mean')
    >>> preproc.inputs.inputspec.func = 'f3.nii'
    >>> preproc.inputs.inputspec.fwhm = 5
    >>> preproc.base_dir = '/tmp'
    >>> preproc.run() # doctest: +SKIP
    """


    from nipype.workflows.fmri.fsl.preprocess import create_susan_smooth
    from nipype import LooseVersion

    version = 0
    if fsl.Info.version() and \
            LooseVersion(fsl.Info.version()) > LooseVersion('5.0.6'):
        version = 507

    featpreproc = pe.Workflow(name=name)

    """
    Set up a node to define all inputs required for the preprocessing workflow
    """

    if highpass:
        inputnode = pe.Node(interface=util.IdentityInterface(fields=['func',
                                                                     'fwhm',
                                                                     'highpass']),
                            name='inputspec')
        if realign:
            outputnode = pe.Node(interface=util.IdentityInterface(fields=['reference',
                                                                          'motion_parameters',
                                                                          'realigned_files',
                                                                          'motion_plots',
                                                                          'mask',
                                                                          'smoothed_files',
                                                                          'highpassed_files',
                                                                          'mean']),
                                 name='outputspec')
        else:
            outputnode = pe.Node(interface=util.IdentityInterface(fields=['reference',
                                                                          'mask',
                                                                          'smoothed_files',
                                                                          'highpassed_files',
                                                                          'mean']),
                                 name='outputspec')
    else:
        inputnode = pe.Node(interface=util.IdentityInterface(fields=['func',
                                                                     'fwhm']),
                            name='inputspec')
        if realign:
            outputnode = pe.Node(interface=util.IdentityInterface(fields=['reference',
                                                                          'motion_parameters',
                                                                          'realigned_files',
                                                                          'motion_plots',
                                                                          'mask',
                                                                          'smoothed_files',
                                                                          'mean']),
                                 name='outputspec')
        else:
            outputnode = pe.Node(interface=util.IdentityInterface(fields=['reference',
                                                                          'mask',
                                                                          'smoothed_files',
                                                                          'mean']),
                                 name='outputspec')


    """
    Set up a node to define outputs for the preprocessing workflow
    """

    """
    This node will write out image files in output directory
    """
    datasink = pe.Node(nio.DataSink(), name='sinker')
    datasink.inputs.base_directory = outputdir

    """
    Convert functional images to float representation. Since there can
    be more than one functional run we use a MapNode to convert each
    run.
    """
    img2float = pe.MapNode(interface=fsl.ImageMaths(out_data_type='float',
                                                    op_string='',
                                                    suffix='_dtype'),
                           iterfield=['in_file'],
                           name='img2float')
    featpreproc.connect(inputnode, 'func', img2float, 'in_file')

    """
    Extract the middle (or what whichvol points to) volume of the first run as the reference
    """

    if whichvol != 'mean':
        extract_ref = pe.Node(interface=fsl.ExtractROI(t_size=1),
                              iterfield=['in_file'],
                              name='extractref')
        featpreproc.connect(img2float, ('out_file', pickfirst), extract_ref, 'in_file')
        featpreproc.connect(img2float, ('out_file', pickvol, 0, whichvol), extract_ref, 't_min')
        featpreproc.connect(extract_ref, 'roi_file', outputnode, 'reference')

    if realign:
        """
        Realign the functional runs to the reference (`whichvol` volume of first run)
        """

        motion_correct = pe.MapNode(interface=fsl.MCFLIRT(save_mats=True,
                                                          save_plots=True,
                                                          interpolation='spline'),
                                    name='realign',
                                    iterfield=['in_file'])
        featpreproc.connect(img2float, 'out_file', motion_correct, 'in_file')
        if whichvol != 'mean':
            featpreproc.connect(extract_ref, 'roi_file', motion_correct, 'ref_file')
        else:
            motion_correct.inputs.mean_vol = True
            featpreproc.connect(motion_correct, ('mean_img', pickfirst), outputnode, 'reference')

        featpreproc.connect(motion_correct, 'par_file', outputnode, 'motion_parameters')
        featpreproc.connect(motion_correct, 'out_file', outputnode, 'realigned_files')

        """
        Plot the estimated motion parameters
        """

        plot_motion = pe.MapNode(interface=fsl.PlotMotionParams(in_source='fsl'),
                                 name='plot_motion',
                                 iterfield=['in_file'])
        plot_motion.iterables = ('plot_type', ['rotations', 'translations'])
        featpreproc.connect(motion_correct, 'par_file', plot_motion, 'in_file')
        featpreproc.connect(plot_motion, 'out_file', outputnode, 'motion_plots')

    """
    Extract the mean volume of the first functional run
    """

    meanfunc = pe.Node(interface=fsl.ImageMaths(op_string='-Tmean',
                                                suffix='_mean'),
                       name='meanfunc')
    if realign:
        featpreproc.connect(motion_correct, ('out_file', pickfirst), meanfunc, 'in_file')
    else:
        featpreproc.connect(img2float, ('out_file', pickfirst), meanfunc, 'in_file')

    """
    Strip the skull from the mean functional to generate a mask
    """

    meanfuncmask = pe.Node(interface=fsl.BET(mask=True,
                                             no_output=True,
                                             frac=0.3),
                           name='meanfuncmask')
    featpreproc.connect(meanfunc, 'out_file', meanfuncmask, 'in_file')

    """
    Mask the functional runs with the extracted mask
    """

    maskfunc = pe.MapNode(interface=fsl.ImageMaths(suffix='_bet',
                                                   op_string='-mas'),
                          iterfield=['in_file'],
                          name='maskfunc')
    if realign:
        featpreproc.connect(motion_correct, 'out_file', maskfunc, 'in_file')
    else:
        featpreproc.connect(img2float, ('out_file', pickfirst), maskfunc, 'in_file')
    featpreproc.connect(meanfuncmask, 'mask_file', maskfunc, 'in_file2')

    """
    Determine the 2nd and 98th percentile intensities of each functional run
    """

    getthresh = pe.MapNode(interface=fsl.ImageStats(op_string='-p 2 -p 98'),
                           iterfield=['in_file'],
                           name='getthreshold')
    featpreproc.connect(maskfunc, 'out_file', getthresh, 'in_file')

    """
    Threshold the first run of the functional data at 10% of the 98th percentile
    """

    threshold = pe.MapNode(interface=fsl.ImageMaths(out_data_type='char',
                                                    suffix='_thresh'),
                           iterfield=['in_file', 'op_string'],
                           name='threshold')
    featpreproc.connect(maskfunc, 'out_file', threshold, 'in_file')

    """
    Define a function to get 10% of the intensity
    """

    featpreproc.connect(getthresh, ('out_stat', getthreshop), threshold, 'op_string')

    """
    Determine the median value of the functional runs using the mask
    """

    medianval = pe.MapNode(interface=fsl.ImageStats(op_string='-k %s -p 50'),
                           iterfield=['in_file', 'mask_file'],
                           name='medianval')
    if realign:
        featpreproc.connect(motion_correct, 'out_file', medianval, 'in_file')
    else:
        featpreproc.connect(img2float, ('out_file', pickfirst), medianval, 'in_file')
    featpreproc.connect(threshold, 'out_file', medianval, 'mask_file')

    """
    Dilate the mask
    """

    dilatemask = pe.MapNode(interface=fsl.ImageMaths(suffix='_dil',
                                                     op_string='-dilF'),
                            iterfield=['in_file'],
                            name='dilatemask')
    featpreproc.connect(threshold, 'out_file', dilatemask, 'in_file')
    featpreproc.connect(dilatemask, 'out_file', outputnode, 'mask')
    featpreproc.connect(dilatemask, 'out_file', datasink, 'mask')

    """
    Mask the motion corrected functional runs with the dilated mask
    """

    maskfunc2 = pe.MapNode(interface=fsl.ImageMaths(suffix='_mask',
                                                    op_string='-mas'),
                           iterfield=['in_file', 'in_file2'],
                           name='maskfunc2')
    if realign:
        featpreproc.connect(motion_correct, 'out_file', maskfunc2, 'in_file')
    else:
        featpreproc.connect(img2float, ('out_file', pickfirst), maskfunc2, 'in_file')
    featpreproc.connect(dilatemask, 'out_file', maskfunc2, 'in_file2')

    """
    Smooth each run using SUSAN with the brightness threshold set to 75%
    of the median value for each run and a mask constituting the mean
    functional
    """

    smooth = create_susan_smooth()

    featpreproc.connect(inputnode, 'fwhm', smooth, 'inputnode.fwhm')
    featpreproc.connect(maskfunc2, 'out_file', smooth, 'inputnode.in_files')
    featpreproc.connect(dilatemask, 'out_file', smooth, 'inputnode.mask_file')

    """
    Mask the smoothed data with the dilated mask
    """

    maskfunc3 = pe.MapNode(interface=fsl.ImageMaths(suffix='_mask',
                                                    op_string='-mas'),
                           iterfield=['in_file', 'in_file2'],
                           name='maskfunc3')
    featpreproc.connect(smooth, 'outputnode.smoothed_files', maskfunc3, 'in_file')

    featpreproc.connect(dilatemask, 'out_file', maskfunc3, 'in_file2')

    concatnode = pe.Node(interface=util.Merge(2),
                         name='concat')
    featpreproc.connect(maskfunc2, ('out_file', tolist), concatnode, 'in1')
    featpreproc.connect(maskfunc3, ('out_file', tolist), concatnode, 'in2')

    """
    The following nodes select smooth or unsmoothed data depending on the
    fwhm. This is because SUSAN defaults to smoothing the data with about the
    voxel size of the input data if the fwhm parameter is less than 1/3 of the
    voxel size.
    """
    selectnode = pe.Node(interface=util.Select(), name='select')

    featpreproc.connect(concatnode, 'out', selectnode, 'inlist')

    featpreproc.connect(inputnode, ('fwhm', chooseindex), selectnode, 'index')
    featpreproc.connect(selectnode, 'out', outputnode, 'smoothed_files')

    """
    Scale the median value of the run is set to 10000
    """

    meanscale = pe.MapNode(interface=fsl.ImageMaths(suffix='_gms'),
                           iterfield=['in_file', 'op_string'],
                           name='meanscale')
    featpreproc.connect(selectnode, 'out', meanscale, 'in_file')

    """
    Define a function to get the scaling factor for intensity normalization
    """

    featpreproc.connect(medianval, ('out_stat', getmeanscale), meanscale, 'op_string')
    featpreproc.connect(meanscale, 'out_file', datasink, 'gms')

    """
    Generate a mean functional image from the first run
    """

    meanfunc3 = pe.Node(interface=fsl.ImageMaths(op_string='-Tmean',
                                                 suffix='_mean'),
                        iterfield=['in_file'],
                        name='meanfunc3')

    featpreproc.connect(meanscale, ('out_file', pickfirst), meanfunc3, 'in_file')
    featpreproc.connect(meanfunc3, 'out_file', outputnode, 'mean')

    """
    Perform temporal highpass filtering on the data
    """

    if highpass:
        highpass = pe.MapNode(interface=fsl.ImageMaths(suffix='_tempfilt'),
                              iterfield=['in_file'],
                              name='highpass')
        featpreproc.connect(inputnode, ('highpass', highpass_operand), highpass, 'op_string')
        featpreproc.connect(meanscale, 'out_file', highpass, 'in_file')

        if version < 507:
            featpreproc.connect(highpass, 'out_file', outputnode, 'highpassed_files')
        else:
            """
            Add back the mean removed by the highpass filter operation as of FSL 5.0.7
            """
            meanfunc4 = pe.MapNode(interface=fsl.ImageMaths(op_string='-Tmean',
                                                            suffix='_mean'),
                                   iterfield=['in_file'],
                                   name='meanfunc4')

            featpreproc.connect(meanscale, 'out_file', meanfunc4, 'in_file')
            addmean = pe.MapNode(interface=fsl.BinaryMaths(operation='add'),
                                 iterfield=['in_file', 'operand_file'],
                                 name='addmean')
            featpreproc.connect(highpass, 'out_file', addmean, 'in_file')
            featpreproc.connect(meanfunc4, 'out_file', addmean, 'operand_file')
            featpreproc.connect(addmean, 'out_file', outputnode, 'highpassed_files')

    return featpreproc


def rs_firstlevel(name="rsworkflow", outputdir=None):

    import nipype.algorithms.modelgen as model  # model generation
    from niflow.nipype1.workflows.fmri.fsl import (create_modelfit_workflow,
                                       create_fixed_effects_flow)


    level1_workflow = pe.Workflow(name='level1flow')

    modelfit = create_modelfit_workflow()
    modelfit.get_node('modelestimate').inputs.smooth_autocorr = False
    modelfit.get_node('modelestimate').inputs.autocorr_noestimate = True
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


    cont1 = ['corr', 'T', ['roi', 'gsr'], [1,0]]
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
                smooth_flow = rs_preprocess(name='featpreproc', highpass=False, realign=False, whichvol='first', outputdir=nii_work_dir)
                smooth_flow.inputs.inputspec.func = op.join(rs_data_dir, ppt, 'func', nii_fn)
                smooth_flow.inputs.inputspec.fwhm = 4
                smooth_flow.base_dir = nii_work_dir

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
                meants.out_file = op.join(roi_out_dir, '{0}.{1}.txt'.format(op.basename(tmp_roi_fn).split('.')[0], nii_fn.split('.')[0]))
                meants.run()

                meants.mask = mask
                meants.out_file = op.join(roi_out_dir, 'gsr.{0}.txt'.format(nii_fn.split('.')[0]))
                meants.run()

                roi_ts = np.atleast_2d(np.loadtxt(op.join(roi_out_dir, '{0}.{1}.txt'.format(op.basename(tmp_roi_fn).split('.')[0], nii_fn.split('.')[0]))))
                gsr_ts = np.atleast_2d(np.loadtxt(op.join(roi_out_dir, 'gsr.{0}.txt'.format(nii_fn.split('.')[0]))))
                subject_info = Bunch(conditions=['roi', 'gsr'], onsets=[list(range(1,len(roi_ts)+1,1)), list(range(1,len(gsr_ts)+1,1))], durations=[[0], [0]], amplitudes=[roi_ts.tolist(), gsr_ts.tolist()])
                #subject_info = Bunch(conditions=['roi', 'gsr'], onsets=[roi_ts.tolist(), gsr_ts.tolist()], durations=[[0], [0]])

                firstlevel = rs_firstlevel(name="firstlevel", outputdir=roi_out_dir)
                firstlevel.inputs.inputspec.func = smooth_fn
                firstlevel.inputs.inputspec.subjectinfo = subject_info

                firstlevel.run()


        if len(nii_files)>1:
            rs_secondlevel()

    rs_grouplevel()
