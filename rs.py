def rs_preprocess():
    #preprocessing workflow
    print('smooth data')
    std_img = pe.Node(interface=fsl.maths.StdImage(), name='stdimg')
    std_img.inputs.in_file = op.join(rs_data_dir, ppt, 'func', nii_fn)
    std_img.inputs.dimension = "T"
    #std_img.out_file = op.join(rs_data_dir, ppt, 'func', nii_fn.split('.')[0] + "_mask.nii.gz")

    bin_img = pe.Node(interface=fsl.maths.UnaryMaths(), name='binimg')
    #bin_img.in_file = op.join(rs_data_dir, ppt, 'func', nii_fn.split('.')[0] + "_mask.nii.gz")
    bin_img.inputs.operation = "bin"
    bin_img.inputs.out_file = op.join(rs_data_dir, ppt, 'func', nii_fn.split('.')[0] + "_mask.nii.gz")

    smoother = pe.Node(interface = create_susan_smooth(), name='smooth')
    smoother.inputs.inputnode.in_files = op.join(rs_data_dir, ppt, 'func', nii_fn)
    smoother.inputs.inputnode.fwhm = 4
    smoother.inputs.inputnode.mask_file = op.join(rs_data_dir, ppt, 'func', nii_fn.split('.')[0] + "_mask.nii.gz")
    smooth.run()

    mask_wf = pe.Workflow(name='genmask')
    mask_wf.base_dir = '.'
    mask_wf.connect(std_img, 'out_file', bin_img, 'in_file')

    mask_wf.run()


def rs_firstlevel():

def rs_secondlevel():

def rs_group_level():

def rs_workflow(rs_data_dir, roi_prefix, tmp_roi_fn):
    #get participants
    ppt_df = pandas.read_csv(rs_data_dir, 'participants.tsv', sep='/t')
    for ppt in ppt_df['participant_id']:
        nii_files = os.listdir(rs_data_dir, ppt, 'func')
        for nii_fn in nii_files:
            #check to see if smoothed data exists
            if not op.isfile(rs_data_dir, 'derivatives', 'smoothed', ppt + nii_fn.split('.')[0] + '.feat', 'filtered_func_data.nii.gz'):
                rs_preprocess()
        else:
            #run analysis
            print('running roi analysis')
