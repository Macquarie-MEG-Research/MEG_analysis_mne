#!/usr/bin/python3
#
# MEG source reconstruction using LCMV beamformer 
# (can use individual MRI or fsaverage)
#
# Authors: Paul Sowman, Judy Zhu

#######################################################################################

# This resource by Christian Brodbeck is very useful:
# https://github.com/christianbrodbeck/Eelbrain/wiki/MNE-Pipeline
# https://github.com/christianbrodbeck/Eelbrain/wiki/Coregistration:-Structural-MRI
# etc.

'''
# Put these commands in .bashrc, so they will be executed upon user login
# https://surfer.nmr.mgh.harvard.edu/fswiki/FS7_wsl_ubuntu
# (best not to be part of the script, as different computers will have different paths)
export FREESURFER_HOME=/usr/local/freesurfer/7-dev # path to your FS installation
export FS_LICENSE=$HOME/Downloads/freesurfer/license.txt # path to FS license file
source $FREESURFER_HOME/SetUpFreeSurfer.sh
export SUBJECTS_DIR=$HOME/analysis_mne/processing/mri/ 
# output from recon-all will be stored here (each subject will have its own subfolder)
# a number of mne functions will also use this to determine path (could also pass in each time explicitly, 
# to avoid the need to rely on env var)

# Alternatively, these commands can be run as part of python script using the following: 
#import os
#os.system('your command here')


# If using individual MRI scans, batch process recon-all on all subjects first 
# on a fast computer (can use spawner script)
my_subject=FTD0185_MEG1441 # a new folder with this name will be created inside $SUBJECTS_DIR, to contain the output from recon-all
my_nifti=/home/jzhu/analysis_mne/data/$my_subject/anat/FTD0185_T1a.nii # specify the input T1 scan
recon-all -i $my_nifti -s $my_subject -all -parallel #-openmp 6 # default 4 threads for parallel, can specify how many here
'''

#######################################################################################

#NOTE: if running from terminal, all plots will close when the script ends;
# in order to keep the figures open, use -i option when running, e.g.
# python3 -i ~/my_GH/MEG_analysis_mne/Source_analysis.py

import os
import os.path as op
import numpy as np
import matplotlib.pyplot as plt
import glob

import mne
from mne.beamformer import make_lcmv, apply_lcmv
from mne.minimum_norm import make_inverse_operator, apply_inverse


# set up file and folder paths here
exp_dir = '/mnt/d/Work/analysis_ME209/' #'/home/jzhu/analysis_mne/'
meg_task = '_dual_stim' #'_localiser' #'_1_oddball' #''
run_name = '' #'_TSPCA'

# specify a name for this run (to save intermediate processing files)
source_method = "dSPM"
#source_method = "beamformer"
#source_method = "beamformer_for_RNN_comparison"

# type of source space (note: beamformer rqs volumetric source space)
src_type = 'surface' #'vol'
spacing = "oct6" # for 'surface' source space only
                 # use a recursively subdivided octahedron: 4 for speed, 6 for real analyses
if src_type != 'surface':
    spacing = ''

# for RNN we need a sparse source space, specify the spacing below (pos)
if source_method == "beamformer_for_RNN_comparison":
    #pos = 30 # use 30mm spacing -> produces about 54 vertices
    #suffix = "_54-sources"   
    pos = 52.3 # use 52.3mm spacing -> produces about 12 vertices
    suffix = "_12-sources"            
else: # for normal source analysis
    pos = 5 # default is 5mm -> produces more than 10000 vertices
    suffix = ""

# set to False if you just want to run the whole script & save results
SHOW_PLOTS = False


# specify which subjects to analyse
subjects = ['20240202_Pilot04_RW', '20240209_Pilot07_AB']

'''
subjects = ['G01','G02','G03','G04','G05','G06','G07','G08','G09','G10',
            'G11','G12','G13','G14','G15','G16','G17','G18','G19','G20',
            'G21','G22','G23','G24','G25','G26','G27','G28','G29','G30',
            'G31','G32']
# subject exclusion
subjects.remove('G12')
'''

# specify the bad marker coils to be removed (up to 2 bad coils for each subject)
bad_coils = {}
'''
bad_coils = {"G01": [0], 
             "G02": [2],
             "G06": [0],
             "G08": [0],
             "G13": [3],
             "G18": [4],
             "G24": [3],
             "G28": [3],
             "G29": [1],
             "G32": [1]}
'''

# adjust mne options to fix rendering issues (only needed in Linux / WSL)
mne.viz.set_3d_options(antialias = False, depth_peeling = False, 
                    smooth_shading = False, multi_samples = 1) 


# loop through the subjects we want to analyse
for subject_MEG in subjects:

    #subject_MEG = 'G14' #'220112_p003' #'FTD0185_MEG1441'
    subject = 'fsaverage' #'FTD0185_MEG1441' # specify subject MRI or use template (e.g. fsaverage)
    

    # All paths below should be automatic

    data_dir = op.join(exp_dir, "data")
    meg_dir = op.join(data_dir, subject_MEG, "meg")
    processing_dir = op.join(exp_dir, "processing")
    results_dir = op.join(exp_dir, "results")
    subjects_dir = op.join('/mnt/d/Work/analysis_ME206/processing', "mri")
    inner_skull = op.join(subjects_dir, subject, "bem", "inner_skull.surf")
    src_fname = op.join(subjects_dir, subject, "bem", subject + suffix + "_" + spacing + src_type + "-src.fif") 

    subject_dir_meg = op.join(processing_dir, "meg", subject_MEG)
    raw_fname = op.join(subject_dir_meg, subject_MEG + "_emptyroom-raw.fif") #"-raw.fif" 
    # just use empty room recording for kit2fiff (embedding hsp for coreg) & for mne.io.read_info()
    trans_fname = op.join(subject_dir_meg, subject_MEG + "-trans.fif")
    epochs_fname = op.join(subject_dir_meg, subject_MEG + meg_task + run_name + "-epo.fif")
    fwd_fname = op.join(subject_dir_meg, subject_MEG + "_" + spacing + src_type + "-fwd.fif")

    save_dir = op.join(subject_dir_meg, source_method + '_' + src_type, suffix[1:])
    #os.system('mkdir -p ' + save_dir) # create the folder if needed
    filters_fname = op.join(save_dir, subject_MEG + meg_task + "-filters-lcmv.h5")
    filters_vec_fname = op.join(save_dir, subject_MEG + meg_task + "-filters_vec-lcmv.h5")
    source_results_dir = op.join(results_dir, 'meg', 'source', meg_task[1:] + run_name, source_method + '_' + src_type)
    stcs_filename = op.join(source_results_dir, subject_MEG)
    stcs_vec_filename = op.join(source_results_dir, subject_MEG + '_vec')
    figures_dir = op.join(source_results_dir, 'Figures') # where to save the figures for all subjects
    #figures_dir = op.join(results_dir, 'meg', 'source', 'Figures') # where to save the figures for all subjects
    os.system('mkdir -p ' + source_results_dir)
    os.system('mkdir -p ' + figures_dir) # create the folder if needed


    # extract info from the raw file (will be used in multiple steps below)
    file_raw = glob.glob(op.join(meg_dir, "*empty*.con"))[0]
    file_elp = glob.glob(op.join(meg_dir, "*.elp"))[0]
    file_hsp = glob.glob(op.join(meg_dir, "*.hsp"))[0]
    file_mrk = glob.glob(op.join(meg_dir, "*ini.mrk"))[0] # use initial marker
    
    #info = mne.io.read_info(raw_fname) # this only supports fif file, and does not allow editing of elp & mrk (e.g. removing bad marker coils)
    
    # read in the mrk & elp ourselves, so we can remove the bad marker coils 
    # before incorporating these info into the "raw" object
    '''
    mrk = mne.io.kit.read_mrk(file_mrk)
    elp = mne.io.kit.coreg._read_dig_kit(file_elp)
    if subject_MEG in bad_coils: # if there are bad marker coils for this subject
        mrk = np.delete(mrk, bad_coils[subject_MEG], 0)
        elp = np.delete(elp, np.array(bad_coils[subject_MEG] + 3), 0) # add 3 to the indices, as elp list contains fiducials as first 3 rows

    raw = mne.io.read_raw_kit(file_raw, mrk=mrk, elp=elp, hsp=file_hsp) # use the edited mrk & elp, rather than supplying the filenames
    info = raw.info
    '''
    raw = mne.io.read_raw_kit(file_raw, mrk=file_mrk, elp=file_elp, hsp=file_hsp)
    info = raw.info


    # Follow the steps here:
    # https://mne.tools/stable/auto_tutorials/forward/30_forward.html


    # ===== Compute head surfaces ===== #

    # Note: these commands require both MNE & Freesurfer
    if not op.exists(inner_skull): # check one of the target files to see if these steps have been run already
        os.system('mne make_scalp_surfaces --overwrite -s ' + subject + ' -d ' + subjects_dir + ' --force')
        os.system('mne watershed_bem -s ' + subject + ' -d ' + subjects_dir)
        os.system('mne setup_forward_model -s ' + subject + ' -d ' + subjects_dir + ' --homog --ico 4')

    # plot the head surface (BEM) computed from MRI
    if SHOW_PLOTS:
        plot_bem_kwargs = dict(
            subject=subject,
            subjects_dir=subjects_dir,
            brain_surfaces="white", # one or more brain surface to plot - should correspond to 
                                    # files in the subject's surf directory
            orientation="coronal",
            slices=[50, 100, 150, 200],
        )
        mne.viz.plot_bem(**plot_bem_kwargs) # plot bem  
   

    # ===== Coregistration ===== #
    # This aligns the MRI scan & the headshape from MEG digitisation into same space

    # For FIF files, hsp info are embedded in it, whereas for KIT data we have a separate .hsp file.
    # So, convert the confile to FIF format first (to embed hsp), which can then be loaded during coreg.
    # (Note: to save disk space, we just use the empty room confile here!)
    if not op.exists(raw_fname):
        os.system('mne kit2fiff --input ' + file_raw + ' --output ' + raw_fname + 
        ' --mrk ' + file_mrk + ' --elp ' + file_elp + ' --hsp ' + file_hsp)   
        # note that bad marker coils are NOT excluded here (as we have to supply
        # the filename, and the mrk file cannot be edited as plain text),
        # but it shouldn't matter here as marker coils are not used for coreg
    
    # Use the GUI for coreg, then save the results as -trans.fif
    if not op.exists(trans_fname):
        mne.gui.coregistration(subject=subject, subjects_dir=subjects_dir)
    # Note: if this gives some issues with pyvista and vtk and the versions of python/mne,
    # just install the missing packages as prompted (traitlets, pyvista, pyvistaqt, pyqt5).
    # Also disable anti-aliasing if head model not rendering (see above); hence we 
    # don't use "mne coreg" from command line (cannot set 3d options)
    # Update June 2023: new version of mne is having rendering issues again in
    # linux - just run the coreg portion in Windows instead

    trans = mne.read_trans(trans_fname)

    # Here we plot the dense head, which isn't used for BEM computations but
    # is useful for checking alignment after coregistration
    if SHOW_PLOTS:
        mne.viz.plot_alignment(
            info,
            trans_fname,
            subject=subject,
            dig=True, # include digitised headshape
            meg=["helmet", "sensors"], # include MEG helmet & sensors
            subjects_dir=subjects_dir,
            surfaces="head-dense", # include head surface from MRI
        )

    # also print out some info on distances
    print(
        "Distance from head origin to MEG origin: %0.1f mm"
        % (1000 * np.linalg.norm(info["dev_head_t"]["trans"][:3, 3]))
    )
    print(
        "Distance from head origin to MRI origin: %0.1f mm"
        % (1000 * np.linalg.norm(trans["trans"][:3, 3]))
    )

    dists = mne.dig_mri_distances(info, trans, subject, subjects_dir=subjects_dir)
    print(
        "Distance from %s digitized points to head surface: %0.1f mm"
        % (len(dists), 1000 * np.mean(dists))
    )


    # ===== Create source space ===== #

    # Note: beamformers are usually computed in a volume source space, 
    # because estimating only cortical surface activation can misrepresent the data
    # https://mne.tools/stable/auto_tutorials/inverse/50_beamformer_lcmv.html#the-forward-model

    if op.exists(src_fname):
        src = mne.read_source_spaces(src_fname)
    else:
        if src_type == 'surface':
            # create source space from cortical surface (by selecting a subset of vertices)
            src = mne.setup_source_space(
                subject, spacing=spacing, add_dist="patch", subjects_dir=subjects_dir
            )
        elif src_type == 'vol':
            # create volume source space using grid spacing (bounded by the bem)
            src = mne.setup_volume_source_space(
                subject, subjects_dir=subjects_dir, pos=pos, 
                surface=inner_skull, add_interpolator=True
            )

        # save to mri folder
        mne.write_source_spaces(src_fname, src)


    # check the source space
    print(src)
    if SHOW_PLOTS:
        plot_bem_kwargs = dict(
            subject=subject,
            subjects_dir=subjects_dir,
            brain_surfaces="white", # one or more brain surface to plot - should correspond to 
                                    # files in the subject's surf directory
            orientation="coronal",
            slices=[50, 100, 150, 200],
        )
        mne.viz.plot_bem(src=src, **plot_bem_kwargs)

    # check alignment
    if SHOW_PLOTS:
        fig = mne.viz.plot_alignment(
            subject=subject,
            subjects_dir=subjects_dir,
            surfaces="white",
            coord_frame="mri",
            src=src,
        )
        mne.viz.set_3d_view(
            fig,
            azimuth=173.78,
            elevation=101.75,
            distance=0.35,
            #focalpoint=(-0.03, -0.01, 0.03),
        )


    # ===== Compute forward solution / leadfield ===== #

    if op.exists(fwd_fname):
        fwd = mne.read_forward_solution(fwd_fname)
    else:
        conductivity = (0.3,)  # single layer: inner skull (good enough for MEG but not EEG)
        # conductivity = (0.3, 0.006, 0.3)  # three layers (use this for EEG)
        model = mne.make_bem_model( # BEM model describes the head geometry & conductivities of diff tissues
            subject=subject, ico=4, conductivity=conductivity, subjects_dir=subjects_dir
        )
        bem = mne.make_bem_solution(model)

        fwd = mne.make_forward_solution(
            info,
            trans=trans,
            src=src,
            bem=bem,
            meg=True,
            eeg=False,
            mindist=5.0,
            n_jobs=1,
            verbose=True,
        )
        # save a copy
        mne.write_forward_solution(fwd_fname, fwd)
        
    print(fwd)

    # apply source orientation constraint (e.g. fixed orientation)
    # note: not applicable to volumetric source space
    '''
    fwd = mne.convert_forward_solution(
        fwd, surf_ori=True, force_fixed=True, use_cps=True
    )
    '''

    # we can explore the content of fwd to access the numpy array that contains the leadfield matrix
    leadfield = fwd["sol"]["data"]
    print("Leadfield size (free orientation): %d sensors x %d dipoles" % leadfield.shape)
    # can save a copy of the leadfield
    if (source_method == "beamformer_for_RNN_comparison"):
        np.save(op.join(save_dir, 'leadfield.npy'), leadfield)
            
    # Forward computation can remove vertices that are too close to (or outside) the inner skull surface,
    # so always use fwd["src"] (rather than just src) when passing to other functions.
    # Let's compare before & after to see if any vertices were removed:
    print(f'Before: {src}')
    print(f'After:  {fwd["src"]}')

    # save a bit of memory
    src = fwd["src"]
    #del fwd


    # ===== Reconstruct source activity ===== #

    # Note: if running this part in Windows, copy everything over 
    # (from both the "mri" and "meg" folders), but can skip 
    # the "-raw.fif" (if large) as we can just use the con file here


    # Run sensor-space analysis script to obtain the epochs (or read from saved file)
    epochs = mne.read_epochs(epochs_fname)
    #epochs = epochs.apply_baseline((None, 0.)) # this is redundant as baseline correction was applied by default when constructing the mne.epochs object

    # use the info from the epochs object, as it contains the 
    # bad channels for each subject & task block
    #info_new = epochs.info
    # update relevant fields to ensure bad marker coils have been removed
    # Note: this doesn't work - some of the following cannot be set directly:
    #info_new['dig'] = info['dig']
    #nfo_new['dev_head_t'] = info['dev_head_t']
    #info_new['hpi_results'] = info['hpi_results']

    # alt: just use the previous info object, but obtain the bad channels from epochs.info
    info_new = info
    info_new['bads'] = epochs.info['bads']


    # compute evoked (averaged over all conditions)
    evoked_allconds = epochs.average()
    #evoked_allconds.plot_joint() # average ERF across all conds

    # compute evoked for each cond
    evokeds = []
    for cond in epochs.event_id:
        evokeds.append(epochs[cond].average())


    # compute source timecourses
    stcs = dict()
    stcs_vec = dict()

    # which method to use?
    if source_method == 'dSPM' or 'mne':
        # https://mne.tools/stable/auto_tutorials/inverse/30_mne_dspm_loreta.html

        noise_cov = mne.compute_covariance(epochs, tmin=-0.1, tmax=0,
                                        method=['shrunk','empirical'])
        #fig_cov, fig_spectra = mne.viz.plot_cov(noise_cov, info_new)

        inverse_operator = make_inverse_operator(
            info_new, fwd, noise_cov)

        for index, evoked in enumerate(evokeds):
            cond = evoked.comment

            snr = 3.
            lambda2 = 1. / snr ** 2
            stcs[cond], residual = apply_inverse(evoked, inverse_operator, lambda2,
                                        method=source_method, pick_ori=None,
                                        return_residual=True, verbose=True)
 
            # save the source estimates
            stcs[cond].save(stcs_filename + '-' + cond, overwrite=True)

    else: # use beamformer
        # https://mne.tools/stable/auto_tutorials/inverse/50_beamformer_lcmv.html
        
        # create the spatial filter
        if op.exists(filters_fname) & op.exists(filters_vec_fname):
            filters = mne.beamformer.read_beamformer(filters_fname)
            filters_vec = mne.beamformer.read_beamformer(filters_vec_fname)
        else:
            # compute cov matrices
            data_cov = mne.compute_covariance(epochs, tmin=-0.01, tmax=0.4,
                                            method='empirical')
            noise_cov = mne.compute_covariance(epochs, tmin=-0.1, tmax=0,
                                            method='empirical')
            #data_cov.plot(info_new)

            # compute the spatial filter (LCMV beamformer) - use common filter for all conds?
            filters = make_lcmv(info_new, fwd, data_cov, reg=0.05,
                                noise_cov=noise_cov, pick_ori='max-power', # 1 estimate per voxel (only preserve the axis with max power)
                                weight_norm='unit-noise-gain', rank=None)
            filters_vec = make_lcmv(info_new, fwd, data_cov, reg=0.05,
                                    noise_cov=noise_cov, pick_ori='vector', # 3 estimates per voxel, corresponding to the 3 axes
                                    weight_norm='unit-noise-gain', rank=None)
            # save the filters for later
            #filters.save(filters_fname, overwrite=True)
            #filters_vec.save(filters_vec_fname, overwrite=True)

        # apply the spatial filter (to get reconstructed source activity)
        for index, evoked in enumerate(evokeds):
            cond = evoked.comment
            stcs[cond] = apply_lcmv(evoked, filters) # timecourses contain both positive & negative values
            stcs_vec[cond] = apply_lcmv(evoked, filters_vec) # timecourses contain both positive & negative values

            # save the source estimates
            stcs[cond].save(stcs_filename + '-' + cond, overwrite=True)        
            stcs_vec[cond].save(stcs_vec_filename + '-' + cond, overwrite=True)

            # can save the source timecourses (vertices x samples) as numpy array file
            if source_method == "beamformer_for_RNN_comparison":
                stcs_vec[cond].data.shape
                np.save(op.join(save_dir, "vec_" + cond + ".npy"), stcs_vec[cond].data)

            ## use the stcs_vec structure but swap in the results from RNN
            # stcs_vec['standard'].data = np.load('standard_rnn_reshaped.npy')
            # stcs_vec['deviant'].data = np.load('deviant_rnn_reshaped.npy')

    
    # Plot the source timecourses
    for index, evoked in enumerate(evokeds):
        cond = evoked.comment

        # depending on the src type, it will create diff types of plots
        if src_type == 'vol':
            fig = stcs[cond].plot(src=src, 
                subject=subject, subjects_dir=subjects_dir, verbose=True,
                #mode='glass_brain',
                initial_time=0.1)
            fig.savefig(op.join(figures_dir, subject_MEG + meg_task + run_name + '-' + cond + '.png'))
            # also see: https://mne.tools/dev/auto_examples/visualization/publication_figure.html

        elif src_type == 'surface':  
            hemi='both'
            #vertno_max, time_max = stcs[cond].get_peak(hemi=hemi, tmin=0.1, tmax=0.27)
            initial_time = 0.16 #time_max
            surfer_kwargs = dict(
                hemi=hemi, subjects_dir=subjects_dir,
                initial_time=initial_time, 
                time_unit='s', title=subject_MEG + ' - ' + cond,
                views=['caudal','ventral','lateral','medial'], 
                show_traces=False, # use False to avoid having the blue dot (peak vertex) showing up on the brain
                smoothing_steps=10)
            brain = stcs[cond].plot(**surfer_kwargs)
            #brain.add_foci(vertno_max, coords_as_verts=True, hemi=hemi, 
            #    color='blue', scale_factor=0.6, alpha=0.5)
            #brain.save_image(op.join(figures_dir, subject_MEG + meg_task + run_name + '-' + cond + '.png'))
            brain.save_movie(op.join(figures_dir, subject_MEG + meg_task + run_name + '-' + cond + '-both.mov'), 
                tmin=0, tmax=0.35, interpolation='linear',
                time_dilation=50, time_viewer=True)

            # Note: if there are any issues with the plot/movie (e.g. showing 
            # horizontal bands), it's probably a rendering issue in Linux. 
            # Try running this script in Windows/Mac!


        # 3d plot (heavy operation - can only do one plot at a time)
        '''
        kwargs = dict(src=src, subject=subject, subjects_dir=subjects_dir, verbose=True,
            initial_time=0.1)
        brain_3d = stcs[cond].plot_3d(
            #clim=dict(kind='value', lims=[0.3, 0.45, 0.6]), # set colour scale
            hemi='both', size=(600, 600),
            #views=['sagittal'], # only show sag view
            view_layout='horizontal', views=['coronal', 'sagittal', 'axial'], # make a 3-panel figure showing all views
            brain_kwargs=dict(silhouette=True),
            **kwargs)
        '''

        # to combine stcs from 3 directions into 1: (all become positive values, 
        # i.e. what the show_traces option gives you in stcs_vec[cond].plot_3d)
        #stcs_vec[cond].magnitude().data
    
    # close all figures before moving onto the next subject
    mne.viz.close_all_3d_figures()



    # = Extract ROI time course from source estimates =

    conds = ['match','mm1','mm2','mm3','mm4']
    avgovertime_table = [] # table to contain the "average over time" values for each ROI
    avgovertime_table.append([""] + conds) # put cond names (i.e. headings) in first row

    # load source space
    src = mne.read_source_spaces(src_fname)


    # for surface source space, we need to create the Label object first
    # by reading from .annot or .label file
    # Can't use the mri file like we do for vol source space, as extract_label_time_course() will throw an error
    # https://mne.tools/stable/generated/mne.extract_label_time_course.html

    # Get labels for FreeSurfer 'aparc' cortical parcellation (69 labels)
    # https://freesurfer.net/fswiki/CorticalParcellation
    labels_parc = mne.read_labels_from_annot(subject, parc='aparc', subjects_dir=subjects_dir) # read all labels from annot file
    labels = [labels_parc[36] + labels_parc[38] + labels_parc[40], labels_parc[60]] # left IFG (combine 3 labels), left STG

    # or use 'aparc.a2009s' parcellation (150 labels)
    #labels_parc = mne.read_labels_from_annot(subject, parc='aparc.a2009s', subjects_dir=subjects_dir)
    #labels = [
    #    labels_parc[18]+labels_parc[20], labels_parc[19]+labels_parc[21], labels_parc[50], labels_parc[51], 
    #    labels_parc[58], labels_parc[59], labels_parc[66], labels_parc[67], labels_parc[84], labels_parc[85], labels_parc[144], labels_parc[145]
    #] # change these as needed

    # or read a single label (e.g. V1, BA44, etc)
    #labels_parc = mne.read_label(op.join(subjects_dir, subject, 'label', 'lh.V1.label'))


    for label in labels:
        label_name = label.name 
        # or set a custom name for combined labels
        if label_name[0:15] == 'parsopercularis':
            label_name = 'inferiorfrontal-lh'

        avgovertime_row = [label_name] # add ROI name in first column
        os.system(f'mkdir -p {op.join(figures_dir, label_name)}')

        # Plot ROI time series
        fig, axes = plt.subplots(1, layout="constrained")    
        for cond in conds:
            stc_file = stcs_filename + '-' + cond + "-lh.stc"
            stc = mne.read_source_estimate(stc_file)
            label_ts = mne.extract_label_time_course(
                [stc], label, src, mode="auto", allow_empty=True
            )
            axes.plot(1e3 * stc.times, label_ts[0][0, :], label=cond)

            # Average over time
            time_interval = range(80,101)
            #stc.times[time_interval] # verify this corresponds to 300-500ms
            avg = np.mean(label_ts[0][0][time_interval], axis=0)
            avgovertime_row.append(avg)

        axes.axvline(linestyle='--') # add verticle line at time 0
        axes.set(xlabel="Time (ms)", ylabel="MNE current (nAm)")
        axes.set(title=label_name)
        axes.legend()

        fig.savefig(op.join(figures_dir, label_name, subject_MEG + ".png"))
        plt.close(fig)

        # add row to table
        avgovertime_table.append(avgovertime_row)

        # write table to file
        avgovertime_dir = op.join(source_results_dir, "ROI_avgovertime")
        os.system(f'mkdir -p {avgovertime_dir}')
        with open(op.join(avgovertime_dir, subject_MEG + ".txt"), "w") as file:
            for row in avgovertime_table:
                file.write("\t".join(map(str, row)) + "\n")


    '''
    # Q: 
    # 1. How do we choose an ROI, i.e. get source activity for A1 only? 
    #    (need to apply the label from freesurfer to work out which vertices belong to A1?)
    # https://mne.tools/stable/auto_examples/inverse/label_source_activations.html
    # See also: 
    # https://mne.tools/stable/auto_examples/visualization/parcellation.html
    # https://mne.tools/stable/auto_tutorials/inverse/60_visualize_stc.html#volume-source-estimates

    # choose atlas for parcellation
    fname_aseg = op.join(subjects_dir, subject, 'mri', 'aparc.a2009s+aseg.mgz')
    rois = ['ctx_rh_G_temp_sup-Lateral']  # can have multiple labels in this list

    #label_names = mne.get_volume_labels_from_aseg(fname_aseg)
    #roi_idx = label_names.index(rois[0])
    #colours = ['b', 'r']

    # choose how to combine the vertices in an ROI
    modes = ('mean')#, 'max') # plain mean will prob cancel things out - try RMS!
                            # Paul: max is prob also not good
                            # for non-volumetric src, there are other options, e.g. PCA
    # make one plot for mean, one plot for max
    for mode in modes:
        fig, ax = plt.subplots(1)
        for cond, value in stcs.items():
            roi_timecourse = np.squeeze((stcs[cond]**2).extract_label_time_course(
                (fname_aseg, rois), src=src, mode=mode)**0.5) # use RMS (square then average then sqrt)
                # Note: this only works for volumetric source space; for surface source result, need to supply the Label objects (see mne doc)
            ax.plot(stcs[cond].times, roi_timecourse, lw=2., alpha=0.5, label=cond)
            ax.set(xlim=stcs[cond].times[[0, -1]],
                xlabel='Time (s)', ylabel='Activation')

        # this would need to be dynamic for multiple rois
        ax.set(title=mode + '_' + rois[0])
        ax.legend()
        for loc in ('right', 'top'):
            ax.spines[loc].set_visible(False)
        fig.tight_layout()

    # Try 'auto' mode: (cf with RMS plot - if similar then prob good?)
    # https://mne.tools/stable/generated/mne.extract_label_time_course.html
    modes = ('auto') 
    for mode in modes:
        fig, ax = plt.subplots(1)
        for cond, value in stcs.items():
            roi_timecourse = np.squeeze(stcs[cond].extract_label_time_course(
                (fname_aseg, rois), src=src, mode=mode))
            ax.plot(stcs[cond].times, roi_timecourse, lw=2., alpha=0.5, label=cond)
            ax.set(xlim=stcs[cond].times[[0, -1]],
                xlabel='Time (s)', ylabel='Activation')

        # this would need to be dynamic for multiple rois
        ax.set(title=mode + '_' + rois[0])
        ax.legend()
        for loc in ('right', 'top'):
            ax.spines[loc].set_visible(False)
        fig.tight_layout()


    # 2. How to compare stcs between 2 conds? atm I'm just plotting each of them separately ...
    # https://mne.tools/stable/auto_tutorials/stats-source-space/20_cluster_1samp_spatiotemporal.html

    # Compare evoked response across conds (can prob do the same for stcs)
    # https://mne.tools/stable/auto_examples/visualization/topo_compare_conditions.html
    #
    # Plotting stcs:
    # https://mne.tools/stable/auto_tutorials/inverse/60_visualize_stc.html
    '''
