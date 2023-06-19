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
exp_dir = '/mnt/d/Work/analysis_ME206/' #'/home/jzhu/analysis_mne/'
subject = 'fsaverage' #'FTD0185_MEG1441' # specify subject MRI or use template (e.g. fsaverage)
subject_MEG = 'G06' #'220112_p003' #'FTD0185_MEG1441'
meg_task = '_localiser' #'_1_oddball' #''
run_name = '_TSPCA'

# specify a name for this run (to save intermediate processing files)
#source_method = "beamformer_TSPCA"
#source_method = "beamformer_for_RNN_comparison"
source_method = "mne"

# type of source space (note: beamformer rqs volumetric source space)
src_type = 'vol' #'surface'
spacing = "oct6" # for 'surface' source space only
                 # use a recursively subdivided octahedron: 4 for speed, 6 for real analyses
if src_type != 'surface':
    spacing = ''

# for RNN we need a sparse source space, specify the spacing below (pos)
if source_method == "beamformer_for_RNN_comparison":
    #pos = 30 # use 30mm spacing -> produces about 54 vertices
    #suffix = "54-sources"   
    pos = 52.3 # use 52.3mm spacing -> produces about 12 vertices
    suffix = "12-sources"            
else: # for normal source analysis
    pos = 5 # default is 5mm -> produces more than 10000 vertices
    suffix = ""

# set to False if you just want to run the whole script & save results
SHOW_PLOTS = True 


# All paths below should be automatic

data_dir = op.join(exp_dir, "data")
meg_dir = op.join(data_dir, subject_MEG, "meg")
processing_dir = op.join(exp_dir, "processing")
subjects_dir = op.join(processing_dir, "mri")
inner_skull = op.join(subjects_dir, subject, "bem", "inner_skull.surf")
src_fname = op.join(subjects_dir, subject, "bem", subject + "_" + suffix + "_" + spacing + src_type + "-src.fif") 

subject_dir_meg = op.join(processing_dir, "meg", subject_MEG)
raw_fname = op.join(subject_dir_meg, subject_MEG + "_emptyroom-raw.fif") #"-raw.fif" 
# just use empty room recording for kit2fiff (embedding hsp for coreg) & for mne.io.read_info()
trans_fname = op.join(subject_dir_meg, subject_MEG + "-trans.fif")
epochs_fname = op.join(subject_dir_meg, subject_MEG + meg_task + run_name + "-epo.fif")
fwd_fname = op.join(subject_dir_meg, subject_MEG + "_" + spacing + src_type + "-fwd.fif")

save_dir = op.join(subject_dir_meg, source_method + run_name, suffix)
figures_dir = op.join(processing_dir, 'meg', 'Figures', 'source', meg_task[1:] + run_name, source_method) # where to save the figures for all subjects
os.system('mkdir -p ' + save_dir) # create the folder if needed
filters_fname = op.join(save_dir, subject_MEG + meg_task + "-filters-lcmv.h5")
filters_vec_fname = op.join(save_dir, subject_MEG + meg_task + "-filters_vec-lcmv.h5")


# adjust mne options to fix rendering issues (only needed in Linux / WSL)
mne.viz.set_3d_options(antialias = 0, depth_peeling = 0) 


# Follow the steps here:
# https://mne.tools/stable/auto_tutorials/forward/30_forward.html


# ===== Compute head surfaces ===== #

# Note: these commands require both MNE & Freesurfer
if not op.exists(inner_skull): # check one of the target files to see if these steps have been run already
    os.system('mne make_scalp_surfaces --overwrite -s ' + subject + ' -d ' + subjects_dir + ' --force')
    os.system('mne watershed_bem -s ' + subject + ' -d ' + subjects_dir)
    os.system('mne setup_forward_model -s ' + subject + ' -d ' + subjects_dir + ' --homog --ico 4')

# specify some settings for plots (will be re-used below)
plot_bem_kwargs = dict(
    subject=subject,
    subjects_dir=subjects_dir,
    brain_surfaces="white", # one or more brain surface to plot - should correspond to 
                            # files in the subject’s surf directory
    orientation="coronal",
    slices=[50, 100, 150, 200],
)

# plot the head surface (BEM) computed from MRI
if SHOW_PLOTS:
    mne.viz.plot_bem(**plot_bem_kwargs) # plot bem


# ===== Coregistration ===== #

# Coregister MRI scan with headshape from MEG digitisation 

# For FIF files, hsp info are embedded in it, whereas for KIT data we have a separate .hsp file.
# So, convert the confile to FIF format first (to embed mrk & hsp), which can then be loaded during coreg.
# (Note: to save disk space, we just use the empty room confile here!)
if not op.exists(raw_fname):
    file_raw = glob.glob(op.join(meg_dir, "*empty*.con"))
    file_elp = glob.glob(op.join(meg_dir, "*.elp"))
    file_hsp = glob.glob(op.join(meg_dir, "*.hsp"))
    file_mrk = glob.glob(op.join(meg_dir, "*.mrk"))
    os.system('mne kit2fiff --input ' + file_raw[0] + ' --output ' + raw_fname + 
    ' --mrk ' + file_mrk[0] + ' --elp ' + file_elp[0] + ' --hsp ' + file_hsp[0])

# Use the GUI for coreg, then save the results as -trans.fif
if not op.exists(trans_fname):
    mne.gui.coregistration(subject=subject, subjects_dir=subjects_dir)
# Note: if this gives some issues with pyvista and vtk and the versions of python/mne,
# just install the missing packages as prompted (traitlets, pyvista, pyvistaqt, pyqt5).
# Also disable anti-aliasing if head model not rendering (see above); hence we 
# don't use "mne coreg" from command line (cannot set 3d options)

trans = mne.read_trans(trans_fname)

# Here we plot the dense head, which isn't used for BEM computations but
# is useful for checking alignment after coregistration
info = mne.io.read_info(raw_fname) # only supports fif file? For con file, you may need to read in the raw first (mne.io.read_raw_kit) then use raw.info
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
if source_method == 'mne':
    # https://mne.tools/stable/auto_tutorials/inverse/30_mne_dspm_loreta.html

    noise_cov = mne.compute_covariance(epochs, tmin=-0.1, tmax=0,
                                    method=['shrunk','empirical'])
    #fig_cov, fig_spectra = mne.viz.plot_cov(noise_cov, info)

    inverse_operator = make_inverse_operator(
        evoked_allconds.info, fwd, noise_cov)

    for index, evoked in enumerate(evokeds):
        cond = evoked.comment

        method = "MNE"
        snr = 3.
        lambda2 = 1. / snr ** 2
        stcs[cond], residual = apply_inverse(evoked, inverse_operator, lambda2,
                                    method=method, pick_ori=None,
                                    return_residual=True, verbose=True)
        
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
        #data_cov.plot(epochs.info)

        # compute the spatial filter (LCMV beamformer) - use common filter for all conds?
        filters = make_lcmv(evoked_allconds.info, fwd, data_cov, reg=0.05,
                            noise_cov=noise_cov, pick_ori='max-power', # 1 estimate per voxel (only preserve the axis with max power)
                            weight_norm='unit-noise-gain', rank=None)
        filters_vec = make_lcmv(evoked_allconds.info, fwd, data_cov, reg=0.05,
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
        fig.savefig(op.join(figures_dir, subject_MEG + meg_task + run_name '-' + cond + '.png'))
        # also see: https://mne.tools/dev/auto_examples/visualization/publication_figure.html

    elif src_type == 'surface':    
        vertno_max, time_max = stcs[cond].get_peak(hemi='rh')
        surfer_kwargs = dict(
            hemi='rh', subjects_dir=subjects_dir,
            clim=dict(kind='value', lims=[8, 12, 15]), views='lateral',
            initial_time=time_max, time_unit='s', size=(800, 800), smoothing_steps=10)
        brain = stcs[cond].plot(**surfer_kwargs)
        brain.add_foci(vertno_max, coords_as_verts=True, hemi='rh', color='blue',
                    scale_factor=0.6, alpha=0.5)
    
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
#
# Compare evoked response across conds (can do the same to compare stcs?)
# https://mne.tools/stable/auto_examples/visualization/topo_compare_conditions.html
#
# Plotting stcs:
# https://mne.tools/stable/auto_tutorials/inverse/60_visualize_stc.html