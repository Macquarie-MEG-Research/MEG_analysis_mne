#!/usr/bin/python3
import os.path as op
import numpy as np
import matplotlib.pyplot as plt
import copy

import mne
from mne.minimum_norm import make_inverse_operator, apply_inverse
from mne.beamformer import make_lcmv, apply_lcmv
from mne_features import univariate

# set up paths
exp_dir = '/home/jzhu/epilepsy_MEG/'
subjects_dir = op.join(exp_dir, 'mri')
subject = 'p0001' # subject name in mri folder
subjects_dir_MEG = op.join(exp_dir, 'meg')
subject_MEG = '0001_JC_ME200_11022022' # subject name in meg folder
meg_task = '_resting_B1_TSPCA'

path_MEG = op.join(subjects_dir_MEG, subject_MEG)
results_dir = op.join(path_MEG, 'source_analysis')
raw_fname = op.join(path_MEG, subject_MEG) + meg_task + '-raw.fif'
raw_emptyroom_fname = op.join(path_MEG, subject_MEG) + '_emptyroom.con'
fname_bem = op.join(subjects_dir, subject, 'bem', 'p0001-5120-bem-sol.fif') # obtained with: mne setup_forward_model -s $my_subject -d $SUBJECTS_DIR --homog --ico 4
fname_trans = op.join(path_MEG, subject_MEG) + "-trans.fif" # obtained with: mne coreg (GUI)
fname_src = op.join(subjects_dir, subject, "bem", subject + "_vol-src.fif") # volumetric source model - can use for both minimum norm & beamformer
fname_fwd = op.join(results_dir, subject_MEG) + "-fwd.fif"
fname_filters = op.join(results_dir, subject_MEG) + meg_task + "-filters-lcmv.h5"
fname_annot = op.join(path_MEG, 'saved-annotations-for_Judy_1Aug22.csv')


# load raw data
raw = mne.io.read_raw_fif(raw_fname, verbose=False, preload=True) # filtering requires preload of raw data

# filtering 1hr recording needs 12GB memory allocation for WSL (if "Killed", you need to allocate more)
#raw.filter(l_freq=1, h_freq=80)
raw.filter(l_freq=3, h_freq=70) # use this for kurtosis beamformer (see Rui's paper)

# read spikes from csv file
my_annot = mne.read_annotations(fname_annot)
# remove 'BAD_' prefix in annotation descriptions
my_annot.rename({'BAD_sw_post' : 'sw_post', 
                        'BAD_postswbroad' : 'postswbroad', 
                        'BAD_sw_lesspost' : 'sw_lesspost', 
                        'BAD_sw_post' : 'sw_post', 
                        'BAD_polysw' : 'polysw', 
                        'BAD_smallpolysw' : 'smallpolysw'}, verbose=None)
print(my_annot)

# convert annotations to events array
raw.set_annotations(my_annot)
events_from_annot, event_dict = mne.events_from_annotations(raw)
print(event_dict)
print(events_from_annot)
#mne.write_events(exp_dir + 'events_from_annot_eve.txt', events_from_annot)

# For more info:
# https://mne.tools/dev/auto_tutorials/raw/30_annotate_raw.html
# https://mne.tools/dev/auto_tutorials/intro/20_events_from_raw.html#the-events-and-annotations-data-structures



### Prepare for source analysis ###

# epoching based on events
epochs = mne.Epochs(
    raw, events_from_annot, event_id=event_dict, tmin=-0.1, tmax=0.6, preload=True
)

# average the epochs
evoked_polysw = epochs['polysw'].average()
evoked_sw_post = epochs['sw_post'].average()
evoked_sw_post.crop(tmin=-0.1, tmax=0.37) # crop based on average length of manually marked spikes in this cateogory

# compute noise covariance matrix
cov_polysw = mne.compute_covariance(epochs['polysw'], tmax=0., method=['shrunk', 'empirical'], rank=None)
cov_sw_post = mne.compute_covariance(epochs['sw_post'], tmax=0., method=['shrunk', 'empirical'], rank=None)



# Method 1: fit a dipole
# https://mne.tools/stable/auto_tutorials/inverse/20_dipole_fit.html
dip = mne.fit_dipole(evoked_polysw, cov_polysw, fname_bem, fname_trans)[0]
dip.save(results_dir + 'polysw.dip')
dip_sw_post = mne.fit_dipole(evoked_sw_post, cov_sw_post, fname_bem, fname_trans)[0]
dip_sw_post.save(results_dir + 'sw_post.dip')

# Plot the result in 3D brain using individual T1
#dip = mne.read_dipole(exp_dir + 'polysw.dip')
dip.plot_locations(fname_trans, subject, subjects_dir, mode='orthoview')
#dip_sw_post = mne.read_dipole(exp_dir + 'sw_post.dip')
dip_sw_post.plot_locations(fname_trans, subject, subjects_dir, mode='orthoview')



# Prep for Methods 2 & 3 - create source space & forward model

# create source space
if op.exists(fname_src):
    src = mne.read_source_spaces(fname_src)
else:
    surface = op.join(subjects_dir, subject, "bem", "inner_skull.surf")
    src = mne.setup_volume_source_space(
        subject, subjects_dir=subjects_dir, surface=surface, add_interpolator=True
    )
    # save to mri folder
    mne.write_source_spaces(
        fname_src, src
    )
print(src)

# plot bem with source points
plot_bem_kwargs = dict(
    subject=subject,
    subjects_dir=subjects_dir,
    brain_surfaces="white", # one or more brain surface to plot - should correspond to 
                            # files in the subject’s surf directory
    orientation="coronal",
    slices=[50, 100, 150, 200],
)
mne.viz.plot_bem(src=src, **plot_bem_kwargs) 

# check alignment after creating source space
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

# compute forward model
if op.exists(fname_fwd):
    fwd = mne.read_forward_solution(fname_fwd)
else:
    conductivity = (0.3,)  # single layer: inner skull (good enough for MEG but not EEG)
    # conductivity = (0.3, 0.006, 0.3)  # three layers (use this for EEG)
    model = mne.make_bem_model( # BEM model describes the head geometry & conductivities of diff tissues
        subject=subject, ico=4, conductivity=conductivity, subjects_dir=subjects_dir
    )
    bem = mne.make_bem_solution(model)

    trans = mne.read_trans(fname_trans)
    fwd = mne.make_forward_solution(
        raw_fname,
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
    mne.write_forward_solution(fname_fwd, fwd)
print(fwd)



# Method 2: minimum norm
# https://mne.tools/stable/auto_tutorials/inverse/30_mne_dspm_loreta.html#inverse-modeling-mne-dspm-on-evoked-and-raw-data

# compute inverse solution
inv = make_inverse_operator(
    evoked_sw_post.info, fwd, cov_sw_post, loose='auto', depth=0.8)

method = "MNE" # “dSPM” | “sLORETA” | “eLORETA”
snr = 3.
lambda2 = 1. / snr ** 2
stc, residual = apply_inverse(evoked_sw_post, inv, lambda2,
                              method=method, pick_ori=None,
                              return_residual=True, verbose=True)    
stc.save(results_dir + '/sw_post')

# plot the results:
# https://mne.tools/stable/auto_tutorials/inverse/60_visualize_stc.html#volume-source-estimates
stc = mne.read_source_estimate(results_dir + '/sw_post')
src = mne.read_source_spaces(fname_src)
stc.plot(src=src, subject=subject, subjects_dir=subjects_dir, initial_time=0.053,
    #clim=dict(kind='percent', lims=[90, 95, 99]),
    #view_layout='horizontal', views=['coronal', 'sagittal', 'axial'], # make a 3-panel figure showing all views
    #brain_kwargs=dict(silhouette=True)
    )
# use plot_3d() for interactive
stc.plot(src, subject=subject, subjects_dir=subjects_dir, mode='glass_brain')



# Method 3: kurtosis beamformer
# use 1-min segment around each manually marked spike, ensure there's no overlap

# select a particular condition
cond = 'sw_post'
cond_id = event_dict.get(cond)
rows = np.where(events_from_annot[:,2] == cond_id)
events_sw_post = events_from_annot[rows]
# calculate timing diff of two consecutive events, find where the diff is shorter than 1 min (i.e. too close)
t = events_sw_post[:,0]
diff = np.diff(t)
too_close = np.where(diff < 60000) # note: indices start from 0
too_close = np.asarray(too_close).flatten() # convert tuple to array
# when 2 events are too close, combine them into one event
for i in reversed(too_close): # do in reversed order as indices will change after each iteration
    #t[i] = (t[i] + t[i+1]) / 2
    #t = np.delete(t, i+1)
    events_sw_post[i,0] = (t[i] + t[i+1]) / 2 # average the timing of the two events
    events_sw_post = np.delete(events_sw_post, i+1, axis=0) # delete the second event

# create 1-min epochs around these events
epochs = mne.Epochs(
    raw, events_sw_post, event_id={cond: cond_id}, tmin=-30, tmax=30, baseline=None, preload=True
)
# downsample to 250Hz, otherwise stc will be too large (due to very long epochs)
# (checked on raw data - spikes are still pretty obvious after downsampling)
epochs.decimate(4) 
# this step is done here as it's better not to epoch after downsampling:
# https://mne.tools/stable/generated/mne.io.Raw.html#mne.io.Raw.resample
# https://mne.tools/stable/auto_tutorials/preprocessing/30_filtering_resampling.html#best-practices

# average the epochs
evoked = epochs[cond].average()
fname_evoked = op.join(results_dir, subject_MEG + '_' + cond + '-ave.fif')
evoked.save(fname_evoked)

# compute data cov
data_cov = mne.compute_covariance(epochs) # use the whole epochs
#data_cov.plot(epochs.info)

# compute noise cov from empty room data
# https://mne.tools/dev/auto_tutorials/forward/90_compute_covariance.html
raw_empty_room = mne.io.read_raw_kit(raw_emptyroom_fname)
raw_empty_room.filter(l_freq=3, h_freq=70) # just to be consistent (not sure if required)
raw_empty_room.decimate(4) # just to be consistent (not sure if required)
noise_cov = mne.compute_raw_covariance(
    raw_empty_room, tmin=0, tmax=None)


# LCMV beamformer
# https://mne.tools/stable/auto_tutorials/inverse/50_beamformer_lcmv.html

fwd = mne.read_forward_solution(fname_fwd)
src = fwd["src"]

# compute the spatial filter (LCMV beamformer) - use common filter for all conds?
if op.exists(fname_filters):
    filters = mne.beamformer.read_beamformer(fname_filters)
else:
    filters = make_lcmv(evoked.info, fwd, data_cov, reg=0.05,
        noise_cov=noise_cov, pick_ori='max-power', # 1 estimate per voxel (only preserve the axis with max power)
        weight_norm='unit-noise-gain', rank=None)
    # save the filters for later
    filters.save(fname_filters)

# save some memory
del raw, raw_empty_room, epochs, fwd

# load the saved evoked
cond = 'sw_post'
fname_evoked = op.join(results_dir, subject_MEG + '_' + cond + '-ave.fif')
evoked = mne.read_evokeds(fname_evoked)
evoked = evoked[0]

# apply the spatial filter
evoked = epochs[cond][4].average() # TODO: apply spatial filter on indi epochs
stcs = dict()
stcs[cond] = apply_lcmv(evoked, filters)

# plot the reconstructed source activity
# (Memory intensive due to 1-min epochs - cannot run if we didn't downsample from 1000Hz)
#lims = [0.3, 0.45, 0.6] # set colour scale
kwargs = dict(src=src, subject=subject, subjects_dir=subjects_dir, verbose=True)
brain = stcs[cond].plot_3d(   
    #clim=dict(kind='value', lims=lims), 
    hemi='both', size=(600, 600),
    #views=['sagittal'], # only show sag view
    view_layout='horizontal', views=['coronal', 'sagittal', 'axial'], # make a 3-panel figure showing all views
    brain_kwargs=dict(silhouette=True),
    **kwargs)

# compute kurtosis for each vertex
kurtosis = univariate.compute_kurtosis(stcs[cond].data)

# find the vertex (or cluster of vertices) with maximum kurtosis
print(np.max(kurtosis))
VS_list = np.where(kurtosis > 6.5) 
VS_list
# note: Rui's paper uses all the local maxima on the kurtosis map, we are just using an absolute cutoff here

# plot the kurtosis value for each vertex on the source model
'''
tmp = copy.copy(stcs[cond]) # make a fake stc by copying the structure
tmp.data = kurtosis.reshape(-1,1) # convert 1d array to 2d
kwargs = dict(src=src, subject=subject, subjects_dir=subjects_dir, verbose=True)
tmp.plot_3d(   
    clim=dict(kind='value', lims=[2.9, 3.3, 3.6]),
    #clim=dict(kind='percent', lims=[90, 95, 99]),
    smoothing_steps=7,
    #hemi='both', size=(600, 600),
    #views=['sagittal'], # only show sag view
    view_layout='horizontal', views=['coronal', 'sagittal', 'axial'], # make a 3-panel figure showing all views
    brain_kwargs=dict(silhouette=True),
    **kwargs)
'''


# now we should visually inspect the stc for each of these vertices
# to see if the timing of spikes match the ones marked in raw data (i.e. around time 0)

# TODO: how to plot the time course for a particular vertex?
# These don't work:
#brain.add_foci(VS_list, coords_as_verts = True, hemi = 'lh')
#brain.plot_time_course(hemi='lh', vertex_id=6442, color='r')
#mne.viz.plot_compare_evokeds(stcs, picks=VS_list)
vertex_id = 2291
plt.plot(stcs[cond].data[vertex_id, :])
plt.show()
# to find the index for a particular vertex (e.g. the one chosen by stc.plot_3d)
#np.where(stcs[cond].vertices[0] == 16453)

# If the spikes seem correct in timing, then we are done?
# (i.e. high kurtosis value == source of the spikes)
# But there are more steps in Rui's paper, what do these steps mean?


# Qs:
# Rui's paper does beamforming on each indi 3-min segment (does this mean it's not
# 3-min epochs, but breaking each 3-min segment into many epochs? I did actual 1-min 
# epochs), so they get a set of kurtosis values for each segment, then pick the vertex
# with max kurtosis value in each segment
