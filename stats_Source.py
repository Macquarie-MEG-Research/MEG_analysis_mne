#!/usr/bin/python3
#
# Statistical analysis in source space
#
# Authors: Judy Zhu

#######################################################################################

import os
import os.path as op
import numpy as np
import matplotlib.pyplot as plt
import glob

import mne
from scipy import stats
from mne.stats import spatio_temporal_cluster_1samp_test, summarize_clusters_stc


# set up file and folder paths here
exp_dir = '/mnt/d/Work/analysis_ME209/' #'/home/jzhu/analysis_mne/'
meg_task = 'dual_stim' #'localiser' #'1_oddball' #''
run_name = '' #'_TSPCA'

subjects_dir = '/mnt/d/Work/analysis_ME206/processing/mri/'
subject = 'fsaverage'

# specify conditions
conds = ['HF', 'LF']

# specify which version of results to read in
source_result = 'dSPM_surface' #"beamformer_vol" #"mne_vol"

# specify the type of source space used for this run 
src_type = 'surface' #'vol'

if src_type == 'vol':
    src_fname = op.join(subjects_dir, subject, "bem", subject + "_vol-src.fif") 
    src_suffix = '-vl.stc'
elif src_type == 'surface':
    src_fname = op.join(subjects_dir, subject, "bem", subject + "_oct6" + src_type + "-src.fif") 
    src_suffix = '-lh.stc'


# All paths below should be automatic
results_dir = op.join(exp_dir, "results")
#source_results_dir = op.join(results_dir, 'meg', 'source', meg_task[1:] + run_name, source_result)
source_results_dir = op.join(results_dir, 'meg', 'source', meg_task + run_name, source_result)
figures_dir = op.join(source_results_dir, 'Figures') # where to save the figure
figures_ROI_dir = op.join(source_results_dir, 'Figures_ROI')
os.system('mkdir -p ' + figures_ROI_dir) # create the folder if needed



# = Grand average of source estimates =
# (can't use mne.grand_average, as that only works for Evoked or TFR objects)

GA_stcs = {}
for cond in conds:

    # find all the saved stc results
    stc_files = glob.glob(op.join(source_results_dir, '*-' + cond + src_suffix))
    # Note: for surface-based stcs, only need to supply the filename for 
    # one hemisphere (e.g. '-lh.stc') and it will look for both
    # https://mne.tools/stable/generated/mne.read_source_estimate.html

    # initialise the sum array to correct size using the first subject's stc
    stc = mne.read_source_estimate(stc_files[0])
    stcs_sum = stc.data # this contains lh & rh vertices combined together
    # there are also separate fields for the 2 hemis (stc.lh_data, stc.rh_data),
    # but their contents cannot be set directly, so just use the combined one

    # read in the stc for each subsequent subject or cond, add to the sum array
    for fname in stc_files[1:]:
        stc = mne.read_source_estimate(fname)
        stcs_sum = stcs_sum + stc.data

    # divide by number of files
    stc.data = stcs_sum / len(stc_files)

    # store in the GA struct
    GA_stcs[cond] = stc


    # Plot the GAs
    # Depending on the src type, use diff types of plots
    if src_type == 'vol':
        src = mne.read_source_spaces(src_fname)
        fig = GA_stcs[cond].plot(src=src, 
            subject=subject, subjects_dir=subjects_dir, verbose=True,
            #mode='glass_brain',
            initial_time=0.08)
        fig.savefig(op.join(figures_dir, 'GA' + meg_task + run_name + '.png'))

    elif src_type == 'surface':
        '''
        hemi='lh'
        vertno_max, time_max = stc.get_peak(hemi=hemi)
        initial_time = time_max
        surfer_kwargs = dict(
            hemi=hemi, subject=subject, subjects_dir=subjects_dir, 
            #clim=dict(kind='value', lims=[8, 12, 15]), 
            initial_time=initial_time, 
            time_unit='s', views='lateral', size=(800, 800), smoothing_steps=10)
        brain = GA_stcs[cond].plot(**surfer_kwargs)
        #brain.add_foci(vertno_max, coords_as_verts=True, hemi=hemi, color='blue',
        #            scale_factor=0.6, alpha=0.5)
        brain.save_image(op.join(figures_dir, 'GA-' + str(initial_time) + 's-' + hemi + '.png'))
        '''

        # ME209 - specify scales for brain plot & time windows for movies
        if 'dual_' in meg_task:
            if '_sac' in meg_task: # saccade-locked analysis
                clim=dict(kind='value', lims=[2.5, 4.5, 6.5]) # can adjust based on final GA plots
                tmin = -0.3
                tmax = 0
            else: # stimulus-locked analysis
                clim=dict(kind='value', lims=[6, 10.5, 15]) # can adjust based on final GA plots
                tmin = 0
                tmax = 0.35
        
        hemi='both'
        #vertno_max, time_max = stcs[cond].get_peak(hemi=hemi, tmin=0.1, tmax=0.27)
        initial_time = 0.16 #time_max
        surfer_kwargs = dict(
            hemi=hemi, subject=subject, subjects_dir=subjects_dir,
            #clim=clim, # explicitly set the scales to make them consistent across conds
            initial_time=initial_time,
            time_unit='s', title='GA_' + cond,
            views=['caudal','ventral','lateral','medial'],
            show_traces=False, # use False to avoid having the blue dot (peak vertex) showing up on the brain
            smoothing_steps=10)
        brain = GA_stcs[cond].plot(**surfer_kwargs)
        #brain.add_foci(vertno_max, coords_as_verts=True, hemi=hemi,
        #    color='blue', scale_factor=0.6, alpha=0.5)
        #brain.save_image(op.join(figures_dir, subject_MEG + meg_task + run_name + '-' + cond + '.png'))
        brain.save_movie(op.join(figures_dir, run_name + 'GA-' + meg_task + '-' + cond + '-' + hemi + '.mov'),
            tmin=tmin, tmax=tmax, interpolation='linear',
            time_dilation=50, time_viewer=True)
        # see also:
        # https://mne.tools/stable/auto_tutorials/inverse/60_visualize_stc.html#surface-source-estimates


# Note: if plots are not working - it's probably a rendering issue in Linux. 
# Try running this script in windows!

# We can also plot the difference between two conditions
GA_stc_diff = GA_stcs['LF'] - GA_stcs['HF']

hemi='both'
initial_time = 0.16
surfer_kwargs = dict(
    hemi=hemi, subject=subject, subjects_dir=subjects_dir,
    initial_time=initial_time,
    time_unit='s', title='GA_' + cond,
    views=['caudal','ventral','lateral','medial'],
    show_traces=False, # use False to avoid having the blue dot (peak vertex) showing up on the brain
    smoothing_steps=10)
brain = GA_stcs[cond].plot(**surfer_kwargs)
brain.save_movie(op.join(figures_dir, run_name + 'GA-' + meg_task + '-diff-' + hemi + '.mov'),
    tmin=tmin, tmax=tmax, interpolation='linear',
    time_dilation=50, time_viewer=True)



# = Extract ROI time courses from source estimates =

# load source space
src = mne.read_source_spaces(src_fname)

os.system(f'mkdir -p {op.join(figures_ROI_dir, "all_ROIs")}')

if src_type == 'surface':
    # Get labels for FreeSurfer 'aparc' cortical parcellation (69 labels)
    # https://freesurfer.net/fswiki/CorticalParcellation
    labels_parc = mne.read_labels_from_annot(subject, parc='aparc', subjects_dir=subjects_dir) # read all labels from annot file
    labels = [labels_parc[12], labels_parc[13]] # 12: fusiform-lh; 13: fusiform-rh
    
    # to visualise the locations of ROIs
    # Note: the rendering doesn't work in Linux/WSL
    '''
    brain = mne.viz.Brain("fsaverage")
    for label in labels:
        brain.add_label(label)
    '''

    for label in labels:
        label_name = label.name
        #os.system(f'mkdir -p {op.join(figures_ROI_dir, label_name)}')

        avgovertime_table = [] # table to contain the "average over time" values for each ROI

        # Plot grand average ROI time series
        fig, axes = plt.subplots(1, layout="constrained")    
        for cond in conds:
            label_ts = mne.extract_label_time_course(
                [GA_stcs[cond]], label, src, mode="auto"
            )
            label_ts = label_ts[0][0]
            axes.plot(1e3 * GA_stcs[cond].times, label_ts, label=cond)

        axes.axvline(linestyle='-', color='k') # add verticle line at time 0
        axes.set(xlabel="Time (ms)", ylabel="Activation")
        axes.legend()

        fig.savefig(op.join(figures_ROI_dir, "all_ROIs", label_name + "_GA.png"))
        plt.close(fig)


        # Extract individual subjects ROI time series
        all_subjects_label_ts = {cond: [] for cond in conds} # initialise an empty struct
        #fig, axes = plt.subplots(1, layout="constrained")    
        for cond in conds:
            # find all the saved stc results
            stc_files = glob.glob(op.join(source_results_dir, '*-' + cond + src_suffix))
        
            for stc_file in stc_files:
                stc = mne.read_source_estimate(stc_file)
                label_ts = mne.extract_label_time_course(
                    [stc], label, src, mode="auto"
                )
                label_ts = label_ts[0][0]
                all_subjects_label_ts[cond].append(label_ts)

            # Sanity check - take the mean over all subjects (plot looks the same as the label_ts extracted from GA stc above)
            #axes.plot(1e3 * stc.times, np.mean(all_subjects_label_ts[cond], axis=0), label=cond)            

            # Average over a particular time window
            if '_sac' in meg_task: # saccade-locked analysis
                time_interval = range(22, 29) # 180-120ms before saccade onset
            else: # stimulus-locked analysis
                time_interval = range(30, 41) # 200-300ms after stim onset
            stc.times[time_interval] # verify the time interval is correct
            avgovertime_row = [cond] # add the condition name to the beginning of the row
            for subject_label_ts in all_subjects_label_ts[cond]:
                avg = np.mean(subject_label_ts[time_interval], axis=0)
                avgovertime_row.append(avg) # append the avgovertime value for each subject
            avgovertime_table.append(avgovertime_row) # add this condition to the overall table
            avgovertime_table = np.transpose(avgovertime_table) # transpose so that each column is a condition

        # write table to file
        avgovertime_dir = op.join(source_results_dir, "ROI_avgovertime")
        os.system(f'mkdir -p {avgovertime_dir}')
        with open(op.join(avgovertime_dir, label_name + ".csv"), "w") as file:
            for row in avgovertime_table:
                file.write(",".join(map(str, row)) + "\n")


        # Automatically detect temporal clusters (as an alternative to avgovertime)
        # https://mne.tools/stable/auto_tutorials/stats-source-space/20_cluster_1samp_spatiotemporal.html#compute-statistic
        
        # calculate the difference between two conditions
        all_subjects_diff = np.array(all_subjects_label_ts['LF']) - np.array(all_subjects_label_ts['HF'])

        X = np.squeeze(all_subjects_diff) # subjects x time
        X = np.expand_dims(X, axis=2) # subjects x time x space
        
        #adjacency = mne.spatial_src_adjacency(src)
        
        # We use a two-tailed threshold, the "1 - p_threshold" is needed
        # because for two-tailed tests we must specify a positive threshold
        p_threshold = 0.001
        df = len(X) - 1  # degrees of freedom for the test
        t_threshold = stats.distributions.t.ppf(1 - p_threshold / 2, df=df)
        
        T_obs, clusters, cluster_p_values, H0 = clu = spatio_temporal_cluster_1samp_test(
            X,
            threshold=t_threshold,
            adjacency=None, # CHECK - is it correct to specify this as None (in order to only look for temporal clusters)? We have no spatial dimension here
            n_permutations=1000,
        )

        # Select the clusters that are statistically significant at p < 0.05
        good_clusters_idx = np.where(cluster_p_values < 0.05)[0]
        good_clusters = [clusters[idx] for idx in good_clusters_idx]
        # Note: a "cluster" is a two-element tuple: the first array contains the time points, 
        # the second array contains the spatial indices

        for i_clu, clu_idx in enumerate(good_clusters_idx):
            # unpack cluster information, get unique indices
            time_inds, space_inds = np.squeeze(clusters[clu_idx])
            time_inds = np.unique(time_inds)

            # find the temporal extent of the cluster
            sig_times = stc.times[time_inds]
            
        '''
        # Visualize the clusters (may not work for temporal-only clusters)
        tstep = stc.tstep
        stc_all_cluster_vis = summarize_clusters_stc(
            clu, tstep=tstep, 
            vertices=[[0]], # we have no spatial dimension here, so just put a dummy value
            subject="fsaverage"
        )
        brain = stc_all_cluster_vis.plot(
            #hemi="both",
            #views="lateral",
            subjects_dir=subjects_dir,
            src=src,
            #time_label="temporal extent (ms)",
            #size=(800, 800),
            #smoothing_steps=5,
            #clim=dict(kind="value", pos_lims=[0, 1, 40]),
        )
        '''

print('All done')
