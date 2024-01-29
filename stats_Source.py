#!/usr/bin/python3
#
# Statistical analysis in source space
#
# Authors: Judy Zhu

#######################################################################################

import os.path as op
import numpy as np
import matplotlib.pyplot as plt
import glob

import mne


# set up file and folder paths here
exp_dir = '/mnt/d/Work/analysis_ME206/' #'/home/jzhu/analysis_mne/'
meg_task = '_localiser' #'_1_oddball' #''
run_name = '_TSPCA'

# specify which version of results to read in
source_result = 'mne_surface' #"beamformer_vol" #"mne_vol"
# specify the type of source space used for this run 
src_type = 'surface' #'vol'

# which cond to look at: (default == average across conditions)
#cond = 'ba'


# All paths below should be automatic
processing_dir = op.join(exp_dir, "processing")
subjects_dir = op.join(processing_dir, "mri")
subject='fsaverage'

results_dir = op.join(exp_dir, "results")
source_results_dir = op.join(results_dir, 'meg', 'source', meg_task[1:] + run_name, source_result)
figures_dir = op.join(source_results_dir, 'Figures') # where to save the figure



# = Grand average of source estimates =
# (can't use mne.grand_average, as that only works for Evoked or TFR objects)

if src_type == 'vol':

    # find all the saved stc results
    stc_files = glob.glob(op.join(source_results_dir, 'G*-vl.stc'))

    # initialise the sum array to correct size using the first subject's stc
    stc = mne.read_source_estimate(stc_files[0])
    stcs_sum = stc.data

    # read in the stc for each subsequent subject or cond, add to the sum array
    for fname in stc_files[1:]:
        stc = mne.read_source_estimate(fname)
        stcs_sum = stcs_sum + stc.data

    # divide by number of files
    stcs_GA = stcs_sum / len(stc_files)

    # feed into the dummy stc structure
    stc.data = stcs_GA


    # Plot the GA stc
    src_fname = op.join(subjects_dir, subject, "bem", subject + "_vol-src.fif") 
    src = mne.read_source_spaces(src_fname)
    fig = stc.plot(src=src, 
        subject=subject, subjects_dir=subjects_dir, verbose=True,
        #mode='glass_brain',
        initial_time=0.08)
    fig.savefig(op.join(figures_dir, 'GA' + meg_task + run_name + '.png'))


elif src_type == 'surface':

    # find all the saved stc results
    stc_files = glob.glob(op.join(source_results_dir, 'G*-lh.stc'))
    # only need to supply the filename for one hemisphere, it will look for both
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
    stcs_GA = stcs_sum / len(stc_files)

    # feed into the dummy stc structure
    stc.data = stcs_GA


    # Plot the GA stc
    hemi='lh'
    vertno_max, time_max = stc.get_peak(hemi=hemi)
    initial_time = time_max
    surfer_kwargs = dict(
        hemi=hemi, subject=subject, subjects_dir=subjects_dir, 
        #clim=dict(kind='value', lims=[8, 12, 15]), 
        initial_time=initial_time, 
        time_unit='s', views='lateral', size=(800, 800), smoothing_steps=10)
    brain = stc.plot(**surfer_kwargs)
    #brain.add_foci(vertno_max, coords_as_verts=True, hemi=hemi, color='blue',
    #            scale_factor=0.6, alpha=0.5)
    brain.save_image(op.join(figures_dir, 'GA-' + str(initial_time) + 's-' + hemi + '.png'))

    # see also:
    # https://mne.tools/stable/auto_tutorials/inverse/60_visualize_stc.html#surface-source-estimates


# Note: if plots are not working - it's probably a rendering issue in Linux. 
# Try running this script in windows!

print('All done')
