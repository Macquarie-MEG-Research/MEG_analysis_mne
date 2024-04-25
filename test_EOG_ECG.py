#!/usr/bin/python3
#
# Test out using EOG & ECG channels to automatically pick ICA components for artifact rejection
#
# Authors: Judy Zhu

#######################################################################################

import os
import mne
import meegkit
import glob
import matplotlib.pyplot as plt
import numpy as np
import copy


# set up file and folder paths here
exp_dir = "/mnt/d/Work/analysis_ME211/"
subject_MEG = 'test_EOG_ECG'
task = 'oddball'; #'_1_oddball' #''
run_name = '_TSPCA'

# the paths below should be automatic
data_dir = exp_dir + "data/"
processing_dir = exp_dir + "processing/"
results_dir = exp_dir + "results/"
meg_dir = data_dir + subject_MEG + "/meg/"
save_dir = processing_dir + "meg/" + subject_MEG + "/" # where to save the epoch files for each subject
ica_fname = save_dir + subject_MEG + '-ica.fif'
figures_dir = results_dir + 'meg/sensor/' + task + run_name + '/Figures/' # where to save the figures for all subjects
epochs_fname = save_dir + subject_MEG + "_" + task + run_name + "-epo.fif"
# create the folders if needed
os.system('mkdir -p ' + save_dir)
os.system('mkdir -p ' + figures_dir)


#%% === Read raw data === #

# find the MEG files
#print(glob.glob(task + ".con"))
fname_raw = glob.glob(meg_dir + "*" + task + ".con")
fname_elp = glob.glob(meg_dir + "*.elp")
fname_hsp = glob.glob(meg_dir + "*.hsp")
fname_mrk = glob.glob(meg_dir + "*.mrk")
# find the EEG files (for EOG & ECG recordings)
fname_eeg = glob.glob(meg_dir + "*" + task + ".vhdr")

# Raw MEG data
raw_meg = mne.io.read_raw_kit(
    fname_raw[0], 
    mrk=fname_mrk[0],
    elp=fname_elp[0],
    hsp=fname_hsp[0],
    stim=[*[166], *range(176, 192), *range(194, 199)], # misc 007 = audio, the rest are triggers
    slope="+",
    stim_code="channel",
    stimthresh=2,  # 2 for adult
    preload=True,
    allow_unknown_format=False,
    verbose=True,
)

# Apply TSPCA for noise reduction
noisy_data = raw_meg.get_data(picks="meg").transpose()
noisy_ref = raw_meg.get_data(picks=[160,161,162]).transpose()
data_after_tspca, idx = meegkit.tspca.tsr(noisy_data, noisy_ref)[0:2]
raw_meg._data[0:160] = data_after_tspca.transpose()

# Raw EEG data
raw_eeg = mne.io.read_raw_brainvision(fname_eeg[0]).load_data()
eeg_renames = {'1':'ECG', '2':'EOG'}
ch_types_map = dict(ECG="ecg", EOG="eog")
raw_eeg.rename_channels(eeg_renames)
raw_eeg.set_channel_types(ch_types_map)
assert raw_eeg.first_time == 0
assert raw_eeg.info["sfreq"] == raw_meg.info["sfreq"]


# Find triggers so we can combine the raw MEG & EEG data
meg_event = mne.find_events(
    raw_meg,
    output="onset",
    consecutive=False,
    min_duration=0,
    shortest_event=1,
    mask=None,
    uint_cast=False,
    mask_type="and",
    initial_event=False,
    verbose=None,
)
eeg_event = mne.events_from_annotations(
    raw_eeg,
)[0]

# find the first trigger on Ch182 (MEG) / Stimulus 22 (EEG)
meg_idx = np.where(meg_event[:, 2] == 182)
meg_samp = meg_event[meg_idx[0][0], 0]
eeg_idx = np.where(eeg_event[:, 2] == 22)
eeg_samp = eeg_event[eeg_idx[0][0], 0]

# Code below for merging MEG & EEG data written by Eric Larson

# Instead of cropping MEG, let's just zero-order hold the first or last EEG
# sample. This will make timing of events align with the original MEG
# data.
if eeg_samp < meg_samp:
    n_pad = meg_samp - eeg_samp
    raw_eeg_pad = raw_eeg.copy().crop(0, (n_pad - 1) / raw_eeg.info["sfreq"])
    assert len(raw_eeg_pad.times) == n_pad
    raw_eeg_pad._data[:] = raw_eeg[:, 0][0]
    raw_eeg_pad.set_annotations(None)
    raw_eeg = mne.concatenate_raws([raw_eeg_pad, raw_eeg])
    del raw_eeg_pad
elif eeg_samp > meg_samp:
    raw_eeg.crop((eeg_samp - meg_samp) / raw_eeg.info["sfreq"], None)
if len(raw_eeg.times) < len(raw_meg.times):
    n_pad = len(raw_meg.times) - len(raw_eeg.times)
    raw_eeg_pad = raw_eeg.copy().crop(0, (n_pad - 1) / raw_eeg.info["sfreq"])
    assert len(raw_eeg_pad.times) == n_pad
    raw_eeg_pad._data[:] = raw_eeg[:, -1][0]
    raw_eeg_pad.set_annotations(None)
    raw_eeg = mne.concatenate_raws([raw_eeg, raw_eeg_pad])
    del raw_eeg_pad
elif len(raw_eeg.times) > len(raw_meg.times):
    raw_eeg.crop(0, (len(raw_meg.times) - 1) / raw_eeg.info["sfreq"])

# fix the info field to be consistent with MEG data
for key in ("dev_head_t", "description"):
    raw_eeg.info[key] = raw_meg.info[key]
with raw_eeg.info._unlock():
    for key in ("highpass", "lowpass"):
        raw_eeg.info[key] = raw_meg.info[key]

# add EOG & ECG channels into MEG data
raw_meg.add_channels([raw_eeg])


#%% === Preprocessing === #

# browse data to identify bad channels
raw_meg.plot()
raw_meg.info["bads"] = ["MEG 043"]

# filtering
raw_meg.filter(l_freq=1, h_freq=30)


# run ICA
if os.path.exists(ica_fname): # if we've run it before, just load the components
    ica = mne.preprocessing.read_ica(ica_fname)
else:
    ica = mne.preprocessing.ICA(n_components=60, max_iter="auto", random_state=97)
    ica.fit(raw_meg)
    ica.save(ica_fname)

# plot ICA results
ica.plot_sources(raw_meg) # plot IC time series  #, picks = [0,1,2,3,4,5,6,17,46]
ica.plot_components() # plot IC topography

# use EOG & ECG channels to automatically select which IC comps to reject
# https://mne.tools/stable/auto_tutorials/preprocessing/40_artifact_correction_ica.html#using-an-eog-channel-to-select-ica-components
ica.exclude = []
eog_indices, eog_scores = ica.find_bads_eog(raw_meg)
ecg_indices, ecg_scores = ica.find_bads_ecg(raw_meg)
ica.exclude = [eog_indices, ecg_indices]

# barplot of ICA component "ECG match" scores
ica.plot_scores(ecg_scores)
# plot diagnostics
ica.plot_properties(raw_meg, picks=ecg_indices)

# Compare raw data before & after IC rejection
raw_orig = copy.deepcopy(raw_meg) # need to make a copy, otherwise the 'before'
    # and 'after' plots become the same (even if you do the 'before' plot
    # first, then apply ICA, it still gets updated to look the same as the 'after' plot)
raw_orig.plot(title='before ICA')
ica.apply(raw_meg) # apply component rejection onto raw (continuous) data
                   # Note: data will be modified in-place
raw_meg.plot(title='after ICA')


#%% === Epoching & sensor space analysis === #


#%% === Source analysis === #

