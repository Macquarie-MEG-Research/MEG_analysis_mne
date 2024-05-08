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
raw = mne.io.read_raw_kit(
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
noisy_data = raw.get_data(picks="meg").transpose()
noisy_ref = raw.get_data(picks=[160,161,162]).transpose()
data_after_tspca, idx = meegkit.tspca.tsr(noisy_data, noisy_ref)[0:2]
raw._data[0:160] = data_after_tspca.transpose()

# Raw EEG data
raw_eeg = mne.io.read_raw_brainvision(fname_eeg[0]).load_data()
eeg_renames = {'1':'ECG', '2':'EOG'}
ch_types_map = dict(ECG="ecg", EOG="eog")
raw_eeg.rename_channels(eeg_renames)
raw_eeg.set_channel_types(ch_types_map)
assert raw_eeg.first_time == 0
assert raw_eeg.info["sfreq"] == raw.info["sfreq"]


# Find triggers so we can combine the raw MEG & EEG data
events = mne.find_events(
    raw,
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
eeg_events = mne.events_from_annotations(
    raw_eeg,
)[0]

# find the first trigger on Ch182 (MEG) / Stimulus 22 (EEG)
meg_idx = np.where(events[:, 2] == 182)
meg_samp = events[meg_idx[0][0], 0]
eeg_idx = np.where(eeg_events[:, 2] == 22)
eeg_samp = eeg_events[eeg_idx[0][0], 0]

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
if len(raw_eeg.times) < len(raw.times):
    n_pad = len(raw.times) - len(raw_eeg.times)
    raw_eeg_pad = raw_eeg.copy().crop(0, (n_pad - 1) / raw_eeg.info["sfreq"])
    assert len(raw_eeg_pad.times) == n_pad
    raw_eeg_pad._data[:] = raw_eeg[:, -1][0]
    raw_eeg_pad.set_annotations(None)
    raw_eeg = mne.concatenate_raws([raw_eeg, raw_eeg_pad])
    del raw_eeg_pad
elif len(raw_eeg.times) > len(raw.times):
    raw_eeg.crop(0, (len(raw.times) - 1) / raw_eeg.info["sfreq"])

# fix the info field to be consistent with MEG data
for key in ("dev_head_t", "description"):
    raw_eeg.info[key] = raw.info[key]
with raw_eeg.info._unlock():
    for key in ("highpass", "lowpass"):
        raw_eeg.info[key] = raw.info[key]

# add EOG & ECG channels into MEG data
raw.add_channels([raw_eeg])


#%% === Preprocessing === #

# browse data to identify bad channels
raw.plot()
raw.info["bads"] = ["MEG 043"]

# filtering
raw.filter(l_freq=1, h_freq=30)


# run ICA
if os.path.exists(ica_fname): # if we've run it before, just load the components
    ica = mne.preprocessing.read_ica(ica_fname)
else:
    ica = mne.preprocessing.ICA(n_components=60, max_iter="auto", random_state=97)
    ica.fit(raw)
    ica.save(ica_fname)

# plot ICA results
ica.plot_sources(raw) # plot IC time series  #, picks = [0,1,2,3,4,5,6,17,46]
ica.plot_components() # plot IC topography

# use EOG & ECG channels to automatically select which IC comps to reject
# https://mne.tools/stable/auto_tutorials/preprocessing/40_artifact_correction_ica.html#using-an-eog-channel-to-select-ica-components
ica.exclude = []
eog_indices, eog_scores = ica.find_bads_eog(raw)
ecg_indices, ecg_scores = ica.find_bads_ecg(raw)
ica.exclude = eog_indices + ecg_indices

# barplot of ICA component "ECG match" scores
ica.plot_scores(ecg_scores)
# plot diagnostics
#ica.plot_properties(raw, picks=ecg_indices)

# Compare raw data before & after IC rejection
raw_orig = copy.deepcopy(raw) # need to make a copy, otherwise the 'before'
    # and 'after' plots become the same (even if you do the 'before' plot
    # first, then apply ICA, it still gets updated to look the same as the 'after' plot)
raw_orig.plot(title='before ICA')
ica.apply(raw) # apply component rejection onto raw (continuous) data
               # Note: data will be modified in-place
raw.plot(title='after ICA')


#%% === Event timing correction === #

# Adjust trigger timing based on audio channel signal 

# get rid of audio triggers for now
events = np.delete(events, np.where(events[:, 2] == 166), 0)

# get raw audio signal from ch166
aud_ch_data_raw = raw.get_data(picks="MISC 007")

def getEnvelope(inputSignal, thresh=0.2):
    # Taking the absolute value
    absoluteSignal = []
    for sample in inputSignal:
        absoluteSignal.append(abs(sample))
    absoluteSignal = absoluteSignal[0]

    # Peak detection
    intervalLength = 15  # Experiment with this number!
    outputSignal = []

    # Like a sample and hold filter
    for baseIndex in range(intervalLength, len(absoluteSignal)):
        maximum = 0
        for lookbackIndex in range(intervalLength):
            maximum = max(absoluteSignal[baseIndex - lookbackIndex], maximum)
        outputSignal.append(maximum)

    outputSignal = np.concatenate(
        (
            np.zeros(intervalLength),
            np.array(outputSignal)[:-intervalLength],
            np.zeros(intervalLength),
        )
    )
    # finally binarise the output at a threshold of 2.5  <-  adjust this 
    # threshold based on diagnostic plot below!
    return np.array([1 if np.abs(x) > thresh else 0 for x in outputSignal])

#raw.load_data().apply_function(getEnvelope, picks="MISC 006")
envelope = getEnvelope(aud_ch_data_raw)
envelope = envelope.tolist() # convert ndarray to list
# detect the beginning of each envelope (set the rest of the envelope to 0)
new_stim_ch = np.clip(np.diff(envelope),0,1)
# find all the 1s (i.e. audio triggers)
stim_tps = np.where(new_stim_ch==1)[0]

# compare number of events from trigger channels & from AD
print("Number of events from trigger channels:", events.shape[0])
print("Number of events from audio channel (166) signal:", stim_tps.shape[0])
print("Note: these numbers won't match here, as omission deviants do not produce an audio signal!")


# calculate the audio delay on each trial
AD_delta = []
for i in range(events.shape[0]):
    idx = np.where((stim_tps >= events[i,0]) & (stim_tps < events[i,0]+100))
    if len(idx[0]): # if an AD trigger exists within 100ms of event trigger
        idx = idx[0][0] # use the first AD trigger (if there are multiple)
        AD_delta.append(stim_tps[idx] - events[i,0]) # keep track of audio delay values (for histogram)

# histogram showing the distribution of audio delays
n, bins, patches = plt.hist(
    x=AD_delta, bins="auto", color="#0504aa", alpha=0.7, rwidth=0.85
)
plt.grid(axis="y", alpha=0.75)
plt.xlabel("Delay (ms)")
plt.ylabel("Frequency")
plt.title("Audio Detector Delays")
plt.text(
    70,
    50,
    r"$mean="
    + str(round(np.mean(AD_delta)))
    + ", std="
    + str(round(np.std(AD_delta)))
    + "$",
)
maxfreq = n.max()
# set a clean upper y-axis limit
plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)

# apply timing correction using the average AD delta (~47ms)
delay = np.median(AD_delta)
print ("The median audio delay is", delay, "ms")
events_corrected = copy.copy(events) # work on a copy so we don't affect the original
for i in range(events_corrected.shape[0]):
    events_corrected[i,0] = events_corrected[i,0] + delay # update event timing


#%% === Epoching === #

# specify the event IDs
event_ids = {
    "standard": 186,
    "short": 191,
    "omitted": 198,
}

if not os.path.exists(epochs_fname):
    epochs = mne.Epochs(raw, events_corrected, event_id=event_ids, tmin=-0.1, tmax=0.41, preload=True)

    conds_we_care_about = ["standard", "short", "omitted"]
    epochs.equalize_event_counts(conds_we_care_about)

    # downsample to 100Hz
    print("Original sampling rate:", epochs.info["sfreq"], "Hz")
    epochs_resampled = epochs.copy().resample(100, npad="auto")
    print("New sampling rate:", epochs_resampled.info["sfreq"], "Hz")

    # save for later use (e.g. in Source_analysis script)
    epochs_resampled.save(epochs_fname, overwrite=True)

# plot ERFs
if not os.path.exists(figures_dir + subject_MEG + '_AEF_butterfly.png'):
    epochs_resampled = mne.read_epochs(epochs_fname)

    fig = epochs_resampled.average().plot(spatial_colors=True, gfp=True)
    fig.savefig(figures_dir + subject_MEG + '_AEF_butterfly.png')

    fig2 = mne.viz.plot_compare_evokeds(
        [
            epochs_resampled["standard"].average(),
            epochs_resampled["short"].average(),
            epochs_resampled["omitted"].average(),
        ],
        #combine = 'mean' # combine channels by taking the mean (default is GFP)
    )
    fig2[0].savefig(figures_dir + subject_MEG + '_AEF_gfp.png')


#%% === Source analysis === #

