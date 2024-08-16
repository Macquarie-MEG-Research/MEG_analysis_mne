'''
17.1.1.	Preprocessing steps based on MNE-BIDS-Pipeline
17.1.1.1.	Data de-noising using TSPCA-based noise reduction based on MEG reference sensors
17.1.1.2.	Downsample to 250Hz
17.1.1.3.	Apply low- and high-pass filters:
17.1.1.3.1.	For visual longterm potentiation and auditory oddball tasks: 0.1-40Hz
17.1.1.3.2.	For resting state: .1-150Hz with 50, 100 and 150 notch filters.
17.1.1.4.	Extract epochs
17.1.1.5.	Automated noisy channel identification and interpolation using RANSAC
17.1.1.6.	Automated artifact noise removal by ICA based on correlations of ICA components with MEG sensors EOG artifacts and ECG related components using cross-trial phase statistic
'''

import os
import mne
import meegkit # for TSPCA
import glob
import matplotlib.pyplot as plt
import numpy as np
import copy

from mne.preprocessing import find_bad_channels_maxwell
from autoreject import get_rejection_threshold  # noqa
from autoreject import Ransac  # noqa
from autoreject.utils import interpolate_bads  # noqa


# set up file and folder paths here
#exp_dir = "/mnt/d/Work/analysis_ME197/"
exp_dir = "C:/sync/OneDrive - Macquarie University/Studies/19_MEG_Microdosing/analysis/meg/"
subject_MEG = '230616_25065_S2'  #'230426_72956_S2'#'220503_87225_S1'  #'220112_p003'
meg_task = '_oddball' #''

# the paths below should be automatic
#data_dir = exp_dir + "data/"
data_dir = "C:/sync/OneDrive - Macquarie University/Studies/19_MEG_Microdosing/data/ACQUISITION/"
processing_dir = exp_dir + "processing/"
results_dir = exp_dir + "results/"
meg_dir = data_dir + subject_MEG + "/"
save_dir = processing_dir + subject_MEG + "/"
figures_dir_meg = results_dir + 'oddball' + '/Figures/' # where to save the figures for all subjects
epochs_fname = save_dir + subject_MEG + meg_task + "-epo.fif"
ica_fname = save_dir + subject_MEG + meg_task + "-ica.fif"
#os.system('mkdir -p ' + save_dir) # create the folder if needed
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
if not os.path.exists(figures_dir_meg):
    os.makedirs(figures_dir_meg)
print("exp_dir",exp_dir)
print("data_dir",data_dir)
print("processing_dir",processing_dir)
print("results_dir",results_dir)
print("meg_dir",meg_dir)
print("save_dir",save_dir)
print("figures_dir_meg",figures_dir_meg)
print("epochs_fname",epochs_fname)
print("ica_fname",ica_fname)


#%% === Read raw data === #

#print(glob.glob("*_oddball.con"))
fname_raw = glob.glob(meg_dir + "*" + meg_task + ".con")
fname_elp = glob.glob(meg_dir + "*.elp")
fname_hsp = glob.glob(meg_dir + "*.hsp")
fname_mrk = glob.glob(meg_dir + "*.mrk")

# Raw extraction ch misc 23-29 = triggers
# ch misc 007 = audio
raw = mne.io.read_raw_kit(
    fname_raw[0],  # change depending on file i want
    mrk=fname_mrk[0],
    elp=fname_elp[0],
    hsp=fname_hsp[0],
    stim=[*[166], *range(182, 190)],
    slope="+",
    stim_code="channel",
    stimthresh=2,  # 2 for adult (1 for child??)
    preload=True,
    allow_unknown_format=False,
    verbose=True,
)

#TEMP: crop for now to speed up processing
#raw.crop(tmax=120)

# Apply TSPCA for noise reduction
print("Starting TSPCA")
noisy_data = raw.get_data(picks="meg").transpose()
noisy_ref = raw.get_data(picks=[160,161,162]).transpose()
data_after_tspca, idx = meegkit.tspca.tsr(noisy_data, noisy_ref)[0:2]
raw._data[0:160] = data_after_tspca.transpose()
print("Finished TSPCA")

print("Starting resampling")
raw.resample(250)
print("Resampled")

print("Starting filter")
raw.filter(l_freq=0.1, h_freq=40)
print("Finished filter")

# browse data
raw.plot()
input("Close plot and then press Enter to continue...") # For reasons I don't understand I need to have this line after each plot to avoid errors on my pc (VP)

print("Finding events")
# Finding events
events = mne.find_events(
    raw,
    output="onset",
    consecutive=False,
    min_duration=0,
    shortest_event=1,  # 5 for adult
    mask=None,
    uint_cast=False,
    mask_type="and",
    initial_event=False,
    verbose=None,
)

# get rid of audio triggers for now
events = np.delete(events, np.where(events[:, 2] == 166), 0)

# re-code standard & deviant trials as '1' and '2'
#events = copy.deepcopy(events)
std_dev_bool = np.insert(np.diff(events[:, 2]) != 0, 0, "True") # find all deviants & mark as "True"
for idx, event in enumerate(std_dev_bool):
    if event and idx > 0: # for all deviants (except for the very first trial, which we won't use)
        events[idx, 2] = 2 # code current trial as '2'
        if events[idx - 1, 2] != 2:
            events[idx - 1, 2] = 1 # code previous trial as '1'

# specify the event IDs (these will be used during epoching)
event_ids = {
    "pre-deviant": 1,
    "deviant": 2,
}


# Adjust trigger timing based on audio channel signal 

# get raw audio signal from ch166
aud_ch_data_raw = raw.get_data(picks="MISC 007")

def getEnvelope(inputSignal):
    # Taking the absolute value
    absoluteSignal = []
    for sample in inputSignal:
        absoluteSignal.append(abs(sample))
    absoluteSignal = absoluteSignal[0]

    # Peak detection
    intervalLength = 5  # Experiment with this number!
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
    return np.array([1 if np.abs(x) > 0.2 else 0 for x in outputSignal])

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

# plot any problematic time period to aid diagnosis
'''
test_time = 454368
span = 10000
plt.figure()
plt.plot(aud_ch_data_raw[0], 'b')
#plt.plot(outputSignal, 'r')
for i in range(events.shape[0]):
   plt.axvline(events[i,0], color='b', lw=2, ls='--')
   plt.axvline(stim_tps[i], color='r', lw=2, ls='--')
plt.xlim(test_time-span, test_time+span)
plt.show()
input("Close plot and then press Enter to continue...")
'''

# if we have downsampled already, need to adjust the indices
decim = 1000 / raw.info['sfreq']

# apply timing correction onto the events array
events_corrected = copy.copy(events) # work on a copy so we don't affect the original

# Missing AD triggers can be handled:
# if there's an AD trigger within 200ms following normal trigger (this ensures 
# we've got the correct trial), update to AD timing;
# if there's no AD trigger in this time range, discard the trial
AD_delta = []
missing = [] # keep track of the trials to discard (due to missing AD trigger)
for i in range(events.shape[0]):
    idx = np.where((stim_tps > events[i,0]) & (stim_tps <= events[i,0] + 200/decim))
    if len(idx[0]): # if an AD trigger exists within 200ms of trigger channel
        idx = idx[0][0] # use the first AD trigger (if there are multiple)
        AD_delta.append(stim_tps[idx] - events[i,0]) # keep track of audio delay values (for histogram)
        events_corrected[i,0] = stim_tps[idx] # update event timing
    else:
        missing.append(i)
# discard events which could not be corrected
events_corrected = np.delete(events_corrected, missing, 0)
print("Could not correct", len(missing), "events - these were discarded!")

# histogram showing the distribution of audio delays
AD_delta = np.array(AD_delta) * decim
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
plt.show()
input("Close plot and then press Enter to continue...")

# a simple plot for testing purposes
#X = range(10)
#plt.plot(X, [x*x for x in X])
#plt.show()


print("Starting epoching")
# ***** NOTE WE SET BASELINE=None BECAUSE MNE HAD OUTPUT SUGGESTING THIS IS BEST PRACTICE WHEN DOING EPOCHING BEFORE ICA
epochs = mne.Epochs(raw, events_corrected, event_id=event_ids, tmin=-0.1, tmax=0.41, preload=True, baseline=None)
conds_we_care_about = ["pre-deviant", "deviant"]
epochs.equalize_event_counts(conds_we_care_about)
print("Finished epoching")


print("Starting RANSAC")
rsc = Ransac()
epochs = rsc.fit_transform(epochs)
print('\n'.join(rsc.bad_chs_)) # print the bad channels identified by RANSAC
print("Finished RANSAC")


print("Starting ICA")
if os.path.exists(ica_fname):
    ica = mne.preprocessing.read_ica(ica_fname)
else:
    # filter again (1Hz high-pass) before ICA?
    # probably only appropriate for continuous data
    
    # could use 'autoreject' to compute a threshold for removing large noise
    '''
    reject = get_rejection_threshold(epochs)
    reject # print the result
    # remove large noise before running ICA
    #epochs.load_data() # to avoid reading epochs from disk multiple times
    epochs.drop_bad(reject=reject)
    '''
    
    # run ICA
    ica = mne.preprocessing.ICA(n_components=60, max_iter=300, random_state=97)
    ica.fit(epochs)
    ica.save(ica_fname, overwrite=True)
print("Finished ICA")


# Automated artifact removal by ICA, based on correlations of ICA components with 
# MEG sensors EOG artifacts and ECG related components using cross-trial phase statistic
print("Starting ICA component rejection")
# plot ICA results
ica.plot_sources(epochs) # plot IC time series
input("Close plot and then press Enter to continue...")
ica.plot_components() # plot IC topography
input("Close plot and then press Enter to continue...")

# find which ICs match the ECG pattern
# ********** NOTE FOR PAUL: THE FOLLOWING LINE NEVER SEEMS TO FIND ANY ECG COMPONENTS. HAVE TRIED MULTIPLE DATASETS.
ecg_indices, ecg_scores = ica.find_bads_ecg(epochs, method="ctps", threshold="auto")
# barplot of ICA component "ECG match" scores
ica.plot_scores(ecg_scores)
input("Close plot and then press Enter to continue...")
# plot diagnostics
# we will need error correction in case no ecg components found
ica.plot_properties(epochs, picks=ecg_indices)
input("Close plot and then press Enter to continue...")
# plot ICs applied to epochs, with ECG matches highlighted
ica.plot_sources(epochs, show_scrollbars=False)
input("Close plot and then press Enter to continue...")

#TODO: find which ICs match the EOG pattern
#eog_indices, eog_scores = ica.find_bads_eog(epochs, ch_name=??)

# set which components to reject
ica.exclude = ecg_indices # should be: eog_indices + ecg_indices

# Compare data before & after IC rejection
epochs_orig = copy.deepcopy(epochs)
epochs_orig.plot(title='before ICA')
ica.apply(epochs) # Note: data will be modified in-place
epochs.plot(title='after ICA')
input("Close plot and then press Enter to continue...")

# save the clean epochs
epochs.save(epochs_fname, overwrite=True)

print("Finished ICA component rejection")


# plot ERFs
fig = epochs.average().plot(spatial_colors=True, gfp=True)
input("Close plot and then press Enter to continue...")
fig.savefig(figures_dir_meg + subject_MEG + '_AEF_butterfly.png')
fig2 = mne.viz.plot_compare_evokeds(
    [
        epochs["pre-deviant"].average(),
        epochs["deviant"].average(),
    ]
)
input("Close plot and then press Enter to continue...")
fig2[0].savefig(figures_dir_meg + subject_MEG + '_AEF_gfp.png')
