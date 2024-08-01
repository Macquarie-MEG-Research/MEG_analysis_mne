#!/usr/bin/python3
#
# MEG sensor space analysis for auditory roving MMF
#
# Authors: Paul Sowman, Judy Zhu

#######################################################################################

print("hello")


import os
import mne
import meegkit # for TSPCA
import glob
import matplotlib.pyplot as plt
import numpy as np
import copy

from mne.preprocessing import find_bad_channels_maxwell
#from autoreject import get_rejection_threshold  # noqa
#from autoreject import Ransac  # noqa
#from autoreject.utils import interpolate_bads  # noqa
from mne.decoding import EMS

import my_preprocessing

# We can use the `decim` parameter to only take every nth time slice.
# This speeds up the computation time. Note however that for low sampling
# rates and high decimation parameters, you might not detect "peaky artifacts"
# (with a fast timecourse) in your data. A low amount of decimation however is
# almost always beneficial at no decrease of accuracy.

#os.chdir("/Users/mq20096022/Downloads/MD_pilot1/")
#os.chdir("/Users/mq20096022/Downloads/220112_p003/")

# set up file and folder paths here
#exp_dir = "/mnt/d/Work/analysis_ME197/"
exp_dir = "C:/sync/OneDrive - Macquarie University/Studies/19_MEG_Microdosing/analysis/meg/"
subject_MEG = '220503_87225_S1' #'230426_72956_S2' #'220112_p003'
meg_task = '_oddball' #'_oddball' #''

# the paths below should be automatic
#data_dir = exp_dir + "data/"
data_dir = "C:/sync/OneDrive - Macquarie University/Studies/19_MEG_Microdosing/data/ACQUISITION/"
processing_dir = exp_dir + "processing/"
results_dir = exp_dir + "results/"
#meg_dir = data_dir + subject_MEG + "/meg/"
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

# Apply TSPCA for noise reduction
noisy_data = raw.get_data(picks="meg").transpose()
noisy_ref = raw.get_data(picks=[160,161,162]).transpose()
data_after_tspca, idx = meegkit.tspca.tsr(noisy_data, noisy_ref)[0:2]
raw._data[0:160] = data_after_tspca.transpose()

# browse data to identify bad sections & bad channels
#raw.plot()


# Filtering & ICA
raw = my_preprocessing.reject_artefact(raw, 0.1, 40, False, ica_fname)


#%% === Trigger detection & timing correction === #

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

# specify the event IDS (these will be used during epoching)
event_ids = {
    "standard": 1,
    "deviant": 2,
}


# Adjust trigger timing based on audio channel signal 

# get raw audio signal from ch166
aud_ch_data_raw = raw.get_data(picks="MISC 007")

# Opt 1: use Jamie script
'''
np.save(save_dir + 'audio_channel_raw.npy', aud_ch_data_raw) 

# NOW we need to manually run "04_process_raw_audio..."" script by Jamie

# then load the results
stim_tps = np.load(save_dir + 'audio_channel_triggers.npy')
'''

# Opt 2: use getEnvelope function

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
'''

# apply timing correction onto the events array
events_corrected = copy.copy(events) # work on a copy so we don't affect the original

# Missing AD triggers can be handled:
# if there's an AD trigger between 100-200ms after normal trigger (this ensures 
# we've got the correct trial), update to AD timing;
# if there's no AD trigger in this time range, discard the trial
AD_delta = []
missing = [] # keep track of the trials to discard (due to missing AD trigger)
for i in range(events.shape[0]):
    idx = np.where((stim_tps > events[i,0]) & (stim_tps <= events[i,0]+200))
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

# Opt 3: use a fixed delay (~150ms)
'''
events_corrected = copy.copy(events)
events_corrected[:,0] = events[:,0] + 150
'''


#%% === Epoching === #

if os.path.exists(epochs_fname):
    epochs_resampled = mne.read_epochs(epochs_fname)
else:
    epochs = mne.Epochs(raw, events_corrected, event_id=event_ids, tmin=-0.1, tmax=0.41, preload=True)

    # Should we do another autoreject / Ransac here? (so far have only done it
    # on the arbitrary epochs created for ICA)
    # if so, should the rejection threhold be based on only these 2 conditions 
    # of interest, or all epochs?

    conds_we_care_about = ["standard", "deviant"]
    epochs.equalize_event_counts(conds_we_care_about)

    # downsample to 100Hz
    print("Original sampling rate:", epochs.info["sfreq"], "Hz")
    epochs_resampled = epochs.copy().resample(250, npad="auto")
    print("New sampling rate:", epochs_resampled.info["sfreq"], "Hz")

    # save for later use (e.g. in Source_analysis script)
    epochs_resampled.save(epochs_fname)

# plot ERFs
#fig0 = mne.viz.plot_evoked(epochs_resampled.average(), gfp="only")
#fig0.savefig(figures_dir_meg + subject_MEG + '_rms.png')
fig = epochs_resampled.average().plot(spatial_colors=True, gfp=True)
fig.savefig(figures_dir_meg + subject_MEG + '_AEF_butterfly.png')
fig2 = mne.viz.plot_compare_evokeds(
    [
        epochs_resampled["standard"].average(),
        epochs_resampled["deviant"].average(),
    ]
)
fig2[0].savefig(figures_dir_meg + subject_MEG + '_AEF_gfp.png')

#############################################################################



#epochs_small = epochs_resampled["deviant"][1:6]
# THESE LINES RUN THE RANSAC ****************************************************************************
from autoreject import Ransac
rsc = Ransac()
epochs_clean = rsc.fit_transform(epochs_resampled)  
print('\n'.join(rsc.bad_chs_))
#epochs_clean.save("C://sync//OneDrive - Macquarie University//Studies//19_MEG_Microdosing//analysis//meg//processing//220503_87225_S1//ransac_small-epo.fif")
epochs_clean.save("C://sync//OneDrive - Macquarie University//Studies//19_MEG_Microdosing//analysis//meg//processing//"+ subject_MEG + "//ransac_big-epo.fif")

epochs_resampled = mne.read_epochs(epochs_fname)
#epochs_small = epochs_resampled["deviant"][1:6]
epochs_clean = mne.read_epochs("C://sync//OneDrive - Macquarie University//Studies//19_MEG_Microdosing//analysis//meg//processing//"+ subject_MEG + "//ransac_big-epo.fif")

ica = mne.preprocessing.read_ica("C://sync//OneDrive - Macquarie University//Studies//19_MEG_Microdosing//analysis//meg//processing//"+ subject_MEG + "//220503_87225_S1_oddball-ica.fif")
ica.exclude = []
# find which ICs match the ECG pattern
ecg_indices, ecg_scores = ica.find_bads_ecg(raw, method="ctps", threshold="auto")
ica.exclude = ecg_indices

# barplot of ICA component "ECG match" scores
ica.plot_scores(ecg_scores)

# plot diagnostics
ica.plot_properties(raw, picks=ecg_indices)

# plot ICs applied to raw data, with ECG matches highlighted
ica.plot_sources(raw, show_scrollbars=False)

# plot ICs applied to the averaged ECG epochs, with ECG matches highlighted
ica.plot_sources(ecg_evoked)




# THESE LINES MADE THE RANSAC PLOT   *********************************************************************
evoked = epochs_small.average()
evoked_clean = epochs_clean.average()

evoked.info['bads'] = ['MEG 005']
evoked_clean.info['bads'] = ['MEG 005']

from autoreject.utils import set_matplotlib_defaults  # noqa
import matplotlib.pyplot as plt  # noqa
set_matplotlib_defaults(plt)

fig, axes = plt.subplots(2, 1, figsize=(6, 6))

for ax in axes:
    ax.tick_params(axis='x', which='both', bottom='off', top='off')
    ax.tick_params(axis='y', which='both', left='off', right='off')

ylim = dict(grad=(-170, 200))
#evoked.pick_types(meg='grad', exclude=[])
evoked.plot(exclude=[], axes=axes[0], ylim=ylim, show=False)
axes[0].set_title('Before RANSAC')
#evoked_clean.pick_types(meg='grad', exclude=[])
evoked_clean.plot(exclude=[], axes=axes[1], ylim=ylim)
axes[1].set_title('After RANSAC')
#fig.tight_layout()

fig3 = mne.viz.plot_compare_evokeds(
    [
        epochs_small.average(),
        epochs_clean.average(),
    ]
)



from autoreject import AutoReject

ar = AutoReject()

epochs_clean = ar.fit_transform(epochs_resampled)

epochs_resampled.pick_types(meg=True, exclude="bads")
X = epochs_resampled.get_data()  # MEG signals: n_epochs, n_channels, n_times

# pca = UnsupervisedSpatialFilter(PCA(0.95), average=False)
# X = pca.fit_transform(X)

y = epochs_resampled.events[:, 2]  # target: Standard or Deviant
# # Setup the data to use it a scikit-learn way:
# X = epochs.get_data()  # The MEG data
# y = epochs.events[:, 2]  # The conditions indices
n_epochs, n_channels, n_times = X.shape


# Initialize EMS transformer
ems = EMS()

# Initialize the variables of interest
X_transform = np.zeros((n_epochs, n_times))  # Data after EMS transformation
filters = list()  # Spatial filters at each time point

# In the original paper, the cross-validation is a leave-one-out. However,
# we recommend using a Stratified KFold, because leave-one-out tends
# to overfit and cannot be used to estimate the variance of the
# prediction within a given fold.

for train, test in StratifiedKFold().split(X, y):
    # In the original paper, the z-scoring is applied outside the CV.
    # However, we recommend to apply this preprocessing inside the CV.
    # Note that such scaling should be done separately for each channels if the
    # data contains multiple channel types.
    X_scaled = X / np.std(X[train])

    # Fit and store the spatial filters
    ems.fit(X_scaled[train], y[train])

    # Store filters for future plotting
    filters.append(ems.filters_)

    # Generate the transformed data
    X_transform[test] = ems.transform(X_scaled[test])

# Average the spatial filters across folds
filters = np.mean(filters, axis=0)

# # Plot individual trials
# plt.figure()
# plt.title('single trial surrogates')
# plt.imshow(X_transform[y.argsort()], origin='lower', aspect='auto',
#            extent=[epochs_resampled.times[0], epochs_resampled.times[-1], 1, len(X_transform)],
#            cmap=None)
# plt.xlabel('Time (ms)')
# plt.ylabel('Trials (reordered by condition)')
# plt.clim(vmin=None, vmax=None)

# Plot average response
plt.figure()
plt.title("Average EMS signal")
mappings = [(key, value) for key, value in event_ids.items()]
for key, value in mappings:
    ems_ave = X_transform[y == value]
    plt.plot(epochs_resampled.times, ems_ave.mean(0), label=key)
plt.xlabel("Time (ms)")
plt.ylabel("a.u.")
plt.legend(loc="best")
plt.show()

# # Visualize spatial filters across time
# evoked = EvokedArray(filters, epochs_resampled.info, tmin=epochs_resampled.tmin)
# evoked.plot_topomap(time_unit="s")

# s = os.path.splitext(folder)[0].split("/")
# np.savetxt(s[-2] + "-filters.csv", filters, delimiter=",")
# np.savetxt(s[-2] + "-trials.csv", X_transform, delimiter=",")
# epochs_resampled.save(s[-2] + "-epo.fif", overwrite=True)
# epochs_resampled.pick_types(meg=True, exclude="bads")

X = epochs_resampled.get_data()  # MEG signals: n_epochs, n_channels, n_times

pca = UnsupervisedSpatialFilter(PCA(0.95), average=False)
X = pca.fit_transform(X)
y = epochs_resampled.events[:, 2]  # target: Standard or Deviant

clf = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis())
time_decod = SlidingEstimator(clf, n_jobs=1, scoring="accuracy")

# scores = cross_val_multiscore(time_decod, X, y, cv=int(X.shape[0]/2),
#                               n_jobs=1) #LOO

scores = cross_val_multiscore(time_decod, X, y, cv=5, n_jobs=1)  # k=5

# Mean scores across cross-validation splits
scores = np.mean(scores, axis=0)

# # Plot
# fig, ax = plt.subplots()
# ax.plot(epochs_resampled.times, scores, label="score")
# ax.axhline(0.5, color="k", linestyle="--", label="chance")
# ax.set_xlabel("Times")
# ax.set_ylabel("AUC")  # Area Under the Curve
# ax.legend()
# ax.axvline(0.0, color="k", linestyle="-")
# ax.set_title("Sensor space decoding")
# plt.show()

# You can retrieve the spatial filters and spatial patterns if you explicitly
# use a LinearModel - DOesn't work for PCA reduced data though as need all chs
# clf = make_pipeline(StandardScaler(), LinearModel(LinearDiscriminantAnalysis()))
# time_decod = SlidingEstimator(clf, n_jobs=1, scoring='roc_auc')

# time_decod.fit(X, y)

# coef = get_coef(time_decod, 'patterns_', inverse_transform=True)
# evoked = mne.EvokedArray(coef, epochs_resampled.info, tmin=epochs.times[0])
# joint_kwargs = dict(ts_args=dict(time_unit='s'),
#                     topomap_args=dict(time_unit='s'))
# evoked.plot_joint(times=np.arange(0., .500, .100), title='patterns',
#                   **joint_kwargs)

s = os.path.splitext(confile)[0].split("/")
np.savetxt(s[-2] + "-filters.csv", filters, delimiter=",")
np.savetxt(s[-2] + "-trials.csv", X_transform, delimiter=",")
np.savetxt(s[-2] + "-decoding.csv", scores, delimiter=",")
epochs_resampled.save(s[-2] + "-epo.fif", overwrite=True)
# except:
#     print(" error with " + confile)
#     errors.append(confile)
#     continue
# # epochs_resampled.p
