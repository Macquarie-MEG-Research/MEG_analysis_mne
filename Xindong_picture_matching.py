#test_script

import os
import mne
import meegkit
import glob
import matplotlib.pyplot as plt
import numpy as np
import copy
from mne.viz import plot_evoked_topo
import my_preprocessing


# set up file and folder paths here
exp_dir = '/mnt/d/Work/analysis_ME207_Xindong/'; #'/Users/zhangxindong/Downloads/ME207/'
subject_MEG = '20231016_f01_Alan'; #'20231011_Pilot03_JZ';
task = 'matching'; #'_1_oddball' #''
run_name = '_TSPCA'

# the paths below should be automatic
data_dir = exp_dir + "data/"
processing_dir = exp_dir + "processing/"
results_dir = exp_dir + "results/"
meg_dir = data_dir + subject_MEG + "/meg/"

save_dir_meg = processing_dir + "meg/" + subject_MEG + "/" # where to save the epoch files for each subject
ica_fname = save_dir_meg + subject_MEG + '-ica.fif'
events_fname = save_dir_meg + subject_MEG + '_eve.txt'
figures_dir_meg = results_dir + 'meg/sensor/' + task + run_name + '/Figures/' # where to save the figures for all subjects
epochs_fname_meg = processing_dir + "meg/" + "epochs/" + subject_MEG + "_" + task + run_name + "-epo.fif"
evokeds_fname_meg = processing_dir + "meg/" + "evokeds/" + subject_MEG + "_" + task + run_name + "-ave.fif"

# create the folders if needed
os.system('mkdir -p ' + save_dir_meg)
os.system('mkdir -p ' + figures_dir_meg)



#%% === Read raw MEG data === #

#print(glob.glob("*_oddball.con"))
fname_raw = glob.glob(meg_dir + "*" + task + ".con")
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
    stim=[*[166], *[176], *[178], *range(194, 199)],
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
raw.plot()

# Filtering & ICA
raw = my_preprocessing.reject_artefact(raw, 1, 40, False, ica_fname)


#%% === Trigger detection === #

# Find events
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
    verbose=True,
)

# specify the event IDs
event_ids = {
    "match": 198,
    "mm1": 194,
    "mm2": 195,
    "mm3": 196,
    "mm4": 197
}

# extract the events on ch176 (i.e. correct button responses) & ch178 (i.e. incorrect responses)
events_correct_resp = events[np.where(events[:, 2] == 176)[0]]
events_incorrect_resp = events[np.where(events[:, 2] == 178)[0]]


# get rid of unused triggers for now (i.e. triggers not representing actual trials)
events = np.delete(events, np.where((events[:, 2] > 165) & (events[:, 2] < 194)), 0)
events = np.delete(events, np.where((events[:, 2] > 200)), 0) # some strange triggers on channel 300+


# export events for manual fix
#mne.write_events(events_fname, events, overwrite=False, verbose=None)

# re-read events from txt file
#events = mne.read_events(events_fname)

#np.where (events[:,2]==196)



#%% === Check for correctness of response on each trial === #

# print some diagnostics
print("Number of correct button responses:", events_correct_resp.shape[0])
print("Number of incorrect button responses:", events_incorrect_resp.shape[0])
print("Number of trials:", events.shape[0])

# Definition of "correct trial": 
# presence of a trigger on ch176 & absence of a trigger on ch178, 
# between 300~2500ms following auditory stimulus onset
RTs = [] # keep track of RTs for diagnostic plot (note - these RTs have not been corrected, so they are only a rough estimation)
incorrect = [] # keep track of the trials to discard
for i in range(events.shape[0]):
    idx_correct = np.where((events_correct_resp[:,0] > events[i,0]+300) & (events_correct_resp[:,0] < events[i,0]+2500))
    idx_incorrect = np.where((events_incorrect_resp[:,0] > events[i,0]+300) & (events_incorrect_resp[:,0] < events[i,0]+2500))
    
    if len(idx_correct[0]) & (len(idx_incorrect[0])==0): # presence of trigger on ch176 but not ch178
        idx_correct = idx_correct[0][0] # use the first one (if there are multiple)
        RTs.append(events_correct_resp[idx_correct,0] - events[i,0])
    else:
        incorrect.append(i)
    # alternatively, if we choose to trust ch178, this can be a workaround for
    # the occasional missing triggers on ch176
    # but note that if there are any missing triggers on ch178, then we would 
    # end up counting those incorrect trials as correct
    '''
    if len(idx_incorrect[0]): # decision is solely based on ch178
        incorrect.append(i)
    elif len(idx_correct[0]): # only collect RTs if ch176 trigger is present
        idx_correct = idx_correct[0][0] # use the first one (if there are multiple)
        RTs.append(events_correct_resp[idx_correct,0] - events[i,0])
    '''

# discard trials with incorrect responses
events = np.delete(events, incorrect, 0)
print("There were", len(incorrect), "trials with incorrect or missing button responses - these trials have been discarded.")
print("Number of trials remaining:", events.shape[0])

# histogram showing the distribution of (rough) RTs
n, bins, patches = plt.hist(
    x=RTs, bins="auto", color="#0504aa", alpha=0.7, rwidth=1
)
plt.grid(axis="y", alpha=0.75)
plt.xlabel("RT (ms)")
plt.ylabel("Frequency")
plt.title("RTs (rough estimation)")
plt.text(
    70,
    50,
    r"$mean="
    + str(round(np.mean(RTs)))
    + ", std="
    + str(round(np.std(RTs)))
    + "$",
)
maxfreq = n.max()
# set a clean upper y-axis limit
plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)



#%% === Trigger timing correction based on audio channel signal === #

# get raw audio signal from ch166
aud_ch_data_raw = raw.get_data(picks="MISC 007")

def getEnvelope(inputSignal, thresh=0.05):
    # Taking the absolute value
    absoluteSignal = []
    for sample in inputSignal:
        absoluteSignal.append(abs(sample))
    absoluteSignal = absoluteSignal[0]

    # Peak detection
    intervalLength = 400  # Experiment with this number!
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

# plot any problematic time period to aid diagnosis
'''
test_time = 20000
span = 5000
plt.figure()
plt.plot(aud_ch_data_raw[0], 'b')
#plt.plot(outputSignal, 'r')
for i in range(events.shape[0]):
   plt.axvline(events[i,0], color='b', lw=2, ls='--')
for i in range(stim_tps.shape[0]):
   plt.axvline(stim_tps[i], color='r', lw=2, ls='--')
#plt.xlim(test_time-span, test_time+span)
plt.show()
'''

# apply timing correction onto the events array
events_corrected = copy.copy(events) # work on a copy so we don't affect the original

# Missing AD triggers can be handled:
# if there's an AD trigger within 50ms following the normal trigger
# (this ensures we've got the correct trial), update to AD timing;
# if there's no AD trigger in this time range, discard the trial
AD_delta = []
missing = [] # keep track of the trials to discard (due to missing AD trigger)
for i in range(events.shape[0]):
    idx = np.where((stim_tps >= events[i,0]-30) & (stim_tps < events[i,0]+350))
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
    x=AD_delta, bins="auto", color="#0504aa", alpha=0.7, rwidth=1
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


#%% === Epoching === #

if not os.path.exists(epochs_fname_meg):
    epochs = mne.Epochs(raw, events, event_id=event_ids, tmin=-0.5, tmax=1.5, preload=True)
    print (epochs)
    conds_we_care_about = ["match", "mm1", "mm2", "mm3", "mm4"]
    #epochs.equalize_event_counts(conds_we_care_about)
    
    # downsample to 100Hz
    print("Original sampling rate:", epochs.info["sfreq"], "Hz")
    epochs_resampled = epochs.copy().resample(100, npad="auto")
    print("New sampling rate:", epochs_resampled.info["sfreq"], "Hz")

    # save for later use (e.g. in Source_analysis script)
    epochs_resampled.save(epochs_fname_meg, overwrite=True)
    
# plot ERFs
if not os.path.exists(figures_dir_meg + subject_MEG + '_AEF_butterfly1025v0.png'):
    epochs_resampled = mne.read_epochs(epochs_fname_meg)

    fig = epochs_resampled.average().plot(spatial_colors=True, gfp=True)
    fig.savefig(figures_dir_meg + subject_MEG + '_AEF_butterfly1025v0.png')

    fig2 = mne.viz.plot_compare_evokeds(
        [
            epochs_resampled["mm1"].average(),
            epochs_resampled["mm2"].average(),
            #epochs_resampled["mm3"].average(),
            epochs_resampled["mm4"].average(),
            #epochs_resampled["match"].average()
        ],
        #combine = 'mean' # combine channels by taking the mean (default is GFP)
    )
    fig2[0].savefig(figures_dir_meg + subject_MEG + '_AEF_gfp1025v0.png')

#save evokeds
evokeds = [epochs[name].average() for name in ("mm1", "mm2", "mm3", "mm4", "match")]

mne.write_evokeds(evokeds_fname_meg, 
                  evokeds, 
                  on_mismatch='raise',
                  overwrite=False, 
                  verbose=None)


#plot evokeds in each channel
'''
colors = "blue", "red", "green"
title = "MNE sample data\nleft vs right (A/V combined)"

plot_evoked_topo(evokeds, color=colors, title=title, background_color="w")

plt.show()
'''
'''
evoked1=epochs_resampled["mm1"].average()
evoked2=epochs_resampled["mm2"].average()
evoked3=epochs_resampled["mm4"].average()
evoked1.plot_joint(picks="meg")
evoked_diff = mne.combine_evoked([evoked1, evoked3], weights=[1,1])
fig2.pick_types(meg="mag").plot_topo(color="r", legend=False)
'''
