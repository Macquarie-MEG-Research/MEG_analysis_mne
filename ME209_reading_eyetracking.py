# -*- coding: utf-8 -*-

#!/usr/bin/python3
#
# MEG sensor space analysis for Lexical Decision Task (high vs low frequency words)
#
# Authors: Paul Sowman, Judy Zhu

#######################################################################################

import os
import mne
import meegkit # for TSPCA
import glob
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

import my_preprocessing


# Make plots interactive when running in interactive window in vscode
#plt.switch_backend('TkAgg') You can use this backend if needed
#plt.ion() 
# %matplotlib qt 


# set up file and folder paths here
exp_dir = "/Volumes/DATA/RSG data/"
subject_MEG = '20240109_Pilot01_LY'
tasks = ['_B4'] #'_B2'

# specify run options here
run_name = '' #'_noICA' #''
do_ICA = True


# the paths below should be automatic
data_dir = exp_dir + "data/"
processing_dir = exp_dir + "processing/"
results_dir = exp_dir + "results/"
meg_dir = data_dir + subject_MEG + "/meg/"
save_dir = processing_dir + "meg/" + subject_MEG + "/" # where to save the epoch files for each subject
figures_dir = results_dir + 'meg/sensor/Figures/' # where to save the figures for all subjects
# create the folders if needed
os.system('mkdir -p ' + save_dir)
os.system('mkdir -p ' + figures_dir)


#%% === Read raw MEG data === #

fname_elp = glob.glob(meg_dir + "*.elp")
fname_hsp = glob.glob(meg_dir + "*.hsp")
fname_mrk = glob.glob(meg_dir + "*.mrk")

#%% Loop over different tasks
for counter, task in enumerate(tasks):
    fname_raw = glob.glob(meg_dir + "*" + task + ".con")

    ica_fname = save_dir + subject_MEG + task + "-ica.fif"
    epochs_fname = save_dir + subject_MEG + task + run_name + "_stim-epo.fif"
    epochs_fname_sac_locked = save_dir + subject_MEG + task + run_name + "_sac-epo.fif"
    #ERFs_fname = save_dir + subject_MEG + task + "-ave.fif"
    ERFs_figure_fname = figures_dir + subject_MEG + task + run_name + "_stim.png"
    ERFs_sac_figure_fname = figures_dir + subject_MEG + task + run_name + "_sac.png"
    butterfly_figure_fname = figures_dir + subject_MEG + task + run_name + '_stim_butterfly.png'
    butterfly_sac_figure_fname = figures_dir + subject_MEG + task + run_name + '_sac_butterfly.png'

    raw = mne.io.read_raw_kit(
        fname_raw[0], 
        mrk=fname_mrk[0],
        elp=fname_elp[0],
        hsp=fname_hsp[0],
        stim=[*range(176, 180), *range(181, 192)], # exclude ch180 (button box trigger), as it sometimes overlaps with other triggers and therefore results in an additional event on channel 300+
        slope="+",
        stim_code="channel",
        stimthresh=1,  # 2 for adults
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
    raw = my_preprocessing.reject_artefact(raw, 0.1, 40, do_ICA, ica_fname)


    #%% === Trigger detection & timing correction === #

    # ch169 (MISC 10) == Photodetector
    
    # B1 (dual LDT):
    # ch177 and 178 (MISC 18 and 19) == HF / LF stim onset
    # ch182 and 183 == saccade onset (right / left)

    # B2 (single LDT):
    # ch177 and 178 (MISC 18 and 19) == HF / LF stim onset

    # B3 (hash-string saccade task):
    # ch179 == stim onset (for both right and left)
    # ch182 and 183 == saccade onset (right / left)
    
    # B4 (dot saccade task):
    # ch185 == stim onset (for both right and left)
    # ch182 and 183 == saccade onset (right / left)

    #%% Find events
    events = mne.find_events(
        raw,
        output="onset",
        consecutive="True", # tmp solution to deal with saccade triggers not being detected
        min_duration=0,
        shortest_event=1,  # 5 for adults
        mask=None,
        uint_cast=False,
        mask_type="and",
        initial_event=False,
        verbose=None,
    )

    # Check if saccade onset triggers have any delays (by comparing to eyetracking data)
    '''
    # find all stim onset & saccade onset triggers
    stim_onset = []
    saccade_onset = []
    for index, event in enumerate(events):
        if task == '_B1':
            if event[2] == 177 or event[2] == 178 or event[2] == 181: # stim onset trigger
                stim_onset.append(event[0])
        if task == '_B3':
            if event[2] == 179: # stim onset trigger
                stim_onset.append(event[0])
        if task == '_B4':
            if event[2] == 185: # stim onset trigger
                stim_onset.append(event[0])
        if event[2] == 182 or event[2] == 183: # saccade onset trigger
            saccade_onset.append(event[0])

    # sac latency = saccade onset minus stim onset
    #assert(len(stim_onset) == len(saccade_onset))
    sac_latencies = []
    missing = []
    for i in range(len(stim_onset)):
        idx = np.where((saccade_onset >= stim_onset[i]) & (saccade_onset < stim_onset[i]+4000))
        if len(idx[0]): # if a saccade onset trigger exists within the specified window
            idx = idx[0][0] # use the first one
            sac_latencies.append(saccade_onset[idx] - stim_onset[i])
        else:
            missing.append(i)
    sac_latencies_avg = np.mean(sac_latencies)
    if missing:
        print("Could not calculate sac latencies for", len(missing), "trials!")
    # now we can compare these values with the sac latencies for each task
    # in the excel table extracted from eyetracking data
    '''

    # recode event IDs so we can spot them more easily in the events array (optional)
    # TODO - update the following when we have the new version of exp
    if task == '_B1' or task == '_B2':
        for index, event in enumerate(events):
            if event[2] == 177: # ch177 == MISC 18
                events[index, 2] = 2 # HF
            elif event[2] == 178: # ch178 == MISC 19
                events[index, 2] = 3 # LF
    elif task == '_B3': # hash saccade
        for index, event in enumerate(events):
            if event[2] == 179:
                events[index, 2] = 2 # stim onset
    elif task == '_B4': # dot saccade
        for index, event in enumerate(events):
            if event[2] == 185:
                events[index, 2] = 2 # stim onset
    # for all blocks, saccade triggers are on the same channels
    for index, event in enumerate(events):
        if event[2] == 182: # ch182 == MISC 23
            events[index, 2] = 6 # saccade right
        elif event[2] == 183: # ch183 == MISC 24
            events[index, 2] = 7 # saccade left

    # specify mappings between exp conditions & event IDs
            
    # TODO - in new version of exp, can use hierarchical event IDs like this:
    #event_id = {'HF/right': 2, 'HF/left': 3, 'LF/right': 4, 'LF/left': 5}
    # https://mne.tools/stable/generated/mne.merge_events.html
    # More examples: https://github.com/mne-tools/mne-python/issues/3599
    if task == '_B1':
        event_ids_stim_locked = {
            "HF": 2,
            "LF": 3,
        }
        event_ids_sac_locked = {
            "right": 6,
            "left": 7,
        }
    elif task == '_B2':
       event_ids_stim_locked = {
            "HF": 2,
            "LF": 3,
        }
       event_ids_sac_locked = {}
    elif task == '_B3' or task == '_B4':
        event_ids_stim_locked = {
            "all": 2,
        }
        event_ids_sac_locked = {
            "right": 6,
            "left": 7,
        }

    # sanity check: extract all trials for a particular cond (can check number of trials etc)
    #events_tmp = events[np.where(events[:, 2] == 2)[0]]


    #%% Adjust trigger timing based on photodetector channel

    # Find times of PD triggers
    # Ensure correct PD channel is entered here, might sometimes be 165
    events_PD = mne.find_events(
        raw, 
        stim_channel=[raw.ch_names[x] for x in [169]], 
        output="onset", 
        consecutive=False,
    )

    # some PD triggers have event ID 1 and some ID 2 (not sure why),
    # here we convert all to 1
    for index, event in enumerate(events_PD):
        events_PD[index, 2] = 1

    combined_events = np.concatenate([events, events_PD])
    combined_events = combined_events[np.argsort(combined_events[:, 0])]

    # find the difference between PD time and trigger time
    pd_delta = []
    for index, event in enumerate(combined_events):
        if (
            index > 0  # PD can't be first event
            and combined_events[index, 2] == 1 # current trigger is PD trigger
            and combined_events[index - 1, 2] != 1 # previous trigger is not PD trigger
        ):
            pd_delta.append(
                combined_events[index, 0] - combined_events[index - 1, 0] # find the time difference
            )
    # for B4, there is an extra PD trigger at the end - remove this
    if task == '_B4' and pd_delta[-1] > 500:
        pd_delta.pop(-1)
    # show histogram of PD delays
    n, bins, patches = plt.hist(
        x=pd_delta, bins="auto", color="#0504aa", alpha=0.7, rwidth=0.85
    )
    plt.grid(axis="y", alpha=0.75)
    plt.xlabel("Delay (ms)")
    plt.ylabel("Frequency")
    plt.title("Photo Detector Delays")
    plt.text(
        70,
        50,
        r"$mean="
        + str(round(np.mean(pd_delta)))
        + ", std="
        + str(round(np.std(pd_delta)))
        + "$",
    )
    maxfreq = n.max()
    # Set a clean upper y-axis limit.
    plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)

    # Use target events to align triggers & avoid outliers using a z-value threshold of 3
    z = np.abs(stats.zscore(pd_delta))
    #TODO: check this part works correctly when we do have outliers!
    if [pd_delta[i] for i in np.where(z > 3)[0]]:
        tmax = -max([pd_delta[i] for i in np.where(z > 3)[0]]) / 1000
    else:
        tmax = 0

    # TODO - update in new version of exp
    if task == '_B1' or task == '_B2':
        events_to_find = [2, 3] # target events
    elif task == '_B3' or task == '_B4':
        events_to_find = [2] # target events

    sfreq = raw.info["sfreq"]  # sampling rate
    tmin = -0.4  # PD occurs after trigger, hence negative
    fill_na = None  # the fill value for non-target
    reference_id = 1  # PD
    
    # loop through events and replace PD events with event class identifier i.e. trigger number
    events_target = {}
    for event in events_to_find:
        new_id = 20 + event # event IDs will now be 22 and 23, need to change it back afterwards
        events_target["event" + str(event)], lag = mne.event.define_target_events(
            combined_events,
            reference_id,
            event,
            sfreq,
            tmin,
            tmax,
            new_id,
            fill_na,
        )

    # TODO - update in new version of exp
    if task == '_B1' or task == '_B2':
        events_corrected = np.concatenate((events_target["event2"], events_target["event3"]))
    elif task == '_B3' or task == '_B4':
        events_corrected = events_target["event2"]
    # note: events_corrected only contains the stim-locked events;
    # no timing correction needed for saccade events, so just use original events array for those

    # change the event IDs back to normal
    for index, event in enumerate(events_corrected):
        if event[2] == 22:
            events_corrected[index, 2] = 2
        if event[2] == 23:
            events_corrected[index, 2] = 3

            

    #%% === Sensor space (ERF) analysis, stimulus-locked === #

    # epoching
    if os.path.exists(epochs_fname):
        epochs_resampled = mne.read_epochs(epochs_fname)
    else:
        epochs = mne.Epochs(raw, events_corrected, event_id=event_ids_stim_locked, 
            tmin=-0.1, tmax=0.41, preload=True)
        epochs.equalize_event_counts(event_ids_stim_locked)

        # sanity check - PD triggers occur at 0ms
        mne.viz.plot_evoked(
            epochs.average(picks="MISC 010")
        ) 

        # downsample to 100Hz
        epochs_resampled = epochs.copy().resample(100, npad="auto")

        # save the epochs to file
        epochs_resampled.save(epochs_fname)


    # plot ERFs
    #mne.viz.plot_evoked(epochs_resampled.average(), gfp="only")
        
    # compute evoked for each cond
    evokeds = []
    for cond in epochs_resampled.event_id:
        evokeds.append(epochs_resampled[cond].average())

    # GFP plot (one line per condition)
    '''
    fig = mne.viz.plot_compare_evokeds(
        [
            epochs_resampled[list(event_ids_locked)[0]].average(),
            epochs_resampled[list(event_ids_locked)[1]].average(),
        ]
    )
    '''
    fig = mne.viz.plot_compare_evokeds(evokeds)
    fig[0].savefig(ERFs_figure_fname)

    # butterfly plot with topography at peak time points
    if task == '_B3' or task == '_B4':
        fig1 = epochs_resampled.average().plot_joint() # avg over conds, as we don't need differentiate between right and left saccades
        fig1.savefig(butterfly_figure_fname)

    # make a separate butterfly plot for each condition (e.g might be useful for B1 & B2)
    '''
    for evoked in evokeds:
        fig2 = evoked.plot_joint()
        fig2.savefig(butterfly_figure_fname[:-4] + '_(' + evoked.comment + ').png')
    '''


    #%% === Sensor space (ERF) analysis, saccade-locked === #

    if task != '_B2': # B2 (single LDT) doesn't have saccades

        # epoching
        if os.path.exists(epochs_fname_sac_locked):
            epochs_sac_resampled = mne.read_epochs(epochs_fname_sac_locked)
        else:
            epochs_sac = mne.Epochs(raw, events, event_id=event_ids_sac_locked, # no timing correction needed for saccade triggers
                tmin=-0.4, tmax=0.01, baseline=None, preload=True) # explicitly disable baseline correction (default setting is to use the entire period before time 0 as baseline)
            epochs_sac.equalize_event_counts(event_ids_sac_locked)

            # downsample to 100Hz
            epochs_sac_resampled = epochs_sac.copy().resample(100, npad="auto")

            # save the epochs to file
            epochs_sac_resampled.save(epochs_fname_sac_locked)


        # plot ERFs
        #mne.viz.plot_evoked(epochs_sac_resampled.average(), gfp="only")
        
        # compute evoked for each cond
        evokeds_sac = []
        for cond in epochs_sac_resampled.event_id:
            evokeds_sac.append(epochs_sac_resampled[cond].average())

        # GFP plot (one line per condition)
        fig = mne.viz.plot_compare_evokeds(evokeds_sac)
        fig[0].savefig(ERFs_sac_figure_fname)

        # butterfly plot with topography at peak time points
        if task == '_B3' or task == '_B4':
            fig1 = epochs_sac_resampled.average().plot_joint() # avg over conds, as we don't need differentiate between right and left saccades
            fig1.savefig(butterfly_sac_figure_fname)



# this is just for pausing the script (using a breakpoint), 
# so that it doesn't exit immediately
print("All done!")
           

# report = mne.Report(title=fname_raw[0])
# report.add_evokeds(
#     evokeds=evoked, titles=["VEP"], n_time_points=25  # Manually specify titles
# )
# report.save(fname_raw[0] + "_report_evoked.html", overwrite=True)
