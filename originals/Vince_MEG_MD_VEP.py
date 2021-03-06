import mne
import glob
import matplotlib.pyplot as plt
import numpy as np

# from mne.preprocessing import find_bad_channels_maxwell
# from autoreject import get_rejection_threshold  # noqa
# from autoreject import Ransac  # noqa
# from autoreject.utils import interpolate_bads  # noqa

# import matplotlib.pyplot as plt
from scipy import stats

# MISC 18 and 19 are triggers


#%% Main function that gets called when this script is run
def main():
    #%% Raw extraction ch misc 23-29 = triggers
    # ch misc 006 = Photodetector
    data_dir = "/Users/mq20096022/Downloads/220112_p003/"
    #%% Loop over conditions pre post post 2
    condition_labels = ["pre", "post1", "post2"]
    pre = [data_dir + "220112_p003_1_LTP_b.con"]
    post1 = [data_dir + "220112_p003_1_LTP2.con"]
    post2 = [data_dir + "220112_p003_1_LTP3.con"]

    conditions = np.concatenate([pre, post1, post2])
    fname_elp = glob.glob(data_dir + "*.elp")
    fname_hsp = glob.glob(data_dir + "*.hsp")
    fname_mrk = glob.glob(data_dir + "*.mrk")
    global epochs
    epochs = {}  # initialise dict to store outputs
    for file_index, file in enumerate(conditions):
        fname_raw = file

        raw = mne.io.read_raw_kit(
            fname_raw,  # change depending on file i want
            mrk=fname_mrk[0],
            elp=fname_elp[0],
            hsp=fname_hsp[0],
            stim=[*range(177, 179)],
            slope="+",
            stim_code="channel",
            stimthresh=1,  # 2 for adults
            preload=True,
            allow_unknown_format=False,
            verbose=True,
        )

        #%% Finding events
        events = mne.find_events(
            raw,
            output="onset",
            consecutive=False,
            min_duration=0,
            shortest_event=1,  # 5 for adults
            mask=None,
            uint_cast=False,
            mask_type="and",
            initial_event=False,
            verbose=None,
        )

        for index, event in enumerate(events):
            if event[2] == 177:
                events[index, 2] = 2
            elif event[2] == 178:
                events[index, 2] = 3

        #%% Find the envelope of the sound recording
        raw.load_data().apply_function(getEnvelope, picks="MISC 006")

        # Find times of audio event
        events_PD = mne.find_events(
            raw, stim_channel=[raw.info["ch_names"][x] for x in [165]], output="onset"
        )

        combined_events = np.concatenate([events, events_PD])
        combined_events = combined_events[np.argsort(combined_events[:, 0])]

        #%% find the difference between AD time and trigger time i.e. the PD delay
        # TODO SANITY CHECK TO MAKE SURE THIS IS CORRECT PD SHOULD COME AFTER TRIGGER?
        # TODO plot PD epoched by triggers
        pd_delta = []
        for index, event in enumerate(combined_events):
            if (
                index > 0  # PD can't be first event
                and combined_events[index, 2] == 1
                and combined_events[index - 1, 2] != 1
            ):
                pd_delta.append(
                    combined_events[index, 0] - combined_events[index - 1, 0]
                )

        z = np.abs(stats.zscore(pd_delta))

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

        # Use target event to align triggers with audio events and avoid outliers using z of 3
        events_to_find = [2, 3]
        sfreq = raw.info["sfreq"]  # sampling rate
        tmin = -0.4  # audio occurs after trigger hence negative
        if [pd_delta[i] for i in np.where(z > 3)[0]]:

            tmax = -max([pd_delta[i] for i in np.where(z > 3)[0]]) / 1000
        else:

            tmax = 0

        fill_na = None  # the fill value for non-target
        reference_id = 1  # audio recording
        events_target = {}  # initialize dictionary

        # loop through events and replace AD events with event class identifier i.e. trigger number
        for event in events_to_find:

            target_id = event  # prevertical
            new_id = 20 + event
            events_target["event" + str(event)], lag = mne.event.define_target_events(
                combined_events,
                reference_id,
                target_id,
                sfreq,
                tmin,
                tmax,
                new_id,
                fill_na,
            )
        events = np.concatenate((events_target["event2"], events_target["event3"]))

        event_ids = {
            "horizontal": 22,
            "vertical": 23,
        }

        epochs[condition_labels[file_index]] = mne.Epochs(
            raw, events, event_id=event_ids, tmin=-0.1, tmax=0.4, preload=True
        )
        conds_we_care_about = ["horizontal", "vertical"]

        epochs[condition_labels[file_index]].equalize_event_counts(conds_we_care_about)
        mne.viz.plot_evoked(epochs[condition_labels[file_index]].average(), gfp="only")
        mne.viz.plot_evoked(
            epochs[condition_labels[file_index]].average(picks="MISC 006")
        )  # TODO change this as this is already corrected so will be zero!!
        mne.viz.plot_compare_evokeds(
            [
                epochs[condition_labels[file_index]]["horizontal"].average(),
                epochs[condition_labels[file_index]]["vertical"].average(),
            ]
        )
    return epochs
    # report = mne.Report(title=fname_raw[0])
    # report.add_evokeds(
    #     evokeds=evoked, titles=["VEP"], n_time_points=25  # Manually specify titles
    # )
    # report.save(fname_raw[0] + "_report_evoked.html", overwrite=True)

    #%% Function to get envelope of signal


def getEnvelope(inputSignal):

    # Taking the absolute value
    absoluteSignal = []
    for sample in inputSignal:
        absoluteSignal.append(abs(sample))

    # Peak detection
    intervalLength = 5  # Experiment with this number
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
    # finally binarise the output at a threshold of 2.5
    return np.array([1 if np.abs(x) > 0.5 else 0 for x in outputSignal])


if __name__ == "__main__":
    main()
