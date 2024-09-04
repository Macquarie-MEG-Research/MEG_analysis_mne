"""
MEG Preprocessing Script
========================

Author: Paul Sowman - I have adapted the ICA component selection to be based on code developed by Lance Abel
https://github.com/LanceAbel/MQ_MEG_Analysis/blob/main/participant.py
Date: September 4, 2024
Version: 1.0

Description:
This script performs preprocessing on Magnetoencephalography (MEG) data.
It includes the following steps:
1. Loading raw MEG data
2. Applying Time-Shift Principal Component Analysis (TSPCA) for noise reduction
3. Preprocessing raw data (resampling and filtering)
4. Finding and adjusting events
5. Creating epochs
6. Applying Random Sample Consensus (RANSAC) for artifact rejection
7. Running Independent Component Analysis (ICA)
8. Detecting and removing EOG and ECG artifacts
9. Creating a comprehensive Quality Assurance (QA) report

The script uses MNE-Python for most MEG data processing operations and
creates an HTML report for quality assessment of the preprocessing steps.

Usage:
Run this script from the command line or an interactive Python environment.
Ensure all required dependencies are installed (see requirements.txt).

Note: This script is designed for research purposes and should be used
in accordance with ethical guidelines for MEG data analysis.
"""

import os
import mne
import meegkit
import matplotlib
import glob
import matplotlib.pyplot as plt
import numpy as np
from mne.preprocessing import ICA, create_ecg_epochs, create_eog_epochs
from autoreject import Ransac
from mne.report import Report
from tqdm import tqdm


mne.viz.set_browser_backend("qt")


matplotlib.use("Agg")

# Constants
ICA_NUM_COMPONENTS = 15
ICA_THRESHOLD_EOG = 0.3
ICA_THRESHOLD_ECG = 0.3
MIN_CHANNELS_ICA = 5


def setup_directories(base_dir, subject_id, meg_task):
    data_dir = os.path.join(base_dir, "Data")
    processing_dir = os.path.join(base_dir, "processing")
    results_dir = os.path.join(base_dir, "results")
    meg_dir = os.path.join(data_dir, subject_id)
    save_dir = os.path.join(processing_dir, subject_id)
    figures_dir_meg = os.path.join(results_dir, "oddball", "Figures")

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(figures_dir_meg, exist_ok=True)

    epochs_fname = os.path.join(save_dir, f"{subject_id}{meg_task}-epo.fif")
    ica_fname = os.path.join(save_dir, f"{subject_id}{meg_task}-ica.fif")

    return data_dir, meg_dir, save_dir, figures_dir_meg, epochs_fname, ica_fname


def load_raw_data(meg_dir, meg_task):
    fname_raw = glob.glob(os.path.join(meg_dir, f"*{meg_task}.con"))[0]
    fname_elp = glob.glob(os.path.join(meg_dir, "*.elp"))[0]
    fname_hsp = glob.glob(os.path.join(meg_dir, "*.hsp"))[0]
    fname_mrk = glob.glob(os.path.join(meg_dir, "*.mrk"))[0]

    raw = mne.io.read_raw_kit(
        fname_raw,
        mrk=fname_mrk,
        elp=fname_elp,
        hsp=fname_hsp,
        stim=[166] + list(range(182, 190)),
        slope="+",
        stim_code="channel",
        stimthresh=2,
        preload=True,
        allow_unknown_format=False,
        verbose=True,
    )

    return raw


def apply_tspca(raw):
    print("Starting TSPCA")
    noisy_data = raw.get_data(picks="meg").T
    noisy_ref = raw.get_data(picks=[160, 161, 162]).T
    data_after_tspca, _ = meegkit.tspca.tsr(noisy_data, noisy_ref)[0:2]
    raw._data[0:160] = data_after_tspca.T
    print("Finished TSPCA")


def preprocess_raw(raw):
    print("Starting resampling")
    raw.resample(250)
    print("Resampled")

    print("Starting filter")
    raw.filter(l_freq=0.1, h_freq=40)
    print("Finished filter")


def find_and_adjust_events(raw):
    print("Finding events")
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

    events = np.delete(events, np.where(events[:, 2] == 166), 0)

    std_dev_bool = np.insert(np.diff(events[:, 2]) != 0, 0, True)
    for idx, event in enumerate(std_dev_bool):
        if event and idx > 0:
            events[idx, 2] = 2
            if events[idx - 1, 2] != 2:
                events[idx - 1, 2] = 1

    return events


def get_envelope(input_signal):
    abs_signal = np.abs(input_signal)
    interval_length = 5
    output_signal = np.maximum.reduceat(
        abs_signal, np.arange(0, len(abs_signal), interval_length)
    )
    output_signal = np.concatenate(
        (np.zeros(interval_length), output_signal, np.zeros(interval_length))
    )
    return np.where(np.abs(output_signal) > 0.2, 1, 0)


def adjust_event_timing(raw, events):
    audio_data = raw.get_data(picks="MISC 007")[0]
    envelope = get_envelope(audio_data)
    new_stim_ch = np.clip(np.diff(envelope), 0, 1)
    stim_tps = np.where(new_stim_ch == 1)[0]

    print("Number of events from trigger channels:", events.shape[0])
    print("Number of events from audio channel (166) signal:", stim_tps.shape[0])

    decim = 1000 / raw.info["sfreq"]
    events_corrected = events.copy()
    ad_delta = []
    missing = []

    for i in range(events.shape[0]):
        idx = np.where(
            (stim_tps > events[i, 0]) & (stim_tps <= events[i, 0] + 200 / decim)
        )
        if len(idx[0]):
            idx = idx[0][0]
            ad_delta.append(stim_tps[idx] - events[i, 0])
            events_corrected[i, 0] = stim_tps[idx]
        else:
            missing.append(i)

    events_corrected = np.delete(events_corrected, missing, 0)
    print("Could not correct", len(missing), "events - these were discarded!")
    ad_delta = np.array(ad_delta) * decim
    # plot_audio_delay_histogram(ad_delta, decim)

    return events_corrected, ad_delta


def plot_audio_delay_histogram(ad_delta):
    fig, ax = plt.subplots()
    n, bins, patches = ax.hist(
        x=ad_delta, bins="auto", color="#0504aa", alpha=0.7, rwidth=0.85
    )
    ax.grid(axis="y", alpha=0.75)
    ax.set_xlabel("Delay (ms)")
    ax.set_ylabel("Frequency")
    ax.set_title("Audio Detector Delays")
    ax.text(70, 50, f"mean={round(np.mean(ad_delta))}, std={round(np.std(ad_delta))}")
    max_freq = n.max()
    ax.set_ylim(ymax=np.ceil(max_freq / 10) * 10 if max_freq % 10 else max_freq + 10)
    return fig


def create_epochs(raw, events, event_ids):
    print("Starting epoching")
    epochs = mne.Epochs(
        raw,
        events,
        event_id=event_ids,
        tmin=-0.1,
        tmax=0.41,
        preload=True,
        baseline=None,
    )
    conds_we_care_about = ["pre-deviant", "deviant"]
    epochs.equalize_event_counts(conds_we_care_about)
    print("Finished epoching")
    return epochs


def apply_ransac(epochs):
    print("Starting RANSAC")
    n_epochs_before = len(epochs)
    rsc = Ransac(verbose=False)
    epochs_clean = rsc.fit_transform(epochs)
    n_epochs_after = len(epochs_clean)
    n_dropped_epochs = n_epochs_before - n_epochs_after

    interpolated_channels = rsc.bad_chs_
    n_interpolated_channels = len(interpolated_channels)

    print(
        f"Finished RANSAC. Dropped {n_dropped_epochs} epochs. Interpolated {n_interpolated_channels} channels."
    )
    return (
        epochs_clean,
        n_dropped_epochs,
        n_interpolated_channels,
        interpolated_channels,
    )


def run_ica(raw, epochs, ica_fname):
    print("Starting ICA")
    if os.path.exists(ica_fname):
        ica = mne.preprocessing.read_ica(ica_fname)
    else:
        ica = ICA(
            n_components=ICA_NUM_COMPONENTS,
            max_iter="auto",
            method="fastica",
            random_state=97,
            verbose=False,
        )
        reject = dict(mag=5e-12)
        picks_meg = mne.pick_types(
            raw.info, meg=True, eeg=False, eog=False, stim=False, exclude="bads"
        )
        ica.fit(raw, picks=picks_meg, reject=reject)
        ica.save(ica_fname, overwrite=True)

    print("Finished ICA")
    return ica


def find_eog_artifacts(raw, ica):
    print("ICA EOG")
    all_eog_scores = {}
    all_eog_indices = {}
    num_channels = len(raw.ch_names)

    for p in tqdm(range(1, 160), desc="Finding EOG artifacts"):
        channel_name = f"MEG {str(p).zfill(3)}"
        eog_indices, eog_scores = ica.find_bads_eog(
            raw,
            ch_name=channel_name,
            measure="correlation",
            threshold=ICA_THRESHOLD_EOG,
            verbose=False,
        )
        all_eog_scores[p] = eog_scores
        all_eog_indices[p] = eog_indices

    eog_bads = []
    eog_bad_ct = {}
    for index_all_channels in all_eog_indices.values():
        for ica_index in index_all_channels:
            eog_bads.append(ica_index)
            eog_bad_ct[ica_index] = eog_bad_ct.get(ica_index, 0) + 1

    eog_bads = list(set(eog_bads))
    print("ALL EOG bads ", eog_bads)
    print("All EOG bad Counts: ", eog_bad_ct)

    max_key = max(eog_bad_ct, key=eog_bad_ct.get) if eog_bad_ct else None
    second_largest_key = None
    if len(eog_bad_ct) > 1:
        second_largest = sorted(eog_bad_ct.values())[-2]
        second_largest_key = next(
            k for k, v in eog_bad_ct.items() if v == second_largest
        )

    print("Max ", max_key, " 2nd: ", second_largest_key)

    eog_exclude = []
    if (
        second_largest_key is not None
        and eog_bad_ct[second_largest_key] >= MIN_CHANNELS_ICA
    ):
        eog_exclude.append(second_largest_key)
    if max_key is not None and eog_bad_ct[max_key] >= MIN_CHANNELS_ICA:
        eog_exclude.append(max_key)
    print("EOG Components to exclude:", eog_exclude)
    return eog_exclude


def find_ecg_artifacts(raw, ica):
    print("ICA ECG")
    all_ecg_indices = {}
    all_ecg_scores = {}

    for p in tqdm(range(1, 160), desc="Finding ECG artifacts"):
        channel_name = f"MEG {str(p).zfill(3)}"
        ecg_indices, ecg_scores = ica.find_bads_ecg(
            raw,
            ch_name=channel_name,
            method="correlation",
            threshold=ICA_THRESHOLD_ECG,
            verbose=False,
        )
        print(f"Channel {channel_name}, ecg indices {ecg_indices}")
        all_ecg_scores[p] = ecg_scores
        all_ecg_indices[p] = ecg_indices

    ecg_bads = []
    ecg_bad_ct = {}
    for index_all_channels in all_ecg_indices.values():
        for ica_index in index_all_channels:
            ecg_bads.append(ica_index)
            ecg_bad_ct[ica_index] = ecg_bad_ct.get(ica_index, 0) + 1

    ecg_bads = list(set(ecg_bads))
    print("ALL ECG bads ", ecg_bads)
    print("ECG BAD Counts: ", ecg_bad_ct)

    sorted_ecg_bad_ct = sorted(ecg_bad_ct.items(), key=lambda x: x[1], reverse=True)

    ecg_exclude = []
    for key, value in sorted_ecg_bad_ct[:2]:  # Get top two components
        if value >= MIN_CHANNELS_ICA:
            ecg_exclude.append(key)

    ecg_exclude = list(set(ecg_exclude))  # Remove any duplicates
    print("ECG Components to exclude:", ecg_exclude)
    return ecg_exclude


def apply_ica(raw, epochs, ica, exclude_components):
    print("ICA components to remove: ", exclude_components)
    ica.exclude = np.unique(exclude_components).tolist()

    # Apply ICA
    ica.apply(raw)
    ica.apply(epochs)


def plot_ica_components(ica, picks=None):
    """Create a static plot of ICA components."""
    n_components = ica.n_components_ if picks is None else len(picks)
    fig, axes = plt.subplots(
        int(np.ceil(n_components / 5)),
        5,
        figsize=(12, 2.5 * int(np.ceil(n_components / 5))),
    )
    ica.plot_components(picks=picks, axes=axes.ravel()[:n_components], show=False)
    fig.tight_layout()
    return fig


def plot_ica_sources(ica, raw, picks=None, window_size=10):
    """
    Create a static plot of ICA sources.

    Parameters:
    - ica: ICA object
    - raw: Raw object
    - picks: list of ICA components to plot
    - window_size: size of the window to plot in seconds (default: 10)
    """
    if picks is None:
        picks = range(ica.n_components_)

    sources = ica.get_sources(raw).get_data()
    times = raw.times
    n_picks = len(picks)

    # Calculate the middle of the time series
    mid_point = len(times) // 2

    # Calculate the number of samples in the window
    samples_in_window = int(window_size * raw.info["sfreq"])

    # Calculate start and end indices for the window
    start_idx = mid_point - (samples_in_window // 2)
    end_idx = start_idx + samples_in_window

    # Ensure indices are within bounds
    start_idx = max(0, start_idx)
    end_idx = min(len(times), end_idx)

    # Select the time window
    times_window = times[start_idx:end_idx]
    sources_window = sources[:, start_idx:end_idx]

    fig, axes = plt.subplots(n_picks, 1, figsize=(12, 2 * n_picks), sharex=True)
    if n_picks == 1:
        axes = [axes]

    for idx, (comp, ax) in enumerate(zip(picks, axes)):
        ax.plot(times_window, sources_window[comp])
        ax.set_title(f"ICA {comp:03d}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")

    # Set the x-axis limits to show only the selected window
    plt.xlim(times_window[0], times_window[-1])

    fig.suptitle(
        f"ICA Sources (10-second window from {times_window[0]:.2f}s to {times_window[-1]:.2f}s)"
    )
    fig.tight_layout()
    return fig


def safe_close(fig):
    """Safely close matplotlib figures, whether it's a single figure or a list of figures."""
    if isinstance(fig, list):
        for f in fig:
            plt.close(f)
    elif fig is not None:
        plt.close(fig)


def create_qa_report(
    raw,
    epochs,
    ica,
    eog_exclude,
    ecg_exclude,
    figures_dir_meg,
    subject_id,
    ad_delta,
    ransac_stats,
):
    report = Report(title=f"QA Report - Subject {subject_id}")

    # Before ICA
    fig_psd_before = raw.compute_psd().plot(
        average=False, picks="data", exclude="bads", amplitude=False
    )
    report.add_figure(fig_psd_before, title="PSD before ICA", section="Before ICA")
    safe_close(fig_psd_before)

    # Add RANSAC statistics to the report
    n_dropped_epochs, n_interpolated_channels, interpolated_channels = ransac_stats
    ransac_text = f"""
    <h3>RANSAC Statistics</h3>
    <p>Number of epochs dropped: {n_dropped_epochs}</p>
    <p>Number of channels interpolated: {n_interpolated_channels}</p>
    <p>Interpolated channels: {', '.join(interpolated_channels) if interpolated_channels else 'None'}</p>
    """
    report.add_html(ransac_text, title="RANSAC Statistics", section="RANSAC")

    # ICA Components (static plot)
    fig_components = plot_ica_components(ica)
    report.add_figure(fig_components, title="ICA Components", section="ICA Components")
    safe_close(fig_components)

    # EOG Components
    for idx in eog_exclude:
        fig_eog_overlay = ica.plot_overlay(raw, exclude=[idx], picks="meg")
        report.add_figure(
            fig_eog_overlay,
            title=f"EOG Component {idx} Overlay",
            section="EOG Components",
        )
        safe_close(fig_eog_overlay)

        fig_eog_properties = ica.plot_properties(
            raw, picks=[idx], psd_args={"fmax": 35.0}
        )
        report.add_figure(
            fig_eog_properties,
            title=f"EOG Component {idx} Properties",
            section="EOG Components",
        )
        safe_close(fig_eog_properties)

    # ECG Components
    for idx in ecg_exclude:
        fig_ecg_overlay = ica.plot_overlay(raw, exclude=[idx], picks="mag")
        report.add_figure(
            fig_ecg_overlay,
            title=f"ECG Component {idx} Overlay",
            section="ECG Components",
        )
        safe_close(fig_ecg_overlay)

        fig_ecg_properties = ica.plot_properties(
            raw, picks=[idx], psd_args={"fmax": 35.0}
        )
        report.add_figure(
            fig_ecg_properties,
            title=f"ECG Component {idx} Properties",
            section="ECG Components",
        )
        safe_close(fig_ecg_properties)

    # Plot ICA sources (static plot)
    fig_sources = plot_ica_sources(ica, raw, picks=ecg_exclude + eog_exclude)
    report.add_figure(fig_sources, title="ICA Sources", section="ICA Sources")
    safe_close(fig_sources)

    # Apply ICA
    raw_copy = raw.copy()
    epochs_copy = epochs.copy()
    ica.apply(raw_copy)
    ica.apply(epochs_copy)

    # After ICA
    fig_psd_after = raw_copy.compute_psd().plot(
        average=False, picks="data", exclude="bads", amplitude=False
    )
    report.add_figure(fig_psd_after, title="PSD after ICA", section="After ICA")
    safe_close(fig_psd_after)

    # ERFs with confidence intervals
    fig_erf = epochs.average().plot(spatial_colors=True, gfp=True)
    report.add_figure(fig_erf, title="Event-Related Fields", section="ERFs")
    safe_close(fig_erf)

    fig_erf_compare = mne.viz.plot_compare_evokeds(
        [epochs["pre-deviant"].average(), epochs["deviant"].average()],
        picks="meg",  # Plot MEG channels
        combine="gfp",  # Combine channels by taking the mean
        ci=True,  # Show confidence intervals
        title="Compared ERFs with Confidence Intervals",
    )
    report.add_figure(fig_erf_compare, title="Compared ERFs with CI", section="ERFs")
    safe_close(fig_erf_compare)

    # Average EOG plot
    try:
        eog_evoked = create_eog_epochs(raw).average()
        eog_evoked.apply_baseline(baseline=(None, -0.2))
        fig_eog = eog_evoked.plot_joint()
        report.add_figure(fig_eog, title="Average EOG", section="EOG")
        safe_close(fig_eog)
    except Exception as e:
        print(f"Failed to create average EOG plot: {str(e)}")

    # Average ECG plot
    try:
        ecg_evoked = create_ecg_epochs(raw).average()
        ecg_evoked.apply_baseline(baseline=(None, -0.2))
        fig_ecg = ecg_evoked.plot_joint()
        report.add_figure(fig_ecg, title="Average ECG", section="ECG")
        safe_close(fig_ecg)
    except Exception as e:
        print(f"Failed to create average ECG plot: {str(e)}")

    # Audio Delay Histogram
    fig_audio_delay = plot_audio_delay_histogram(ad_delta)
    report.add_figure(
        fig_audio_delay, title="Audio Detector Delays", section="Audio Delays"
    )
    safe_close(fig_audio_delay)

    # Save report
    report_fname = os.path.join(figures_dir_meg, f"{subject_id}_QA_report.html")
    report.save(report_fname, overwrite=True)

    print(f"QA report saved to {report_fname}")


def plot_and_save_erfs(epochs, figures_dir_meg, subject_id):
    fig = epochs.average().plot(spatial_colors=True, gfp=True)
    fig.savefig(os.path.join(figures_dir_meg, f"{subject_id}_AEF_butterfly.png"))

    fig2 = mne.viz.plot_compare_evokeds(
        [
            epochs["pre-deviant"].average(),
            epochs["deviant"].average(),
        ]
    )
    fig2[0].savefig(os.path.join(figures_dir_meg, f"{subject_id}_AEF_gfp.png"))


def main():
    try:
        base_dir = "/Users/em18033/Library/CloudStorage/OneDrive-AUTUniversity/Projects/Microdosing/"
        subject_id = "230618_17454_S2"
        meg_task = "_oddball"

        data_dir, meg_dir, save_dir, figures_dir_meg, epochs_fname, ica_fname = (
            setup_directories(base_dir, subject_id, meg_task)
        )

        print("Loading raw data...")
        raw = load_raw_data(meg_dir, meg_task)

        print("Applying TSPCA...")
        apply_tspca(raw)

        print("Preprocessing raw data...")
        preprocess_raw(raw)

        print("Finding and adjusting events...")
        events = find_and_adjust_events(raw)
        events_corrected, ad_delta = adjust_event_timing(raw, events)

        print("Creating epochs...")
        event_ids = {"pre-deviant": 1, "deviant": 2}
        epochs = create_epochs(raw, events_corrected, event_ids)

        print("Applying RANSAC...")
        epochs, n_dropped_epochs, n_interpolated_channels, interpolated_channels = (
            apply_ransac(epochs)
        )
        ransac_stats = (
            n_dropped_epochs,
            n_interpolated_channels,
            interpolated_channels,
        )

        print("Running ICA...")
        ica = run_ica(raw, epochs, ica_fname)

        print("Finding EOG artifacts...")
        eog_exclude = find_eog_artifacts(raw, ica)

        print("Finding ECG artifacts...")
        ecg_exclude = find_ecg_artifacts(raw, ica)

        exclude_components = eog_exclude + ecg_exclude

        print("Creating QA report...")
        create_qa_report(
            raw.copy(),
            epochs.copy(),  # Now this is correct as epochs is no longer a tuple
            ica,
            eog_exclude,
            ecg_exclude,
            figures_dir_meg,
            subject_id,
            ad_delta,
            ransac_stats,
        )

        print("Applying ICA...")
        apply_ica(raw, epochs, ica, exclude_components)

        print("Saving epochs...")
        epochs.save(epochs_fname, overwrite=True)

        print("Plotting and saving ERFs...")
        plot_and_save_erfs(epochs, figures_dir_meg, subject_id)

        # Close all remaining plots
        plt.close("all")

        print("Processing complete!")

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
