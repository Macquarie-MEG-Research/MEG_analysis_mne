import mne
import numpy as np

# Load the epochs
epochs = mne.read_epochs('E:/vince/OneDrive - Macquarie University/Studies/19_MEG_Microdosing/processing/oddball/220204_p004/220204_p004_ICA_20_EOGmeasure_zscore_EOGthreshold_3.5-epo.fif', preload=True)

# Define the time windows of interest
time_windows = [(0.1, 0.2), (0.2, 0.3)]

# Get the evoked response from trials that are not deviants and not predeviants
evoked_other = epochs['other'].average()
#evoked_other = epochs.average() # pretend these are 'other' trials

# Get the list of MEG channels
meg_channels = [ch for ch in epochs.info["chs"] if ch["kind"] == mne.io.constants.FIFF.FIFFV_MEG_CH]

# Identify the sensors belonging to each hemisphere (based on x-coordinate of the channel)
left_meg_sensors = [ch["ch_name"] for ch in meg_channels if ch["loc"][0] < 0]
right_meg_sensors = [ch["ch_name"] for ch in meg_channels if ch["loc"][0] > 0]
#print(left_meg_sensors)
#print(right_meg_sensors)

# Get indices for left and right sensors
left_indices = [epochs.ch_names.index(ch) for ch in left_meg_sensors]
right_indices = [epochs.ch_names.index(ch) for ch in right_meg_sensors]


# Initialise a dictionary to store the sensors with largest absolute amplitude in each hemisphere
largest_sensors = {window: {'left': [], 'right': []} for window in time_windows}

# Loop through each time window
for window in time_windows:
    # Crop to the time window of interest
    data = evoked_other.copy().crop(tmin=window[0], tmax=window[1]).data

    # QUESTION - take absolute value first or average over time first??

    # Average over time
    data = np.mean(data, axis=1)
    # Convert to absolute value
    abs_amplitude = np.abs(data)  
    
    # Find the 10 sensors with largest absolute amplitude in each hemisphere
    # Get absolute amplitudes for left and right sensors
    left_amplitudes = abs_amplitude[left_indices]
    right_amplitudes = abs_amplitude[right_indices]
    # Find indices of top 10 sensors in each hemisphere
    top_left_idx = np.argsort(left_amplitudes)[-10:][::-1]
    top_right_idx = np.argsort(right_amplitudes)[-10:][::-1]
    # Get sensor names for top 10 sensors
    left_sensors = [left_meg_sensors[i] for i in top_left_idx]
    right_sensors = [right_meg_sensors[i] for i in top_right_idx]

    # Alternative method:
    #left_sensors = sorted(left_meg_sensors, key=lambda ch: abs_amplitude[evoked_other.ch_names.index(ch)], reverse=True)[:10]
    #right_sensors = sorted(right_meg_sensors, key=lambda ch: abs_amplitude[evoked_other.ch_names.index(ch)], reverse=True)[:10]

    # Store the sensor indices
    largest_sensors[window]['left'] = left_sensors
    largest_sensors[window]['right'] = right_sensors


# Initialise a dictionary to store mismatch amplitudes
mismatch_amplitudes = {window: {'left': [], 'right': []} for window in time_windows}

# Loop through each time window
for window in time_windows:
    # Get the evoked response for deviant and predeviant
    evoked_deviant = epochs['deviant'].average()
    evoked_predeviant = epochs['pre-deviant'].average()
    
    # Crop to the desired time window
    data_deviant = evoked_deviant.copy().crop(tmin=window[0], tmax=window[1]).data
    data_predeviant = evoked_predeviant.copy().crop(tmin=window[0], tmax=window[1]).data
    
    # Average over time
    data_deviant = np.mean(data_deviant, axis=1)
    data_predeviant = np.mean(data_predeviant, axis=1)
    
    # Calculate the mismatch amplitude
    mismatch_amplitude = data_deviant - data_predeviant
    
    # Average these over the selected sensors in each hemisphere
    channel_indices_left = [epochs.ch_names.index(ch) for ch in largest_sensors[window]['left']]
    mismatch_amplitudes[window]['left'] = np.mean(mismatch_amplitude[channel_indices_left], axis=0)
    channel_indices_right = [epochs.ch_names.index(ch) for ch in largest_sensors[window]['right']]
    mismatch_amplitudes[window]['right'] = np.mean(mismatch_amplitude[channel_indices_right], axis=0)

print('Done')
