import os
import os.path as op
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import mne
from mne.stats import spatio_temporal_cluster_test
from mne.viz import plot_compare_evokeds
from mne.channels import find_ch_adjacency


# set up file and folder paths here
exp_dir = '/mnt/d/Work/analysis_ME209/'
meg_task = '_dual_stim'

# which conditions to include in stats
conditions = ['HF', 'LF']


# All paths below should be automatic
processing_dir = op.join(exp_dir, 'processing', 'meg')
stats_dir = op.join(exp_dir, 'stats')
figures_dir = op.join(stats_dir, 'sensor', 'Figures')
os.system('mkdir -p ' + figures_dir) # create the folder if needed

# find all subjects in the processing dir
subjects = [f for f in os.listdir(processing_dir) if f.startswith('2024')]


# initialise with an empty list for each condition
evokeds = {cond: [] for cond in conditions} # keep the Evoked structure
evokeds_data = {cond: [] for cond in conditions} # keep values only

# read the saved epochs for each subject, average to get ERF, and append to subject list
for subject_MEG in subjects:
    epochs_fname = op.join(processing_dir, subject_MEG, subject_MEG + meg_task + '-epo.fif')
    epochs = mne.read_epochs(epochs_fname, preload = True)

    # interpolate bad channels
    if epochs.info['bads']:
        epochs.interpolate_bads(reset_bads=True)
                   
    for cond in conditions:
        evoked = epochs[cond].average()
        evokeds[cond].append(evoked)
        
        data = evoked.get_data() # channels x time
        data = [np.transpose(data, (1, 0))] # time x channels
        evokeds_data[cond].append(data)

# prepare data for cluster permutation
X = []
for cond in conditions:
    array = np.squeeze(np.array(evokeds_data[cond])) # subjects x time x channels
    X.append(array)

# prepare data for plotting
GA = {cond: [] for cond in conditions}
for cond in conditions:
    GA[cond] = mne.grand_average(evokeds[cond])



# Tutorial:
# https://mne.tools/stable/auto_tutorials/stats-sensor-space/75_cluster_ftest_spatiotemporal.html


# Calculate adjacency matrix at sensor level
adjacency, ch_names = find_ch_adjacency(epochs.info, ch_type="mag")
#mne.viz.plot_ch_adjacency(epochs.info, adjacency, ch_names)
#plt.show()


# Cluster-based permutation test

# We are running an F test, so we look at the upper tail
# see also: https://stats.stackexchange.com/a/73993
tail = 1

# We want to set a critical test statistic (here: F), to determine when
# clusters are being formed. Using Scipy's percent point function of the F
# distribution, we can conveniently select a threshold that corresponds to
# some alpha level that we arbitrarily pick.
alpha_cluster_forming = 0.001

# For an F test we need the degrees of freedom for the numerator
# (number of conditions - 1) and the denominator (number of observations
# - number of conditions):
n_conditions = len(X)
n_observations = len(subjects)
dfn = n_conditions - 1
dfd = n_observations - n_conditions

# Note: we calculate 1 - alpha_cluster_forming to get the critical value
# on the right tail
f_thresh = scipy.stats.f.ppf(1 - alpha_cluster_forming, dfn=dfn, dfd=dfd)

# run the cluster based permutation analysis
cluster_stats = spatio_temporal_cluster_test(
    X,
    n_permutations=1000, 
    threshold=f_thresh,
    tail=tail,
    #max_step=2, # can set this so that time samples do not have to be strictly continuous to be considered adjacent
    n_jobs=None,
    buffer_size=None,
    adjacency=adjacency,
)
F_obs, clusters, p_values, _ = cluster_stats


# Visualise clusters

# We subselect clusters that we consider significant at an arbitrarily
# picked alpha level: "p_accept".
# NOTE: remember the caveats with respect to "significant" clusters that
# we mentioned in the introduction of this tutorial!
p_accept = 0.05
good_cluster_inds = np.where(p_values < p_accept)[0]

# configure variables for visualization
colors = {"HF": "crimson", "LF": "steelblue"}

# loop over clusters
for i_clu, clu_idx in enumerate(good_cluster_inds):
    # unpack cluster information, get unique indices
    time_inds, space_inds = np.squeeze(clusters[clu_idx])
    ch_inds = np.unique(space_inds)
    time_inds = np.unique(time_inds)

    # get topography for F stat
    f_map = F_obs[time_inds, ...].mean(axis=0)

    # get signals at the sensors contributing to the cluster
    sig_times = epochs.times[time_inds]

    # create spatial mask
    mask = np.zeros((f_map.shape[0], 1), dtype=bool)
    mask[ch_inds, :] = True

    # initialize figure
    fig, ax_topo = plt.subplots(1, 1, figsize=(10, 3), layout="constrained")
    
    # plot average test statistic and mark significant sensors
    epochs = epochs.pick('meg') # fix an error complaining about number of channels
    f_evoked = mne.EvokedArray(f_map[:, np.newaxis], epochs.info, tmin=0)
    f_evoked.plot_topomap(
        times=0,
        mask=mask,
        axes=ax_topo,
        cmap="Reds",
        vlim=(np.min, np.max),
        show=False,
        colorbar=False,
        mask_params=dict(markersize=10),
    )
    image = ax_topo.images[0]

    # remove the title that would otherwise say "0.000 s"
    ax_topo.set_title("")

    # create additional axes (for ERF and colorbar)
    divider = make_axes_locatable(ax_topo)

    # add axes for colorbar
    ax_colorbar = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(image, cax=ax_colorbar)
    ax_topo.set_xlabel(
        "Averaged F-map ({:0.3f} - {:0.3f} s)".format(*sig_times[[0, -1]])
    )

    # add new axis for time courses and plot time courses
    ax_signals = divider.append_axes("right", size="300%", pad=1.2)
    title = f"Cluster #{i_clu + 1}, {len(ch_inds)} sensor"
    #title = "Cluster #{0}, {1} sensor".format(i_clu + 1, len(ch_inds))
    if len(ch_inds) > 1:
        title += "s (mean)"
    plot_compare_evokeds(
        GA,
        title=title,
        picks=ch_inds,
        axes=ax_signals,
        colors=colors,
        show=False,
        split_legend=True,
        truncate_yaxis="auto",

    )

    # plot temporal cluster extent
    ymin, ymax = ax_signals.get_ylim()
    ax_signals.fill_betweenx(
        (ymin, ymax), sig_times[0], sig_times[-1], color="orange", alpha=0.3
    )

plt.show()

print('All done')
