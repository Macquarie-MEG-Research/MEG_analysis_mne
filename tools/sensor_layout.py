# plot MEG sensor layout

import mne

lay = mne.channels.read_layout("KIT-160.lay")
lay.plot()

print("Done!")