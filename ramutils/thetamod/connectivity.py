import functools

import mne
import numpy as np
import pandas as pd

import cmlreaders
from cmlreaders.readers.eeg import milliseconds_to_events, \
    samples_to_milliseconds

__all__ = [
    'countdown_to_resting',
    'get_countdown_events',
    'get_resting_state_connectivity',
    'read_eeg_data',
]

FREQUENCY_BANDS = {
    'theta': [4., 8.],
    'theta_cwt': np.arange(4., 8.+1, 1),
    'alpha': [9., 13.],
    'alpha_cwt': np.arange(9., 13.+1, 1),
    'theta-alpha': np.arange(4, 14, 1),
    'theta-alpha_cwt': [4., 13.],
    'beta': [16., 28.],
    'beta_cwt': np.arange(16., 28.+1, 2),
    'lowgamma': [30., 60.],
    'lowgamma_cwt': np.arange(30., 60.+1, 5),
    'highgamma': [60., 120.],
    'highgamma_cwt': np.arange(70., 90.+1, 10),
    'hfa_cwt': np.array([75., 80., 85., 90., 95., 100.]),
    'hfa': [30., 120.],
    'all_freqs': np.array([4., 5., 6., 7., 8.,
                           9., 10., 11., 12., 13.,
                           16., 18., 20., 22., 24., 26., 28.,
                           30., 32., 34., 36., 38., 40., 42., 44., 46., 48.,
                           50.]),
    'all_freqs_ext': np.array([4., 5., 6., 7., 8.,
                               9., 10., 11., 12., 13.,
                               16., 18., 20., 22., 24., 26., 28.,
                               30., 32., 34., 36., 38., 40., 42., 44., 46., 48.,
                               50., 55., 60., 65., 70., 75., 80.])
}


def get_countdown_events(reader):
    """Get all COUNTDOWN_START events.

    Returns
    -------
    countdowns : pd.DataFrame

    """
    events = reader.load('events')
    countdowns = events[events.type == 'COUNTDOWN_START']
    return countdowns


def countdown_to_resting(events, samplerate=1000):
    """Convert countdown events to "resting" events: this selects 3 EEG epochs
    of 1 s each starting at offsets of 1, 3, and 7 seconds from the beginning of
    the countdown phase.

    Parameters
    ----------
    events : pd.DataFrame
    samplerate : float
        Sample rate in samples per second.

    Returns
    -------
    resting_events : pd.DataFrame
        A DataFrame consisting of only the ``eegoffset`` field (the only one
        needed to convert events to epochs).

    """
    to_millis = functools.partial(samples_to_milliseconds,
                                  sample_rate=samplerate)
    msoffsets = []
    eegfiles=  []
    for _, event in events.iterrows():
        msoffsets += [to_millis(event.eegoffset) + s * 1000 for s in (1, 4, 7)]
        eegfiles += [event.eegfile] * 3
    new_offsets = milliseconds_to_events(msoffsets, samplerate)
    new_events = pd.DataFrame({'eegfile': eegfiles,
                              'eegoffset': new_offsets.values.squeeze()})
    return new_events


def read_eeg_data(reader, events, reref=True):
    """Read EEG data from events in a single session.

    Parameters
    ----------
    reader : CMLReader
        The reader object.
    events : pd.DataFrame
        Events to read.
    reref : bool
        When True (the default), try to rereference data. This will fail when
        data were recorded in bipolar mode.

    Returns
    -------
    eeg
        EEG timeseries data.

    Notes
    -----
    This assumes a countdown phase of at least 10 seconds in length.

    """
    if reref:
        scheme = reader.load('pairs').sort_values(by=['contact_1', 'contact_2'])
    else:
        scheme = None

    eeg = reader.load_eeg(events=events, rel_start=0, rel_stop=1000,
                          scheme=scheme)

    return eeg


def get_resting_state_connectivity(array, samplerate):
    """Compute resting state connectivity coherence matrix.

    Parameters
    ----------
    array : mne.EpochsArray

    samplerate: int

    Returns
    -------
    Coherence matrix.

    """
    freqs = FREQUENCY_BANDS['theta-alpha']
    fmin, fmax = freqs[0], freqs[-1]
    # fmin, fmax = 5., 13.
    out = mne.connectivity.spectral_connectivity(array,
                                                 method='coh',
                                                 mode='multitaper',
                                                 sfreq=samplerate,
                                                 fmin=fmin, fmax=fmax,
                                                 faverage=True,
                                                 tmin=0.0,
                                                 mt_adaptive=False,
                                                 n_jobs=1,
                                                 verbose=False)
    con, freqs, times, n_epochs, n_tapers = out

    # copied directly from Ethan's code
    cons_rec = con[:, :, 0]

    # Symmetrize average network
    mu = cons_rec
    mu_full = np.nansum(np.array([mu, mu.T]), 0)
    mu_full[np.diag_indices_from(mu_full)] = np.finfo(mu_full.dtype).eps
    return mu_full
