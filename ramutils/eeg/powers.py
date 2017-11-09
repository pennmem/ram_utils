""" Helper functions for computing powers from a set of EEG signals """

import time
import numpy as np

from ptsa.data.readers import EEGReader
from ptsa.data.filters import (
    MonopolarToBipolarMapper,
    MorletWaveletFilterCpp,
    MorletWaveletFilter,
    ButterworthFilter
)

try:
    from typing import List
except ImportError:
    pass

from ramutils.log import get_logger
from ramutils.parameters import FRParameters
from ramutils.utils import timer

logger = get_logger()


def compute_single_session_powers(session, all_events, params, bipolar_pairs=None):
    """Compute powers for a single session.

    Parameters
    ----------
    session : int
        Session number to compute powers for
    all_events : np.recarray
        Events from all sessions
    params : ExperimentParameters

    Returns
    -------
    powers : np.ndarray

    """
    events = all_events[all_events.session == session]

    logger.info("Loading EEG data for session %d", session)
    eeg_reader = EEGReader(events=events,
                           start_time=params.start_time,
                           end_time=params.end_time)
    eeg = eeg_reader.read()

    if eeg_reader.removed_bad_data():
        logger.warning('PTSA EEG reader elected to remove some bad events')
        events = np.concatenate((events[events.session != session], 
                                    eeg['events'].data.view(np.recarray))).view(np.recarray)
        event_fields = events.dtype.names
        order = tuple(f for f in ['session', 'list', 'mstime'] if f in event_fields)
        ev_order = np.argsort(events, order=order)
        events = events[ev_order]

    eeg.add_mirror_buffer(params.buf)

    # Use bipolar pairs if they exist and recording is not already bipolar
    if 'bipolar_pairs' not in eeg.coords:
        eeg = MonopolarToBipolarMapper(time_series=eeg, bipolar_pairs=bipolar_pairs).filter()

    # Butterworth filter to remove line noise
    eeg = eeg.filtered(freq_range=[58., 62.], filt_type='stop', order=params.filt_order)

    with timer("Total wavelet decomposition time: %f s"):
        eeg.data = np.ascontiguousarray(eeg.data)
        wavelet_filter = MorletWaveletFilterCpp(time_series=eeg,
                                                freqs=params.freqs,
                                                output='power',
                                                width=params.width,
                                                cpus=25)  # FIXME: why 25?
        sess_pow_mat, phase_mat = wavelet_filter.filter()

    sess_pow_mat = sess_pow_mat.remove_buffer(params.buf).data + np.finfo(np.float).eps/2.

    if params.log_powers:
        np.log10(sess_pow_mat, sess_pow_mat)

    sess_pow_mat = np.nanmean(sess_pow_mat.transpose(2, 1, 0, 3), -1)
    return sess_pow_mat, events


# FIXME: pass in params as single object
def compute_powers(events, start_time, end_time, buffer_time, freqs,
                   log_powers, monopolar_channels=None, bipolar_pairs=None,
                   ComputePowers=None, filt_order=4, width=5):
    """ Compute powers using a Morlet wavelet filter

    Parameters
    ----------
    events:
    start_time:
    end_time:
    buffer_time:
    freqs:
    log_powers:
    monopolar_channels=None:
    bipolar_pairs=None:
    ComputePowers=None:
    filt_order=4:
    width=5:

    Returns
    -------

    """
    # TODO: This should really be split up into a few smaller functions: load,
    # ButterworthFilter, WaveletFilter, etc.

    if (bipolar_pairs is not None) and (not isinstance(bipolar_pairs, np.recarray)):
        # it expects to receive a list
        bipolar_pairs = np.array(bipolar_pairs, dtype=[('ch0', 'S3'), ('ch1', 'S3')]).view(np.recarray)
    elif (bipolar_pairs is not None) and (isinstance(bipolar_pairs, np.recarray)):
        # to get the same treatment if we get recarray , we will convert it to a list and then back to
        # recarray with correct dtype
        bipolar_pairs = np.array(list(bipolar_pairs), dtype=[('ch0', 'S3'), ('ch1', 'S3')]).view(np.recarray)

    # since it's already not guaranteed that there will be a time series for each event
    n_events = len(events)
    events = events[events['eegoffset'] >= 0]

    if n_events != len(events):
        logger.warning('Removed %s events with negative offsets', (n_events - len(events)))

    sessions = np.unique(events.session)
    pow_mat = None

    with timer("Total time for computing powers: %f"):
        for sess in sessions:
            # TODO: change this function's signature to take a params object
            # FIXME: generalize to other experiments
            params = FRParameters()
            powers, events = compute_single_session_powers(sess, events, params, bipolar_pairs)
            pow_mat = powers if pow_mat is None else np.concatenate((pow_mat, powers))

        pow_mat = pow_mat.reshape((len(events), -1))

    return pow_mat, events
