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
from ramutils.utils import timer

logger = get_logger()


def compute_single_session_powers(session, all_events, start_time, end_time,
                                  buffer_time, freqs, log_powers,
                                  filt_order, width, bipolar_pairs):
    """Compute powers for a single session """
    # PTSA will sometimes modify events when reading the eeg, so we ultimately
    # need to return the updated events. In case no events are removed, return
    # the original set of events
    updated_events = all_events
    session_events = all_events[all_events.session == session]

    logger.info("Loading EEG data for session %d", session)
    eeg_reader = EEGReader(events=session_events,
                           start_time=start_time,
                           end_time=end_time,
                           )
    try:
        eeg = eeg_reader.read()
    # recording was done in bipolar mode, and the channels are different than
    # what we expect
    except IndexError:
        eeg_reader.channels = np.array([])
        eeg = eeg_reader.read()

    if eeg_reader.removed_bad_data():
        logger.warning('PTSA EEG reader elected to remove some bad events')
        # TODO: Use the event utility functions here
        updated_events = np.concatenate(
            (all_events[all_events.session != session],
             eeg['events'].data.view(np.recarray))).view(np.recarray)
        event_fields = updated_events.dtype.names
        order = tuple(f for f in ['session', 'list', 'mstime'] if f in event_fields)
        ev_order = np.argsort(updated_events, order=order)
        updated_events = updated_events[ev_order]
        updated_events = updated_events.view(np.recarray)

    eeg = eeg.add_mirror_buffer(buffer_time)

    # Use bipolar pairs if they exist and recording is not already bipolar
    if 'bipolar_pairs' not in eeg.coords:
        eeg = MonopolarToBipolarMapper(time_series=eeg,
                                       bipolar_pairs=bipolar_pairs).filter()

    # Butterworth filter to remove line noise
    eeg = eeg.filtered(freq_range=[58., 62.],
                       filt_type='stop',
                       order=filt_order)
    with timer("Total wavelet decomposition time: %f s"):
        eeg.data = np.ascontiguousarray(eeg.data)
        wavelet_filter = MorletWaveletFilterCpp(time_series=eeg,
                                                freqs=freqs,
                                                output='power',
                                                width=width,
                                                cpus=25)  # FIXME: why 25?
        sess_pow_mat, phase_mat = wavelet_filter.filter()

    sess_pow_mat = sess_pow_mat.remove_buffer(buffer_time).data + np.finfo(
        np.float).eps/2.

    if log_powers:
        np.log10(sess_pow_mat, sess_pow_mat)
    sess_pow_mat = np.nanmean(sess_pow_mat.transpose(2, 1, 0, 3), -1)

    return sess_pow_mat, updated_events


def compute_powers(events, start_time, end_time, buffer_time, freqs,
                   log_powers, filt_order=4, width=5, bipolar_pairs=None):
    """
        Compute powers (or log powers) using a Morlet wavelet filter and
        Butterworth Filter to get rid of line noise


    Parameters
    ----------
    events: np.recarray
        Events to consider when computing powers
    start_time: float
        Start of the period in the EEG to consider for each event
    end_time: float
        End of the period to consider
    buffer_time: float
        Buffer time
    freqs: array_like
        List of frequencies to use when applying Wavelet Filter
    log_powers: bool
        Whether to take the logarithm of the powers
    filt_order: Int
        Filter order to use in Butterworth filter
    width: Int
        Wavelet width to use in Wavelet Filter
    bipolar_pairs: array_like
        List of bipolar pairs to use if converting a monopolar EEG recording to
        bipolar recording

    Returns
    -------
    np.ndarray
        Calculated powers of shape n_events X (freqs * n_channels) where
        n_channels is determined when loading the EEG
    np.recarray
        Set of events after 'bad' events were removed while loading the EEG.
        Currently, removal of these events is a side effect of loading the
        EEG, so the 'cleaned' events must be caught. In an ideal world,
        this side effect would not exist and bad events would be remove prior
        to computing powers.
    -------

    """
    if (bipolar_pairs is not None) and \
            (not isinstance(bipolar_pairs, np.recarray)):
        # it expects to receive a list
        bipolar_pairs = np.array(bipolar_pairs,
                                 dtype=[('ch0', 'S3'),
                                        ('ch1', 'S3')]).view(np.recarray)

    elif (bipolar_pairs is not None) and \
            (isinstance(bipolar_pairs, np.recarray)):
        # to get the same treatment if we get recarray , we will convert it to
        # a list and then back to recarray with correct dtype
        bipolar_pairs = np.array(list(bipolar_pairs),
                                 dtype=[('ch0', 'S3'),
                                        ('ch1', 'S3')]).view(np.recarray)
    sessions = np.unique(events.session)
    pow_mat = None

    with timer("Total time for computing powers: %f"):
        for sess in sessions:
            powers, updated_events = compute_single_session_powers(sess,
                                                                   events,
                                                                   start_time,
                                                                   end_time,
                                                                   buffer_time,
                                                                   freqs,
                                                                   log_powers,
                                                                   filt_order,
                                                                   width,
                                                                   bipolar_pairs)
            pow_mat = powers if pow_mat is None else np.concatenate((pow_mat,
                                                                     powers))

        pow_mat = pow_mat.reshape((len(events), -1))

    return pow_mat, events
