""" Helper functions for computing powers from a set of EEG signals """

import numpy as np
import pandas as pd
from ptsa.data.filters import (
    MonopolarToBipolarMapper,
    MorletWaveletFilterCpp
)
from ptsa.data.readers import EEGReader
from scipy.stats import zscore, ttest_ind
from statsmodels.sandbox.stats.multicomp import multipletests
import io
try:
    from typing import List
except ImportError:
    pass

from ramutils.log import get_logger
from ramutils.utils import timer
from ramutils.events import get_recall_events_mask, extract_sessions, \
    partition_events, concatenate_events_for_single_experiment, \
    get_partition_masks
from ramutils.montage import generate_pairs_for_ptsa, extract_monopolar_from_bipolar


logger = get_logger()


def compute_single_session_powers(session, all_events, start_time, end_time,
                                  buffer_time, freqs, log_powers,
                                  filt_order, width, normalize, bipolar_pairs):
    """Compute powers for a single session """
    # PTSA will sometimes modify events when reading the eeg, so we ultimately
    # need to return the updated events. In case no events are removed, return
    # the original set of events
    eeg, updated_events = load_single_session_eeg(session, all_events, start_time, end_time, bipolar_pairs)

    eeg = eeg.add_mirror_buffer(buffer_time)

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
        # At this point, pow mat has dimensions: frequency, bipolar_pairs,
        # events, time
        sess_pow_mat, phase_mat = wavelet_filter.filter()

    sess_pow_mat = sess_pow_mat.remove_buffer(buffer_time).data + np.finfo(
        np.float).eps/2.

    if log_powers:
        np.log10(sess_pow_mat, sess_pow_mat)

    # Re-ordering dimensions to be events, frequencies, electrodes with the
    # mean calculated over the time dimension
    updated_session_events = updated_events[updated_events.session == session]
    sess_pow_mat = np.nanmean(sess_pow_mat.transpose(2, 1, 0, 3), -1)
    sess_pow_mat = sess_pow_mat.reshape((len(updated_session_events), -1))

    if normalize:
        sess_pow_mat = zscore(sess_pow_mat, axis=0, ddof=1)

    return sess_pow_mat, updated_events


def load_single_session_eeg(session, all_events, start_time, end_time, bipolar_pairs):
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
        updated_events = np.rec.array(np.concatenate(
            (all_events[all_events.session != session],
             np.rec.array(eeg['events'].data))))
        event_fields = updated_events.dtype.names
        order = tuple(f for f in ['session', 'list', 'mstime'] if f in event_fields)
        ev_order = np.argsort(updated_events, order=order)
        updated_events = updated_events[ev_order]
        updated_events = np.rec.array(updated_events)

    # Use bipolar pairs if they exist and recording is not already bipolar
    if 'bipolar_pairs' not in eeg.coords:
        monopolar_channels = extract_monopolar_from_bipolar(bipolar_pairs)
        eeg_reader = EEGReader(events=session_events,
                               start_time=start_time,
                               end_time=end_time,
                               channels=monopolar_channels)
        eeg = eeg_reader.read()
        # Check for removal of bad data again and update events
        if eeg_reader.removed_bad_data():
            logger.warning('PTSA EEG reader elected to remove some bad events')
            # TODO: Use the event utility functions here
            updated_events = np.rec.array(np.concatenate(
                (all_events[all_events.session != session],
                 np.rec.array(eeg['events'].data))))
            event_fields = updated_events.dtype.names
            order = tuple(f for f in ['session', 'list', 'mstime'] if f in event_fields)
            ev_order = np.argsort(updated_events, order=order)
            updated_events = updated_events[ev_order]
            updated_events = np.rec.array(updated_events)

        eeg = MonopolarToBipolarMapper(time_series=eeg,
                                       bipolar_pairs=bipolar_pairs).filter()
    return eeg, updated_events


def compute_powers(events, start_time, end_time, buffer_time, freqs,
                   log_powers, filt_order=4, width=5,
                   normalize=True, bipolar_pairs=None):
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
    normalize: bool
        Whether power matrix should be zscored using mean and std. dev by
        electrode (row)
    bipolar_pairs: OrderedDoct
        OrderedDict of bipolar pairs to use if converting a monopolar EEG
        recording to bipolar recording

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
        bipolar_pairs = generate_pairs_for_ptsa(bipolar_pairs)

    sessions = np.unique(events.session)
    pow_mat = None

    with timer("Total time for computing powers: %f"):
        updated_events = events.copy()
        for sess in sessions:
            powers, updated_events = compute_single_session_powers(sess,
                                                                   updated_events,
                                                                   start_time,
                                                                   end_time,
                                                                   buffer_time,
                                                                   freqs,
                                                                   log_powers,
                                                                   filt_order,
                                                                   width,
                                                                   normalize,
                                                                   bipolar_pairs)
            pow_mat = powers if pow_mat is None else np.concatenate((pow_mat,
                                                                     powers))

    return pow_mat, updated_events


def compute_normalized_powers(events, **kwargs):
    """ Compute powers by session, encoding/retrieval, and FR vs. PAL

    Notes
    -----
    There are different start times, end time, and buffer times for each
    subset type, so those are passed in as kwargs and looked up prior to
    calling the more general compute_powers function

    """

    event_partitions = partition_events(events)
    cleaned_event_partitions = []
    power_partitions = {}

    if 'bipolar_pairs' not in kwargs.keys():
        kwargs['bipolar_pairs'] = None

    for subset_name, event_subset in event_partitions.items():
        if len(event_subset) == 0:
            continue

        if subset_name == 'fr_encoding':
            start_time = kwargs['encoding_start_time']
            end_time = kwargs['encoding_end_time']
            buffer_time = kwargs['encoding_buf']

        elif subset_name == 'fr_retrieval':
            start_time = kwargs['retrieval_start_time']
            end_time = kwargs['retrieval_end_time']
            buffer_time = kwargs['retrieval_buf']

        elif subset_name == 'pal_encoding':
            start_time = kwargs['pal_start_time']
            end_time = kwargs['pal_end_time']
            buffer_time = kwargs['pal_buf_time']

        elif subset_name == 'pal_retrieval':
            start_time = kwargs['pal_retrieval_start_time']
            end_time = kwargs['pal_retrieval_end_time']
            buffer_time = kwargs['pal_retrieval_buf']

        elif subset_name == 'post_stim':
            start_time = kwargs['post_stim_start_time']
            end_time = kwargs['post_stim_end_time']
            buffer_time = kwargs['post_stim_buf']

        else:
            raise RuntimeError("Unexpected event subset was encountered")

        powers, cleaned_events = compute_powers(event_subset,
                                                start_time,
                                                end_time,
                                                buffer_time,
                                                kwargs['freqs'],
                                                kwargs['log_powers'],
                                                filt_order=kwargs['filt_order'],
                                                normalize=kwargs[
                                                    'normalize_powers'],
                                                width=kwargs['width'],
                                                bipolar_pairs=kwargs[
                                                    'bipolar_pairs'])
        cleaned_event_partitions.append(cleaned_events)
        power_partitions[subset_name] = powers

    cleaned_events = concatenate_events_for_single_experiment(
        cleaned_event_partitions)

    partition_masks = get_partition_masks(cleaned_events)

    # Ensure that the rows of the power matrix match the order of the events.
    # This works by creating masks for each of the event types from the
    # sorted events structure
    n_features = powers.shape[1]
    normalized_powers = np.empty((len(cleaned_events), n_features))
    for subset_name, power_subset in power_partitions.items():
        partition_event_mask = partition_masks[subset_name]
        normalized_powers[partition_event_mask, :] = power_subset

    return normalized_powers, cleaned_events


def reduce_powers(powers, channel_mask, n_frequencies, frequency_mask=None):
    """ Create a subset of the full power matrix by excluding certain electrodes

    Parameters
    ----------
    powers: np.ndarray
        Original power matrix
    channel_mask: array_like
        Boolean array of size n_channels
    n_frequencies: int
        Number of frequencies used in calculating the power matrix. This is
        needed to be able to properly reshape the array
    frequency_mask: array_like
        Boolean array of size n_frequencies

    Returns
    -------
    np.ndarray
        Subsetted power matrix

    """
    if frequency_mask is not None and (len(frequency_mask) != n_frequencies):
        raise RuntimeError("Size of frequency mask must match number of "
                           "frequencies")

    # Reshape into 3-dimensional array (n_events, n_electrodes, n_frequencies)
    reduced_powers = powers.reshape((len(powers), -1, n_frequencies))

    if frequency_mask is not None:
        reduced_powers = reduced_powers[:, channel_mask, frequency_mask]
    else:
        reduced_powers = reduced_powers[:, channel_mask, :]

    # Reshape back to 2D representation so it can be used as a feature matrix
    reduced_powers = reduced_powers.reshape((len(reduced_powers), -1))

    return reduced_powers


def get_trigger_frequency_mask(trigger_frequency, frequencies):
    """
        Returns a boolean mask identifying a single frequency in a list of
        frequencies
    """
    return [True if int(freq) == trigger_frequency else False for freq in
            frequencies]


def normalize_powers_by_session(pow_mat, events):
    """ z-score powers within session. Utility function used by legacy reports

    Parameters
    ----------
    pow_mat: np.ndarray
        Power matrix, i.e. the data matrix for the classifier (features)
    events: pd.DataFrame
        Behavioral events data

    Returns
    -------
    pow_mat: np.ndarray
        Normalized power matrix (features)

    Notes
    -----
    This function can be removed once the legacy reporting pipeline is fully
    replaced since those are the only places where it is currently used

    """

    sessions = np.unique(events.session)
    for sess in sessions:
        sess_event_mask = (events.session == sess)
        pow_mat[sess_event_mask] = zscore(pow_mat[sess_event_mask],
                                          axis=0,
                                          ddof=1)

    return pow_mat


def reshape_powers_to_3d(powers, n_frequencies):
    """
        Make power matrix a 3D structure:
        n_events x n_electrodes x n_frequencies
    """
    reshaped_powers = powers.reshape((len(powers), -1, n_frequencies))
    return reshaped_powers


def reshape_powers_to_2d(powers):
    """
        Make power matrix a 2D structure
        n_events x (n_electrodes x n_frequencies)
    """
    reshaped_powers = powers.reshape((len(powers), -1))
    return reshaped_powers

def save_power_plot(powers,session,full_path):
    """
    Plots the feature matrix to a file path or file-like object
    :param powers:
    :param full_path:
    :return:
    """
    from matplotlib import pyplot as plt

    plt.imshow(reshape_powers_to_2d(powers),cmap='RdBu_r',aspect='auto',)
    cmin,cmax = powers.min(),powers.max()
    clim = max(abs(cmin),abs(cmax))
    plt.clim(-clim,clim)
    cbar = plt.colorbar()
    cbar.ax.set_xlabel('Z-Score')
    cbar.ax.xaxis.set_label_position('top')
    plt.ylabel('Event Number')
    plt.xlabel('Feature Number')
    plt.title('Session %s'%session)
    plt.savefig(full_path,
                format="png",
                dpi=300,
                bbox_inches="tight",
                )
    plt.close()
    return full_path


def load_eeg(all_events, start_time, end_time, bipolar_pairs=None):
    full_eeg = []
    for session in np.unique(all_events.session):
        eeg,_  = load_single_session_eeg(session, all_events, start_time, end_time, bipolar_pairs)
        if bipolar_pairs is None and 'bipolar_pairs' in eeg.dims:
            bipolar_pairs = eeg.bipolar_pairs.values
        time = eeg.time.values
        full_eeg.append(eeg)
    full_eeg = np.concatenate([e.data for e in full_eeg],axis=1)
    return full_eeg


def save_eeg_by_channel_plot(bipolar_pairs, full_eeg, time=None, full_path=None):
    from matplotlib import pyplot as plt
    if full_path is None:
        full_path = io.BytesIO()
    if time is None:
        time = np.arange(full_eeg.shape[-1])

    ylen = int(np.sqrt(full_eeg.shape[0]))
    xlen = int(len(bipolar_pairs) / ylen) + 1
    plt.figure(figsize=(20, 15))
    for i in range(0, len(bipolar_pairs)):
        plt.subplot(xlen, ylen, i + 1)
        plt.plot(time, full_eeg[i].squeeze().T,color='grey', alpha=0.05)
        plt.ylim(-30000, 30000)
        plt.xlabel('%s'%(bipolar_pairs[i]))
    plt.tight_layout()
    plt.savefig(full_path,
                format='png',
                dpi=200,
                bbox_inches='tight')
    plt.close()
    return full_path


def calculate_delta_hfa_table(pairs_metadata_table, normalized_powers, events,
                              frequencies, hfa_cutoff=65, trigger_freq=110):
    """
        Calculate tstats and pvalues from a ttest comparing HFA activity of
        recalled versus non-recalled items
    """
    powers_3d = reshape_powers_to_3d(normalized_powers, len(frequencies))
    hfa_mask = [True if freq > hfa_cutoff else False for freq in frequencies]
    hfa_powers = powers_3d[:, :, hfa_mask]

    # Average powers across frequencies. New shape is n_events x n_electrodes
    hfa_powers = np.nanmean(hfa_powers, axis=-1)

    recall_mask = get_recall_events_mask(events)
    recalled_pow_mat = hfa_powers[recall_mask, :]
    non_recalled_pow_mat = hfa_powers[~recall_mask, :]

    tstats, pvals = ttest_ind(recalled_pow_mat, non_recalled_pow_mat, axis=0)
    sig_mask, pvals, _ , _ = multipletests(pvals, method='fdr_bh')

    pairs_metadata_table['hfa_t_stat'] = tstats
    pairs_metadata_table['hfa_p_value'] = pvals

    # Repeat for 110hz. Actual frequency is a decimal, so convert to int when
    #  checking for equality
    trigger_freq_mask = [True if int(freq) == trigger_freq else False for
                         freq in frequencies]
    single_freq_powers = powers_3d[:, :, trigger_freq_mask]
    single_freq_powers = np.nanmean(single_freq_powers, axis=-1)

    recalled_single_freq_powers = single_freq_powers[recall_mask, :]
    non_recalled_single_freq_powers = single_freq_powers[~recall_mask, :]

    tstats, pvals = ttest_ind(recalled_single_freq_powers,
                              non_recalled_single_freq_powers, axis=0)
    sig_mask, pvals, _ , _ = multipletests(pvals, method='fdr_bh')
    pairs_metadata_table['110_t_stat'] = tstats
    pairs_metadata_table['110_p_value'] = pvals

    # Pairs that do not have a label do not need to have the stats displayed
    pairs_metadata_table = pairs_metadata_table.dropna(subset=['label'])

    return pairs_metadata_table

