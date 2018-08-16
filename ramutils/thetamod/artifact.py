import numpy as np
from cmlreaders.path_finder import PathFinder
import os
from glob import glob
from scipy.stats import ttest_rel
from scipy.stats import levene
import itertools

"""
Three-stage filter for rejecting artifactual signals:

1. Reject any channel labelled as bad in `electrode_categories.txt`
2. Reject any trial that shows amplifier saturation
3. Reject any channel that shows too large a difference between the power spectra
pre- and post-stim, as measured using a paired t-test

"""

__all__ = ['get_saturated_events_mask', 'get_bad_channel_names',
           'get_bad_events_mask',
           'get_channel_exclusion_pvals', 'invalidate_eeg']


def get_bad_channel_names(subject, montage, just_bad=None, rhino_root='/'):
    finder = PathFinder(rootdir=rhino_root,subject=subject, montage=montage)
    fn = finder.find('electrode_categories')
    with open(fn, 'r') as fh:
        lines = [mystr.replace('\n', '') for mystr in fh.readlines()]

    if just_bad is True:
        bidx = len(lines)
        try:
            bidx = [s.lower().replace(':', '').strip() for s in lines].index('bad electrodes')
        except:
            try:
                bidx = [s.lower().replace(':', '').strip() for s in lines].index('broken leads')
            except:
                lines = []
        lines = lines[bidx:]
    return lines


def get_bad_events_mask(eegs, events):
    saturation_mask = get_saturated_events_mask(eegs)
    adjacent_events_mask = get_adjacent_events_mask(events)

    return saturation_mask | adjacent_events_mask[:, None]


def get_adjacent_events_mask(events):
    msdiff = np.diff(events.mstime)
    msdiff = np.append(msdiff, np.nan)
    for idx, i in enumerate(msdiff):
        if i < 1500:
            msdiff[idx] = np.nan
            msdiff[idx+1] = np.nan
    return np.isnan(msdiff)


def get_saturated_events_mask(eegs):
    # Return array of chans x events with 1s where saturation is found

    def zero_runs(a):
        a = np.array(a)
        # Create an array that is 1 where a is 0, and pad each end with an extra 0.
        iszero = np.concatenate(([0], np.equal(a, 0).view(np.int8), [0]))
        absdiff = np.abs(np.diff(iszero))
        # Runs start and end where absdiff is 1.
        ranges = np.where(absdiff == 1)[0].reshape(-1, 2)
        return ranges

    sat_events = np.zeros([eegs.shape[0], eegs.shape[1]])

    for i in range(eegs.shape[0]):
        for j in range(eegs.shape[1]):
            ts = eegs[i, j]
            zr = zero_runs(np.diff(np.array(ts)))
            numzeros = zr[:, 1] - zr[:, 0]
            if (numzeros > 9).any():
                sat_events[i, j] = 1
                continue

    return sat_events.astype(bool)


def get_channel_exclusion_mask(pre_eeg, post_eeg, samplerate, threshold=1e-4):
    pvals, _ = get_channel_exclusion_pvals(pre_eeg, post_eeg, samplerate)
    return pvals < threshold


def get_channel_exclusion_pvals(pre_eeg, post_eeg, samplerate):
    """
    Estimate which channels show broadband DC shift post-stimulation
    using T-test and Levene variance test.

    Parameters
    ----------
    pre_eeg: np.ndarray
       Pre-stimulus EEG signals
    post_eeg
       Post-stimulus EEG signals
    samplerate
        EEG sampling rate

    Returns
    -------

    pvals: np.ndarray
        P-values from paired t-test between pre-stim EEG and post-stim EEG

    lev_pvals: np.ndarray
        P-values from Levene variance test between pre-stim EEG and post-stim
        EEG.
    """

    def justfinites(arr):
        return arr[np.isfinite(arr)]
    pre_eeg = pre_eeg[..., int(-0.35 * samplerate):].mean(-1)
    post_eeg = post_eeg[..., :int(0.35 * samplerate)].mean(-1)

    pvals = []
    lev_pvals = []
    assert pre_eeg.shape == post_eeg.shape  # FIXME: RAISE A PROPER EXCEPTION
    for i in range(pre_eeg.shape[1]):
        eeg_t_chan, eeg_p_chan = ttest_rel(post_eeg[:, i, ...],
                                           pre_eeg[:, i, ...], nan_policy='omit')
        pvals.append(eeg_p_chan)
        try:
            lev_t, lev_p = levene(justfinites(post_eeg[:, i]),
                                  justfinites(pre_eeg[:, i]))
            lev_pvals.append(lev_p)
        except Exception as e:
            lev_pvals.append(0.0)

    return np.array(pvals), np.array(lev_pvals)


def invalidate_eeg(reader, pre_eeg, post_eeg, rhino_root, thresh=1e-5):
    label0, label1 = list(
        zip(*[x.split('-') for x in reader.load('pairs').label])
    )
    saturation_mask = get_saturated_events_mask(post_eeg.data)
    bad_channel_list = get_bad_channel_names(reader.subject, reader.montage,
                                             rhino_root=rhino_root)
    pvals, _ = get_channel_exclusion_pvals(pre_eeg.data,
                                           post_eeg.data,
                                           pre_eeg.samplerate
                                           )
    excluded_dc_shift = pvals <= thresh
    bad_channel_mask = np.in1d(label0,bad_channel_list) | np.in1d(label1, bad_channel_list)
    bad_channel_mask |= excluded_dc_shift

    pre_eeg.data[:, bad_channel_mask, :] = np.nan
    pre_eeg.data[saturation_mask, :] = np.nan
    post_eeg.data[:, bad_channel_mask, :] = np.nan
    post_eeg.data[saturation_mask, :] = np.nan

    return pre_eeg, post_eeg
