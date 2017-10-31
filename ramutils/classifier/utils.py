""" Utility functions used during classifier training """

import h5py
import numpy as np
from scipy.stats.mstats import zscore


def normalize_sessions(pow_mat, events):
    """ z-score powers within session

    Parameters:
    ----------
    pow_mat: (np.ndarray) Power matrix, i.e. the data matrix for the classifier (features)
    events: (pd.DataFrame) Behavioral events data

    Returns
    -------
    pow_mat: np.ndarray
        Normalized features

    """

    sessions = np.unique(events.session)
    for sess in sessions:
        sess_event_mask = (events.session == sess)
        pow_mat[sess_event_mask] = zscore(pow_mat[sess_event_mask], axis=0, ddof=1)

    return pow_mat


def get_sample_weights(events, encoding_multiplier):
    """ Calculate class weights based on recall/non-recall in given events data

    Parameters:
    -----------
    events: pd.DataFrame
    encoding_multiplier: (int) Scalar to determine how much more to weight encoding samples

    Returns
    -------
    weights: np.ndarray

    """
    enc_mask = (events.type == 'WORD')
    retrieval_mask = ((events.type == 'REC_BASE') | (events.type == 'REC_WORD'))

    n_enc_0 = events[enc_mask & (events.recalled == 0)].shape[0]
    n_enc_1 = events[enc_mask & (events.recalled == 1)].shape[0]

    n_ret_0 = events[events.type == 'REC_BASE'].shape[0]
    n_ret_1 = events[events.type == 'REC_WORD'].shape[0]

    n_vec = np.array([1.0/n_enc_0, 1.0/n_enc_1, 1.0/n_ret_0, 1.0/n_ret_1 ], dtype=np.float)
    n_vec /= np.mean(n_vec)

    n_vec[:2] *= encoding_multiplier

    n_vec /= np.mean(n_vec)

    # Initialize observatoins weights to 1
    weights = np.ones(events.shape[0], dtype=np.float)

    weights[enc_mask & (events.recalled == 0)] = n_vec[0]
    weights[enc_mask & (events.recalled == 1)] = n_vec[1]
    weights[retrieval_mask & (events.type == 'REC_BASE')] = n_vec[2]
    weights[retrieval_mask & (events.type == 'REC_WORD')] = n_vec[3]


    return weights


def get_pal_sample_weights(events, pal_sample_weights, encoding_sample_weights):
    """ Calculate sample weights based on PAL-specific scheme

    Parameters:
    -----------
    events: pd.DataFrame
    pal_sample_weights: (int) Scalar to determine how much more to weight pal samples
    encoding_sample_weights: (int) Scalar to determine how much more to weight encoding samples

    Returns
    -------
    weights: np.ndarray

    """

    enc_mask = (events.type == 'WORD')
    retrieval_mask = (events.type == 'REC_EVENT')

    pal_mask = (events.exp_name == 'PAL1')
    fr_mask = ~pal_mask

    pal_n_enc_0 = events[pal_mask & enc_mask & (events.correct == 0)].shape[0]
    pal_n_enc_1 = events[pal_mask & enc_mask & (events.correct == 1)].shape[0]

    pal_n_ret_0 = events[pal_mask & retrieval_mask & (events.correct == 0)].shape[0]
    pal_n_ret_1 = events[pal_mask & retrieval_mask & (events.correct == 1)].shape[0]

    fr_n_enc_0 = events[fr_mask & enc_mask & (events.correct == 0)].shape[0]
    fr_n_enc_1 = events[fr_mask & enc_mask & (events.correct == 1)].shape[0]

    fr_n_ret_0 = events[fr_mask & retrieval_mask & (events.correct == 0)].shape[0]
    fr_n_ret_1 = events[fr_mask & retrieval_mask & (events.correct == 1)].shape[0]

    ev_count_list = [pal_n_enc_0, pal_n_enc_1, pal_n_ret_0, pal_n_ret_1,
                     fr_n_enc_0, fr_n_enc_1, fr_n_ret_0, fr_n_ret_1]

    n_vec = np.array([0.0] * 8, dtype=np.float)

    for i, ev_count in enumerate(ev_count_list):
        n_vec[i] = 1. / ev_count if ev_count else 0.0

    n_vec /= np.mean(n_vec)

    # scaling PAL1 task
    n_vec[0:4] *= pal_sample_weights
    n_vec /= np.mean(n_vec)

    # scaling encoding
    n_vec[[0, 1, 4, 5]] *= encoding_sample_weights
    n_vec /= np.mean(n_vec)

    weights = np.ones(events.shape[0], dtype=np.float)

    weights[pal_mask & enc_mask & (events.correct == 0)] = n_vec[0]
    weights[pal_mask & enc_mask & (events.correct == 1)] = n_vec[1]
    weights[pal_mask & retrieval_mask & (events.correct == 0)] = n_vec[2]
    weights[pal_mask & retrieval_mask & (events.correct == 1)] = n_vec[3]

    weights[fr_mask & enc_mask & (events.correct == 0)] = n_vec[4]
    weights[fr_mask & enc_mask & (events.correct == 1)] = n_vec[5]
    weights[fr_mask & retrieval_mask & (events.correct == 0)] = n_vec[6]
    weights[fr_mask & retrieval_mask & (events.correct == 1)] = n_vec[7]

    return weights
