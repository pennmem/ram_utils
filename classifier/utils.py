""" Utility functions used during classifier training """

import numpy as np
from scipy.stats.mstats import zscore



def normalize_sessions(pow_mat, events):
    """ z-score powers within session """

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

    n_ret_0 = events[retrieval_mask & (events.type == 'REC_BASE')].shape[0]
    n_ret_1 = events[retrieval_mask & (events.type == 'REC_WORD')].shape[0]

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


