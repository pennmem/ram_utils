import numpy as np

from scipy.stats.mstats import zscore, zmap
from numpy.linalg import norm


def standardize_pow_mat(stripped_pow_mat, events, sessions, outsample_session=None, outsample_list=None):
    zpow_mat = np.array(stripped_pow_mat)
    outsample_mask = None
    for session in sessions:
        sess_event_mask = (events.session == session)
        if session == outsample_session:
            outsample_mask = (events.list == outsample_list) & sess_event_mask
            insample_mask = ~outsample_mask & sess_event_mask
            zpow_mat[outsample_mask] = zmap(zpow_mat[outsample_mask], zpow_mat[insample_mask], axis=0, ddof=1)
            zpow_mat[insample_mask] = zscore(zpow_mat[insample_mask], axis=0, ddof=1)
        else:
            zpow_mat[sess_event_mask] = zscore(zpow_mat[sess_event_mask], axis=0, ddof=1)
    return zpow_mat, outsample_mask


def normalize_pow_mat(stripped_pow_mat, events, sessions, outsample_session=None, outsample_list=None):
    normal_mat = np.array(stripped_pow_mat)
    outsample_mask = None
    for session in sessions:
        sess_event_mask = (events.session == session)
        if session == outsample_session:
            outsample_mask = (events.list == outsample_list) & sess_event_mask
            insample_mask = ~outsample_mask & sess_event_mask
            insample_median = np.median(normal_mat[insample_mask], axis=0)
            normal_mat[insample_mask] -= insample_median
            insample_norm = norm(normal_mat[insample_mask], axis=0)
            normal_mat[insample_mask] /= insample_norm
            normal_mat[outsample_mask] -= insample_median
            normal_mat[outsample_mask] /= insample_norm
        else:
            med = np.median(normal_mat[sess_event_mask], axis=0)
            normal_mat[sess_event_mask] -= med
            nrm = norm(normal_mat[sess_event_mask], axis=0)
            normal_mat[sess_event_mask] /= nrm
    return normal_mat, outsample_mask
