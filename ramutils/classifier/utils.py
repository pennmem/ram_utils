""" Utility functions used during classifier training """

import os
import numpy as np
from glob import glob
from scipy.stats.mstats import zscore
from classiflib.container import ClassifierContainer


def reload_classifier(subject, task, session, mount_point='/'):
    """Loads the actual classifier used by Ramulator for a particular session

    Parameters
    ----------
    subject: str
        Subject ID
    task: str
        ex: FR5, FR6, PAL1, etc
    session: int
        Session number
    mount_point: str, default '/'
        Mount point for RHINO

    Returns
    -------
    classifier_container: classiflib.container.ClassifierContainer

    """
    base_path = os.path.join(mount_point, 'data', 'eeg', subject, 'behavioral',
                             task, 'session_{}'.format(str(session)),
                             'host_pc')

    # FIXME: this needs a data quality check to confirm that all classifiers in
    # a session are the same!
    # We take the final timestamped directory because in principle retrained
    # classifiers can be different depending on artifact detection. In
    # reality, stim sessions should never be restarted (apart from issues
    # getting things started in the first place).
    config_path = os.path.join(base_path, 'config_files')
    if 'retrained_classifier' in os.listdir(config_path):
        classifier_path = glob(os.path.join(config_path, 'retrained_classifier', '*classifier*.zip'))[0]
    else:
        classifier_path = glob(os.path.join(config_path, '*classifier*.zip'))[0]
    classifier_container = ClassifierContainer.load(classifier_path)

    return classifier_container


def normalize_powers_by_session(pow_mat, events):
    """ z-score powers within session

    Parameters
    ----------
    pow_mat: (np.ndarray) Power matrix, i.e. the data matrix for the classifier (features)
    events: (pd.DataFrame) Behavioral events data

    Returns
    -------
    pow_mat: np.ndarray
        Normalized power matrix (features)

    """

    sessions = np.unique(events.session)
    for sess in sessions:
        sess_event_mask = (events.session == sess)
        pow_mat[sess_event_mask] = zscore(pow_mat[sess_event_mask],
                                          axis=0,
                                          ddof=1)

    return pow_mat


def get_sample_weights(events, **kwargs):
    """
        Calculate class weights based on recall/non-recall in given events
        data. Sample weights calculation can vary by experiment, so this
        function serves as a dispatcher for the correct sample weighting
        function. By default, it will try to choose the weighting scheme
        based on the given events. If the user specifies as scheme kwarg,
        then that scheme will be used instead.

    Parameters
    ----------
    experiment: str
    events: pd.DataFrame

    Returns
    -------
    weights: np.ndarray

    """
    scheme = determine_weighting_scheme_from_events(events)

    if scheme not in ['PAL', 'FR', 'EQUAL']:
        raise NotImplementedError("The requested weighting scheme has not "
                                  "been implemented.")

    if scheme == 'FR':
        weights = get_fr_sample_weights(events, kwargs['encoding_multiplier'])

    elif scheme == 'PAL':
        weights = get_pal_sample_weights(events, kwargs['pal_multiplier'])

    else:
        weights = get_equal_weights(events)

    return weights


def determine_weighting_scheme_from_events(events):
    """ Identify the weighting scheme to be used based on the event types
    present in events

    Parameters:
    -----------
    events:

    Returns:
    --------
    str
        The suggested weighting scheme based on the given events

    """

    observed_event_types = np.unique(events.type)
    experiments = np.unique(events.experiment)

    if ("PAL" in experiments):
        scheme = "PAL"
        event_types_to_check = ['STUDY_PAIR', 'WORD', 'REC_EVENT']

    elif "FR" in experiments:
        scheme = "FR"
        event_types_to_check = ['WORD', 'REC_BASE', 'REC_WORD']

    else:
        scheme = 'EQUAL'
        event_types_to_check = []

    for event_type in event_types_to_check:
        if event_type not in observed_event_types:
            raise RuntimeError("Unable to use desired weighting scheme based "
                               "on the experiment. Check that your events "
                               "were created with the correct event types")

    return scheme


def get_equal_weights(events):
    """ Return a vector of ones the same length as events """

    weights = np.ones(len(events))
    return weights


def get_fr_sample_weights(events, encoding_multiplier):
    """ Create sample weights based on FR scheme """
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

    # Initialize observation weights to 1
    weights = np.ones(events.shape[0], dtype=np.float)

    weights[enc_mask & (events.recalled == 0)] = n_vec[0]
    weights[enc_mask & (events.recalled == 1)] = n_vec[1]
    weights[retrieval_mask & (events.type == 'REC_BASE')] = n_vec[2]
    weights[retrieval_mask & (events.type == 'REC_WORD')] = n_vec[3]

    return weights


def get_pal_sample_weights(events, encoding_multiplier, pal_multiplier):
    """ Calculate sample weights based on PAL-specific scheme

    Parameters
    ----------
    events: pd.DataFrame
    pal_multiplier: (int) Scalar for weighting PAL samples
    encoding_multiplier: (int) Scalar for weighting encoding samples

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
    n_vec[0:4] *= pal_multiplier
    n_vec /= np.mean(n_vec)

    # scaling encoding
    n_vec[[0, 1, 4, 5]] *= encoding_multiplier
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
