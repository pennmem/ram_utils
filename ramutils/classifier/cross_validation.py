""" Collection of cross-validation functions """
import numpy as np

from random import shuffle
from copy import deepcopy
from sklearn.metrics import roc_auc_score

from ramutils.classifier.weighting import get_sample_weights
from ramutils.events import get_encoding_mask, select_encoding_events
from ramutils.log import get_logger
from ramutils.tasks.classifier import logger

try:
    from typing import Dict, Union, Tuple
except ImportError:
    pass

logger = get_logger()

__all__ = [
    'permuted_lolo_cross_validation',
    'perform_lolo_cross_validation',
    'permuted_loso_cross_validation',
    'perform_loso_cross_validation'
]


def permuted_lolo_cross_validation(classifier, powers, events, n_permutations, **kwargs):
    """Permuted leave-one-list-out cross validation

    Parameters
    ----------
    classifier:
        sklearn model object, usually logistic regression classifier
    powers: np.ndarray
        power matrix
    events : recarray
    n_permutations: int
        number of permutation trials
    kwargs: dict
        Optional keyword arguments. These are passed to get_sample_weights.
        See that function for more details.

    Returns
    -------
    AUCs: list
        List of AUCs from performing leave-one-list-out cross validation
        n_permutations times where the AUC is based on encoding events only

    """
    recalls = events.recalled
    permuted_recalls = np.array(recalls)
    auc_results = np.empty(shape=n_permutations, dtype=np.float)
    encoding_mask = get_encoding_mask(events)

    sessions = np.unique(events.session)
    for i in range(n_permutations):
        for sess in sessions:
            sess_lists = np.unique(events[events.session == sess].list)
            for lst in sess_lists:
                # Permute recall outcome within each session/list
                sel = (events.session == sess) & (events.list == lst)
                list_permuted_recalls = permuted_recalls[sel]
                shuffle(list_permuted_recalls)
                permuted_recalls[sel] = list_permuted_recalls

        # The probabilities returned here are only for encoding events
        probs = perform_lolo_cross_validation(classifier, powers, events,
                                              recalls, **kwargs)

        encoding_recalls = permuted_recalls[encoding_mask]
        auc_results[i] = roc_auc_score(encoding_recalls, probs)

    return auc_results


def perform_lolo_cross_validation(classifier, powers, events, recalls, **kwargs):
    """Perform a single iteration of leave-one-list-out cross validation

    Parameters
    ----------
    classifier: sklearn model object
    powers: mean powers to use as features
    events: set of events for the session
    recalls: vector of recall outcomes
    kwargs: dict
         Optional keyword arguments. These are passed to get_sample_weights.
        See that function for more details.

    Returns
    -------
    probs: np.array
        Predicted probabilities for encoding events across all lists

    Notes
    -----
    Be careful when passing a classifier object to this function since it's
    .fit() method will be called. If you use the classifier object after
    calling this function, the internal state may have changed. To avoid this
    problem, make a copy of the classifier object and pass the copy to this
    function.
    """
    classifier_copy = deepcopy(classifier)
    encoding_mask = get_encoding_mask(events)
    encoding_events = select_encoding_events(events)

    probs = np.empty_like(recalls[encoding_mask], dtype=np.float)
    lists = np.unique(events.list)

    for lst in lists:
        insample_mask = (events.list != lst)
        insample_pow_mat = powers[insample_mask]
        insample_recalls = recalls[insample_mask]
        insample_weights = get_sample_weights(events[insample_mask], **kwargs)

        # We don't want to call fit on the passed classifier because this will
        # have side-effects for the user/program that calls this function
        classifier_copy.fit(insample_pow_mat, insample_recalls,
                            insample_weights)

        # Out of sample predictions need to be on encoding only
        outsample_mask = ~insample_mask & encoding_mask
        outsample_pow_mat = powers[outsample_mask]

        outsample_encoding_event_mask = (encoding_events.list == lst)
        probs[outsample_encoding_event_mask] = classifier_copy.predict_proba(
            outsample_pow_mat)[:, 1]

    return probs


def permuted_loso_cross_validation(classifier, powers, events, n_permutations, **kwargs):
    """ Perform permuted leave one session out cross validation

    Parameters
    ----------
    classifier:
        sklearn model object, usually logistic regression classifier
    powers: np.ndarray
        power matrix
    events : recarray
    n_permutations: int
        number of permutation trials
    kwargs: dict
        Optional keyword arguments. These are passed to get_sample_weights.
        See that function for more details.

    Returns
    -------
    AUCs: list
        List of AUCs from performing leave-one-list-out cross validation
        n_permutations times where the AUCs are based on encoding events only

    """
    recalls = events.recalled
    sessions = np.unique(events.session)

    encoding_mask = get_encoding_mask(events)

    permuted_recalls = np.array(recalls)
    auc_results = np.empty(shape=n_permutations, dtype=np.float)

    for i in range(n_permutations):
        # Shuffle recall outcomes within session
        for session in sessions:
            in_session_mask = (events.session == session)
            session_permuted_recalls = permuted_recalls[in_session_mask]
            shuffle(session_permuted_recalls)
            permuted_recalls[in_session_mask] = session_permuted_recalls

        # The probabilities returned here will be only for encoding events
        probs = perform_loso_cross_validation(classifier, powers, events,
                                              permuted_recalls, **kwargs)

        # Evaluation should happen only on encoding events
        encoding_recalls = permuted_recalls[encoding_mask]
        auc_results[i] = roc_auc_score(encoding_recalls, probs)

    return auc_results


def perform_loso_cross_validation(classifier, powers, events, recalls, **kwargs):
    """ Perform single iteration of leave-one-session-out cross validation

    Parameters
    ----------
    classifier:
        sklearn model object, usually logistic regression classifier
    powers: np.ndarray
        power matrix
    events : np.recarray
    recalls: array_like
        List of recall/not-recalled boolean values for each event
    kwargs: dict
        Optional keyword arguments. These are passed to get_sample_weights.
        See that function for more details.

    Returns
    -------
    probs: np.array
        Predicted probabilities for encoding events across all sessions

    """
    classifier_copy = deepcopy(classifier)
    sessions = np.unique(events.session)
    encoding_mask = get_encoding_mask(events)
    encoding_events = select_encoding_events(events)

    # Predicted probabilities should be assessed only on encoding words
    probs = np.empty_like(recalls[encoding_mask], dtype=np.float)

    for sess_idx, sess in enumerate(sessions):
        # training data
        insample_mask = (events.session != sess)
        insample_pow_mat = powers[insample_mask]
        insample_recalls = recalls[insample_mask]
        insample_samples_weights = get_sample_weights(events[insample_mask],
                                                      **kwargs)
        classifier_copy.fit(insample_pow_mat, insample_recalls,
                            insample_samples_weights)

        # testing data -- Only look at encoding events
        outsample_mask = ~insample_mask & encoding_mask
        outsample_pow_mat = powers[outsample_mask]

        outsample_probs = classifier_copy.predict_proba(outsample_pow_mat)[:, 1]

        outsample_encoding_event_mask = (encoding_events.session == sess)
        probs[outsample_encoding_event_mask] = outsample_probs

    return probs


def perform_cross_validation(classifier, events, n_permutations, pow_mat, recalls, sessions, **kwargs):
    if len(sessions) > 1:
        permuted_auc_values = permuted_loso_cross_validation(classifier,
                                                             pow_mat,
                                                             events,
                                                             n_permutations,
                                                             **kwargs)
        probs = perform_loso_cross_validation(classifier, pow_mat, events,
                                              recalls, **kwargs)

    else:
        logger.info("Performing LOLO cross validation")
        permuted_auc_values = permuted_lolo_cross_validation(classifier,
                                                             pow_mat,
                                                             events,
                                                             n_permutations,
                                                             **kwargs)
        probs = perform_lolo_cross_validation(classifier, pow_mat, events,
                                              recalls, **kwargs)
    return permuted_auc_values, probs