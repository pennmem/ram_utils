""" Collection of cross-validation functions """
import numpy as np

from random import shuffle
from sklearn.metrics import roc_auc_score

from ramutils.classifier import ModelOutput
from ramutils.classifier.weighting import get_sample_weights
from ramutils.log import get_logger

try:
    from typing import Dict, Union, Tuple
except ImportError:
    pass

logger = get_logger()

__all__ = [
    'perform_cross_validation',
    'permuted_lolo_cross_validation',
    'perform_lolo_cross_validation',
    'permuted_loso_cross_validation',
    'perform_loso_cross_validation'
]


def perform_cross_validation(classifier, pow_mat, events, n_permutations,
                             **kwargs):
    """Perform LOSO or LOLO cross validation on a classifier.

    Parameters
    ----------
    classifier : sklearn model object
    pow_mat : np.ndarray
    events : np.recarray
    n_permutations: int
    kwargs: dict
        Extra keyword arguments that are passed to get_sample_weights. See
        that function for more details

    Returns
    -------
    xval : dict
        Results of cross validation.

    """
    recalls = events.recalled

    # Stores cross validation output. Keys are sessions or 'all' for all session
    # cross validation.
    xval = {}  # type: Dict[Union[str, int], ModelOutput]
    sessions = np.unique(events.session)

    # Run leave-one-session-out cross validation when we have > 1 session
    if len(sessions) > 1:
        perm_AUCs = permuted_loso_cross_validation(classifier, pow_mat, events,
                                                   n_permutations, **kwargs)
        probs, xval = perform_loso_cross_validation(classifier, pow_mat, events,
                                                    recalls, xval, **kwargs)

        # Store model output statistics
        auc = roc_auc_score(recalls, probs)
        xval['all'] = auc

    # ... otherwise run leave-one-list-out cross validation
    else:
        logger.info("Performing LOLO cross validation")
        session = sessions[0]
        perm_AUCs = permuted_lolo_cross_validation(classifier, pow_mat, events,
                                                   n_permutations, **kwargs)
        probs = perform_lolo_cross_validation(classifier, pow_mat, events, recalls, **kwargs)

        # Store model output statistics
        auc = roc_auc_score(recalls, probs)
        xval['all'] = xval[session] = auc

    pvalue = np.count_nonzero((perm_AUCs >= xval['all'])) / float(len(
        perm_AUCs))
    logger.info("Permutation test p-value = %f", pvalue)

    recall_prob = classifier.predict_proba(pow_mat)[:, 1]
    insample_auc = roc_auc_score(recalls, recall_prob)
    logger.info("in-sample AUC = %f", insample_auc)
    return xval


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
        n_permutations times

    """
    recalls = events.recalled
    permuted_recalls = np.array(recalls)
    auc_results = np.empty(shape=n_permutations, dtype=np.float)
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

        probs = perform_lolo_cross_validation(classifier, powers, events, recalls, **kwargs)
        auc_results[i] = roc_auc_score(permuted_recalls, probs)

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
        Predicted probabilities for all lists

    """
    probs = np.empty_like(recalls, dtype=np.float)
    lists = np.unique(events.list)

    for lst in lists:
        insample_mask = (events.list != lst)
        insample_pow_mat = powers[insample_mask]
        insample_recalls = recalls[insample_mask]
        insample_weights = get_sample_weights(events[insample_mask], **kwargs)
        classifier.fit(insample_pow_mat, insample_recalls, insample_weights)
        outsample_mask = ~insample_mask
        outsample_pow_mat = powers[outsample_mask]
        probs[outsample_mask] = classifier.predict_proba(outsample_pow_mat)[:, 1]

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
        n_permutations times

    """
    recalls = events.recalled
    permuted_recalls = np.array(recalls)
    auc_results = np.empty(shape=n_permutations, dtype=np.float)
    sessions = np.unique(events.session)
    xval = {}

    for i in range(n_permutations):
        try:
            # Shuffle recall outcomes within session
            for session in sessions:
                in_session_mask = (events.session == session)
                session_permuted_recalls = permuted_recalls[in_session_mask]
                shuffle(session_permuted_recalls)
                permuted_recalls[in_session_mask] = session_permuted_recalls

            # We don't need to capture the auc by session for each permutation
            probs, _ = perform_loso_cross_validation(classifier, powers, events,
                                                     permuted_recalls, xval, **kwargs)
            auc_results[i] = roc_auc_score(permuted_recalls, probs)
        except ValueError:
            auc_results[i] = np.nan

    return auc_results


def perform_loso_cross_validation(classifier, powers, events, recalls, xval, **kwargs):
    """Perform single iteration of leave-one-session-out cross validation

    Parameters
    ----------
    classifier:
        sklearn model object, usually logistic regression classifier
    powers: np.ndarray
        power matrix
    events : np.recarray
    recalls: array_like
        List of recall/not-recalled boolean values for each event
    xval: dict_like
        Object for storing aucs from cross validation
    kwargs: dict
        Optional keyword arguments. These are passed to get_sample_weights.
        See that function for more details.

    Returns
    -------
    probs: np.array
        Predicted probabilities for all sessions

    """
    probs = np.empty_like(recalls, dtype=np.float)
    sessions = np.unique(events.session)

    for sess_idx, sess in enumerate(sessions):
        # training data
        insample_mask = (events.session != sess)
        insample_pow_mat = powers[insample_mask]
        insample_recalls = recalls[insample_mask]
        insample_samples_weights = get_sample_weights(events[insample_mask],
                                                      **kwargs)
        classifier.fit(insample_pow_mat, insample_recalls,
                       insample_samples_weights)

        # testing data
        outsample_mask = ~insample_mask
        outsample_recalls = recalls[outsample_mask]
        outsample_pow_mat = powers[outsample_mask]

        outsample_probs = classifier.predict_proba(outsample_pow_mat)[:, 1]
        probs[outsample_mask] = outsample_probs

        auc = roc_auc_score(outsample_recalls, outsample_probs)
        xval[sess] = auc

    return probs, xval


