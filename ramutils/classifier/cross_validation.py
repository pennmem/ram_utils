""" Collection of cross-validation functions """
import numpy as np

from random import shuffle
from sklearn.metrics import roc_auc_score

from ramutils.classifier.utils import get_sample_weights

__all__ = [
    'permuted_lolo_AUCs',
    'run_lolo_xval',
    'permuted_loso_AUCs',
    'run_loso_xval'
]


def permuted_lolo_AUCs(classifier, powers, events, n_permutations, **kwargs):
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

    Returns
    -------
    AUCs: list
        List of AUCs from performing leave-one-list-out cross validation n_permutations times

    """
    recalls = events.recalled
    permuted_recalls = np.array(recalls)
    AUCs = np.empty(shape=n_permutations, dtype=np.float)
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

        probs = run_lolo_xval(classifier, powers, events, recalls, **kwargs)
        AUCs[i] = roc_auc_score(permuted_recalls, probs)

    return AUCs


def run_lolo_xval(classifier, powers, events, recalls, **kwargs):
    """Perform a single iteration of leave-one-list-out cross validation

    Parameters
    ----------
    classifier: sklearn model object
    powers: mean powers to use as features
    events: set of events for the session
    recalls: vector of recall outcomes

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


def permuted_loso_AUCs(classifier, powers, events, n_permutations, **kwargs):
    recalls = events.recalled
    permuted_recalls = np.array(recalls)
    AUCs = np.empty(shape=n_permutations, dtype=np.float)

    for i in range(n_permutations):
        try:
            for sess in np.unique(events.session):
                sel = (events.session == sess)
                sess_permuted_recalls = permuted_recalls[sel]
                shuffle(sess_permuted_recalls)
                permuted_recalls[sel] = sess_permuted_recalls

            probs = run_loso_xval(classifier, powers, events, recalls, **kwargs)
            AUCs[i] = roc_auc_score(recalls, probs)
        except ValueError:
            AUCs[i] = np.nan

    return AUCs


def run_loso_xval(classifier, powers, events, recalls, **kwargs):
    """Perform leave-one-session-out cross validation.

    Parameters
    ----------
    classifier : LogisticRegression
        sklearn model object
    powers : np.array
        mean powers to use as features
    events : np.recarray
        set of events for the session
    recalls : np.ndarray
        vector of recall outcomes

    Returns
    -------
    probs: np.array
        Predicted probabilities for all lists

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
        outsample_pow_mat = powers[outsample_mask]

        outsample_probs = classifier.predict_proba(outsample_pow_mat)[:, 1]
        probs[outsample_mask] = outsample_probs

    return probs
