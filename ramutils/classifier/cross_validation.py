""" Collection of cross-validation functions """
import numpy as np

from random import shuffle
from sklearn.metrics import roc_auc_score


def permuted_lolo_AUCs(classifier, powers, events, n_permutations):
    """ Permuted leave one list out cross validation

    Parameters
    ----------
    classifier: sklearn model object, usually logistic regression classifier
    powers: power matrix
    events: recarray
    n_permutations: number of permutation trials

    Returns
    -------
    AUCs: list
        List of AUCs from performing leave-one-list-out cross validation n_permutations times

    """
    recalls = events.recalled
    permuted_recalls = np.random.randint(2,size=recalls.shape)
    permuted_recalls = np.array(recalls)
    AUCs = np.empty(shape=n_permutations, dtype=np.float)
    sessions = np.unique(events.session)
    for i in range(n_permutations):
        for sess in sessions:
            sess_lists = np.unique(events[events.session==sess].list)
            for lst in sess_lists:
                # Permute recall outcome within each session/list
                sel = (events.session==sess) & (events.list==lst)
                list_permuted_recalls = permuted_recalls[sel]
                shuffle(list_permuted_recalls)
                permuted_recalls[sel] = list_permuted_recalls

        probs = run_lolo_xval(classifier, powers, events, recalls)
        AUCs[i] = roc_auc_score(permuted_recalls, probs)

    return AUCs

def run_lolo_xval(classifier, powers, events, recalls):
    """ Perform single iteration of leave-one-list-out cross validation

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
        classifier.fit(insample_pow_mat, insample_recalls)
        outsample_mask = ~insample_mask
        outsample_pow_mat = powers[outsample_mask]
        probs[outsample_mask] = classifier.predict_proba(outsample_pow_mat)[:, 1]

    return probs
