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


def permuted_lolo_AUCs(classifier, powers, events, n_permutations):
    """Permuted leave-one-list-out cross validation

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

        probs = run_lolo_xval(classifier, powers, events, recalls)
        AUCs[i] = roc_auc_score(permuted_recalls, probs)

    return AUCs


def run_lolo_xval(classifier, powers, events, recalls):
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
        classifier.fit(insample_pow_mat, insample_recalls)
        outsample_mask = ~insample_mask
        outsample_pow_mat = powers[outsample_mask]
        probs[outsample_mask] = classifier.predict_proba(outsample_pow_mat)[:, 1]

    return probs


def permuted_loso_AUCs(classifier, powers, events, n_permutations):
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

            probs = run_loso_xval(classifier, powers, events, recalls)
            AUCs[i] = roc_auc_score(recalls, probs)
        except ValueError:
            AUCs[i] = np.nan

    return AUCs


def run_loso_xval(classifier, powers, events, recalls, encoding_sample_weight):
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
    encoding_sample_weight : int
        Scalar to determine how much more to weight encoding samples

    Returns
    -------
    probs: np.array
        Predicted probabilities for all lists

    """
    probs = np.empty_like(recalls, dtype=np.float)
    # encoding_probs = np.empty_like(events[events.type == 'WORD'], dtype=np.float)

    sessions = np.unique(events.sesssion)

    # auc_encoding = np.empty(sessions.shape[0], dtype=np.float)
    # auc_retrieval = np.empty(sessions.shape[0], dtype=np.float)
    # auc_both = np.empty(sessions.shape[0], dtype=np.float)

    for sess_idx, sess in enumerate(sessions):
        # training data
        insample_mask = (events.session != sess)
        insample_pow_mat = powers[insample_mask]
        insample_recalls = recalls[insample_mask]
        insample_samples_weights = get_sample_weights(events[events.session != sess],
                                                      encoding_sample_weight)

        classifier.fit(insample_pow_mat, insample_recalls,
                       insample_samples_weights)

        # testing data
        outsample_mask = ~insample_mask
        outsample_pow_mat = powers[outsample_mask]
        # outsample_recalls = recalls[outsample_mask]

        outsample_probs = classifier.predict_proba(outsample_pow_mat)[:, 1]
        probs[outsample_mask] = outsample_probs

        # FIXME: this should be computed elsewhere
        # outsample_encoding_mask = (events.session == sess) & (events.type == 'WORD')
        # outsample_retrieval_mask = (events.session == sess) & ((events.type == 'REC_BASE') | (events.type == 'REC_WORD'))
        # outsample_both_mask = (events.session == sess)
        #
        # auc_encoding[sess_idx] = self.get_auc(
        #     classifier=classifier, features=powers, recalls=recalls, mask=outsample_encoding_mask)
        # encoding_probs[events[events.type == 'WORD'].session == sess] = classifier.predict_proba(powers[outsample_encoding_mask])[:, 1]
        #
        # auc_retrieval[sess_idx] = self.get_auc(
        #     classifier=classifier, features=powers, recalls=recalls, mask=outsample_retrieval_mask)
        #
        # auc_both[sess_idx] = self.get_auc(
        #     classifier=classifier, features=powers, recalls=recalls, mask=outsample_both_mask)

        return probs
