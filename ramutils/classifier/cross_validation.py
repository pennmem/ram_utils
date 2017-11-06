""" Collection of cross-validation functions """
import numpy as np

from random import shuffle
from sklearn.metrics import roc_auc_score


def permuted_loso_AUCs(classifier, powers, events, n_permutations):
    """ Perform leave-one-session out cross validation """

    permuted_recalls = np.array(events.recalled)
    AUCs = np.empty(shape=n_permutations, dtype=np.float)
    for i in range(n_permutations):
        sessions = np.unique(events.sessions)
        for session in sessions:
            sel = (events.session == session)
            sess_permuted_recalls = permuted_recalls[sel]
            shuffle(sess_permuted_recalls)
            permuted_recalls[sel] = sess_permuted_recalls
        probs = classifier.predict_proba(powers)[:,0]
        AUCs[i] = roc_auc_score(permuted_recalls, probs)
    return AUCs


def permuted_lolo_AUCs(classifier, powers, events, n_permutations):
    """ Perform leave one list out cross-validation """
    recalls = events.recalled
    permuted_recalls = np.random.randint(2,size=recalls.shape)
    permuted_recalls = np.array(recalls)
    AUCs = np.empty(shape=n_permutations, dtype=np.float)
    sessions = np.unique(events.session)
    for i in range(n_permutations):
        for sess in sessions:
            sess_lists = np.unique(events[events.session==sess].list)
            for lst in sess_lists:
                sel = (events.session==sess) & (events.list==lst)
                list_permuted_recalls = permuted_recalls[sel]
                shuffle(list_permuted_recalls)
                permuted_recalls[sel] = list_permuted_recalls
        probs = classifier.predict_proba(powers)[:,0]
        AUCs[i] = roc_auc_score(permuted_recalls, probs)
    return AUCs