from random import shuffle

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from classiflib import ClassifierContainer

from ramutils.classifier import ModelOutput
from ramutils.classifier.cross_validation import *
from ramutils.classifier.utils import normalize_sessions, get_sample_weights
from ramutils.log import get_logger
from ramutils.tasks import task
from ramutils.utils import save_array_to_hdf5

try:
    from typing import List
except ImportError:
    pass

logger = get_logger()


@task()
def compute_auc(classifier, features, recalls, mask):
    """Compute the AUC score.

    :param classifier:
    :param np.ndarray features:
    :param np.recarray recalls:
    :param mask:
    :returns: computed AUC value
    :rtype: float

    """
    masked_recalls = recalls[mask]
    probs = classifier.predict_proba(features[mask])[:, 1]
    auc = roc_auc_score(masked_recalls, probs)
    return auc


@task()
def compute_permuted_lolo_aucs(sess, event_lists, recalls, params,
                               sample_weights=None):
    """Compute permuted leave-one-list-out AUC scores.

    :param int sess: session number
    :param ??? event_lists:
    :param ??? recalls:
    :param ExperimentParameters params:
    :param np.ndarray sample_weights: sample weights

    """
    n_perm = params.n_perm
    permuted_recalls = np.array(recalls)
    AUCs = np.empty(shape=n_perm, dtype=np.float)
    for i in range(n_perm):
        for lst in event_lists:
            sel = (event_lists == lst)
            list_permuted_recalls = permuted_recalls[sel]
            shuffle(list_permuted_recalls)
            permuted_recalls[sel] = list_permuted_recalls
        probs = run_lolo_xval(sess, event_lists, permuted_recalls, permuted=True, sample_weights=sample_weights)
        AUCs[i] = roc_auc_score(recalls, probs)
        logger.info('AUC = %f', AUCs[i])
    return AUCs


@task()
def compute_classifier(events, pow_mat, params, paths=None):
    """Compute the classifier.

    :param str subject:
    :param np.recarray events:
    :param np.ndarray pow_mat:
    :param ExperimentParameters params:
    :param FilePaths paths:
        used for accessing the ``dest`` parameter for storing storing debug
        data to

    """
    encoding_mask = events.type == 'WORD'
    pow_mat[encoding_mask] = normalize_sessions(pow_mat[encoding_mask], events[encoding_mask])
    pow_mat[~encoding_mask] = normalize_sessions(pow_mat[~encoding_mask], events[~encoding_mask])
    classifier = LogisticRegression(C=params.C,
                                    penalty=params.penalty_type,
                                    solver='liblinear')

    event_sessions = events.session

    recalls = events.recalled
    recalls[events.type == 'REC_WORD'] = 1
    recalls[events.type == 'REC_BASE'] = 0

    sample_weights = get_sample_weights(events, params.encoding_samples_weight)
    sessions = np.unique(event_sessions)

    if len(sessions > 1):
        raise RuntimeError("LOSO x-val not yet implemented")
    else:
        sess = sessions[0]
        event_lists = events.list
        perm_AUCs = permuted_lolo_AUCs(sess, event_lists, recalls, params.n_perm)
        # FIXME: run_lolo_xval()

    # FIXME: pvalue

    classifier.fit(pow_mat, recalls, sample_weights)
    recall_prob = classifier.predict_proba(pow_mat)[:, 1]
    insample_auc = roc_auc_score(recalls, recall_prob)
    logger.info("in-sample AUC = %f", insample_auc)

    try:
        save_array_to_hdf5(paths.dest + "-debug_data.h5", "model_output",
                           recall_prob, overwrite=True)
        save_array_to_hdf5(paths.dest + "-debug_data.h5", "model_weights",
                           classifier.coef_, overwrite=True)
    except Exception:
        logger.error('could not save debug data', exc_info=True)


@task(cache=False)
def serialize_classifier(classifier, pairs, features, events, sample_weights,
                         xval_output, subject):
    """Serialize the classifier.

    :param LogisticRegression classifier:
    :param np.ndarray features:
    :param np.recarray events:
    :param np.ndarray sample_weights:
    :param List[ModelOutput] xval_output:
    :param str subject:
    :rtype: ClassifierContainer

    """
    return ClassifierContainer(
        classifier=classifier,
        pairs=pairs,
        features=features,
        events=events,
        sample_weight=sample_weights,
        classifier_info={
            'auc': xval_output[-1].auc,
            'subject': subject
        }
    )
