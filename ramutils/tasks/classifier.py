from __future__ import division

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
    from typing import Dict, Union, Tuple
except ImportError:
    pass

logger = get_logger()


@task(nout=3)
def compute_classifier(events, pow_mat, params, paths=None):
    """Compute the classifier.

    :param str subject:
    :param np.recarray events:
    :param np.ndarray pow_mat:
    :param ExperimentParameters params:
    :param FilePaths paths:
        used for accessing the ``dest`` parameter for storing storing debug
        data to (if not given, no debug data will be written)
    :returns: trained classifier and cross-validation output
    :rtype: Tuple[LogisticRegression, Dict[Union[str, int], ModelOutput], np.ndarray]

    """
    encoding_mask = events.type == 'WORD'

    # z-score powers within sessions
    pow_mat[encoding_mask] = normalize_sessions(pow_mat[encoding_mask], events[encoding_mask])
    pow_mat[~encoding_mask] = normalize_sessions(pow_mat[~encoding_mask], events[~encoding_mask])

    classifier = LogisticRegression(C=params.C,
                                    penalty=params.penalty_type,
                                    solver=params.solver)

    # Stores cross validation output. Keys are sessions or 'all' for all session
    # cross validation.
    xval = {}  # type: Dict[Union[str, int], ModelOutput]

    event_sessions = events.session

    recalls = events.recalled
    recalls[events.type == 'REC_WORD'] = 1
    recalls[events.type == 'REC_BASE'] = 0

    # FIXME: make sample_weights an input
    sample_weights = get_sample_weights(events, params.encoding_samples_weight)
    sessions = np.unique(event_sessions)

    # Run leave-one-session-out cross validation when we have > 1 session
    if len(sessions > 1):
        logger.info("Performing LOSO cross validation")
        perm_AUCs = permuted_loso_AUCs(classifier, pow_mat, events, params.n_perm)
        probs = run_loso_xval(classifier, pow_mat, events, recalls, params.encoding_samples_weight)

        # Store model output statistics
        output = ModelOutput(true_labels=recalls, probs=probs)
        output.compute_metrics()
        xval['all'] = output

    # ... otherwise run leave-one-list-out cross validation
    else:
        logger.info("Performing LOLO cross validation")
        session = sessions[0]
        perm_AUCs = permuted_lolo_AUCs(session, pow_mat, events, params.n_perm)
        probs = run_lolo_xval(classifier, pow_mat, events, recalls)

        # Store model output statistics
        output = ModelOutput(true_labels=recalls, probs=probs)
        output.compute_metrics()
        xval['all'] = xval[session] = output

    pvalue = np.sum(perm_AUCs >= xval['all'].auc) / len(perm_AUCs)
    logger.info("Permutation test p-value = %f", pvalue)

    classifier.fit(pow_mat, recalls, sample_weights)
    recall_prob = classifier.predict_proba(pow_mat)[:, 1]
    insample_auc = roc_auc_score(recalls, recall_prob)
    logger.info("in-sample AUC = %f", insample_auc)

    if paths is not None:
        try:
            save_array_to_hdf5(paths.dest + "-debug_data.h5", "model_output",
                               recall_prob, overwrite=True)
            save_array_to_hdf5(paths.dest + "-debug_data.h5", "model_weights",
                               classifier.coef_, overwrite=True)
        except Exception:
            logger.error('could not save debug data', exc_info=True)
    else:
        logger.warning("No debug data written since paths not given")

    return classifier, xval, sample_weights


# FIXME: update signature to be more in line with other tasks
@task(cache=False)
def serialize_classifier(classifier, pairs, features, events, sample_weights,
                         xval_output, subject):
    """Serialize the classifier.

    :param LogisticRegression classifier:
    :param np.ndarray features:
    :param np.recarray events:
    :param np.ndarray sample_weights:
    :param Dict[ModelOutput] xval_output:
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
            'auc': xval_output['all'].auc,
            'subject': subject
        }
    )
