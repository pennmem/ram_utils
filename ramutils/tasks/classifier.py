from __future__ import division

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from classiflib import ClassifierContainer

from ramutils.classifier import ModelOutput
from ramutils.classifier.cross_validation import *
from ramutils.classifier.utils import get_sample_weights as \
    get_sample_weights_core
from ramutils.log import get_logger
from ramutils.tasks import task


try:
    from typing import Dict, Union, Tuple
except ImportError:
    pass

logger = get_logger()

__all__ = [
    'get_sample_weights',
    'train_classifier',
    'perform_cross_validation',
    'serialize_classifier',
]


@task()
def get_sample_weights(events, **kwargs):
    """Calculate class weights based on recall/non-recall in given events data.

    Parameters
    ----------
    events : np.recarray
    kwargs :

    Returns
    -------
    sample_weights : np.ndarray

    """
    sample_weights = get_sample_weights_core(events,
                                             **kwargs)
    return sample_weights


@task()
def train_classifier(pow_mat, events, sample_weights, **kwargs):
    """Train a classifier.

    Parameters
    ----------
    pow_mat : np.ndarray
    events : np.recarray
    sample_weights : np.ndarray
    params : ExperimentParameters

    Returns
    -------
    classifier : LogisticRegression
        Trained classifier

    """
    recalls = events.recalled
    classifier = LogisticRegression(C=kwargs['C'],
                                    penalty=kwargs['penalty_type'],
                                    solver=kwargs['solver'])
    classifier.fit(pow_mat, recalls, sample_weights)
    return classifier


@task()
def perform_cross_validation(classifier, pow_mat, events, n_permutations,
                             **kwargs):
    """Perform LOSO or LOLO cross validation on a classifier.

    Parameters
    ----------
    classifier : sklearn model object
    pow_mat : np.ndarray
    events : np.recarray
    n_permutations: int

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
        logger.info("Performing LOSO cross validation")
        perm_AUCs = permuted_loso_AUCs(classifier, pow_mat, events,
                                       n_permutations, **kwargs)
        probs = run_loso_xval(classifier, pow_mat, events, recalls)

        # Store model output statistics
        output = ModelOutput(true_labels=recalls, probs=probs)
        output.compute_metrics()
        xval['all'] = output

    # ... otherwise run leave-one-list-out cross validation
    else:
        logger.info("Performing LOLO cross validation")
        session = sessions[0]
        perm_AUCs = permuted_lolo_AUCs(classifier, pow_mat, events,
                                       n_permutations, **kwargs)
        probs = run_lolo_xval(classifier, pow_mat, events, recalls)

        # Store model output statistics
        output = ModelOutput(true_labels=recalls, probs=probs)
        output.compute_metrics()
        xval['all'] = xval[session] = output

    pvalue = np.sum(perm_AUCs >= xval['all'].auc) / len(perm_AUCs)
    logger.info("Permutation test p-value = %f", pvalue)

    recall_prob = classifier.predict_proba(pow_mat)[:, 1]
    insample_auc = roc_auc_score(recalls, recall_prob)
    logger.info("in-sample AUC = %f", insample_auc)
    return xval


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

    container = ClassifierContainer(
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
    return container
