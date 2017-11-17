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
    sample_weights = get_sample_weights_core(events,
                                             **kwargs)
    return sample_weights


@task()
def train_classifier(pow_mat, events, sample_weights, penalty_param,
                     penalty_type, solver):
    """Train a classifier.

    Parameters
    ----------
    pow_mat : np.ndarray
    events : np.recarray
    sample_weights : np.ndarray
    penalty_param: Float
        Penalty parameter to use
    penalty_type: str
        Type of penalty to use for regularized model (ex: L2)
    solver: str
        Solver to use when fitting the model (ex: liblinear)

    Returns
    -------
    classifier : LogisticRegression
        Trained classifier

    """
    recalls = events.recalled
    classifier = LogisticRegression(C=penalty_param,
                                    penalty=penalty_type,
                                    solver=solver)
    classifier.fit(pow_mat, recalls, sample_weights)
    return classifier


# FIXME: Remove the reliance on that weird ModelOutput object. Pull methods out
# as functions and track results with a dict if that is really necessary
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
        logger.info("Performing LOSO cross validation")
        perm_AUCs = permuted_loso_AUCs(classifier, pow_mat, events,
                                       n_permutations, **kwargs)
        probs = run_loso_xval(classifier, pow_mat, events, recalls, **kwargs)

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
        probs = run_lolo_xval(classifier, pow_mat, events, recalls, **kwargs)

        # Store model output statistics
        output = ModelOutput(true_labels=recalls, probs=probs)
        output.compute_metrics()
        xval['all'] = xval[session] = output

    pvalue = np.count_nonzero((perm_AUCs >= xval['all'].auc)) / len(perm_AUCs)
    logger.info("Permutation test p-value = %f", pvalue)

    recall_prob = classifier.predict_proba(pow_mat)[:, 1]
    insample_auc = roc_auc_score(recalls, recall_prob)
    logger.info("in-sample AUC = %f", insample_auc)
    return xval


# FIXME: update signature to be more in line with other tasks
@task(cache=False)
def serialize_classifier(classifier, pairs, features, events, sample_weights,
                         xval_output, subject):

    """ Serialize classifier into a container object

    Parameters
    ----------
    classifier: sklearn Estimator
        Model used during training
    pairs: array_like
        bipolar pairs used for training
    features: np.ndarray
        Normalized power matrix used as features to the classifier
    events: np.recarray
        Set of events used for training
    sample_weights: array_like
        Weights used for each of the event
    xval_output: ModelOutput
        Object used for calculating and storing cross-validation-related metrics
    subject: str
        Subject identifier

    Returns
    -------
    ClassififerContainer
        Object representing all meta-data associated with training a classifier

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
