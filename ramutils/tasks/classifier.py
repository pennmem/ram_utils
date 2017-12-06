from __future__ import division

import numpy as np
from classiflib import ClassifierContainer
from sklearn.metrics import roc_auc_score

from ramutils.classifier.cross_validation import permuted_loso_cross_validation, perform_loso_cross_validation, logger, \
    permuted_lolo_cross_validation, perform_lolo_cross_validation
from ramutils.classifier.utils import train_classifier as train_classifier_core
from ramutils.classifier.weighting import \
    get_sample_weights as get_sample_weights_core
from ramutils.reports.summary import ClassifierSummary
from ramutils.log import get_logger
from ramutils.tasks import task


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
    classifier = train_classifier_core(pow_mat, events, sample_weights,
                                       penalty_param, penalty_type, solver)
    return classifier


@task(cache=False)
def serialize_classifier(classifier, pairs, features, events, sample_weights,
                         classifier_summary, subject):

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
    classifier_summary: ClassifierSummary
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
            'auc': classifier_summary.auc,
            'subject': subject
        }
    )
    return container


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

    classifier_summary = ClassifierSummary()
    sessions = np.unique(events.session)

    # Run leave-one-session-out cross validation when we have > 1 session, otherwise leave-one-list-out
    if len(sessions) > 1:
        permuted_auc_values = permuted_loso_cross_validation(classifier, pow_mat, events,
                                                             n_permutations, **kwargs)
        probs = perform_loso_cross_validation(classifier, pow_mat, events, recalls, **kwargs)
        classifier_summary.populate(recalls, probs, permuted_auc_values)

    else:
        logger.info("Performing LOLO cross validation")
        permuted_auc_values = permuted_lolo_cross_validation(classifier, pow_mat, events, n_permutations, **kwargs)
        probs = perform_lolo_cross_validation(classifier, pow_mat, events, recalls, **kwargs)

        # Store model output statistics
        classifier_summary.populate(recalls, probs, permuted_auc_values)

    logger.info("Permutation test p-value = %f", classifier_summary.pvalue)
    recall_prob = classifier.predict_proba(pow_mat)[:, 1]
    insample_auc = roc_auc_score(recalls, recall_prob)
    logger.info("in-sample AUC = %f", insample_auc)

    return classifier_summary
