from __future__ import division

import numpy as np
from classiflib import ClassifierContainer

from ramutils.classifier.utils import train_classifier as train_classifier_core
from ramutils.classifier.cross_validation import perform_cross_validation as \
    perform_cross_validation_core
from ramutils.classifier.weighting import \
    get_sample_weights as get_sample_weights_core
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


@task()
def perform_cross_validation(classifier, pow_mat, events, n_permutations,
                             **kwargs):
    xval = perform_cross_validation_core(classifier, pow_mat, events,
                                         n_permutations, **kwargs)
    return xval


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
            'auc': xval_output['all'],
            'subject': subject
        }
    )
    return container
