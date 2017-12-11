from __future__ import division

import numpy as np
from classiflib import ClassifierContainer
from sklearn.metrics import roc_auc_score

from ramutils.classifier.cross_validation import permuted_loso_cross_validation, \
    perform_loso_cross_validation, logger, \
    permuted_lolo_cross_validation, perform_lolo_cross_validation
from ramutils.classifier.utils import train_classifier as train_classifier_core
from ramutils.classifier.utils import reload_classifier
from ramutils.classifier.weighting import \
    get_sample_weights as get_sample_weights_core
from ramutils.reports.summary import ClassifierSummary
from ramutils.events import extract_sessions, get_nonstim_events_mask
from ramutils.montage import compare_recorded_with_all_pairs
from ramutils.powers import reduce_powers
from ramutils.log import get_logger
from ramutils.tasks import task


logger = get_logger()

__all__ = [
    'get_sample_weights',
    'train_classifier',
    'perform_cross_validation',
    'serialize_classifier',
    'post_hoc_classifier_evaluation',
    'reload_used_classifiers'
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
    classifier_summary : ClassifierSummary
        Results of cross validation as a summary object

    """
    recalls = events.recalled

    classifier_summary = ClassifierSummary()
    sessions = extract_sessions(events)

    # Run leave-one-session-out cross validation when we have > 1 session,
    # otherwise leave-one-list-out
    if len(sessions) > 1:
        permuted_auc_values = permuted_loso_cross_validation(classifier,
                                                             pow_mat,
                                                             events,
                                                             n_permutations,
                                                             **kwargs)
        probs = perform_loso_cross_validation(classifier, pow_mat, events,
                                              recalls, **kwargs)
        classifier_summary.populate(recalls, probs, permuted_auc_values)

    else:
        logger.info("Performing LOLO cross validation")
        permuted_auc_values = permuted_lolo_cross_validation(classifier,
                                                             pow_mat,
                                                             events,
                                                             n_permutations,
                                                             **kwargs)
        probs = perform_lolo_cross_validation(classifier, pow_mat, events,
                                              recalls, **kwargs)

        # Store model output statistics
        classifier_summary.populate(recalls, probs, permuted_auc_values)

    logger.info("Permutation test p-value = %f", classifier_summary.pvalue)
    recall_prob = classifier.predict_proba(pow_mat)[:, 1]
    insample_auc = roc_auc_score(recalls, recall_prob)
    logger.info("in-sample AUC = %f", insample_auc)

    return classifier_summary


@task()
def reload_used_classifiers(subject, experiment, sessions, root):
    """ Reload the actual classifiers used in each session of an experiment

    Parameters
    ----------
    subject: str
    experiment: str
    sessions: list
    root: str

    Returns
    -------
    list
        List of ClassifierContainer objects of length n_sessions

    """
    used_classifiers = []
    for session in sessions:
        try:
            classifier = reload_classifier(subject, experiment, session, root)
        except Exception:
            logger.warning('Unable to load classifier for {}, '
                           '{}, session {}'.format(subject, experiment,
                                                   session))
            classifier = None
        used_classifiers.append(classifier)

    return used_classifiers


@task()
def post_hoc_classifier_evaluation(events, powers, all_pairs, classifiers,
                                   n_permutations, retrained_classifier,
                                   **kwargs):
    """ Evaluate a trained classifier

    Notes
    -----
    Different channels could be excluded based on results of artifact detection
    and stim parameters. Extract the used pairs from the serialized classifier
    that was used/retrained in order to correct assess the classifier. The
    default behavior is to use the retrained classifier for any sessions
    where the actual classifier was not found or is unusable

    """
    sessions = extract_sessions(events)
    if len(sessions) != len(classifiers):
        raise RuntimeError('The number of sessions for evaluation must match '
                           'the number of classifiers')

    if (any([classifier is None for classifier in classifiers]) and
            retrained_classifier is None):
        raise RuntimeError('A retrained classifier must be passed if any '
                           'sessions have missing classifiers')
    recalls = events.recalled

    non_stim_mask = get_nonstim_events_mask(events)

    classifier_summaries = []
    for i, session in enumerate(sessions):
        classifier_summary = ClassifierSummary()

        if classifiers[i] is None:
            classifier_container = retrained_classifier

        else:
            classifier_container = classifiers[i]

        classifier = classifier_container.classifier
        recorded_pairs = classifier_container.pairs

        used_mask = compare_recorded_with_all_pairs(all_pairs, recorded_pairs)

        session_mask = (events.session == session)
        session_events = events[(session_mask & non_stim_mask)]
        session_recalls = recalls[(session_mask & non_stim_mask)]

        session_powers = powers[(session_mask & non_stim_mask)]
        session_powers = reduce_powers(session_powers, used_mask,
                                       len(kwargs['freqs']))

        # Manually pass in the weighting scheme here, otherwise the cross
        # validation procedures will try to determine it for you
        permuted_auc_values = permuted_lolo_cross_validation(classifier,
                                                             session_powers,
                                                             session_events,
                                                             n_permutations,
                                                             scheme='EQUAL',
                                                             **kwargs)

        probs = perform_lolo_cross_validation(classifier,
                                              session_powers,
                                              session_events,
                                              session_recalls,
                                              scheme='EQUAL',
                                              **kwargs)

        # Store model output statistics
        classifier_summary.populate(session_recalls, probs, permuted_auc_values)
        classifier_summaries.append(classifier_summary)

    return classifier_summaries
