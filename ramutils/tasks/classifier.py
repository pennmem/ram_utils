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
from ramutils.events import extract_sessions, get_nonstim_events_mask, \
    get_encoding_mask, extract_event_metadata
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
                             tag='classifier', **kwargs):
    """Perform LOSO or LOLO cross validation on a classifier.

    Parameters
    ----------
    classifier : sklearn model object
    pow_mat : np.ndarray
    events : np.recarray
    n_permutations: int
    tag: str
        Tag to assign the resulting classifier summary (default:
        ``'classifier'``)
    kwargs: dict
        Extra keyword arguments that are passed to get_sample_weights. See
        that function for more details

    Returns
    -------
    classifier_summary : ClassifierSummary
        Results of cross validation as a summary object

    """
    recalls = events.recalled
    encoding_event_mask = get_encoding_mask(events)
    encoding_recalls = recalls[encoding_event_mask]

    classifier_summary = ClassifierSummary()
    subject, experiment, sessions = extract_event_metadata(events)

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
        classifier_summary.populate(subject, experiment,
                                    sessions, encoding_recalls, probs,
                                    permuted_auc_values, tag=tag)

    else:
        logger.info("Performing LOLO cross validation")
        permuted_auc_values = permuted_lolo_cross_validation(classifier,
                                                             pow_mat,
                                                             events,
                                                             n_permutations,
                                                             **kwargs)
        probs = perform_lolo_cross_validation(classifier, pow_mat, events,
                                              recalls, **kwargs)

        classifier_summary.populate(subject, experiment,
                                    sessions, encoding_recalls, probs,
                                    permuted_auc_values, tag=tag)

    logger.info("Permutation test p-value = %f", classifier_summary.pvalue)
    recall_prob = classifier.predict_proba(pow_mat)[:, 1]
    insample_auc = roc_auc_score(recalls, recall_prob)
    logger.info("in-sample AUC = %f", insample_auc)

    return classifier_summary


@task()
def reload_used_classifiers(subject, experiment, events, root):
    """ Reload the actual classifiers used in each session of an experiment

    Parameters
    ----------
    subject: str
        Subject identifier
    experiment: str
        Name of the experiment
    sessions: list
        List of sessions to try reloading a classifier
    root: str
        Base path of where to find RHINO files

    Returns
    -------
    list
        List of ClassifierContainer objects of length n_sessions

    Notes
    -----
    If a classifier is not found or is unable to be reloaded (legacy storage
    format, or other issues), then the list of ClassifierContainer objects
    will have None as the entry for that session.

    """
    used_classifiers = []
    sessions = extract_sessions(events)
    for session in sessions:
        classifier = reload_classifier(subject, experiment, session, root)
        used_classifiers.append(classifier)

    return used_classifiers


@task()
def post_hoc_classifier_evaluation(events, powers, post_stim_events,
                                   post_stim_powers, all_pairs, classifiers,
                                   n_permutations, retrained_classifier,
                                   **kwargs):
    """ Evaluate a trained classifier

    Parameters
    ----------
    events: np.recarray
        Task events associated with the stim sessesion to be evaluated
    powers: np.ndarray
        Normalized mean powers
    post_stim_events: np.recarray
        Post-stimulation events associated with the stim sessesion to be
        evaluated
    post_stim_powers: np.ndarray
        Normalized mean powers for post_stim period events
    all_pairs: OrderedDict
        All pairs based on recorded electrodes combine from config file
    classifiers: List
        List of classifiers corresponding to each session
    n_permutations: int
        Number of permutations to use for cross validation
    retrained_classifier: classiflib.container.ClassifierContainer
        classifier container object based on a retrained classifier
    kwargs

    Returns
    -------
    dict
        A dictionary of summary objects that are needed in subsequent parts
        of the processing pipeline. The dictionary will be in the following
        format:
            {'cross_session_summary': MultiSessionClassifierSummary,
             'session_summaries': List of ClassifierSummary objects,
             'session_summaries_stim_table': List of ClassifierSummary
             objects built using all encoding events,
             'post_stim_predicted_probs': Classifier output during post stim
             period
            }

    Notes
    -----
    Different channels could be excluded based on results of artifact detection
    and stim parameters. Extract the used pairs from the serialized classifier
    that was used/retrained in order to correctly assess the classifier. The
    default behavior is to use the retrained classifier for any sessions
    where the actual classifier was not found or was unable to be loaded.
    Legacy-formatted classifiers are not supported for re-loading. In cases
    where a stim session was restarted, the default behavior is to use the
    original classifier (i.e. the classifier before artifact detection) rather
    than trying to guess which classifier to load.

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
    post_stim_recalls = post_stim_events.recalled

    # Masks for encoding events
    encoding_mask = get_encoding_mask(events)

    # This takes care of sub-setting events to encoding non-stim events
    non_stim_mask = get_nonstim_events_mask(events)
    non_stim_recalls = recalls[non_stim_mask]

    classifier_summaries = []
    encoding_classifier_summaries = []
    predicted_probs = []
    post_stim_predicted_probs = []
    for i, session in enumerate(sessions):
        classifier_summary = ClassifierSummary()

        # Be sure to work with a copy of the classifier object because it will
        # be re-fit as part of the lolo cross validation and if you pass
        # a reference, the AUCs will be wacky
        if classifiers[i] is None:
            classifier_container = retrained_classifier
            logger.info("Using the retrained classifier for session {}".format(session))

        else:
            classifier_container = classifiers[i]
            logger.info("Using actual classifier for session {}".format(session))

        classifier = classifier_container.classifier
        recorded_pairs = classifier_container.pairs

        used_mask = compare_recorded_with_all_pairs(all_pairs, recorded_pairs)

        session_mask = (events.session == session)
        session_events = events[(session_mask & non_stim_mask)]
        session_recalls = recalls[session_mask & non_stim_mask]

        session_powers = powers[(session_mask & non_stim_mask)]
        reduced_session_powers = reduce_powers(session_powers,
                                               used_mask,
                                               len(kwargs['freqs']))

        # Manually pass in the weighting scheme here, otherwise the cross
        # validation procedures will try to determine it for you
        permuted_auc_values = permuted_lolo_cross_validation(classifier,
                                                             reduced_session_powers,
                                                             session_events,
                                                             n_permutations,
                                                             scheme='EQUAL',
                                                             **kwargs)

        session_probs = classifier.predict_proba(reduced_session_powers)[:, 1]
        predicted_probs.append(session_probs)

        # Calculate classifier outputs during the post stim period. This is
        # used downstream in the reports to see if stimulation affected the
        # biomarker
        post_stim_session_mask = (post_stim_events.session == session)
        post_stim_session_powers = post_stim_powers[post_stim_session_mask]
        post_stim_reduced_session_powers = reduce_powers(post_stim_session_powers,
                                                         used_mask,
                                                         len(kwargs['freqs']))
        post_stim_probs = classifier.predict_proba(
            post_stim_reduced_session_powers)[:, 1]
        post_stim_predicted_probs.append(post_stim_probs)

        subject, experiment, sessions = extract_event_metadata(session_events)
        classifier_summary.populate(subject, experiment, sessions,
                                    session_recalls, session_probs,
                                    permuted_auc_values, tag=session)
        classifier_summaries.append(classifier_summary)
        logger.info('AUC for session {}: {}'.format(session,
                                                    classifier_summary.auc))

        # Get a classifier summary for all encoding events
        session_encoding_powers = powers[(session_mask & encoding_mask)]
        reduced_session_encoding_powers = reduce_powers(
            session_encoding_powers, used_mask, len(kwargs['freqs']))

        session_encoding_probs = classifier.predict_proba(
            reduced_session_encoding_powers)[:, 1]

        session_encoding_recalls = recalls[session_mask & encoding_mask]

        encoding_classifier_summary = ClassifierSummary()
        encoding_classifier_summary.populate(subject, experiment, sessions,
                                             session_encoding_recalls,
                                             session_encoding_probs,
                                             None)
        encoding_classifier_summaries.append(encoding_classifier_summary)

    # Combine session-specific predicted probabilities into 1D array
    all_predicted_probs = np.array(predicted_probs).flatten()

    if len(sessions) > 1:
        permuted_auc_values = permuted_loso_cross_validation(
          retrained_classifier.classifier, powers, events, n_permutations,
            scheme='EQUAL', **kwargs)

    subject, experiment, sessions = extract_event_metadata(events)
    cross_session_summary = ClassifierSummary()
    cross_session_summary.populate(subject, experiment, sessions,
                                   non_stim_recalls, all_predicted_probs,
                                   permuted_auc_values, tag='Combined Sessions')
    classifier_summaries.append(cross_session_summary)
    logger.info("Combined AUC: {}".format(cross_session_summary.auc))

    result_dict = {
        'cross_session_summary': cross_session_summary,
        'session_summaries': classifier_summaries,
        'session_summaries_stim_table': encoding_classifier_summaries,
        'post_stim_predicted_probs': post_stim_predicted_probs
    }

    return result_dict
