""" Pipeline for create aggregated reports """

from ramutils.tasks import *
from ramutils.utils import extract_experiment_series
from ramutils.events import find_subjects
from ramutils.log import get_logger

logger = get_logger()


def make_aggregated_report(subjects=None, experiments=None, sessions=None, paths=None):
    """ Create an aggregated stim session report """
    # Case 1: Given subject, then experiment must be given. Not sure how to
    # arbitrarily combine sessions for single subject
    # Case 2: Given experiment, combine all subject/sessions who completed that
    # experiment. If joint, then do both FR and catFR

    all_classifier_evaluation_results, all_session_summaries, all_math_summaries, target_selection_table = [], [], [], None

    # All subjects completing a given experiment(s)
    if subjects is None and experiments is not None:
        subject = 'combined'
        experiment = "_".join(experiments)
        for experiment in experiments:
            exp_subjects = find_subjects(experiment, paths.root)
            for subject in exp_subjects:
                target_selection_table, classifier_evaluation_results, \
                session_summaries, math_summaries, hmm_results = \
                    load_existing_results(subject, experiment, sessions, True,
                                          paths.data_db, rootdir=paths.root).compute()
                if all([val is None for val in [target_selection_table,
                                                classifier_evaluation_results,
                                                session_summaries, math_summaries]]):
                    logger.warning('Unable to find underlying data for {}, experiment {}'.format(subject, experiment))
                    continue

                all_classifier_evaluation_results.extend(classifier_evaluation_results)
                all_session_summaries.extend(session_summaries)
                all_math_summaries.extend(math_summaries)

    # Set of subject(s) completing a specific set of experiment(s)
    elif subjects is not None and experiments is not None and sessions is None:
        subject = '_'.join([subjects])
        experiment = "_".join([experiments])
        for subject in subjects:
            for experiment in experiments:
                target_selection_table, classifier_evaluation_results, \
                session_summaries, math_summaries, hmm_results = \
                    load_existing_results(subject, experiment, sessions, True,
                                          paths.data_db, rootdir=paths.root).compute()

                if all([val is None for val in [target_selection_table,
                                                classifier_evaluation_results,
                                                session_summaries, math_summaries]]):
                    logger.warning('Unable to find underlying data for {}, experiment {}'.format(subject, experiment))

                all_classifier_evaluation_results.extend(classifier_evaluation_results)
                all_session_summaries.extend(session_summaries)
                all_math_summaries.extend(math_summaries)

    # Single subject/experiment and a subset of sessions
    elif subjects is not None and experiments is not None and sessions is not None:
        if len(subjects) > 1 or len(experiments) > 1:
            raise RuntimeError("When specifying sessions, only single subject and experiment are allowed")

        subject = subjects[0]
        experiment = experiments[0]

        target_selection_table, classifier_evaluation_results, \
        session_summaries, math_summaries, hmm_results = \
            load_existing_results(subjects, experiments, sessions, True,
                                  paths.data_db, rootdir=paths.root).compute()

        if all([val is None for val in [target_selection_table,
                                        classifier_evaluation_results,
                                        session_summaries, math_summaries]]):
            logger.warning('Unable to find underlying data for {}, experiment {}'.format(subject, experiment))

            all_classifier_evaluation_results.extend(classifier_evaluation_results)
            all_session_summaries.extend(session_summaries)
            all_math_summaries.extend(math_summaries)

    else:
        raise RuntimeError('The requested type of aggregation is not currently supported')

    # Fit new hmm model on the joint data and save resulting images
    hmm_results = estimate_effects_of_stim(subject, 'FR3', all_session_summaries)
    output = save_all_output(subject, experiment, [], [], [], paths.data_db,
                             behavioral_results=hmm_results)

    report = build_static_report(subject, experiment, all_session_summaries,
                                 all_math_summaries, target_selection_table,
                                 all_classifier_evaluation_results,
                                 paths.dest, hmm_results=output,
                                 save=True)

    return report.compute()
