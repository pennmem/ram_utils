""" Pipeline for create aggregated reports """

from ramutils.tasks import *
from ramutils.utils import extract_experiment_series


def make_aggregated_report(subject, experiment, paths, joint=True, sessions=None):
    """ Create an aggregated stim session report """
    experiments = [experiment]
    if joint:
        series_num = extract_experiment_series(experiment)
        experiments = ['FR' + series_num, 'catFR' + series_num]

    all_classifier_evaluation_results, all_session_summaries, all_math_summaries = [], [], []
    for experiment in experiments:

        target_selection_table, classifier_evaluation_results, \
        session_summaries, math_summaries, hmm_results = \
            load_existing_results(subject, experiment, sessions, True,
                                  paths.data_db, rootdir=paths.root).compute()

        # Check if only None values were returned. Processing will continue
        # undeterred
        if all([val is None for val in [target_selection_table,
                                        classifier_evaluation_results,
                                        session_summaries, math_summaries]]):
            raise RuntimeError('Unable to find all necessary data for requested '
                               'aggregate report. Run individual session reports '
                               'first')
        all_classifier_evaluation_results.extend(classifier_evaluation_results)
        all_session_summaries.extend(session_summaries)
        all_math_summaries.extend(math_summaries)

    # Fit new hmm model on the joint data and save resulting images
    hmm_results = estimate_effects_of_stim(subject, experiment, all_session_summaries)
    output = save_all_output(subject, experiment, [], [], [], paths.data_db,
                             behavioral_results=hmm_results)

    report = build_static_report(subject, experiment, all_session_summaries,
                                 all_math_summaries, target_selection_table,
                                 all_classifier_evaluation_results,
                                 paths.dest, hmm_results=output,
                                 save=True)

    return report.compute()
