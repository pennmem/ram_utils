"""Pipeline for creating reports."""

import pandas as pd

from ramutils.tasks import *


def make_report(subject, experiment, paths, joint_report=False,
                retrain=False, stim_params=None, exp_params=None,
                sessions=None, vispath=None, rerun=False):
    """Run a report.

    Parameters
    ----------
    subject : str
        Subject ID
    experiment : str
        Experiment to generate report for
    paths : FilePaths
    joint_report: Bool
        If True, catFR/FR sessions will be combined in the report
    retrain: Bool
        If True, retrain classifier rather than trying to load from disk
    stim_params : List[StimParameters]
        Stimulation parameters (empty list for non-stim experiments).
    exp_params : ExperimentParameters
        When given, overrides the inferred default parameters to use for an
        experiment.
    sessions : list or None
        For reports that span sessions, sessions to read data from.
        When not given, all available sessions are used for reports.
    vispath : str
        Filename for task graph visualization.
    rerun: bool
        If True, do not attempt to load data from long-term storage. If any
        necessary data is not found, everything will be rerun

    Returns
    -------
    report_path : str
        Path to generated report.

    Notes
    -----
    Eventually this will return an object that summarizes all output of the
    report rather than the report itself.

    """
    kwargs = exp_params.to_dict()

    # Lower case 'c' is expected for reading events. The reader should probably
    # just be case insensitive
    if 'Cat' in experiment:
        experiment = experiment.replace('Cat', 'cat')

    stim_report = is_stim_experiment(experiment).compute()

    # PS runs so quickly and has a much more nested event structure, so it is
    # better to always just re-run
    if not rerun and 'PS' not in experiment:
        target_selection_table, classifier_evaluation_results, \
        session_summaries, math_summaries = load_existing_results(subject,
                                                                  experiment,
                                                                  sessions,
                                                                  stim_report,
                                                                  paths.data_db,
                                                                  rootdir=paths.root).compute()

        # Check if only None values were returned. Processing will continue
        # undeterred
        if all([val is None for val in [target_selection_table,
                                        classifier_evaluation_results,
                                        session_summaries, math_summaries]]):
            pass
        else:
            report = build_static_report(subject, experiment, session_summaries,
                                         math_summaries, target_selection_table,
                                         classifier_evaluation_results, paths.dest)
            return report.compute()

    # TODO: allow using different localization, montage numbers
    ec_pairs = get_pairs(subject, experiment, paths)
    excluded_pairs = reduce_pairs(ec_pairs, stim_params, True)
    final_pairs = generate_pairs_for_classifier(ec_pairs, excluded_pairs)
    used_pair_mask = get_used_pair_mask(ec_pairs, excluded_pairs)
    pairs_metadata_table = generate_montage_metadata_table(subject,
                                                           experiment,
                                                           ec_pairs,
                                                           root=paths.root)

    if 'PS' not in experiment:
        # all_events are used for producing math summaries. Task events are only
        # used by the stim reports. Non-stim reports create a different set of
        # events. Stim params are used in building the stim session
        # summaries. PS experiments do not have an all_events.json file,
        # which is what these subsets are built from, so PS has it's own
        # build_*_data function
        all_events, task_events, stim_data = build_test_data(subject,
                                                             experiment,
                                                             paths,
                                                             joint_report,
                                                             sessions=sessions,
                                                             **kwargs).compute()

    target_selection_table = pd.DataFrame(columns=['type', 'contact0',
                                                   'contact1', 'label',
                                                   'p_value', 'tstat',
                                                   'mni_x', 'mni_y', 'mni_z',
                                                   'controllability'])
    repetition_ratio_dict = {}
    if not stim_report:
        if joint_report or (experiment == 'catFR1'):
            repetition_ratio_dict = get_repetition_ratio_dict(paths)

        # This logic is very similar to what is done in config generation except
        # that events are not combined by default
        if not joint_report:
            kwargs['combine_events'] = False

        kwargs['encoding_only'] = False
        all_task_events = build_training_data(subject,
                                              experiment,
                                              paths,
                                              sessions=sessions,
                                              **kwargs)

        powers, final_task_events = compute_normalized_powers(
            all_task_events, bipolar_pairs=ec_pairs, **kwargs)

        reduced_powers = reduce_powers(powers, used_pair_mask,
                                       len(kwargs['freqs']))

        sample_weights = get_sample_weights(final_task_events, **kwargs)

        classifier = train_classifier(reduced_powers,
                                      final_task_events,
                                      sample_weights,
                                      kwargs['C'],
                                      kwargs['penalty_type'],
                                      kwargs['solver'])

        joint_classifier_summary = perform_cross_validation(classifier,
                                                            reduced_powers,
                                                            final_task_events,
                                                            kwargs['n_perm'],
                                                            tag='Joint',
                                                            **kwargs)

        # Subset events, powers, etc to get encoding-only classifier summary
        kwargs['scheme'] = 'EQUAL'
        encoding_only_mask = get_word_event_mask(final_task_events, True)
        final_encoding_task_events = subset_events(final_task_events,
                                                   encoding_only_mask)
        encoding_reduced_powers = subset_powers(powers, encoding_only_mask)

        encoding_sample_weights = get_sample_weights(final_encoding_task_events,
                                                     **kwargs)

        encoding_classifier = train_classifier(encoding_reduced_powers,
                                               final_encoding_task_events,
                                               encoding_sample_weights,
                                               kwargs['C'],
                                               kwargs['penalty_type'],
                                               kwargs['solver'])

        encoding_classifier_summary = perform_cross_validation(
            encoding_classifier, encoding_reduced_powers,
            final_encoding_task_events, kwargs['n_perm'],
            tag='Encoding', **kwargs)

        # TODO: Add distanced-based ranking of electrodes to prior stim results
        target_selection_table = create_target_selection_table(
            pairs_metadata_table, powers, final_task_events, kwargs['freqs'],
            hfa_cutoff=kwargs['hfa_cutoff'], root=paths.root)

        session_summaries = summarize_nonstim_sessions(all_events,
                                                       final_task_events,
                                                       joint=joint_report,
                                                       repetition_ratio_dict=repetition_ratio_dict)
        math_summaries = summarize_math(all_events, joint=joint_report)

    if stim_report and 'PS' not in experiment:
        # We need post stim period events/powers
        post_stim_mask = get_post_stim_events_mask(all_events)
        post_stim_events = subset_events(all_events, post_stim_mask)
        post_stim_powers, final_post_stim_events = compute_normalized_powers(
           post_stim_events, bipolar_pairs=ec_pairs, **kwargs)

        powers, final_task_events = compute_normalized_powers(
            task_events, bipolar_pairs=ec_pairs, **kwargs)

        used_classifiers = reload_used_classifiers(subject,
                                                   experiment,
                                                   final_task_events,
                                                   paths.root).compute()
        # Retraining occurs on-demand or if any session-specific classifiers
        # failed to load
        retrained_classifier = None
        if retrain or any([classifier is None for classifier in used_classifiers]):
            # Intentionally not passing 'sessions' so that training takes place
            # on the full set of record only events
            training_events = build_training_data(subject, experiment, paths,
                                                  **kwargs)

            training_powers, final_training_events = compute_normalized_powers(
                training_events, bipolar_pairs=ec_pairs, **kwargs)

            training_reduced_powers = reduce_powers(training_powers,
                                                    used_pair_mask,
                                                    len(kwargs['freqs']))

            sample_weights = get_sample_weights(final_training_events, **kwargs)

            retrained_classifier = train_classifier(training_reduced_powers,
                                                    final_training_events,
                                                    sample_weights,
                                                    kwargs['C'],
                                                    kwargs['penalty_type'],
                                                    kwargs['solver'])

            training_classifier_summaries = perform_cross_validation(
                retrained_classifier, training_reduced_powers,
                final_training_events, kwargs['n_perm'],
                tag='Original Classifier', **kwargs)

            retrained_classifier = serialize_classifier(retrained_classifier,
                                                        final_pairs,
                                                        training_reduced_powers,
                                                        training_events,
                                                        sample_weights,
                                                        training_classifier_summaries,
                                                        subject)

        post_hoc_results = post_hoc_classifier_evaluation(final_task_events,
                                                          powers,
                                                          final_post_stim_events,
                                                          post_stim_powers,
                                                          ec_pairs,
                                                          used_classifiers,
                                                          kwargs['n_perm'],
                                                          retrained_classifier,
                                                          **kwargs)

        session_summaries = summarize_stim_sessions(
            all_events, final_task_events, stim_data,
            post_hoc_results['encoding_classifier_summaries'],
            post_hoc_results['post_stim_predicted_probs'],
            pairs_metadata_table)

        math_summaries = summarize_math(all_events, joint=joint_report)

        # TODO: Commented out until we have a clean way to plot results from
        # the traces
        # behavioral_results = estimate_effects_of_stim(subject, experiment,
        #     session_summaries).compute()

    elif stim_report and 'PS' in experiment:
        ps_events = build_ps_data(subject, experiment, 'ps4_events',
                                  sessions, paths.root)
        session_summaries = summarize_ps_sessions(ps_events)
        math_summaries = [] # No math summaries for PS4
        classifier_evaluation_results = []

    # TODO: Add task that saves out all necessary underlying data

    if not stim_report:
        classifier_evaluation_results = [encoding_classifier_summary,
                                         joint_classifier_summary]
    elif stim_report and 'PS' not in experiment:
        classifier_evaluation_results = post_hoc_results[
            'classifier_summaries']

    if 'PS' not in experiment:
        output = save_all_output(subject, experiment, session_summaries,
                                 math_summaries, target_selection_table,
                                 classifier_evaluation_results,
                                 paths.data_db).compute()

    report = build_static_report(subject, experiment, session_summaries,
                                 math_summaries, target_selection_table,
                                 classifier_evaluation_results, paths.dest)

    if vispath is not None:
        report.visualize(filename=vispath)

    return report.compute()
