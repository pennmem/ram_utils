"""Pipeline for creating reports."""


from ramutils.tasks import *
import pandas as pd


def make_report(subject, experiment, paths, joint_report=False,
                retrain=False, stim_params=None, exp_params=None,
                sessions=None, vispath=None):
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
    sessions : list
        For reports that span sessions, sessions to read data from.
        When not given, all available sessions are used for reports.
    vispath : str
        Filename for task graph visualization.

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

    # TODO: Add method that will check if the necessary underlying data already
    # exists to avoid re-running

    stim_report = is_stim_experiment(experiment).compute()

    ec_pairs = generate_pairs_from_electrode_config(subject, paths)
    excluded_pairs = reduce_pairs(ec_pairs, stim_params, True)
    final_pairs = generate_pairs_for_classifier(ec_pairs, excluded_pairs)
    used_pair_mask = get_used_pair_mask(ec_pairs, excluded_pairs)
    pairs_metadata_table = generate_montage_metadata_table(subject, ec_pairs,
                                                           root=paths.root)

    # all_events are used for producing math summaries. Task events are only
    # used by the stim reports. Non-stim reports create a different set of
    # events
    all_events, task_events = build_test_data(subject, experiment, paths,
                                              joint_report, sessions=sessions,
                                              **kwargs)

    delta_hfa_table = pd.DataFrame(columns=['type', 'contact0',
                                            'contact1','label',
                                            'p_value', 'tstat'])
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

        powers, final_task_events = compute_normalized_powers(all_task_events,
                                                              **kwargs)
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
                                                            tag='Joint Classifier',
                                                            **kwargs)

        # Subset events, powers, etc to get encoding-only classifier summary
        kwargs['scheme'] = 'EQUAL'
        encoding_only_mask = get_word_event_mask(all_task_events, True)
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
            tag='Encoding Classifier', **kwargs)

        delta_hfa_table = calculate_delta_hfa_table(pairs_metadata_table,
                                                    powers,
                                                    final_task_events,
                                                    kwargs['freqs'],
                                                    hfa_cutoff=kwargs['hfa_cutoff'])

        # TODO: Modal Controllability Table calculation here
        # TODO: Optimal stim target table based on prior stim results table here

    if stim_report:
        powers, final_task_events = compute_normalized_powers(task_events,
                                                              **kwargs)
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
                training_events, **kwargs)

            training_reduced_powers = reduce_powers(training_powers,
                                                    used_pair_mask,
                                                    len(kwargs['freqs']))

            sample_weights = get_sample_weights(training_events, **kwargs)

            retrained_classifier = train_classifier(training_reduced_powers,
                                                    final_training_events,
                                                    sample_weights,
                                                    kwargs['C'],
                                                    kwargs['penalty_type'],
                                                    kwargs['solver'])

            training_classifier_summaries = perform_cross_validation(
                retrained_classifier, training_reduced_powers, training_events,
                kwargs['n_perm'], tag='Original Classifier', **kwargs)

            retrained_classifier = serialize_classifier(retrained_classifier,
                                                        final_pairs,
                                                        training_reduced_powers,
                                                        training_events,
                                                        sample_weights,
                                                        training_classifier_summaries,
                                                        subject)

        post_hoc_results = post_hoc_classifier_evaluation(final_task_events,
                                                          powers,
                                                          ec_pairs,
                                                          used_classifiers,
                                                          kwargs['n_perm'],
                                                          retrained_classifier,
                                                          **kwargs)

        stim_session_summaries = summarize_stim_sessions(
            all_events, final_task_events,
            post_hoc_results['session_summaries_stim_table'],
            pairs_metadata_table).compute() # TODO: Remove this forced

        # TODO: Add stimulation evaluation task that uses the HMM code

    # TODO: Add task that saves out all necessary underlying data
    session_summaries = summarize_sessions(all_events,
                                           final_task_events,
                                           joint=joint_report,
                                           repetition_ratio_dict=repetition_ratio_dict)

    math_summaries = summarize_math(all_events, joint=joint_report)

    if not stim_report:
        results = [encoding_classifier_summary, joint_classifier_summary]
    else:
        results = post_hoc_results['session_summaries']

    report = build_static_report(subject, experiment, session_summaries,
                                 math_summaries, delta_hfa_table,
                                 results, paths.dest)

    if vispath is not None:
        report.visualize(filename=vispath)

    return report.compute()
