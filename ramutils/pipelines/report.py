"""Pipeline for creating reports."""


from ramutils.tasks import *


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
    used_pair_mask = get_used_pair_mask(ec_pairs, excluded_pairs)
    pairs_metadata_table = generate_montage_metadata_table(subject, ec_pairs,
                                                           root=paths.root)

    # all_events are used for producing math summaries. Task events are only
    # used by the stim reports. Non-stim reports create a different set of
    # events
    all_events, task_events = build_test_data(subject, experiment, paths,
                                              joint_report, sessions,
                                              **kwargs)
    delta_hfa_table = []
    if not stim_report:
        final_pairs = generate_pairs_for_classifier(ec_pairs, excluded_pairs)
        # This logic is very similar to what is done in config generation except
        # that events are not combined by default
        kwargs['combine_events'] = False
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

        classifier_summaries = perform_cross_validation(classifier,
                                                        reduced_powers,
                                                        final_task_events,
                                                        kwargs['n_perm'],
                                                        **kwargs)

        # FIXME: We don't technically need this right now, but in the future,
        # this object will be saved out somewhere for easier reloading,
        # so go ahead and create it
        classifier = serialize_classifier(classifier,
                                          final_pairs,
                                          reduced_powers,
                                          final_task_events,
                                          sample_weights,
                                          classifier_summaries,
                                          subject)

        # Everything else is specific to the reports and does not follow the
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
                                                   sessions,
                                                   paths.root).compute()
        # Retraining occurs on-demand or if any session-specific classifiers
        # failed to load
        retrained_classifier = None
        if retrain or any([classifier is None for classifier in used_classifiers]):
            final_pairs = generate_pairs_for_classifier(ec_pairs,
                                                        excluded_pairs)

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
                kwargs['n_perm'], **kwargs)

            retrained_classifier = serialize_classifier(retrained_classifier,
                                                        final_pairs,
                                                        training_reduced_powers,
                                                        training_events,
                                                        sample_weights,
                                                        training_classifier_summaries,
                                                        subject)

        classifier_summaries = post_hoc_classifier_evaluation(final_task_events,
                                                              powers,
                                                              ec_pairs,
                                                              used_classifiers,
                                                              kwargs['n_perm'],
                                                              retrained_classifier,
                                                              **kwargs)

        stim_session_summaries = summarize_stim_sessions()

        # TODO: Add build_stim_table task
        # TODO: Add stimulation evaluation task that uses the HMM code

    # TODO: Add task that saves out all necessary underlying data
    session_summaries = summarize_sessions(final_task_events,
                                           joint=joint_report)
    math_summaries = summarize_math(all_events, joint=joint_report)
    report = build_static_report(session_summaries, math_summaries,
                                 delta_hfa_table, classifier_summaries,
                                 paths.dest)

    if vispath is not None:
        report.visualize(filename=vispath)

    return report.compute()
