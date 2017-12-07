"""Pipeline for creating reports."""


from ramutils.tasks import *


def make_report(subject, experiment, paths, joint_report=False, stim_params=None, classifier_container=None,
                exp_params=None, sessions=None, vispath=None):
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
    stim_params : List[StimParameters]
        Stimulation parameters (empty list for non-stim experiments).
    classifier_container : ClassifierContainer
        For experiments that ran with a classifier, the container detailing the
        classifier that was actually used. When not given, a new classifier will
        be trained to (hopefully) recreate what was actually used.
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

    ec_pairs = generate_pairs_from_electrode_config(subject, paths)
    excluded_pairs = reduce_pairs(ec_pairs, stim_params, True)
    used_pair_mask = get_used_pair_mask(ec_pairs, excluded_pairs)

    pairs_metadata_table = generate_montage_metadata_table(subject, ec_pairs, root=paths.root)

    if joint_report and 'FR' in experiment:
        fr_events = load_events(subject, 'FR1', rootdir=paths.root)
        cleaned_fr_events = clean_events(fr_events,
                                         start_time=kwargs['baseline_removal_start_time'],
                                         end_time=kwargs['retrieval_time'],
                                         duration=kwargs['empty_epoch_duration'],
                                         pre=kwargs['pre_event_buf'],
                                         post=kwargs['post_event_buf'])

        catfr_events = load_events(subject, 'catFR1', rootdir=paths.root)
        cleaned_catfr_events = clean_events(catfr_events,
                                            start_time=kwargs['baseline_removal_start_time'],
                                            end_time=kwargs['retrieval_time'],
                                            duration=kwargs['empty_epoch_duration'],
                                            pre=kwargs['pre_event_buf'],
                                            post=kwargs['post_event_buf'])

        all_events = concatenate_events_across_experiments([fr_events, catfr_events])
        task_events = concatenate_events_across_experiments(
            [cleaned_fr_events, cleaned_catfr_events])

    elif not joint_report and 'FR' in experiment:
        all_events = load_events(subject, experiment, sessions=sessions, rootdir=paths.root)
        task_events = clean_events(all_events,
                                   start_time=kwargs['baseline_removal_start_time'],
                                   end_time=kwargs['retrieval_time'],
                                   duration=kwargs['empty_epoch_duration'],
                                   pre=kwargs['pre_event_buf'],
                                   post=kwargs['post_event_buf'])

    else:
        all_events = load_events(subject, experiment, sessions=sessions, rootdir=paths.root)
        task_events = clean_events(all_events)

    # Both powers and task_events need to be saved out somewhere
    powers, final_task_events = compute_normalized_powers(task_events,
                                                          **kwargs)

    # There are a set of things that need to be done both by session and across sessions.
    delta_hfa_table = calculate_delta_hfa_table(pairs_metadata_table,
                                                powers,
                                                task_events,
                                                kwargs['freqs'],
                                                hfa_cutoff=kwargs['hfa_cutoff'])

    # Classifier training/evaluation will be different for stim reports versus non-stim
    # Create another task here called build_classifier(events, powers) and returning a trained classifier
    # that can be passed to cross-validation. Then, we can train a set of classifiers on an arbitrary set of
    # events/powers
    reduced_powers = reduce_powers(powers, used_pair_mask, len(kwargs['freqs']))
    sample_weights = get_sample_weights(final_task_events, **kwargs)

    # For FR1/catFR1 report, we need to be able to train the encoding only and joint classifier for comparison
    # This means different event subsets as well as different power subsets
    classifier = train_classifier(reduced_powers,
                                  final_task_events,
                                  sample_weights,
                                  kwargs['C'],
                                  kwargs['penalty_type'],
                                  kwargs['solver'])

    cross_validation_results = perform_cross_validation(classifier,
                                                        reduced_powers,
                                                        final_task_events,
                                                        kwargs['n_perm'],
                                                        **kwargs)

    if sessions is None:
        session_summaries = [
            summarize_session(final_task_events[final_task_events.session == session])
            for session in sessions]
    else:
        session_summaries = summarize_session(final_task_events)

    if vispath is not None:
        pass

    math_summaries = summarize_math(all_events)
    report = build_static_report([session_summaries], [math_summaries], cross_validation_results, delta_hfa_table)

    return report.compute()
