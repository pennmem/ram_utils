"""Pipeline for creating reports."""


from ramutils.tasks import *


# TODO: Are record-only report different enough from stim report that we should have two different pipelines and just
# unify in the command line interface?
def make_report(subject, experiment, paths, stim_params=None, classifier_container=None,
                exp_params=None, sessions=None, vispath=None):
    """Run a report.

    Parameters
    ----------
    subject : str
        Subject ID
    experiment : str
        Experiment to generate report for
    paths : FilePaths
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
    # Note: All of these pairs variables are of type OrderedDict, which is
    # crucial for preserving the initial order of the electrodes in the
    # config file
    ec_pairs = generate_pairs_from_electrode_config(subject, paths)
    excluded_pairs = reduce_pairs(ec_pairs, stim_params, True)
    used_pair_mask = get_used_pair_mask(ec_pairs, excluded_pairs)
    final_pairs = generate_pairs_for_classifier(ec_pairs, excluded_pairs)

    pairs_metadata_table = generate_montage_metadata_table(subject, ec_pairs, root=paths.root)

    if classifier_container is not None:
        # Load task events, reduced powers, trained classifier, etc all from the given serialized classifier container
        classifier = classifier_container.classifier
        events = classifier_container.events

    kwargs = exp_params.to_dict()
    # Use a different function here? generate_event_subsets() -- We need all events in order to create MathSummary
    events = preprocess_events(subject, experiment, kwargs['baseline_removal_start_time'], kwargs['retrieval_time'],
                               kwargs['empty_epoch_duration'], kwargs['pre_event_buf'], kwargs['post_event_buf'],
                               sessions=sessions, combine_events=kwargs['combine_events'], root=paths.root)

    # Both powers and task_events need to be saved out somewhere
    powers, task_events = compute_normalized_powers(events,
                                                    **kwargs)

    delta_hfa_table = calculate_delta_hfa_table(pairs_metadata_table,
                                                powers,
                                                events,
                                                kwargs['freqs'],
                                                hfa_cutoff=kwargs['hfa_cutoff'])

    # Create another task here called build_classifier(events, powers) and returning a trained classifier
    # that can be passed to cross-validation. Then, we can train a set of classifiers on an arbitrary set of
    # events/powers

    reduced_powers = reduce_powers(powers, used_pair_mask, len(kwargs['freqs']))

    sample_weights = get_sample_weights(task_events, **kwargs)

    # For FR1/catFR1 report, we need to be able to train the encoding only and joint classifier for comparison
    # This means different event subsets as well as different power subsets
    classifier = train_classifier(reduced_powers,
                                  task_events,
                                  sample_weights,
                                  kwargs['C'],
                                  kwargs['penalty_type'],
                                  kwargs['solver'])

    cross_validation_results = perform_cross_validation(classifier,
                                                        reduced_powers,
                                                        task_events,
                                                        kwargs['n_perm'],
                                                        **kwargs)

    if sessions is None:
        session_summaries = [
            summarize_session(events[events.session == session])
            for session in sessions]
    else:
        session_summaries = [summarize_session(events)]

    if vispath is not None:
        pass

    math_summaries = []
    classifier_summaries = [cross_validation_results]
    report_path = build_static_report(session_summaries, math_summaries, classifier_summaries, delta_hfa_table)

    return report_path.compute()
