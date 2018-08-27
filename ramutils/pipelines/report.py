"""Pipeline for creating reports."""
from collections import namedtuple

import pandas as pd

from ramutils.events import dataframe_to_recarray
from ramutils.tasks import *
from ramutils.utils import extract_experiment_series
from ramutils.stim_artifact import get_tstats
from .hooks import PipelineCallback
import json

ReportData = namedtuple('ReportData', 'session_summaries, math_summaries, '
                                      'target_selection_table, classifier_evaluation_results,'
                                      'trained_classifier, repetition_ratio_dict, '
                                      'retrained_classifier, behavioral_results')


def make_report(subject, experiment, paths, joint_report=False,
                retrain=False, stim_params=None, exp_params=None,
                sessions=None, vispath=None, rerun=False,
                trigger_electrode=None, use_classifier_excluded_leads=False,
                pipeline_name="report"):
    """ Constructs a report and saves out all the necessary data to re-construct the report

    This pipeline should be used for generating single session reports for both record-only and
    stimulation sessions. However, the current pipeline also support combining sessions of record-only
    experiments into a single report. In the future, this capability may be moved to
    `ramutils.pipelines.aggregated_report.make_aggregated_report` since that is a more natural location

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
    trigger_electrode: str
        The label for the bipolar pair to be used for triggering stimulation
        in PS5
    use_classifier_excluded_leads: bool
        Use contents of classifier_excluded_leads.txt to exclude channels from
        classifier training
    pipeline_name : str
        Name to use for status updates.

    Returns
    -------
    report_path : str
        Path to generated report.

    Notes
    -----
    Eventually this will return an object that summarizes all output of the
    report rather than the report itself.

    """
    # Lower case 'c' is expected for reading events. The reader should probably
    # just be case insensitive
    if 'Cat' in experiment:
        experiment = experiment.replace('Cat', 'cat')

    ec_pairs = get_pairs(subject, experiment, sessions, paths)

    if use_classifier_excluded_leads:
        classifier_excluded_leads = get_classifier_excluded_leads(
            subject, ec_pairs, paths.root).compute()
        if stim_params is None:
            stim_params = []
        stim_params.extend(classifier_excluded_leads)
    excluded_pairs = reduce_pairs(ec_pairs, stim_params, return_excluded=True)

    # PS4 is such a special beast, that we just return its own sub-pipeline
    # in order to simplify the branching logic for generating all other reports
    if "PS4" in experiment:
        return generate_ps4_report(subject, experiment, sessions, ec_pairs,
                                   excluded_pairs, paths)

    kwargs = exp_params.to_dict()

    stim_report = is_stim_experiment(experiment).compute()
    series_num = extract_experiment_series(experiment)

    if not rerun:
        print('Loading results from %s' % paths.data_db)
        pre_built_results = load_existing_results(subject, experiment, sessions, stim_report,
                                                  paths.data_db,
                                                  joint_report,
                                                  rootdir=paths.root).compute()

        # Check if only None values were returned. Processing will continue
        # undeterred
        if all([val is None for val in [pre_built_results['target_selection_table'],
                                        pre_built_results['classifier_evaluation_results'],
                                        pre_built_results['session_summaries'],
                                        pre_built_results['math_summaries']]]):
            pass
        else:
            report = build_static_report(subject,
                                         experiment,
                                         pre_built_results['session_summaries'],
                                         pre_built_results['math_summaries'],
                                         pre_built_results['target_selection_table'],
                                         pre_built_results['classifier_evaluation_results'],
                                         paths.dest,
                                         hmm_results=pre_built_results['hmm_results'])
            return report.compute()

    final_pairs = generate_pairs_for_classifier(ec_pairs, excluded_pairs)
    used_pair_mask = get_used_pair_mask(ec_pairs, excluded_pairs)
    pairs_metadata_table = generate_montage_metadata_table(subject,
                                                           experiment,
                                                           sessions,
                                                           ec_pairs,
                                                           root=paths.root).compute()

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
                                                   'hfa_p_value', 'hfa_tstat',
                                                   '110_p_value', '110_tstat',
                                                   'mni_x', 'mni_y', 'mni_z',
                                                   'controllability'])

    if not stim_report:
        data = generate_data_for_nonstim_report(subject, experiment, sessions,
                                                joint_report, paths, ec_pairs,
                                                used_pair_mask, excluded_pairs,
                                                final_pairs, pairs_metadata_table,
                                                all_events,
                                                **kwargs)
    elif experiment.find("PS5") != -1:
        data = generate_data_for_ps5_report(subject, experiment, joint_report,
                                            pairs_metadata_table,
                                            trigger_electrode, ec_pairs,
                                            excluded_pairs, all_events,
                                            task_events, stim_data, paths,
                                            **kwargs)
    elif "LocationSearch" in experiment:
        import dask.config
        dask.config.set(scheduler="synchronous")
        data = generate_data_for_location_search_report(
            subject, experiment, pairs_metadata_table, ec_pairs,excluded_pairs,
            all_events, stim_data, paths, **kwargs
        )

    else:
        data = generate_data_for_stim_report(subject, experiment, joint_report,
                                             retrain, paths, ec_pairs,
                                             excluded_pairs,
                                             used_pair_mask, final_pairs,
                                             pairs_metadata_table, all_events,
                                             task_events, stim_data,
                                             **kwargs)

    output = save_all_output(subject, experiment, data.session_summaries,
                             data.math_summaries, data.classifier_evaluation_results,
                             paths.data_db,
                             retrained_classifier=data.retrained_classifier,
                             target_selection_table=data.target_selection_table,
                             behavioral_results=data.behavioral_results).compute()

    report = build_static_report(subject, experiment, data.session_summaries,
                                 data.math_summaries, data.target_selection_table,
                                 data.classifier_evaluation_results,
                                 hmm_results=output, dest=paths.dest)

    if vispath is not None:
        report.visualize(filename=vispath)

    with PipelineCallback(pipeline_name):
        return report.compute()


def generate_ps4_report(subject, experiment, sessions, ec_pairs,
                        excluded_pairs, paths):
    """ PS4-specific report generation pipeline """
    ps_events = build_ps_data(subject, experiment, 'ps4_events',
                              sessions, paths.root)
    session_summaries = summarize_ps_sessions(
        ps_events, ec_pairs, excluded_pairs)

    # PS4 doesn't have most of the same data/requirements as other experiments,
    # but we want to still be able to call the same build_static_report function
    math_summaries = []
    classifier_evaluation_results = []
    target_selection_table = pd.DataFrame(columns=['type', 'contact0',
                                                   'contact1', 'label',
                                                   'hfa_p_value', 'hfa_tstat',
                                                   '110_p_value', '110_tstat',
                                                   'mni_x', 'mni_y', 'mni_z',
                                                   'controllability'])

    report = build_static_report(subject, experiment, session_summaries,
                                 math_summaries, target_selection_table,
                                 classifier_evaluation_results, paths.dest)
    return report.compute()


def generate_data_for_nonstim_report(subject, experiment, sessions,
                                     joint_report, paths, ec_pairs,
                                     used_pair_mask, excluded_pairs,
                                     final_pairs, pairs_metadata_table,
                                     all_events, **kwargs):
    """ Report generation sub-pipeline that is shared by all nonstim reports """
    repetition_ratio_dict = {}
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

    pairinfo = dataframe_to_recarray(pairs_metadata_table[['label',
                                                           'location',
                                                           'region']],
                                     [('label', 'S256'),
                                      ('location', 'S256'),
                                      ('region', 'S256')])[used_pair_mask.compute()]

    joint_classifier_summary = summarize_classifier(classifier,
                                                    reduced_powers,
                                                    final_task_events,
                                                    kwargs['n_perm'],
                                                    tag='Joint',
                                                    pairs=pairinfo,
                                                    **kwargs)
    # Serialize the classifier here
    trained_classifier = serialize_classifier(classifier,
                                              final_pairs,
                                              reduced_powers,
                                              final_task_events,
                                              sample_weights,
                                              joint_classifier_summary,
                                              subject)

    # Subset events, powers, etc to get encoding-only classifier summary
    kwargs['scheme'] = 'EQUAL'
    encoding_only_mask = get_word_event_mask(final_task_events, True)
    final_encoding_task_events = subset_events(final_task_events,
                                               encoding_only_mask)
    encoding_reduced_powers = subset_powers(reduced_powers, encoding_only_mask)

    encoding_sample_weights = get_sample_weights(final_encoding_task_events,
                                                 **kwargs)

    encoding_classifier = train_classifier(encoding_reduced_powers,
                                           final_encoding_task_events,
                                           encoding_sample_weights,
                                           kwargs['C'],
                                           kwargs['penalty_type'],
                                           kwargs['solver'])

    encoding_classifier_summary = summarize_classifier(
        encoding_classifier, encoding_reduced_powers,
        final_encoding_task_events, kwargs['n_perm'], pairs=pairinfo,
        tag='Encoding', **kwargs)

    target_selection_table = create_target_selection_table(
        pairs_metadata_table, powers, final_task_events, kwargs['freqs'],
        hfa_cutoff=kwargs['hfa_cutoff'], trigger_freq=kwargs['trigger_freq'],
        root=paths.root)

    session_summaries = summarize_nonstim_sessions(all_events,
                                                   final_task_events,
                                                   ec_pairs,
                                                   excluded_pairs,
                                                   powers,
                                                   joint=joint_report,
                                                   repetition_ratio_dict=repetition_ratio_dict)

    math_summaries = summarize_math(all_events, joint=joint_report)
    classifier_evaluation_results = [encoding_classifier_summary,
                                     joint_classifier_summary]

    data = ReportData(session_summaries, math_summaries, target_selection_table,
                      classifier_evaluation_results, trained_classifier, repetition_ratio_dict,
                      trained_classifier, None)

    return data


def generate_data_for_stim_report(subject, experiment, joint_report, retrain,
                                  paths, ec_pairs, excluded_pairs,
                                  used_pair_mask, final_pairs,
                                  pairs_metadata_table, all_events,
                                  task_events, stim_data, **kwargs):
    """ Report generation sub-pipeline shared by all stim reports """
    series_num = extract_experiment_series(experiment)
    # FR2 does not have STIM_OFF events, so until we can identify them more
    # easily, do not compute post stim powers for now
    if series_num != '2':
        # We need post stim period events/powers
        post_stim_mask = get_post_stim_events_mask(all_events)
        post_stim_events = subset_events(all_events, post_stim_mask)
        post_stim_powers, final_post_stim_events = compute_normalized_powers(
            post_stim_events, bipolar_pairs=ec_pairs, **kwargs)
        post_stim_eeg = load_post_stim_eeg(post_stim_events,
                                           bipolar_pairs=ec_pairs,
                                           **kwargs
                                           )
    else:
        final_post_stim_events = None
        post_stim_powers = None
        post_stim_eeg = None

    powers, final_task_events = compute_normalized_powers(
        task_events, bipolar_pairs=ec_pairs, **kwargs)

    pairinfo = dataframe_to_recarray(pairs_metadata_table[['label',
                                                           'location',
                                                           'region']],
                                     [(str('label'), 'U256'),
                                      (str('location'), 'U256'),
                                      (str('region'), 'U256')])

    used_classifiers = reload_used_classifiers(subject,
                                               experiment,
                                               final_task_events,
                                               paths.root).compute()

    # Retraining occurs on-demand or if any session-specific classifiers
    # failed to load.
    retrained_classifier = None
    if retrain or any([classifier is None for classifier in used_classifiers]):
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

        training_classifier_summaries = summarize_classifier(
            retrained_classifier, training_reduced_powers,
            final_training_events, kwargs['n_perm'],
            tag='Original Classifier', pairs=pairinfo,
            **kwargs)

        retrained_classifier = serialize_classifier(retrained_classifier,
                                                    final_pairs,
                                                    training_reduced_powers,
                                                    final_training_events,
                                                    sample_weights,
                                                    training_classifier_summaries,
                                                    subject)

    post_hoc_results = post_hoc_classifier_evaluation(final_task_events,
                                                      powers,
                                                      ec_pairs,
                                                      used_classifiers,
                                                      kwargs['n_perm'],
                                                      retrained_classifier,
                                                      use_retrained=retrain,
                                                      post_stim_events=final_post_stim_events,
                                                      post_stim_powers=post_stim_powers,
                                                      pairs=pairinfo,
                                                      **kwargs)
    excluded_pairs = extract_rejected_pairs(subject, used_classifiers, ec_pairs,
                                            used_pair_mask)

    pairs_metadata_table['stim_tstats'], pairs_metadata_table['stim_pvals'] = get_artifact_tstats(
        all_events[all_events['type'] == 'STIM_ON'], ec_pairs, return_pvalues=True).compute()

    session_summaries = summarize_stim_sessions(all_events, final_task_events,
                                                stim_data, pairs_metadata_table,
                                                ec_pairs, excluded_pairs,
                                                powers,
                                                post_hoc_results[
                                                    'encoding_classifier_summaries'],
                                                post_hoc_results[
                                                    'post_stim_predicted_probs'],
                                                post_stim_eeg=post_stim_eeg)

    math_summaries = summarize_math(all_events, joint=joint_report)

    # Note: This task modifies the session summaries by adding the result traces after the
    # models are fit. In general, it would be better to create the session summary objects
    # in summarize_stim_sessions and not modify them afterwards, but this task currently
    # needs the session summaries to exist in order to work
    behavioral_results = estimate_effects_of_stim(subject, experiment,
                                                  session_summaries)

    classifier_evaluation_results = post_hoc_results[
        'classifier_summaries']
    data = ReportData(session_summaries, math_summaries, None,
                      classifier_evaluation_results, None, None,
                      retrained_classifier, behavioral_results)

    return data


def generate_data_for_location_search_report(subject, experiment,
                                             pairs_metadata_table,
                                             ec_pairs,
                                             excluded_pairs,
                                             all_events,
                                             stim_data,
                                             paths, **kwargs):
    connectivity = get_resting_connectivity(
        subject, rootdir=paths.root
    )
    stim_events = all_events[all_events.type == 'STIM_ON']

    pre_psd, post_psd, emask, cmask = get_psd_data(
        pd.DataFrame(stim_events), paths.root)

    post_stim_eeg = load_post_stim_eeg(all_events,bipolar_pairs=ec_pairs, **kwargs)

    pairs_metadata_table['stim_tstats'], pairs_metadata_table['stim_pvals'] = get_artifact_tstats(
        all_events[all_events['type'] == 'STIM_ON'], ec_pairs, return_pvalues=True, before_experiment=False).compute()

    session_summaries = summarize_location_search_sessions(stim_data,
                                                           pairs_metadata_table,
                                                           excluded_pairs,
                                                           connectivity,
                                                           pre_psd,
                                                           post_psd,
                                                           emask,
                                                           cmask,
                                                           post_stim_eeg=post_stim_eeg,
                                                           )
    return ReportData(session_summaries, [], None,
                      [], None, dict(), None, dict())


def generate_data_for_ps5_report(subject, experiment, joint_report,
                                 pairs_metadata_table,
                                 trigger_electrode, ec_pairs, excluded_pairs,
                                 all_events, task_events, stim_data, paths,
                                 **kwargs):
    """
        Report generating sub-pipeline for PS5. This is an odd mix
        of the non-stim report and the stim report sub-pipelines
    """
    trigger_electrode_mask = get_trigger_electrode_mask(pairs_metadata_table,
                                                        trigger_electrode)
    trigger_frequency_mask = get_trigger_frequency_mask(kwargs['trigger_freq'],
                                                        kwargs['freqs'])

    # Calculate post stim and encoding powers similar to stim report, but with
    # just the one electrode/frequency
    post_stim_mask = get_post_stim_events_mask(all_events)
    post_stim_events = subset_events(all_events, post_stim_mask)
    post_stim_powers, final_post_stim_events = compute_normalized_powers(
        post_stim_events, bipolar_pairs=ec_pairs, **kwargs)
    post_stim_reduced_powers = reduce_powers(post_stim_powers,
                                             trigger_electrode_mask,
                                             len(kwargs['freqs']),
                                             frequency_mask=trigger_frequency_mask)

    powers, final_task_events = compute_normalized_powers(
        task_events, bipolar_pairs=ec_pairs, **kwargs).compute()
    reduced_powers = reduce_powers(powers, trigger_electrode_mask,
                                   len(kwargs['freqs']),
                                   frequency_mask=trigger_frequency_mask)

    session_summaries = summarize_stim_sessions(all_events, task_events,
                                                stim_data,
                                                pairs_metadata_table,
                                                ec_pairs, excluded_pairs,
                                                powers,
                                                trigger_output=reduced_powers,
                                                post_stim_trigger_output=post_stim_reduced_powers).compute()
    math_summaries = summarize_math(all_events)
    classifier_evaluation_results = []

    data = ReportData(session_summaries, math_summaries, None,
                      classifier_evaluation_results, None, None,
                      None, None)

    return data
