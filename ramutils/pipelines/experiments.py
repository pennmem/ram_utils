from collections import namedtuple
import pandas as pd
from ramutils.events import dataframe_to_recarray
from ramutils.tasks import *
from ramutils.utils import extract_experiment_series
from ramutils.stim_artifact import get_tstats
from .hooks import PipelineCallback
import json


# Old fields:
# 'session_summaries, math_summaries, '
#                                       'target_selection_table, classifier_evaluation_results,'
#                                       'trained_classifier, repetition_ratio_dict, '
#                                       'retrained_classifier, behavioral_results')

class ReportData:
    '''
    Intermediate datastructure from which the report is generated. Thus,
    the 'keys' on this object are important for compatibility with downstream
    formatting functions.

    For this reason, keys should generally map to collections of objects with
    labels, so that duplicates of a particular type of data may be supported.
    To the extent possible, the formatting functions should not feed back
    onto the design of this class. All data needed for a particular analysis
    should be accessible
    '''
    # TODO: construct with subject, experiment, and session data
    def __init__(self, load=False, **kwargs):

        for k, v in kwargs.items():
            setattr(self, k, v)


    def save_data(self):
        raise NotImplementedError()

    def load_data(self):
        raise NotImplementedError()

    def train_classifier(self, name):
        raise NotImplementedError()


def generate_data_for_repfr_report(subject, experiment, sessions,
                                     joint_report, paths, ec_pairs,
                                     used_pair_mask, excluded_pairs,
                                     final_pairs, pairs_metadata_table,
                                     all_events, **kwargs):
    '''Architecture for experiments moving forward. This function defines the
    data processing for the repFR paradigm. All versions (non stim, open
    loop stim, closed loop stim) are all handled by branches from this
    function. The end result is returned as a ReportData named tuple that
    contains null values for the data not generated for this paradigm.

    At some point, this function should become a class to handle stim versions
    '''

    if joint_report:
        raise Exception("repFR does not support joint reports")

    repetition_ratio_dict = None  # NOTE: only for catFR experiments
    kwargs['encoding_only'] = False
    kwargs['combine_events'] = False

    pairinfo = dataframe_to_recarray(pairs_metadata_table[['label',
                                                           'location',
                                                           'region']],
                                     [('label', 'S256'),
                                      ('location', 'S256'),
                                      ('region', 'S256')])[used_pair_mask]

    all_task_events = build_training_data(subject,
                                          experiment,
                                          paths,
                                          sessions=sessions,
                                          **kwargs)

    powers, final_task_events = compute_normalized_powers(all_task_events, bipolar_pairs=ec_pairs, **kwargs)

    pres_counts = (1, 2, 3)
    classifier_summaries = []
    classifiers = []

    for p in pres_counts:
        events_mask = final_task_events['repeats'] == p
        events_subset = subset_events(final_task_events, events_mask)
        powers_subset = subset_powers(powers, events_mask)

        reduced_powers = reduce_powers(powers_subset, used_pair_mask,
                                       len(kwargs['freqs']))

        sample_weights = get_sample_weights(events_subset, **kwargs)

        classifier = train_classifier(reduced_powers,
                                      events_subset,
                                      sample_weights,
                                      kwargs['C'],
                                      kwargs['penalty_type'],
                                      kwargs['solver'])

        joint_classifier_summary = summarize_classifier(classifier,
                                                        reduced_powers,
                                                        events_subset,
                                                        kwargs['n_perm'],
                                                        tag='{}p'.format(p),
                                                        pairs=pairinfo,
                                                        **kwargs)

        trained_classifier = serialize_classifier(classifier,
                                                  final_pairs,
                                                  reduced_powers,
                                                  events_subset,
                                                  sample_weights,
                                                  joint_classifier_summary,
                                                  subject)

        classifiers.append(trained_classifier)
        classifier_summaries.append(joint_classifier_summary)

        #### Encoding Classifiers

        encoding_only_mask = get_word_event_mask(events_subset, True)
        final_encoding_task_events = subset_events(events_subset,
                                                   encoding_only_mask)
        encoding_reduced_powers = subset_powers(reduced_powers, encoding_only_mask)

        encoding_sample_weights = get_sample_weights(final_encoding_task_events,
                                                     scheme='EQUAL',
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
                tag='{}p Encoding'.format(p), scheme='EQUAL', **kwargs)

        classifiers.append(encoding_classifier)
        classifier_summaries.append(encoding_classifier_summary)

        del encoding_reduced_powers


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

    data = ReportData(session_summaries=session_summaries,
                      target_selection_table=target_selection_table,
                      classifier_evaluation_results=classifier_summaries,
                      trained_classifier=classifiers,
                      repetition_ratio_dict=repetition_ratio_dict)

                        
    return data

def generate_data_for_dboy_report(subject, experiment, sessions,
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
                                      ('region', 'S256')])[used_pair_mask]

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

    classifier_evaluation_results = [encoding_classifier_summary,
                                     joint_classifier_summary]

    return ReportData(session_summaries=session_summaries,
                      target_selection_table=target_selection_table,
                      classifier_evaluation_results=classifier_evaluation_results,
                      trained_classifier=trained_classifier,
                      repetition_ratio_dict=repetition_ratio_dict,
                      retrained_classifier=trained_classifier,
                      behavioral_results=None)


def generate_data_for_efrcourier_report(subject, experiment, sessions,
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
                                      ('region', 'S256')])[used_pair_mask]

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

    classifier_evaluation_results = [encoding_classifier_summary,
                                     joint_classifier_summary]

    return ReportData(session_summaries=session_summaries,
                      target_selection_table=target_selection_table,
                      classifier_evaluation_results=classifier_evaluation_results,
                      trained_classifier=trained_classifier,
                      repetition_ratio_dict=repetition_ratio_dict,
                      retrained_classifier=trained_classifier,
                      behavioral_results=None)
