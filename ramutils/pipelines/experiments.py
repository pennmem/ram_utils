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

# TODO: use a more flexible data structure that this named tuple, no sense in having
#       unused fields and limiting ourselves

def generate_data_for_repfr_report(subject, experiment, sessions,
                                     joint_report, paths, ec_pairs,
                                     used_pair_mask, excluded_pairs,
                                     final_pairs, pairs_metadata_table,
                                     all_events, **kwargs):
    '''
    Architecture for experiments moving forward. This function defines the
    data processing for the repFR paradigm. All versions (non stim, open
    loop stim, closed loop stim) are all handled by branches from this
    function. The end result is returned as a ReportData named tuple that
    contains null values for the data not generated for this paradigm.

    At some point, this function should become a class to handle stim versions
    '''

    if joint_report:
        raise Exception("repFR does not support joint reports")

    repetition_ratio_dict = None # NOTE: only for catFR experiments

    kwargs['encoding_only'] = False

    # FIXME: why do I need this?
    kwargs['combine_events'] = False
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

    joint_classifier_summary = None

    trained_classifier = serialize_classifier(classifier,
                                              final_pairs,
                                              reduced_powers,
                                              final_task_events,
                                              sample_weights,
                                              joint_classifier_summary,
                                              subject)


    pairinfo = dataframe_to_recarray(pairs_metadata_table[['label',
                                                           'location',
                                                           'region']],
                                     [('label', 'S256'),
                                      ('location', 'S256'),
                                      ('region', 'S256')])[used_pair_mask.compute()]



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

    math_summaries = None # NOTE: as designed, this task has no math
    
    behavioral_results = None # NOTE: only for stim experiments

    classifier_evaluation_results = [encoding_classifier_summary]

    data = ReportData(session_summaries, math_summaries, target_selection_table,
                    classifier_evaluation_results, trained_classifier, repetition_ratio_dict,
                    trained_classifier, behavioral_results)
                        
    return data

def generate_data_for_dboy_report(subject, experiment, sessions,
                                     joint_report, paths, ec_pairs,
                                     used_pair_mask, excluded_pairs,
                                     final_pairs, pairs_metadata_table,
                                     all_events, **kwargs):

    return ReportData(None, None, None, None, None, None, None, None)

