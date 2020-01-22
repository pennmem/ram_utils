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
    Architecture for experiments moving forward. Thus function defines the
    data processing for the repFR paradigm. All versions (non stim, open
    loop stim, closed loop stim) are all handled by branches from this
    function. The end result is returned as a ReportData named tuple that
    contains null values for the data not generated for this paradigm.
    '''

    if joint_report:
        raise Exception("repFR does not support joint reports")

    repetition_ratio_dict = None # NOTE: only for catFR experiments

    kwargs['encoding_only'] = False

    powers, final_task_events = compute_normalized_powers(
        all_task_events, bipolar_pairs=ec_pairs, **kwargs)

    reduced_powers = reduce_powers(powers, used_pair_mask,
                                   len(kwargs['freqs']))

    trained_classifier = None
    classifier_evaluation_results = None # TODO: will be implemented at a later point

    pairinfo = dataframe_to_recarray(pairs_metadata_table[['label',
                                                           'location',
                                                           'region']],
                                     [('label', 'S256'),
                                      ('location', 'S256'),
                                      ('region', 'S256')])[used_pair_mask.compute()]

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

