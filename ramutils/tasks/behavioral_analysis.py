from __future__ import division

import pandas as pd

from ramutils.hmm import HierarchicalModel
from ramutils.log import get_logger
from ramutils.tasks import task


logger = get_logger()

__all__ = [
    'estimate_effects_of_stim'
]


@task()
def estimate_effects_of_stim(subject, experiment, stim_session_summaries):
    """
        fit a bayesian hierarchical model for estimating the effect of stim
        on recall

    Parameters:
    -----------
    session_summaries: list
        List of FRStimSessionSummary objects

    Returns:
    --------


    """
    result_traces = {}

    session_dataframes = []
    for session_summary in stim_session_summaries:
        session_df = session_summary.to_dataframe()
        session_dataframes.append(session_df)

    df = pd.concat(session_dataframes)

    # serialpos, list, session all need to be 0-indexed for the vectorized
    # implementation of the model to work
    df['session_idx'] = df.groupby(by=['subject', 'experiment', 'session']).grouper.group_info[0]
    df["serialpos"] = df["serialpos"] - (df["serialpos"].min())

    df = df[df["list"] > 3] # drop the first 3 lists since they are not technically part of the experiment

    # Turn list into a % session completed variable to that subjects who
    # complete only partial sessions can still be compared to full sessions
    df["list"] = ((df["list"] - (df["list"].min())) /
                    (len(df["list"].unique())))

    serialpos_dummies = pd.get_dummies(df.serialpos)
    df = pd.concat([df, serialpos_dummies], axis=1)

    df["low_biomarker"] = (df["classifier_output"] < df["thresh"])

    # Stim Lists vs. Non-stim Lists
    stim_list_model = HierarchicalModel(df, subject, experiment,
                                        item_comparison='list')
    stim_list_trace = stim_list_model.fit()
    result_traces['list'] = stim_list_trace

    # Stim items vs. low bio non stim items
    stim_or_low_bio_df = df[((df["is_stim_item"] == True) |
                            ((df["low_biomarker"] == True) &
                            (df["is_stim_item"] == False)))]
    stim_item_model = HierarchicalModel(stim_or_low_bio_df,
                                        subject,
                                        experiment,
                                        item_comparison='stim')
    stim_item_trace = stim_item_model.fit()
    result_traces['stim_item'] = stim_item_trace

    # Post Stim Items vs. Low Biomarker Non-stim Items
    post_stim_or_low_bio_df = df[((df["is_post_stim_item"] == True) |
                                  ((df["low_biomarker"] == True) &
                                  (df["is_post_stim_item"] == False)))]
    post_stim_item_model = HierarchicalModel(post_stim_or_low_bio_df,
                                             subject,
                                             experiment,
                                             item_comparison='post_stim')
    post_stim_item_trace = post_stim_item_model.fit()
    result_traces['post_stim_item'] = post_stim_item_trace

    # In an ideal world, this task would not have the side-effect of updating a member variable
    # of the stim session summary, but I do not currently see a way around it
    for summary in stim_session_summaries:
        summary.model_metadata = result_traces

    return result_traces


