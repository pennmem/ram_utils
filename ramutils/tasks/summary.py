"""Tasks related to summarizing an experiment. Used primarily in reporting
results.

"""

import numpy as np
import pandas as pd

from ._wrapper import task
from ramutils.log import get_logger
from ramutils.utils import combine_tag_names, sanitize_comma_sep_list

logger = get_logger()


@task()
def generate_stim_table(events, all_events, fr_stim_prob, bp_tal_structs,
                        eval_output, ps_events=None):
    """Generate table of stim info for stim experiments.

    Parameters
    ----------
    events : np.recarray
    all_events : np.recarray
    fr_stim_prob
    bp_tal_structs
    eval_output
    ps_events

    Returns
    -------
    stim_table : pd.DataFrame

    """
    try:
        ps_sessions = np.unique(ps_events.session)
    except KeyError:
        ps_sessions = []

    n_events = len(events)

    is_stim_item = np.zeros(n_events, dtype=np.bool)
    is_post_stim_item = np.zeros(n_events, dtype=np.bool)
    is_ps4_session = np.in1d(events.session, ps_sessions)

    sessions = np.unique(events.session)
    all_events = all_events[np.in1d(all_events.session, sessions) &
                            ((all_events.phase == 'STIM')
                             | (all_events.phase == 'NON-STIM')
                             | (all_events.phase == 'BASELINE')
                             | (all_events.phase == 'PRACTICE'))]

    for session in np.unique(all_events.session):
        all_sess_events = all_events[all_events.session == session]
        for lst in np.unique(all_sess_events.list):
            all_lst_events = all_sess_events[all_sess_events.list == lst]
            lst_stim_words = np.zeros(len(all_lst_events[all_lst_events.type == 'WORD']))
            lst_post_stim_words = np.zeros(len(all_lst_events[all_lst_events.type == 'WORD']))
            j = 0
            for i, ev in enumerate(all_lst_events):
                if ev.type == 'WORD':
                    # FIXME: cleanup conditionals
                    if ((all_lst_events[i+1].type == 'STIM_ON')
                         or (all_lst_events[i+1].type == 'WORD_OFF' and
                            (all_lst_events[i+2].type == 'STIM_ON' or (all_lst_events[i+2].type == 'DISTRACT_START'
                                                                              and all_lst_events[i+3].type == 'STIM_ON')))):
                        lst_stim_words[j] = True
                    if ((all_lst_events[i-1].type == 'STIM_OFF') or (all_lst_events[i+1].type == 'STIM_OFF')
                         or (all_lst_events[i-2].type == 'STIM_OFF' and all_lst_events[i-1].type == 'WORD_OFF')):
                        lst_post_stim_words[j] = True
                    j += 1
            lst_mask = (events.session == session) & (events.list == lst)
            if sum(lst_mask) != len(lst_stim_words):
                new_mask = np.in1d(all_lst_events[all_lst_events.type == 'WORD'].item_name,
                                                  events[lst_mask].item_name)
                lst_stim_words = lst_stim_words[new_mask]
                lst_post_stim_words = lst_post_stim_words[new_mask]
            is_stim_item[lst_mask] = lst_stim_words
            is_post_stim_item[lst_mask] = lst_post_stim_words

    stim_table = pd.DataFrame()
    stim_table['item'] = events.item_name
    stim_table['session'] = events.session
    stim_table['list'] = events.list
    stim_table['serialpos'] = events.serialpos
    stim_table['phase'] = events.phase
    stim_table['item_name'] = events.item_name
    stim_table['is_stim_list'] = [e.phase == 'STIM' for e in events]
    stim_table['is_post_stim_item'] = is_post_stim_item
    stim_table['is_stim_item'] = is_stim_item
    stim_table['recalled'] = events.recalled
    stim_table['thresh'] = 0.5
    stim_table['is_ps4_session'] = is_ps4_session

    for (session, lst), _ in stim_table.groupby(('session', 'list')):
        sess_list = (stim_table.session == session) & (stim_table.list == lst)
        fr_stim_sess_list_table = stim_table.loc[sess_list]
        post_is_stim = np.concatenate(([False], fr_stim_sess_list_table.is_stim_item.values[:-1].astype(bool)))
        stim_table.loc[sess_list,'is_post_stim_item'] = post_is_stim

    if eval_output:
        stim_table['prob'] = fr_stim_prob
    else:
        stim_table['prob'] = -999

    # Calculate stim params on an event-by-event basis
    stim_param_data = {
        'session': [],
        'list': [],
        'amplitude': [],
        'pulse_freq': [],
        'stim_duration': [],
        'stimAnodeTag': [],
        'stimCathodeTag': [],
    }

    for i in range(len(all_events)):
        stim_params = all_events[i].stim_params
        stim_param_data['session'].append(all_events[i].session)
        stim_param_data['list'].append(all_events[i].list)
        stim_param_data['amplitude'].append(",".join([str(stim_params[k].amplitude) for k in range(len(stim_params))]))
        stim_param_data['pulse_freq'].append(",".join([str(stim_params[k].pulse_freq) for k in range(len(stim_params))]))
        stim_param_data['stim_duration'].append(",".join([str(stim_params[k].stim_duration) for k in range(len(stim_params))]))
        stim_param_data['stimAnodeTag'].append(",".join([str(stim_params[k].anode_label) for k in range(len(stim_params))]))
        stim_param_data['stimCathodeTag'].append(",".join([str(stim_params[k].cathode_label) for k in range(len(stim_params))]))

    # Convert to dataframe for easier last-minute munging
    stim_df = pd.DataFrame.from_dict(stim_param_data)
    stim_df = stim_df.drop_duplicates()
    stim_df['stimAnodeTag'] = stim_df['stimAnodeTag'].replace(',', np.nan) # this will allow us to drop non-stim information
    stim_df = stim_df.dropna(how='any')
    stim_df['stimAnodeTag'] = stim_df['stimAnodeTag'].str.rstrip(',')
    stim_df['stimCathodeTag'] = stim_df['stimCathodeTag'].str.rstrip(',')
    stim_df['pair'] = stim_df['stimAnodeTag'] + '-' + stim_df['stimCathodeTag']

    # Remove zeros from these values since the event files store nulls this way
    for col in ['amplitude', 'pulse_freq', 'stim_duration']:
        stim_df[col] = stim_df[col].apply(sanitize_comma_sep_list)

    stim_table = stim_table.merge(stim_df, on=['session', 'list'], how='left')

    # Create the list of stim targets
    grouped = stim_df.groupby(by=['stimAnodeTag', 'stimCathodeTag'])
    targets = grouped.groups.keys()
    targets = combine_tag_names(targets)

    # Add region to the stim table
    target_location_map = {}
    for target in targets:
        if target.find(":") != -1:
            continue
        location = bp_tal_structs.loc[target].bp_atlas_loc
        target_location_map[target] = location

    stim_table['region'] = stim_table['pair'].map(target_location_map)

    return stim_table
