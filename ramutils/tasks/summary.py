"""Tasks related to summarizing an experiment. Used primarily in reporting
results.

"""

from itertools import chain
import time

import numpy as np
import pandas as pd
from statsmodels.stats.proportion import proportions_chisquare

from ._wrapper import task
from ramutils.log import get_logger
from ramutils.utils import (
    combine_tag_names, sanitize_comma_sep_list, safe_divide, join_tag_tuple
)

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


@task()
def construct_session_table(all_events, stim_table):
    """Constructs a basic summary of sessions including date, length,
    number of lists, etc.

    Parameters
    ----------
    all_events : np.recarray
    stim_table

    Returns
    -------
    List of session info.

    """
    session_data = []

    # TODO: This should be done by stim target pairs, i.e. if multiple
    # multi-site targets are chosen, then we should calculate this table for
    # each set of sessions corresponding
    fr_stim_table_by_session = stim_table.groupby(['session'])
    for session, fr_stim_session_table in fr_stim_table_by_session:
        session_all_events = all_events[all_events.session == session]
        timestamps = session_all_events.mstime
        first_time_stamp = timestamps.min()
        last_time_stamp = np.max(timestamps)
        session_length = '%.2f' % ((last_time_stamp - first_time_stamp) / 60000.0)
        session_date = time.strftime('%d-%b-%Y', time.localtime(last_time_stamp/1000))
        n_lists = len(fr_stim_session_table.list.unique())
        pc_correct_words = 100.0 * fr_stim_session_table.recalled.mean()

        # There should be an easier way to get this set of unique amplitudes
        # excluding 0 and nan
        all_amplitudes = fr_stim_session_table['amplitude'].unique()
        all_amplitudes = [amp for amp in all_amplitudes if amp > 0]
        all_amplitudes = [str(amp).split(",") for amp in all_amplitudes]
        all_amplitudes = list(chain.from_iterable(all_amplitudes))
        all_amplitudes = [el for el in set(all_amplitudes) if el != "0"]
        amplitude = ",".join(all_amplitudes)
        session_data.append([session, session_date, session_length, n_lists, '$%.2f$\\%%' % pc_correct_words, amplitude])

    return session_data


def summarize_session(experiment, all_events, math_events, intr_events,
                      rec_events, record_only_events, stim_table):
    """Summarize a single session.

    Parameters
    ----------
    experiment
    all_events
    math_events
    intr_events
    rec_events
    record_only_events : np.recarray
        Events from associated record-only experiments.
    stim_table

    Returns
    -------

    """
    stim_table['prev_prob'] = stim_table['prob'].shift(1)
    stim_table['prob_diff'] = stim_table['prob'] - stim_table['prev_prob']
    stim_table = stim_table.loc[~stim_table['is_ps4_session']]

    session_summary_array = self.get_passed_object('fr_session_summary')
    fr_stim_table_by_phase = stim_table.loc[~stim_table['is_ps4_session']]

    fr_stim_group_table_group = fr_stim_table_by_phase.groupby(['session'])
    for session, fr_stim_session_table in fr_stim_group_table_group:
        # Session-level summary information
        session_summary = FR6SessionSummary()
        session_summary.session = session

        session_summary.n_words = len(fr_stim_session_table)
        session_summary.n_correct_words = fr_stim_session_table.recalled.sum()
        session_summary.pc_correct_words = 100*session_summary.n_correct_words / float(session_summary.n_words)

        sess_sel = lambda x: np.in1d(x,session_summary.session)
        sess_rec_events = rec_events[sess_sel(rec_events.session)]
        n_sess_rec_events = fr_stim_session_table.recalled.sum()
        sess_intr_events = intr_events[sess_sel(intr_events.session)]
        sess_math_events = math_events[sess_sel(math_events.session)]

        fr_stim_table_by_session_list = fr_stim_session_table.groupby(['session','list'])
        fr_stim_stim_item_table_by_session_list = fr_stim_session_table.loc[fr_stim_session_table.is_stim_item==True].groupby(['session','list'])
        fr_stim_nostim_item_table_by_session_list = fr_stim_session_table.loc[fr_stim_session_table.is_stim_item==False].groupby(['session','list'])
        session_summary.n_lists = len(fr_stim_table_by_session_list)

        session_summary.n_pli = np.sum(sess_intr_events.intrusion > 0)
        session_summary.pc_pli = 100*session_summary.n_pli / float(session_summary.n_words)
        session_summary.n_eli = np.sum(sess_intr_events.intrusion == -1)
        session_summary.pc_eli = 100*session_summary.n_eli / float(session_summary.n_words)

        session_summary.n_math = len(sess_math_events)
        session_summary.n_correct_math = np.sum(sess_math_events.iscorrect)
        session_summary.pc_correct_math = 100*session_summary.n_correct_math / float(session_summary.n_math)
        session_summary.math_per_list = session_summary.n_math / float(session_summary.n_lists)

        # Calculate nonstim recall information outside of target-specific info
        fr_non_stim_list_table = fr_stim_session_table[~fr_stim_session_table.is_stim_list & (fr_stim_session_table['list']>=4)]
        session_summary.n_correct_nonstim = fr_non_stim_list_table.recalled.sum()
        session_summary.n_total_nonstim = len(fr_non_stim_list_table)
        session_summary.pc_from_nonstim = 100 * session_summary.n_correct_nonstim / float(session_summary.n_total_nonstim)

        # Calculate non stim intrusions once per session
        non_stim_lists = fr_non_stim_list_table['list'].unique()
        session_summary.n_nonstim_intr = (record_only_events.intrusion==1).sum()
        session_summary.pc_from_nonstim_intr = 100*session_summary.n_nonstim_intr / float(session_summary.n_total_nonstim)

        # Non-stim recall rates can be calculated once per session
        low_state_mask = (fr_non_stim_list_table['prob'] < fr_non_stim_list_table['thresh'])
        post_low_state_mask = low_state_mask.shift(1).fillna(False)
        post_low_state_mask[fr_non_stim_list_table['serialpos']==1] = False
        fr_non_stim_list_low_table = fr_non_stim_list_table[low_state_mask]
        fr_non_stim_list_post_low_table = fr_non_stim_list_table[post_low_state_mask]
        fr_non_stim_list_high_table = fr_non_stim_list_table[fr_non_stim_list_table['prob']>fr_non_stim_list_table['thresh']]

        non_stim_list_recall_rate_low = fr_non_stim_list_low_table['recalled'].mean()
        non_stim_list_recall_rate_post_low = fr_non_stim_list_post_low_table['recalled'].mean()
        non_stim_list_recall_rate_high = fr_non_stim_list_high_table['recalled'].mean()

        session_summary.n_correct_nonstim_low_bio_items = fr_non_stim_list_low_table['recalled'].sum()
        session_summary.n_total_nonstim_low_bio_items = len(fr_non_stim_list_low_table)
        session_summary.pc_nonstim_low_bio_items = safe_divide(100*session_summary.n_correct_nonstim_low_bio_items, float(session_summary.n_total_nonstim_low_bio_items))

        session_summary.n_correct_nonstim_post_low_bio_items = fr_non_stim_list_post_low_table['recalled'].sum()
        session_summary.n_total_nonstim_post_low_bio_items = len(fr_non_stim_list_post_low_table)
        session_summary.pc_nonstim_post_low_bio_items = safe_divide(100*session_summary.n_correct_nonstim_post_low_bio_items, float(session_summary.n_total_nonstim_post_low_bio_items))

        session_summary.control_mean_prob_diff_all = fr_non_stim_list_table['prob_diff'].mean()
        session_summary.control_sem_prob_diff_all = fr_non_stim_list_table['prob_diff'].sem()
        session_summary.control_mean_prob_diff_low = fr_non_stim_list_low_table['prob_diff'].mean()
        session_summary.control_sem_prob_diff_low = fr_non_stim_list_low_table['prob_diff'].sem()

        # List-type level information, i.e. target A, target B, target A+B, nostim
        fr_stim_target_group = fr_stim_session_table.groupby(by=['stimAnodeTag', 'stimCathodeTag'])
        for target, fr_stim_target_table in fr_stim_target_group:
            # self.pass_object("sample_target_table_{}".format(target), fr_stim_target_table)
            target = join_tag_tuple(target)

            # Target summary info
            session_summary.stimtag[target] = target
            session_summary.region_of_interest[target] = ""
            session_summary.region_of_interest[target] = fr_stim_target_table.region.unique()[0]
            session_summary.frequency[target] = fr_stim_target_table.pulse_freq.values[0]
            session_summary.amplitude[target] = fr_stim_target_table.amplitude.values[0]

            # Probability of recall and probability of first recall by list
            # type, i.e. target
            fr_stim_table_by_pos = fr_stim_target_table.groupby('serialpos')
            session_summary.prob_recall[target] = fr_stim_target_table.groupby('serialpos').recalled.mean()
            session_summary.prob_stim_recall[target] = fr_stim_target_table.loc[fr_stim_target_table.is_stim_item == True].groupby('serialpos').recalled.mean()
            session_summary.prob_nostim_recall[target] = fr_stim_target_table.loc[fr_stim_target_table.is_stim_item == False].groupby('serialpos').recalled.mean()
            session_summary.prob_stim[target] = fr_stim_target_table.loc[fr_stim_target_table['is_stim_list'] == True].groupby('serialpos').is_stim_item.mean().values

            session_summary.prob_first_recall[target] = np.zeros(len(fr_stim_table_by_pos), dtype=float)
            session_summary.prob_first_stim_recall[target] = np.zeros(len(fr_stim_table_by_pos), dtype=float)
            session_summary.prob_first_nostim_recall[target] = np.zeros(len(fr_stim_table_by_pos), dtype=float)

            first_recall_counter = np.zeros(len(fr_stim_table_by_pos), dtype=int)

            fr_stim_table_by_session_list = fr_stim_target_table.groupby(['session', 'list'])
            session_summary.list_number[target] = np.empty(len(fr_stim_table_by_session_list), dtype=int)
            session_summary.n_recalls_per_list[target] = np.empty(len(fr_stim_table_by_session_list), dtype=int)
            session_summary.n_stims_per_list[target] = np.zeros(len(fr_stim_table_by_session_list), dtype=int)
            session_summary.is_stim_list[target] = np.zeros(len(fr_stim_table_by_session_list), dtype=np.bool)
            session_summary.is_nonstim_list[target] = np.zeros(len(fr_stim_table_by_session_list), dtype=np.bool)
            session_summary.is_baseline_list[target] = np.zeros(len(fr_stim_table_by_session_list), dtype=np.bool)
            session_summary.is_ps_list[target] = np.zeros(len(fr_stim_table_by_session_list), dtype=np.bool)

            for list_idx, (sess_list,fr_stim_sess_list_table) in enumerate(fr_stim_table_by_session_list):
                session = sess_list[0]
                lst = sess_list[1]
                list_rec_events = rec_events[(rec_events.session == session) & (rec_events['list'] == lst) & (rec_events['intrusion'] == 0)]
                if list_rec_events.size > 0:
                    item_nums = fr_stim_sess_list_table.item_name.values == list_rec_events[0].item_name
                    tmp = np.where(item_nums)[0]
                    if tmp.size > 0:
                        first_recall_idx = tmp[0]
                        if fr_stim_sess_list_table.iloc[tmp[0]].is_stim_item:
                            session_summary.prob_first_stim_recall[target][first_recall_idx]+=1
                        else:
                            session_summary.prob_first_nostim_recall[target][first_recall_idx]+=1
                        session_summary.prob_first_recall[target][first_recall_idx] += 1
                        first_recall_counter[first_recall_idx] += 1

                session_summary.list_number[target][list_idx] = lst
                session_summary.n_recalls_per_list[target][list_idx] = fr_stim_sess_list_table.recalled.sum()
                session_summary.n_stims_per_list[target][list_idx] = fr_stim_sess_list_table.is_stim_item.sum()
                session_summary.is_stim_list[target][list_idx] = fr_stim_sess_list_table.is_stim_list.any()
                session_summary.is_baseline_list[target][list_idx] = (fr_stim_sess_list_table['phase'] == 'BASELINE').any()
                session_summary.is_ps_list[target][list_idx] = (fr_stim_sess_list_table['phase'] == 'PS4').any()
                session_summary.is_nonstim_list[target][list_idx] = (fr_stim_sess_list_table['phase'] == 'NON-STIM').any()

            session_summary.prob_first_recall[target] /= float(len(fr_stim_table_by_session_list))
            session_summary.prob_first_stim_recall[target] /= float(len(fr_stim_stim_item_table_by_session_list))
            session_summary.prob_first_nostim_recall[target] /= float(len(fr_stim_nostim_item_table_by_session_list))

            fr_stim_stim_list_table = fr_stim_target_table.loc[fr_stim_target_table.is_stim_list == True]
            session_summary.n_correct_stim[target] = fr_stim_stim_list_table.recalled.sum()
            session_summary.n_total_stim[target] = len(fr_stim_stim_list_table)

            # Could have division by zero in the case of the no stim target
            try:
                session_summary.pc_from_stim[target] = 100 * session_summary.n_correct_stim[target] / float(session_summary.n_total_stim[target])
            except ZeroDivisionError:
                print("Zero division for target: %s" % target)

            session_summary.chisqr[target], session_summary.pvalue[target], _ = (
                proportions_chisquare([session_summary.n_correct_stim[target],
                                       session_summary.n_correct_nonstim],
                                      [session_summary.n_total_stim[target],
                                       session_summary.n_total_nonstim])
            )

            stim_lists = fr_stim_stim_list_table['list'].unique()
            session_summary.n_stim_intr[target] = 0
            for ev in sess_intr_events:
                if ev.intrusion in stim_lists:
                    session_summary.n_stim_intr[target] += 1

            # Could have division by zero in the case of the no stim target
            try:
                session_summary.pc_from_stim_intr[target] = 100*session_summary.n_stim_intr[target] / float(session_summary.n_total_stim[target])
            except ZeroDivisionError:
                print("Zero division for target: %s" % target)

            fr_stim_stim_list_stim_item_table = fr_stim_stim_list_table[fr_stim_stim_list_table.is_stim_item.values.astype(bool)]
            fr_stim_stim_list_stim_item_low_table = fr_stim_stim_list_stim_item_table.loc[fr_stim_stim_list_stim_item_table['prev_prob']<fr_stim_stim_list_stim_item_table['thresh']]

            fr_stim_stim_list_post_stim_item_table = fr_stim_stim_list_table.loc[fr_stim_stim_list_table['is_post_stim_item']]
            fr_stim_stim_list_post_stim_item_low_table = fr_stim_stim_list_post_stim_item_table.loc[fr_stim_stim_list_post_stim_item_table['prev_prob']<fr_stim_stim_list_post_stim_item_table['thresh']]

            session_summary.mean_prob_diff_all_stim_item[target] = fr_stim_stim_list_stim_item_table['prob_diff'].mean()
            session_summary.sem_prob_diff_all_stim_item[target] = fr_stim_stim_list_stim_item_table['prob_diff'].sem()
            session_summary.mean_prob_diff_low_stim_item[target] = fr_stim_stim_list_stim_item_low_table['prob_diff'].mean()
            session_summary.sem_prob_diff_low_stim_item[target] = fr_stim_stim_list_stim_item_low_table['prob_diff'].sem()

            session_summary.mean_prob_diff_all_post_stim_item[target] = fr_stim_stim_list_post_stim_item_table['prob_diff'].mean()
            session_summary.sem_prob_diff_all_post_stim_item[target] = fr_stim_stim_list_post_stim_item_table['prob_diff'].sem()
            session_summary.mean_prob_diff_low_post_stim_item[target] = fr_stim_stim_list_post_stim_item_low_table['prob_diff'].mean()
            session_summary.sem_prob_diff_low_post_stim_item[target] = fr_stim_stim_list_post_stim_item_low_table['prob_diff'].sem()

            stim_item_recall_rate = fr_stim_stim_list_stim_item_table['recalled'].mean()
            post_stim_item_recall_rate = fr_stim_stim_list_post_stim_item_table['recalled'].mean()

            recall_rate = session_summary.n_correct_words / float(session_summary.n_words)
            stim_pc_diff_from_mean = 100.0 * (stim_item_recall_rate - non_stim_list_recall_rate_low) / recall_rate
            post_stim_pc_diff_from_mean = 100.0 * (post_stim_item_recall_rate - non_stim_list_recall_rate_post_low) / recall_rate
            session_summary.pc_diff_from_mean[target] = (stim_pc_diff_from_mean, post_stim_pc_diff_from_mean)

            session_summary.n_correct_stim_items[target] = fr_stim_stim_list_stim_item_table['recalled'].sum()
            session_summary.n_total_stim_items[target] = len(fr_stim_stim_list_stim_item_table)

            # Could have division by zero in the case of the no stim target
            try:
                session_summary.pc_stim_items[target] = 100*session_summary.n_correct_stim_items[target] / float(session_summary.n_total_stim_items[target])
            except ZeroDivisionError:
                print("Zero division for target: %s" % target)

            session_summary.n_correct_post_stim_items[target] = fr_stim_stim_list_post_stim_item_table['recalled'].sum()
            session_summary.n_total_post_stim_items[target] = len(fr_stim_stim_list_post_stim_item_table)
            # Could have division by zero in the case of the no stim target
            try:
                session_summary.pc_post_stim_items[target] = 100*session_summary.n_correct_post_stim_items[target] / float(session_summary.n_total_post_stim_items[target])
            except ZeroDivisionError:
                print("Zero division for target: %s" % target)

            session_summary.chisqr_stim_item[target], session_summary.pvalue_stim_item[target], _ = proportions_chisquare(
                [session_summary.n_correct_stim_items[target], session_summary.n_correct_nonstim_low_bio_items],
                [session_summary.n_total_stim_items[target], session_summary.n_total_nonstim_low_bio_items])

            session_summary.chisqr_post_stim_item[target], session_summary.pvalue_post_stim_item[target], _ = proportions_chisquare(
                [session_summary.n_correct_post_stim_items[target], session_summary.n_correct_nonstim_post_low_bio_items],
                [session_summary.n_total_post_stim_items[target], session_summary.n_total_nonstim_post_low_bio_items])

        session_summary_array.append(session_summary)

    return session_summary_array
