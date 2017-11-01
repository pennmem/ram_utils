import time
import numpy as np
import operator
import pandas as pd
from scipy import stats
from statsmodels.stats.proportion import proportions_chisquare

from ramutils.utils import safe_divide
from ReportUtils import ReportRamTask
from TexUtils.matrix2latex import matrix2latex
from SessionSummary import FR6SessionSummary, PS4SessionSummary

operator.div = np.true_divide


class ComposeSessionSummary(ReportRamTask):
    PS4 = 'PS'
    BASELINE = 'BASELINE'
    STIM_LIST = 'STIM'
    NONSTIM = 'NON-STIM'

    def __init__(self, params, mark_as_completed=True):
        super(ComposeSessionSummary, self).__init__(mark_as_completed)
        self.params = params

    def run(self):
        fr_stim_table = self.get_passed_object('fr_stim_table')
        self.pass_object('fr_session_summary',[])
        self.compose_ps_fr_session_summary()
        self.compose_fr_session_summary()

    def compose_ps_fr_session_summary(self):
        task = 'FR'
        math_events = self.get_passed_object(task + '_math_events')
        intr_events = self.get_passed_object(task + '_intr_events')

        fr1_intr_events = self.get_passed_object('FR1_intr_events')
        rec_events = self.get_passed_object(task + '_rec_events')
        all_events = self.get_passed_object('all_events')
        monopolar_channels = self.get_passed_object('monopolar_channels')

        fr1_events = self.get_passed_object('FR1_events')

        stim_params_to_sess = self.get_passed_object('stim_params_to_sess')
        fr_stim_table = self.get_passed_object('fr_stim_table')
        fr_stim_table['prev_prob'] = fr_stim_table['prob'].shift(1)
        fr_stim_table['prob_diff'] = fr_stim_table['prob'] - fr_stim_table['prev_prob']

        sessions = sorted(fr_stim_table.session.unique())

        self.pass_object('NUMBER_OF_FR_SESSIONS', len(sessions))
        self.pass_object('NUMBER_OF_ELECTRODES', len(monopolar_channels))

        session_data = []

        fr_stim_table = fr_stim_table.loc[fr_stim_table['is_ps4_session']]
        fr_stim_table_by_session = fr_stim_table.groupby(['session'])
        for session,fr_stim_session_table in fr_stim_table_by_session:
            session_all_events = all_events[all_events.session == session]
            first_time_stamp = session_all_events[session_all_events.type=='INSTRUCT_START'][0].mstime
            timestamps = session_all_events.mstime
            last_time_stamp = np.max(timestamps)
            session_length = '%.2f' % ((last_time_stamp - first_time_stamp) / 60000.0)
            session_date = time.strftime('%d-%b-%Y', time.localtime(last_time_stamp/1000))
            n_lists = len(fr_stim_session_table.list.unique())
            pc_correct_words = 100.0 * fr_stim_session_table.recalled.mean()
            amplitude = fr_stim_session_table['Amplitude'].values[-1]
            session_data.append([session, session_date, session_length, n_lists, '$%.2f$\\%%' % pc_correct_words, amplitude])

        self.pass_object('session_table', session_data)

        session_summary_array = self.get_passed_object('fr_session_summary')
        fr_stim_group_table = fr_stim_table.loc[fr_stim_table['is_ps4_session']]

        fr_stim_group_table_group = fr_stim_group_table.groupby(['stimAnodeTag','stimCathodeTag'])
        for stim_param,fr_stim_session_table in fr_stim_group_table_group:
            print 'Stim param: ',stim_param
            session_summary = FR6SessionSummary()
            session_summary.session = sorted(np.unique(fr_stim_session_table.session))
            session_summary.stimtag = fr_stim_session_table.stimAnodeTag.values[0] + '-' + fr_stim_session_table.stimCathodeTag.values[0]
            session_summary.region_of_interest = fr_stim_session_table.Region.values[0]
            session_summary.frequency = fr_stim_session_table.Pulse_Frequency.values[0]
            session_summary.n_words = len(fr_stim_session_table)
            session_summary.n_correct_words = fr_stim_session_table.recalled.sum()
            session_summary.pc_correct_words = 100*session_summary.n_correct_words / float(session_summary.n_words)
            session_summary.amplitude = fr_stim_session_table['Amplitude'].values[-1]

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
            session_summary.pc_pli = 100*session_summary.n_pli / float(n_sess_rec_events)
            session_summary.n_eli = np.sum(sess_intr_events.intrusion == -1)
            session_summary.pc_eli = 100*session_summary.n_eli / float(n_sess_rec_events)

            session_summary.n_math = len(sess_math_events)
            session_summary.n_correct_math = np.sum(sess_math_events.iscorrect)
            session_summary.pc_correct_math = 100*session_summary.n_correct_math / float(session_summary.n_math)
            session_summary.math_per_list = session_summary.n_math / float(session_summary.n_lists)

            fr_stim_table_by_pos = fr_stim_session_table.groupby('serialpos')
            session_summary.prob_recall = fr_stim_table_by_pos.recalled.mean()
            session_summary.prob_stim_recall = fr_stim_table_by_pos.loc[fr_stim_table_by_pos.is_stim_item==1].recalled.sum().values.astype(np.float)
            session_summary.prob_nostim_recall = fr_stim_table_by_pos.loc[fr_stim_table_by_pos.is_stim_item==0].recalled.sum().values.astype(np.float)
            session_summary.prob_stim = fr_stim_table_by_pos[fr_stim_session_table.is_stim_list==1].is_stim_item.mean().values

            session_summary.prob_stim_recall /= (fr_stim_session_table.is_stim_item==1).sum().astype(np.float)
            session_summary.prob_nostim_recall /= (fr_stim_session_table.is_stim_item==0).sum().astype(np.float)

            print 'session_summary.prob_stim:',session_summary.prob_stim

            session_summary.prob_first_recall = np.zeros(len(fr_stim_table_by_pos), dtype=float)
            session_summary.prob_first_stim_recall = np.zeros(len(fr_stim_table_by_pos), dtype=float)
            session_summary.prob_first_nostim_recall = np.zeros(len(fr_stim_table_by_pos), dtype=float)
            first_recall_counter = np.zeros(len(fr_stim_table_by_pos), dtype=int)
            session_summary.list_number = np.empty(len(fr_stim_table_by_session_list), dtype=int)
            session_summary.n_recalls_per_list = np.empty(len(fr_stim_table_by_session_list), dtype=int)
            session_summary.n_stims_per_list = np.zeros(len(fr_stim_table_by_session_list), dtype=int)
            session_summary.is_stim_list = np.zeros(len(fr_stim_table_by_session_list), dtype=np.bool)
            session_summary.is_baseline_list = np.zeros(len(fr_stim_table_by_session_list), dtype=np.bool)
            session_summary.is_ps_list = np.zeros(len(fr_stim_table_by_session_list), dtype=np.bool)
            session_summary.is_nonstim_list = np.zeros(len(fr_stim_table_by_session_list), dtype=np.bool)

            session_irt_within_cat = []
            session_irt_between_cat = []

            for list_idx, (sess_list,fr_stim_sess_list_table) in enumerate(fr_stim_table_by_session_list):
                session = sess_list[0]
                lst = sess_list[1]

                list_rec_events = rec_events[(rec_events.session==session) & (rec_events['list']==lst) & ~(rec_events['intrusion']>0)]
                if list_rec_events.size > 0:
                    item_nums = fr_stim_sess_list_table.item_name == list_rec_events[0].item_name
                    tmp = np.where(item_nums)[0]
                    if tmp.size > 0:
                        first_recall_idx = tmp[0]
                        if fr_stim_sess_list_table.iloc[tmp[0]].is_stim_item:
                            session_summary.prob_first_stim_recall[first_recall_idx]+=1
                        else:
                            session_summary.prob_first_nostim_recall[first_recall_idx]+=1
                        session_summary.prob_first_recall[first_recall_idx] += 1
                        first_recall_counter[first_recall_idx] += 1

                session_summary.list_number[list_idx] = lst
                session_summary.n_recalls_per_list[list_idx] = fr_stim_sess_list_table.recalled.sum()
                session_summary.n_stims_per_list[list_idx] = fr_stim_sess_list_table.is_stim_item.sum()
                session_summary.is_stim_list[list_idx] = fr_stim_sess_list_table.is_stim_list.any()
                session_summary.is_baseline_list[list_idx] = (fr_stim_sess_list_table['phase']==self.BASELINE).any()
                session_summary.is_ps_list[list_idx] = (fr_stim_sess_list_table['phase'] == self.PS4).any()
                session_summary.is_nonstim_list[list_idx] = (fr_stim_sess_list_table['phase'] == self.NONSTIM).any()

            session_summary.prob_first_recall /= float(len(fr_stim_session_table))
            session_summary.prob_first_stim_recall /= (fr_stim_session_table.is_stim_item==1).sum()
            session_summary.prob_first_nostim_recall /= (fr_stim_session_table.is_stim_item==0).sum()
            fr_stim_stim_list_table = fr_stim_session_table
            fr_stim_non_stim_list_table = pd.DataFrame.from_records([e for e in fr1_events[fr1_events.type=='WORD']],columns=fr1_events.dtype.names)

            all_events_table = pd.DataFrame.from_records([e for e in all_events],columns = all_events.dtype.names)
            lists = np.unique(all_events[all_events.type=='WORD'].list)
            lists = lists[lists>0]
            session_summary.n_recalls_per_list = all_events_table.loc[all_events_table.type=='WORD'].groupby('list').recalled.sum()[lists].values
            session_summary.n_stims_per_list = all_events_table.loc[all_events_table.type=='STIM_ON'].groupby('list').recalled.sum()[lists].values
            session_summary.is_stim_list = all_events_table.loc[all_events_table.type=='WORD'].groupby('list').apply(
                lambda x: (x.phase=='STIM').any())
            session_summary.is_stim_list = session_summary.is_stim_list[lists].values
            session_summary.is_nonstim_list = all_events_table.loc[all_events_table.type == 'WORD'].groupby('list').apply(
                lambda x: (x.phase == 'NON-STIM').any())[lists].values
            session_summary.is_ps_list = all_events_table.loc[all_events_table.type == 'WORD'].groupby('list').apply(
                lambda x: (x.phase == 'PS').any())[lists].values
            session_summary.is_baseline_list = all_events_table.loc[all_events_table.type == 'WORD'].groupby('list').apply(
                lambda x: (x.phase == 'BASELINE').any())[lists].values

            session_summary.n_correct_stim = fr_stim_stim_list_table.recalled.sum()
            session_summary.n_total_stim = len(fr_stim_stim_list_table)
            session_summary.pc_from_stim = 100 * session_summary.n_correct_stim / float(session_summary.n_total_stim)
            session_summary.n_correct_nonstim = fr1_events.recalled.sum()
            session_summary.n_total_nonstim = len(fr1_events)
            session_summary.pc_from_nonstim = 100*session_summary.n_correct_nonstim/float(session_summary.n_total_nonstim)
            last_fr1_session = np.unique(fr1_events.session)[-1]
            fr1_last_sess_events = fr1_events[fr1_events.session == last_fr1_session]
            session_summary.n_correct_nonstim_last = fr1_last_sess_events.recalled.sum()
            session_summary.n_total_nonstim_last = len(fr1_last_sess_events)
            session_summary.chisqr_last, session_summary.pvalue_last, _ = proportions_chisquare(
                [session_summary.n_correct_stim, session_summary.n_correct_nonstim_last],
                [session_summary.n_total_stim, session_summary.n_total_nonstim_last])
            session_summary.last_recall_table = [['', '{}/{} ({:2.2}% from final FR1 list'.format(
                session_summary.n_correct_nonstim_last, session_summary.n_total_nonstim_last,
                session_summary.n_correct_nonstim_last / float(session_summary.n_total_nonstim_last)),
                                                 (session_summary.chisqr_last),
                                                (session_summary.pvalue_last)]]

            session_summary.chisqr, session_summary.pvalue, _ = proportions_chisquare([session_summary.n_correct_stim, session_summary.n_correct_nonstim], [session_summary.n_total_stim, session_summary.n_total_nonstim])

            stim_lists = fr_stim_stim_list_table['list'].unique()
            non_stim_lists = fr_stim_non_stim_list_table['list'].unique()

            session_summary.n_stim_intr = len(sess_intr_events)
            session_summary.n_nonstim_intr = len(fr1_intr_events)
            session_summary.pc_from_stim_intr = 100*session_summary.n_stim_intr / float(session_summary.n_total_stim)
            session_summary.pc_from_nonstim_intr = 100*session_summary.n_nonstim_intr / float(session_summary.n_total_nonstim)

            fr_stim_stim_list_stim_item_table = fr_stim_stim_list_table[fr_stim_stim_list_table['is_stim_item']]
            fr_stim_stim_list_stim_item_low_table = fr_stim_stim_list_stim_item_table[fr_stim_stim_list_stim_item_table['prev_prob']<fr_stim_stim_list_stim_item_table['thresh']]
            fr_stim_stim_list_stim_item_high_table = fr_stim_stim_list_stim_item_table[fr_stim_stim_list_stim_item_table['prev_prob']>fr_stim_stim_list_stim_item_table['thresh']]

            fr_stim_stim_list_post_stim_item_table = fr_stim_stim_list_table[fr_stim_stim_list_table['is_post_stim_item']]
            fr_stim_stim_list_post_stim_item_low_table = fr_stim_stim_list_post_stim_item_table[fr_stim_stim_list_post_stim_item_table['prev_prob']<fr_stim_stim_list_post_stim_item_table['thresh']]
            fr_stim_stim_list_post_stim_item_high_table = fr_stim_stim_list_post_stim_item_table[fr_stim_stim_list_post_stim_item_table['prev_prob']>fr_stim_stim_list_post_stim_item_table['thresh']]

            session_summary.mean_prob_diff_all_stim_item = fr_stim_stim_list_stim_item_table['prob_diff'].mean()
            session_summary.sem_prob_diff_all_stim_item = fr_stim_stim_list_stim_item_table['prob_diff'].sem()
            session_summary.mean_prob_diff_low_stim_item = fr_stim_stim_list_stim_item_low_table['prob_diff'].mean()
            session_summary.sem_prob_diff_low_stim_item = fr_stim_stim_list_stim_item_low_table['prob_diff'].sem()

            session_summary.mean_prob_diff_all_post_stim_item = fr_stim_stim_list_post_stim_item_table['prob_diff'].mean()
            session_summary.sem_prob_diff_all_post_stim_item = fr_stim_stim_list_post_stim_item_table['prob_diff'].sem()
            session_summary.mean_prob_diff_low_post_stim_item = fr_stim_stim_list_post_stim_item_low_table['prob_diff'].mean()
            session_summary.sem_prob_diff_low_post_stim_item = fr_stim_stim_list_post_stim_item_low_table['prob_diff'].sem()

            stim_item_recall_rate = fr_stim_stim_list_stim_item_table['recalled'].mean()
            stim_item_recall_rate_low = fr_stim_stim_list_stim_item_low_table['recalled'].mean()
            stim_item_recall_rate_high = fr_stim_stim_list_stim_item_high_table['recalled'].mean()

            post_stim_item_recall_rate = fr_stim_stim_list_post_stim_item_table['recalled'].mean()
            post_stim_item_recall_rate_low = fr_stim_stim_list_post_stim_item_low_table['recalled'].mean()
            post_stim_item_recall_rate_high = fr_stim_stim_list_post_stim_item_high_table['recalled'].mean()

            probs = self.get_passed_object('xval_output')[-1].probs
            threshold = self.get_passed_object('xval_output')[-1].jstat_thresh
            is_low_item = probs<threshold
            is_post_low_item = np.append([0],is_low_item[1:])

            non_stim_list_recall_rate_low = fr1_events[is_low_item].recalled.mean()
            non_stim_list_recall_rate_post_low = fr1_events[is_post_low_item].recalled.mean()
            non_stim_list_recall_rate_high = fr1_events[~is_low_item].recalled.mean()

            recall_rate = session_summary.n_correct_words / float(session_summary.n_words)

            stim_pc_diff_from_mean = 100.0 * (stim_item_recall_rate-non_stim_list_recall_rate_low) / recall_rate
            post_stim_pc_diff_from_mean = 100.0 * (post_stim_item_recall_rate-non_stim_list_recall_rate_post_low) / recall_rate
            session_summary.pc_diff_from_mean = (stim_pc_diff_from_mean, post_stim_pc_diff_from_mean)

            session_summary.n_correct_stim_items = fr_stim_stim_list_stim_item_table['recalled'].sum()
            session_summary.n_total_stim_items = len(fr_stim_stim_list_stim_item_table)
            session_summary.pc_stim_items = 100*session_summary.n_correct_stim_items / float(session_summary.n_total_stim_items)

            session_summary.n_correct_post_stim_items = fr_stim_stim_list_post_stim_item_table['recalled'].sum()
            session_summary.n_total_post_stim_items = len(fr_stim_stim_list_post_stim_item_table)
            session_summary.pc_post_stim_items = 100*session_summary.n_correct_post_stim_items / float(session_summary.n_total_post_stim_items)

            session_summary.n_correct_nonstim_low_bio_items = fr1_events[probs<threshold].recalled.sum()
            session_summary.n_total_nonstim_low_bio_items = len(fr1_events)
            session_summary.pc_nonstim_low_bio_items = 100*session_summary.n_correct_nonstim_low_bio_items / float(session_summary.n_total_nonstim_low_bio_items)

            nonstim_post_low_items = fr1_events[np.append([0],(probs<threshold)[:-1])]
            session_summary.n_correct_nonstim_post_low_bio_items = nonstim_post_low_items['recalled'].sum()
            session_summary.n_total_nonstim_post_low_bio_items = len(nonstim_post_low_items)
            session_summary.pc_nonstim_post_low_bio_items = 100*session_summary.n_correct_nonstim_post_low_bio_items / float(session_summary.n_total_nonstim_post_low_bio_items)

            session_summary.chisqr_stim_item, session_summary.pvalue_stim_item, _ = proportions_chisquare([session_summary.n_correct_stim_items, session_summary.n_correct_nonstim_low_bio_items], [session_summary.n_total_stim_items, session_summary.n_total_nonstim_low_bio_items])
            session_summary.chisqr_post_stim_item, session_summary.pvalue_post_stim_item, _ = proportions_chisquare([session_summary.n_correct_post_stim_items, session_summary.n_correct_nonstim_post_low_bio_items], [session_summary.n_total_post_stim_items, session_summary.n_total_nonstim_post_low_bio_items])

            if (fr_stim_session_table['recognized'] != -999).any():
                session_summary.n_stim_hits = (fr_stim_session_table[fr_stim_session_table['is_stim_item']].recognized>0).sum()
                session_summary.n_nonstim_hits = (fr_stim_session_table[~fr_stim_session_table['is_stim_item']].recognized>0).sum()
                session_summary.n_false_alarms = (fr_stim_session_table.rejected==0).sum()
                session_summary.pc_stim_hits = 100*session_summary.n_stim_hits/float(len(fr_stim_session_table))
                session_summary.pc_nonstim_hits = 100*session_summary.n_nonstim_hits/float(len(fr_stim_session_table))
                session_summary.pc_false_alarms = 100*session_summary.n_false_alarms/float(len(fr_stim_session_table))
                session_summary.dprime = (stats.norm.ppf(session_summary.pc_stim_hits+session_summary.pc_nonstim_hits) -
                                          stats.norm.ppf(session_summary.pc_false_alarms/100.))
                if np.isnan(session_summary.dprime):
                    session_summary.dprime = -999.


            session_summary_array.append(session_summary)
        self.pass_object('fr_session_summary', session_summary_array)

    def compose_ps_session_summary(self):
        events = self.get_passed_object('ps_events')
        if not len(events):
            self.pass_object('ps_session_data', False)
            self.pass_object('ps_session_summary', False)
        else:

            events = pd.DataFrame.from_records([e for e in events],columns=events.dtype.names)

            ps_session_summaries = {}
            session_data = []
            for session, sess_events in events.groupby('session'):

                first_time_stamp = sess_events.ix[0].mstime
                timestamps = sess_events.mstime
                last_time_stamp = np.max(timestamps)
                session_length = '%.2f' % ((last_time_stamp - first_time_stamp) / 60000.0)
                session_date = time.strftime('%d-%b-%Y', time.localtime(last_time_stamp/1000))
                n_lists = len(sess_events.list.unique())

                session_data.append([session, session_date, session_length, n_lists])

                session_summary = PS4SessionSummary()
                for (i,(location,loc_events)) in enumerate(sess_events.groupby(['anode_label','cathode_label'])):
                    if location[0] and location[1]:
                        session_summary.locations.append('%s-%s'%(location[0],location[1]))
                        opts = loc_events.loc[loc_events['type']=='OPTIMIZATION']
                        session_summary.amplitudes.append(opts['amplitude'].values)
                        session_summary.delta_classifiers.append(opts['delta_classifier'].values)
                        if (loc_events['type']=='OPTIMIZATION_DECISION').any():
                            decision_event = loc_events.loc[loc_events['type']=='OPTIMIZATION_DECISION']
                            session_summary.preferred_location = decision_event['location']
                            session_summary.preferred_amplitude = decision_event['amplitude']
                            session_summary.tstat = decision_event['tstat']
                            session_summary.pvalue = decision_event['pvalue']
                ps_session_summaries[session] = session_summary

            self.pass_object('ps_session_data',session_data)
            self.pass_object('ps_session_summary',ps_session_summaries)

    def compose_fr_session_summary(self):
        task = 'FR'
        math_events = self.get_passed_object(task + '_math_events')
        intr_events = self.get_passed_object(task + '_intr_events')
        rec_events = self.get_passed_object(task + '_rec_events')
        all_events = self.get_passed_object('all_events')
        monopolar_channels = self.get_passed_object('monopolar_channels')

        fr1_events = self.get_passed_object('FR1_events')

        stim_params_to_sess = self.get_passed_object('stim_params_to_sess')
        fr_stim_table = self.get_passed_object('fr_stim_table')
        fr_stim_table['prev_prob'] = fr_stim_table['prob'].shift(1)
        fr_stim_table['prob_diff'] = fr_stim_table['prob'] - fr_stim_table['prev_prob']
        fr_stim_table = fr_stim_table.loc[~fr_stim_table['is_ps4_session']]

        sessions = sorted(fr_stim_table.session.unique())

        self.pass_object('NUMBER_OF_FR_SESSIONS', len(sessions))
        self.pass_object('NUMBER_OF_ELECTRODES', len(monopolar_channels))

        session_data = []

        # TODO: This should be done by stim target pairs, i.e. if multiple multi-site targets are 
        # chosen, then we should calculate this table for each set of sessions corresponding
        stim_targets = []
        fr_stim_table_by_session = fr_stim_table.groupby(['session'])
        for session,fr_stim_session_table in fr_stim_table_by_session:
            session_all_events = all_events[all_events.session == session]
            timestamps = session_all_events.mstime
            first_time_stamp = timestamps.min()
            last_time_stamp = np.max(timestamps)
            session_length = '%.2f' % ((last_time_stamp - first_time_stamp) / 60000.0)
            session_date = time.strftime('%d-%b-%Y', time.localtime(last_time_stamp/1000))
            n_lists = len(fr_stim_session_table.list.unique())
            pc_correct_words = 100.0 * fr_stim_session_table.recalled.mean()
            amplitude = fr_stim_session_table['Amplitude'].values[-1]

            session_data.append([session, session_date, session_length, n_lists, '$%.2f$\\%%' % pc_correct_words, amplitude])

        self.pass_object('session_table', session_data)

        session_summary_array = self.get_passed_object('fr_session_summary')
        fr_stim_table_by_phase = fr_stim_table.loc[~fr_stim_table['is_ps4_session']]

    
        fr_stim_group_table_group = fr_stim_table_by_phase.groupby(['session'])
        for session, fr_stim_session_table in fr_stim_group_table_group:
            # Session-level summary information
            session_summary = FR6SessionSummary()
            session_summary.session = sorted(fr_stim_session_table.session.unique())

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
            session_summary.pc_pli = 100*session_summary.n_pli / float(n_sess_rec_events)
            session_summary.n_eli = np.sum(sess_intr_events.intrusion == -1)
            session_summary.pc_eli = 100*session_summary.n_eli / float(n_sess_rec_events)

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
            session_summary.n_nonstim_intr = (fr1_events.intrusion==1).sum()
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
                target = "-".join(target)

                # Target summary info
                session_summary.stimtag[target] = target
                session_summary.region_of_interest[target] = fr_stim_target_table.Region.values[0] # both channels will be in the same region
                session_summary.frequency[target] = fr_stim_target_table.Pulse_Frequency.values[0]
                session_summary.amplitude[target] = fr_stim_target_table.Amplitude.values[0]
                
                # Probability of recall and probability of first recall by list type, i.e. target
                fr_stim_table_by_pos = fr_stim_target_table.groupby('serialpos')
                session_summary.prob_recall[target] = fr_stim_target_table.groupby('serialpos').recalled.mean()
                session_summary.prob_stim_recall[target] = fr_stim_target_table.loc[fr_stim_target_table.is_stim_item==True].groupby('serialpos').recalled.mean()
                session_summary.prob_nostim_recall[target] = fr_stim_target_table.loc[fr_stim_target_table.is_stim_item==False].groupby('serialpos').recalled.mean()
                session_summary.prob_stim[target] = fr_stim_target_table.loc[fr_stim_target_table['is_stim_list']==True].groupby('serialpos').is_stim_item.mean().values

                session_summary.prob_first_recall[target] = np.zeros(len(fr_stim_table_by_pos), dtype=float)
                session_summary.prob_first_stim_recall[target] = np.zeros(len(fr_stim_table_by_pos), dtype=float)
                session_summary.prob_first_nostim_recall[target] = np.zeros(len(fr_stim_table_by_pos), dtype=float)
                
                first_recall_counter = np.zeros(len(fr_stim_table_by_pos), dtype=int)

                fr_stim_table_by_session_list = fr_stim_target_table.groupby(['session','list'])
                session_summary.list_number[target] = np.empty(len(fr_stim_table_by_session_list), dtype=int)
                session_summary.n_recalls_per_list[target] = np.empty(len(fr_stim_table_by_session_list), dtype=int)
                session_summary.n_stims_per_list[target] = np.zeros(len(fr_stim_table_by_session_list), dtype=int)
                session_summary.is_stim_list[target] = np.zeros(len(fr_stim_table_by_session_list), dtype=np.bool)
                session_summary.is_nonstim_list[target] = np.zeros(len(fr_stim_table_by_session_list), dtype=np.bool)
                session_summary.is_baseline_list[target] = np.zeros(len(fr_stim_table_by_session_list), dtype=np.bool)
                session_summary.is_ps_list[target] = np.zeros(len(fr_stim_table_by_session_list), dtype=np.bool)

                session_irt_within_cat = []
                session_irt_between_cat = []

                for list_idx, (sess_list,fr_stim_sess_list_table) in enumerate(fr_stim_table_by_session_list):
                    session = sess_list[0]
                    lst = sess_list[1]
                    list_rec_events = rec_events[(rec_events.session==session) & (rec_events['list']==lst) & (rec_events['intrusion']==0)]
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
                    session_summary.is_baseline_list[target][list_idx] = (fr_stim_sess_list_table['phase']==self.BASELINE).any()
                    session_summary.is_ps_list[target][list_idx] = (fr_stim_sess_list_table['phase'] == self.PS4).any()
                    session_summary.is_nonstim_list[target][list_idx] = (fr_stim_sess_list_table['phase'] == self.NONSTIM).any()

                session_summary.prob_first_recall[target] /= float(len(fr_stim_table_by_session_list))
                session_summary.prob_first_stim_recall[target] /= float(len(fr_stim_stim_item_table_by_session_list))
                session_summary.prob_first_nostim_recall[target] /= float(len(fr_stim_nostim_item_table_by_session_list))
                
                fr_stim_stim_list_table = fr_stim_target_table.loc[fr_stim_target_table.is_stim_list==True]
                session_summary.n_correct_stim[target] = fr_stim_stim_list_table.recalled.sum()
                session_summary.n_total_stim[target] = len(fr_stim_stim_list_table)

                # Could have division by zero in the case of the no stim target
                try:
                    session_summary.pc_from_stim[target] = 100 * session_summary.n_correct_stim[target] / float(session_summary.n_total_stim[target])
                except ZeroDivisionError:
                    print("Zero division for target: %s" % target)

                session_summary.chisqr[target], session_summary.pvalue[target], _ = proportions_chisquare([session_summary.n_correct_stim[target], 
                                                                                                           session_summary.n_correct_nonstim],
                                                                                                           [session_summary.n_total_stim[target],
                                                                                                            session_summary.n_total_nonstim])

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
                fr_stim_stim_list_stim_item_high_table = fr_stim_stim_list_stim_item_table.loc[fr_stim_stim_list_stim_item_table['prev_prob']>fr_stim_stim_list_stim_item_table['thresh']]

                fr_stim_stim_list_post_stim_item_table = fr_stim_stim_list_table.loc[fr_stim_stim_list_table['is_post_stim_item']]
                fr_stim_stim_list_post_stim_item_low_table = fr_stim_stim_list_post_stim_item_table.loc[fr_stim_stim_list_post_stim_item_table['prev_prob']<fr_stim_stim_list_post_stim_item_table['thresh']]
                fr_stim_stim_list_post_stim_item_high_table = fr_stim_stim_list_post_stim_item_table.loc[fr_stim_stim_list_post_stim_item_table['prev_prob']>fr_stim_stim_list_post_stim_item_table['thresh']]

                session_summary.mean_prob_diff_all_stim_item[target] = fr_stim_stim_list_stim_item_table['prob_diff'].mean()
                session_summary.sem_prob_diff_all_stim_item[target] = fr_stim_stim_list_stim_item_table['prob_diff'].sem()
                session_summary.mean_prob_diff_low_stim_item[target] = fr_stim_stim_list_stim_item_low_table['prob_diff'].mean()
                session_summary.sem_prob_diff_low_stim_item[target] = fr_stim_stim_list_stim_item_low_table['prob_diff'].sem()

                session_summary.mean_prob_diff_all_post_stim_item[target] = fr_stim_stim_list_post_stim_item_table['prob_diff'].mean()
                session_summary.sem_prob_diff_all_post_stim_item[target] = fr_stim_stim_list_post_stim_item_table['prob_diff'].sem()
                session_summary.mean_prob_diff_low_post_stim_item[target] = fr_stim_stim_list_post_stim_item_low_table['prob_diff'].mean()
                session_summary.sem_prob_diff_low_post_stim_item[target] = fr_stim_stim_list_post_stim_item_low_table['prob_diff'].sem()

                stim_item_recall_rate, stim_item_recall_rate_low, stim_item_recall_rate_high = {}, {}, {}
                stim_item_recall_rate[target] = fr_stim_stim_list_stim_item_table['recalled'].mean()
                stim_item_recall_rate_low[target] = fr_stim_stim_list_stim_item_low_table['recalled'].mean()
                stim_item_recall_rate_high[target] = fr_stim_stim_list_stim_item_high_table['recalled'].mean()

                post_stim_item_recall_rate, post_stim_item_recall_rate_low, post_stim_item_recall_rate_high = {}, {}, {}
                post_stim_item_recall_rate[target] = fr_stim_stim_list_post_stim_item_table['recalled'].mean()
                post_stim_item_recall_rate_low[target] = fr_stim_stim_list_post_stim_item_low_table['recalled'].mean()
                post_stim_item_recall_rate_high[target] = fr_stim_stim_list_post_stim_item_high_table['recalled'].mean()

                recall_rate = session_summary.n_correct_words / float(session_summary.n_words)

                stim_pc_diff_from_mean, post_stim_pc_diff_from_mean = {}, {}
                stim_pc_diff_from_mean[target] = 100.0 * (stim_item_recall_rate[target] - non_stim_list_recall_rate_low) / recall_rate
                post_stim_pc_diff_from_mean[target] = 100.0 * (post_stim_item_recall_rate[target] - non_stim_list_recall_rate_post_low) / recall_rate
                session_summary.pc_diff_from_mean[target] = (stim_pc_diff_from_mean[target], post_stim_pc_diff_from_mean[target])

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
        self.pass_object('fr_session_summary', session_summary_array)
        assert 1 == 0
