import numpy as np
from RamPipeline import *
from SessionSummary import FR5SessionSummary,PS4SessionSummary

import numpy as np
import time

from statsmodels.stats.proportion import proportions_chisquare



from ReportUtils import ReportRamTask
import operator
import pandas as pd
from scipy import stats
from  TexUtils.matrix2latex import matrix2latex
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
            session_summary = FR5SessionSummary()


            session_summary.sessions = sorted(np.unique(fr_stim_session_table.session))
            session_summary.stimtag = fr_stim_session_table.stimAnodeTag.values[0] + '-' + fr_stim_session_table.stimCathodeTag.values[0]
            session_summary.region_of_interest = fr_stim_session_table.Region.values[0]
            session_summary.frequency = fr_stim_session_table.Pulse_Frequency.values[0]
            session_summary.n_words = len(fr_stim_session_table)
            session_summary.n_correct_words = fr_stim_session_table.recalled.sum()
            session_summary.pc_correct_words = 100*session_summary.n_correct_words / float(session_summary.n_words)
            session_summary.amplitude = fr_stim_session_table['Amplitude'].values[-1]

            sess_sel = lambda x: np.in1d(x,session_summary.sessions)
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
            # session_summary.prob_stim_recall = fr_stim_session_table.loc[fr_stim_session_table.is_stim_item==1].groupby('serialpos').recalled.sum().values.astype(np.float)
            # session_summary.prob_nostim_recall = fr_stim_session_table.loc[fr_stim_session_table.is_stim_item==0].groupby('serialpos').recalled.sum().values.astype(np.float)
            session_summary.prob_stim = fr_stim_table_by_pos[fr_stim_session_table.is_stim_list==1].is_stim_item.mean().values

            session_summary.prob_stim_recall /= (fr_stim_session_table.is_stim_item==1).sum().astype(np.float)
            session_summary.prob_nostim_recall /= (fr_stim_session_table.is_stim_item==0).sum().astype(np.float)



            # fr_stim_table_by_pos = fr_stim_session_table.groupby('serialpos')
            # session_summary.prob_recall = np.empty(len(fr_stim_table_by_pos), dtype=float)
            # session_summary.prob_stim_recall = np.empty(len(fr_stim_table_by_pos), dtype=float)
            # session_summary.prob_nostim_recall = np.empty(len(fr_stim_table_by_pos), dtype=float)
            # session_summary.prob_stim = np.empty(len(fr_stim_table_by_pos), dtype=float)
            # for i, (pos,fr_stim_pos_table) in enumerate(fr_stim_table_by_pos):
            #     session_summary.prob_recall[i] = fr_stim_pos_table.recalled.sum() / float(len(fr_stim_pos_table))
            #     fr_stim_item_pos_table =fr_stim_pos_table.loc[fr_stim_pos_table.is_stim_item==True]
            #     try:
            #         session_summary.prob_stim_recall[i]=fr_stim_item_pos_table.recalled.sum()/float(len(fr_stim_item_pos_table))
            #     except ZeroDivisionError:
            #         session_summary.prob_stim_recall[i] = np.nan
            #     session_summary.prob_stim[i] = (fr_stim_pos_table.is_stim_item.astype(np.float).sum()
            #                                     /fr_stim_pos_table.is_stim_list.astype(np.float).sum())
            #     print '# stim items: ',fr_stim_pos_table.is_stim_item.astype(np.float).sum()
            #     print '# stim list items: ', fr_stim_pos_table.is_stim_list.astype(np.float).sum()
            #     fr_nostim_item_pos_table = fr_stim_pos_table.loc[fr_stim_pos_table.is_stim_item==False]
            #     session_summary.prob_nostim_recall[i] = fr_nostim_item_pos_table.recalled.sum()/float(len(fr_nostim_item_pos_table))
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



                # if 'cat' in task:
                #     # list_rec_events = session_rec_events[session_rec_events.list == lst]
                #     for i in xrange(1, len(list_rec_events)):
                #         cur_ev = list_rec_events[i]
                #         prev_ev = list_rec_events[i - 1]
                #         # if (cur_ev.intrusion == 0) and (prev_ev.intrusion == 0):
                #         dt = cur_ev.mstime - prev_ev.mstime
                #         if cur_ev.category == prev_ev.category:
                #             session_irt_within_cat.append(dt)
                #         else:
                #             session_irt_between_cat.append(dt)

                session_summary.list_number[list_idx] = lst
                session_summary.n_recalls_per_list[list_idx] = fr_stim_sess_list_table.recalled.sum()
                session_summary.n_stims_per_list[list_idx] = fr_stim_sess_list_table.is_stim_item.sum()
                session_summary.is_stim_list[list_idx] = fr_stim_sess_list_table.is_stim_list.any()
                session_summary.is_baseline_list[list_idx] = (fr_stim_sess_list_table['phase']==self.BASELINE).any()
                session_summary.is_ps_list[list_idx] = (fr_stim_sess_list_table['phase'] == self.PS4).any()
                session_summary.is_nonstim_list[list_idx] = (fr_stim_sess_list_table['phase'] == self.NONSTIM).any()

            #
            # session_summary.irt_within_cat = sum(session_irt_within_cat) / len(
            #     session_irt_within_cat) if session_irt_within_cat else 0.0
            # session_summary.irt_between_cat = sum(session_irt_between_cat) / len(
            #     session_irt_between_cat) if session_irt_between_cat else 0.0
            #
            # irt_within_cat += session_irt_within_cat
            # irt_between_cat += session_irt_between_cat

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
            # session_summary.n_total_stim = session_summary.n_total_stim if session_summary.n_total_stim else session_summary.n_correct_stim
            session_summary.pc_from_stim = 100 * session_summary.n_correct_stim / float(session_summary.n_total_stim)
            # session_summary.pc_from_stim = 100 * session_summary.n_correct_stim / (float(session_summary.n_total_stim) if session_summary.n_total_stim else 4)
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

            # session_summary.n_stim_intr = 0
            # session_summary.n_nonstim_intr = 0
            # for ev in sess_intr_events:
            #     if ev.intrusion in stim_lists:
            #         session_summary.n_stim_intr += 1
            #     if ev.intrusion in non_stim_lists:
            #         session_summary.n_nonstim_intr += 1
            # if not len(fr_stim_non_stim_list_table):
            #     session_summary.n_nonstim_intr = (fr1_events.intrusion==1).sum()
            #
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

            #fr_stim_non_stim_list_table = fr_stim_non_stim_list_table[(~fr_stim_non_stim_list_table['is_stim_list']) & (fr_stim_non_stim_list_table['serialpos']>1)]
            #
            # low_state_mask = (fr_stim_non_stim_list_table['probs']<fr_stim_non_stim_list_table['thresh'])
            # post_low_state_mask = low_state_mask.shift(1).fillna(False)
            # post_low_state_mask[fr_stim_non_stim_list_table['serialpos']==1] = False
            #
            # fr_stim_non_stim_list_low_table = fr_stim_non_stim_list_table[low_state_mask]
            # fr_stim_non_stim_list_post_low_table = fr_stim_non_stim_list_table[post_low_state_mask]
            # fr_stim_non_stim_list_high_table = fr_stim_non_stim_list_table[fr_stim_non_stim_list_table['prob']>fr_stim_non_stim_list_table['thresh']]
            #
            # session_summary.control_mean_prob_diff_all = fr_stim_non_stim_list_table['prob_diff'].mean()
            # session_summary.control_sem_prob_diff_all = fr_stim_non_stim_list_table['prob_diff'].sem()
            # session_summary.control_mean_prob_diff_low = fr_stim_non_stim_list_low_table['prob_diff'].mean()
            # session_summary.control_sem_prob_diff_low = fr_stim_non_stim_list_low_table['prob_diff'].sem()

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
                # pc_correct_words = 100.0 * sess_events[sess_events.recalled != -999].recalled.mean()

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

        self.pass_object('fr5_session_table', session_data)

        session_summary_array = self.get_passed_object('fr_session_summary')
        fr_stim_table_by_phase = fr_stim_table.loc[~fr_stim_table['is_ps4_session']]

        fr_stim_group_table_group = fr_stim_table_by_phase.groupby(['stimAnodeTag','stimCathodeTag'])
        for stim_param,fr_stim_session_table in fr_stim_group_table_group:
            print 'Stim param: ',stim_param
            session_summary = FR5SessionSummary()


            session_summary.sessions = sorted(fr_stim_session_table.session.unique())
            session_summary.stimtag = fr_stim_session_table.stimAnodeTag.values[0] + '-' + fr_stim_session_table.stimCathodeTag.values[0]
            session_summary.region_of_interest = fr_stim_session_table.Region.values[0]
            session_summary.frequency = fr_stim_session_table.Pulse_Frequency.values[0]
            session_summary.n_words = len(fr_stim_session_table)
            session_summary.n_correct_words = fr_stim_session_table.recalled.sum()
            session_summary.pc_correct_words = 100*session_summary.n_correct_words / float(session_summary.n_words)
            session_summary.amplitude = fr_stim_session_table['Amplitude'].values[-1]

            sess_sel = lambda x: np.in1d(x,session_summary.sessions)
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
            session_summary.prob_recall = fr_stim_session_table.groupby('serialpos').recalled.mean()
            session_summary.prob_stim_recall = fr_stim_session_table.loc[fr_stim_session_table.is_stim_item==True].groupby('serialpos').recalled.mean()
            session_summary.prob_nostim_recall = fr_stim_session_table.loc[fr_stim_session_table.is_stim_item==False].groupby('serialpos').recalled.mean()


            session_summary.prob_stim = fr_stim_session_table.loc[fr_stim_session_table['is_stim_list']==True].groupby('serialpos').is_stim_item.mean().values

            # session_summary.prob_recall = np.empty(len(fr_stim_table_by_pos), dtype=float)
            # session_summary.prob_stim_recall = fr_stim_table
            # session_summary.prob_nostim_recall = np.empty(len(fr_stim_table_by_pos), dtype=float)
            # session_summary.prob_stim = np.empty(len(fr_stim_table_by_pos), dtype=float)
            # for i, (pos,fr_stim_pos_table) in enumerate(fr_stim_table_by_pos):
            #     # session_summary.prob_recall[i] = fr_stim_pos_table.recalled.sum() / float(len(fr_stim_pos_table))
            #     fr_stim_item_pos_table =fr_stim_pos_table.loc[fr_stim_pos_table.is_stim_item==True]
            #     try:
            #         session_summary.prob_stim_recall[i]=fr_stim_item_pos_table.recalled.sum()/float(len(fr_stim_item_pos_table))
            #     except ZeroDivisionError:
            #         session_summary.prob_stim_recall[i] = np.nan
            #     session_summary.prob_stim[i] = (fr_stim_pos_table.is_stim_item.astype(np.float).sum()
            #                                     /fr_stim_pos_table.is_stim_list.astype(np.float).sum())
            #     print '# stim items: ',fr_stim_pos_table.is_stim_item.astype(np.float).sum()
            #     print '# stim list items: ', fr_stim_pos_table.is_stim_list.astype(np.float).sum()
            #     fr_nostim_item_pos_table = fr_stim_pos_table.loc[fr_stim_pos_table.is_stim_item==False]
            #     session_summary.prob_nostim_recall[i] = fr_nostim_item_pos_table.recalled.sum()/float(len(fr_nostim_item_pos_table))
            print 'session_summary.prob_stim:',session_summary.prob_stim


            session_summary.prob_first_recall = np.zeros(len(fr_stim_table_by_pos), dtype=float)
            session_summary.prob_first_stim_recall = np.zeros(len(fr_stim_table_by_pos), dtype=float)
            session_summary.prob_first_nostim_recall = np.zeros(len(fr_stim_table_by_pos), dtype=float)
            first_recall_counter = np.zeros(len(fr_stim_table_by_pos), dtype=int)
            session_summary.list_number = np.empty(len(fr_stim_table_by_session_list), dtype=int)
            session_summary.n_recalls_per_list = np.empty(len(fr_stim_table_by_session_list), dtype=int)
            session_summary.n_stims_per_list = np.zeros(len(fr_stim_table_by_session_list), dtype=int)
            session_summary.is_stim_list = np.zeros(len(fr_stim_table_by_session_list), dtype=np.bool)
            session_summary.is_nonstim_list = np.zeros(len(fr_stim_table_by_session_list), dtype=np.bool)
            session_summary.is_baseline_list = np.zeros(len(fr_stim_table_by_session_list), dtype=np.bool)
            session_summary.is_ps_list = np.zeros(len(fr_stim_table_by_session_list), dtype=np.bool)

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
                            session_summary.prob_first_stim_recall[first_recall_idx]+=1
                        else:
                            session_summary.prob_first_nostim_recall[first_recall_idx]+=1
                        session_summary.prob_first_recall[first_recall_idx] += 1
                        first_recall_counter[first_recall_idx] += 1


                # if 'cat' in task:
                #     # list_rec_events = session_rec_events[session_rec_events.list == lst]
                #     for i in xrange(1, len(list_rec_events)):
                #         cur_ev = list_rec_events[i]
                #         prev_ev = list_rec_events[i - 1]
                #         # if (cur_ev.intrusion == 0) and (prev_ev.intrusion == 0):
                #         dt = cur_ev.mstime - prev_ev.mstime
                #         if cur_ev.category == prev_ev.category:
                #             session_irt_within_cat.append(dt)
                #         else:
                #             session_irt_between_cat.append(dt)

                session_summary.list_number[list_idx] = lst
                session_summary.n_recalls_per_list[list_idx] = fr_stim_sess_list_table.recalled.sum()
                session_summary.n_stims_per_list[list_idx] = fr_stim_sess_list_table.is_stim_item.sum()
                session_summary.is_stim_list[list_idx] = fr_stim_sess_list_table.is_stim_list.any()
                session_summary.is_baseline_list[list_idx] = (fr_stim_sess_list_table['phase']==self.BASELINE).any()
                session_summary.is_ps_list[list_idx] = (fr_stim_sess_list_table['phase'] == self.PS4).any()
                session_summary.is_nonstim_list[list_idx] = (fr_stim_sess_list_table['phase'] == self.NONSTIM).any()
            #
            # session_summary.irt_within_cat = sum(session_irt_within_cat) / len(
            #     session_irt_within_cat) if session_irt_within_cat else 0.0
            # session_summary.irt_between_cat = sum(session_irt_between_cat) / len(
            #     session_irt_between_cat) if session_irt_between_cat else 0.0
            #
            # irt_within_cat += session_irt_within_cat
            # irt_between_cat += session_irt_between_cat

            session_summary.prob_first_recall /= float(len(fr_stim_table_by_session_list))
            session_summary.prob_first_stim_recall /= float(len(fr_stim_stim_item_table_by_session_list))
            session_summary.prob_first_nostim_recall /= float(len(fr_stim_nostim_item_table_by_session_list))
            fr_stim_stim_list_table = fr_stim_session_table.loc[fr_stim_session_table.is_stim_list==True]
            fr_stim_non_stim_list_table = fr_stim_session_table[~fr_stim_session_table.is_stim_list & (fr_stim_session_table['list']>=4)]


            session_summary.n_correct_stim = fr_stim_stim_list_table.recalled.sum()
            session_summary.n_total_stim = len(fr_stim_stim_list_table)
            # session_summary.n_total_stim = session_summary.n_total_stim if session_summary.n_total_stim else session_summary.n_correct_stim
            session_summary.pc_from_stim = 100 * session_summary.n_correct_stim / float(session_summary.n_total_stim)
            # session_summary.pc_from_stim = 100 * session_summary.n_correct_stim / (float(session_summary.n_total_stim) if session_summary.n_total_stim else 4)

            if len(fr_stim_non_stim_list_table):
                session_summary.n_correct_nonstim = fr_stim_non_stim_list_table.recalled.sum()
                session_summary.n_total_nonstim = len(fr_stim_non_stim_list_table)

            session_summary.pc_from_nonstim = 100 * session_summary.n_correct_nonstim / float(
                session_summary.n_total_nonstim)

            session_summary.chisqr, session_summary.pvalue, _ = proportions_chisquare([session_summary.n_correct_stim, session_summary.n_correct_nonstim], [session_summary.n_total_stim, session_summary.n_total_nonstim])

            stim_lists = fr_stim_stim_list_table['list'].unique()
            non_stim_lists = fr_stim_non_stim_list_table['list'].unique()

            session_summary.n_stim_intr = 0
            session_summary.n_nonstim_intr = 0
            for ev in sess_intr_events:
                if ev.intrusion in stim_lists:
                    session_summary.n_stim_intr += 1
                if ev.intrusion in non_stim_lists:
                    session_summary.n_nonstim_intr += 1
            if not len(fr_stim_non_stim_list_table):
                session_summary.n_nonstim_intr = (fr1_events.intrusion==1).sum()
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

            #fr_stim_non_stim_list_table = fr_stim_non_stim_list_table[(~fr_stim_non_stim_list_table['is_stim_list']) & (fr_stim_non_stim_list_table['serialpos']>1)]

            low_state_mask = (fr_stim_non_stim_list_table['prob']<fr_stim_non_stim_list_table['thresh'])
            post_low_state_mask = low_state_mask.shift(1).fillna(False)
            post_low_state_mask[fr_stim_non_stim_list_table['serialpos']==1] = False

            fr_stim_non_stim_list_low_table = fr_stim_non_stim_list_table[low_state_mask]
            fr_stim_non_stim_list_post_low_table = fr_stim_non_stim_list_table[post_low_state_mask]
            fr_stim_non_stim_list_high_table = fr_stim_non_stim_list_table[fr_stim_non_stim_list_table['prob']>fr_stim_non_stim_list_table['thresh']]

            session_summary.control_mean_prob_diff_all = fr_stim_non_stim_list_table['prob_diff'].mean()
            session_summary.control_sem_prob_diff_all = fr_stim_non_stim_list_table['prob_diff'].sem()
            session_summary.control_mean_prob_diff_low = fr_stim_non_stim_list_low_table['prob_diff'].mean()
            session_summary.control_sem_prob_diff_low = fr_stim_non_stim_list_low_table['prob_diff'].sem()

            stim_item_recall_rate = fr_stim_stim_list_stim_item_table['recalled'].mean()
            stim_item_recall_rate_low = fr_stim_stim_list_stim_item_low_table['recalled'].mean()
            stim_item_recall_rate_high = fr_stim_stim_list_stim_item_high_table['recalled'].mean()

            post_stim_item_recall_rate = fr_stim_stim_list_post_stim_item_table['recalled'].mean()
            post_stim_item_recall_rate_low = fr_stim_stim_list_post_stim_item_low_table['recalled'].mean()
            post_stim_item_recall_rate_high = fr_stim_stim_list_post_stim_item_high_table['recalled'].mean()

            non_stim_list_recall_rate_low = fr_stim_non_stim_list_low_table['recalled'].mean()
            non_stim_list_recall_rate_post_low = fr_stim_non_stim_list_post_low_table['recalled'].mean()
            non_stim_list_recall_rate_high = fr_stim_non_stim_list_high_table['recalled'].mean()

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

            session_summary.n_correct_nonstim_low_bio_items = fr_stim_non_stim_list_low_table['recalled'].sum()
            session_summary.n_total_nonstim_low_bio_items = len(fr_stim_non_stim_list_low_table)
            session_summary.pc_nonstim_low_bio_items = 100*session_summary.n_correct_nonstim_low_bio_items / float(session_summary.n_total_nonstim_low_bio_items)

            session_summary.n_correct_nonstim_post_low_bio_items = fr_stim_non_stim_list_post_low_table['recalled'].sum()
            session_summary.n_total_nonstim_post_low_bio_items = len(fr_stim_non_stim_list_post_low_table)
            session_summary.pc_nonstim_post_low_bio_items = 100*session_summary.n_correct_nonstim_post_low_bio_items / float(session_summary.n_total_nonstim_post_low_bio_items)

            session_summary.chisqr_stim_item, session_summary.pvalue_stim_item, _ = proportions_chisquare([session_summary.n_correct_stim_items, session_summary.n_correct_nonstim_low_bio_items], [session_summary.n_total_stim_items, session_summary.n_total_nonstim_low_bio_items])
            session_summary.chisqr_post_stim_item, session_summary.pvalue_post_stim_item, _ = proportions_chisquare([session_summary.n_correct_post_stim_items, session_summary.n_correct_nonstim_post_low_bio_items], [session_summary.n_total_post_stim_items, session_summary.n_total_nonstim_post_low_bio_items])


            sess_recog_lures = session_all_events[session_all_events.type=='RECOG_LURE']
            sess_rec_targets = session_all_events[session_all_events.type=='RECOG_TARGET']
            if (fr_stim_session_table['recognized']!=-999).any():
                session_summary.n_stim_hits = (sess_rec_targets[sess_rec_targets.stim_list==1].recognized==1).sum()
                session_summary.n_nonstim_hits =  (sess_rec_targets[sess_rec_targets.stim_list==0].recognized==1).sum()
                session_summary.n_false_alarms = (sess_recog_lures.rejected==0).sum()
                print session_summary.n_false_alarms
                session_summary.pc_stim_hits =  (sess_rec_targets[sess_rec_targets.stim_list==1].recognized==1).mean()
                session_summary.pc_nonstim_hits = np.nanmean((sess_rec_targets[sess_rec_targets.stim_list==0].recognized==1))
                session_summary.pc_false_alarms = (sess_recog_lures.rejected==0).mean()
                session_summary.dprime = '{:03f}'.format(stats.norm.ppf(
                    (session_summary.n_stim_hits+session_summary.n_nonstim_hits)/float(len(sess_rec_targets))) -
                                          stats.norm.ppf(session_summary.pc_false_alarms))

                session_summary.n_stim_item_hits = (fr_stim_stim_list_stim_item_table.recognized==1).values.sum()
                session_summary.n_low_biomarker_hits = (fr_stim_non_stim_list_low_table.recognized==1).values.sum()
                session_summary.pc_stim_item_hits = session_summary.n_stim_item_hits/float(
                    ((fr_stim_stim_list_stim_item_table.recognized==0) | (fr_stim_stim_list_stim_item_table.recognized==1)).sum())
                session_summary.pc_low_biomarker_hits = session_summary.n_low_biomarker_hits/float(
                    ((fr_stim_non_stim_list_low_table.recognized == 1) |(fr_stim_non_stim_list_low_table.recognized==0)).values.sum()
                )
                session_summary.pc_stim_hits *= 100
                session_summary.pc_nonstim_hits *= 100
                session_summary.pc_stim_item_hits *= 100
                session_summary.pc_low_biomarker_hits *=100
                session_summary.pc_false_alarms *= 100



            session_summary_array.append(session_summary)
        self.pass_object('fr_session_summary', session_summary_array)

        # if 'cat' in task:
        #     repetition_ratios = self.get_passed_object('repetition_ratios')
        #     stim_rrs = []
        #     nostim_rrs = []
        #     self.pass_object('mean_rr',np.nanmean(repetition_ratios))
        #     for s_num,session in enumerate(np.unique(rec_events.session)):
        #         sess_events = rec_events[rec_events.session == session]
        #         stim_lists = np.unique(sess_events[sess_events.stim_list==True].list)
        #         print 'stim_lists:',stim_lists
        #         nostim_lists = np.unique(sess_events[sess_events.stim_list==False].list)
        #         print 'nonstim lists',nostim_lists
        #         stim_rrs.append(repetition_ratios[s_num][stim_lists[stim_lists>0]-1])
        #         nostim_rrs.append(repetition_ratios[s_num][nostim_lists[nostim_lists>0]-1])
        #     self.pass_object('stim_mean_rr',np.nanmean(np.hstack(stim_rrs)))
        #     self.pass_object('nostim_mean_rr',np.nanmean(np.hstack(nostim_rrs)))

