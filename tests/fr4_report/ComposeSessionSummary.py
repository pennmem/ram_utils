from RamPipeline import *
from SessionSummary import SessionSummary

import numpy as np
import time

from statsmodels.stats.proportion import proportions_chisquare



from ReportUtils import ReportRamTask

class ComposeSessionSummary(ReportRamTask):
    def __init__(self, params, mark_as_completed=True):
        super(ComposeSessionSummary, self).__init__(mark_as_completed)
        self.params = params

    def run(self):
        subject = self.pipeline.subject
        task = self.pipeline.task

        #events = self.get_passed_object(task + '_events')
        math_events = self.get_passed_object(task + '_math_events')
        intr_events = self.get_passed_object(task + '_intr_events')
        rec_events = self.get_passed_object(task + '_rec_events')
        all_events = self.get_passed_object(task + '_all_events')
        monopolar_channels = self.get_passed_object('monopolar_channels')
        xval_output = self.get_passed_object('xval_output')
        thresh = xval_output[-1].jstat_thresh

        stim_params_to_sess = self.get_passed_object('stim_params_to_sess')
        fr4_table = self.get_passed_object('fr4_table')

        sessions = sorted(fr4_table.session.unique())

        self.pass_object('NUMBER_OF_SESSIONS', len(sessions))
        self.pass_object('NUMBER_OF_ELECTRODES', len(monopolar_channels))

        session_data = []

        fr4_table_by_session = fr4_table.groupby(['session'])
        for session,fr4_session_table in fr4_table_by_session:
            session_all_events = all_events[all_events.session == session]
            first_time_stamp = session_all_events[session_all_events.type=='INSTRUCT_VIDEO'][0].mstime
            timestamps = session_all_events.mstime
            last_time_stamp = np.max(timestamps)
            session_length = '%.2f' % ((last_time_stamp - first_time_stamp) / 60000.0)
            session_date = time.strftime('%d-%b-%Y', time.localtime(last_time_stamp/1000))
            n_lists = len(fr4_session_table.list.unique())
            pc_correct_words = 100.0 * fr4_session_table.recalled.sum() / len(fr4_session_table)

            session_data.append([session, session_date, session_length, n_lists, '$%.2f$\\%%' % pc_correct_words])

        self.pass_object('SESSION_DATA', session_data)

        session_summary_array = []

        fr4_table_by_stim_param = fr4_table.groupby(['stimAnodeTag','stimCathodeTag','Pulse_Frequency'])
        for stim_param,fr4_session_table in fr4_table_by_stim_param:
            session_summary = SessionSummary()

            session_summary.sessions = sorted(fr4_session_table.session.unique())
            session_summary.stimtag = fr4_session_table.stimAnodeTag.values[0] + '-' + fr4_session_table.stimCathodeTag.values[0]
            session_summary.region_of_interest = fr4_session_table.Region.values[0]
            session_summary.frequency = fr4_session_table.Pulse_Frequency.values[0]
            session_summary.n_words = len(fr4_session_table)
            session_summary.n_correct_words = fr4_session_table.recalled.sum()
            session_summary.pc_correct_words = 100*session_summary.n_correct_words / float(session_summary.n_words)

            sess_sel = np.vectorize(lambda sess: sess in session_summary.sessions)
            sess_rec_events = rec_events[sess_sel(rec_events.session)]
            n_sess_rec_events = len(sess_rec_events)
            sess_intr_events = intr_events[sess_sel(intr_events.session)]
            sess_math_events = math_events[sess_sel(math_events.session)]

            fr4_table_by_session_list = fr4_session_table.groupby(['session','list'])
            session_summary.n_lists = len(fr4_table_by_session_list)

            session_summary.n_pli = np.sum(sess_intr_events.intrusion > 0)
            session_summary.pc_pli = 100*session_summary.n_pli / float(n_sess_rec_events)
            session_summary.n_eli = np.sum(sess_intr_events.intrusion == -1)
            session_summary.pc_eli = 100*session_summary.n_eli / float(n_sess_rec_events)

            session_summary.n_math = len(sess_math_events)
            session_summary.n_correct_math = np.sum(sess_math_events.iscorrect)
            session_summary.pc_correct_math = 100*session_summary.n_correct_math / float(session_summary.n_math)
            session_summary.math_per_list = session_summary.n_math / float(session_summary.n_lists)

            fr4_table_by_pos = fr4_session_table.groupby('serialpos')
            session_summary.prob_recall = np.empty(len(fr4_table_by_pos), dtype=float)
            for i, (pos,fr4_pos_table) in enumerate(fr4_table_by_pos):
                session_summary.prob_recall[i] = fr4_pos_table.recalled.sum() / float(len(fr4_pos_table))

            session_summary.prob_first_recall = np.zeros(len(fr4_table_by_pos), dtype=float)
            first_recall_counter = np.zeros(len(fr4_table_by_pos), dtype=int)
            session_summary.list_number = np.empty(len(fr4_table_by_session_list), dtype=int)
            session_summary.n_recalls_per_list = np.empty(len(fr4_table_by_session_list), dtype=int)
            session_summary.n_stims_per_list = np.zeros(len(fr4_table_by_session_list), dtype=int)
            session_summary.is_stim_list = np.zeros(len(fr4_table_by_session_list), dtype=np.bool)
            for list_idx, (sess_list,fr4_sess_list_table) in enumerate(fr4_table_by_session_list):
                session = sess_list[0]
                lst = sess_list[1]

                list_rec_events = rec_events[(rec_events.session==session) & (rec_events['list']==lst) & (rec_events['intrusion']==0)]
                if list_rec_events.size > 0:
                    tmp = np.where(fr4_sess_list_table.itemno.values == list_rec_events[0].itemno)[0]
                    if tmp.size > 0:
                        first_recall_idx = tmp[0]
                        session_summary.prob_first_recall[first_recall_idx] += 1
                        first_recall_counter[first_recall_idx] += 1

                session_summary.list_number[list_idx] = lst
                session_summary.n_recalls_per_list[list_idx] = fr4_sess_list_table.recalled.sum()
                session_summary.n_stims_per_list[list_idx] = fr4_sess_list_table.is_stim_item.sum()
                session_summary.is_stim_list[list_idx] = fr4_sess_list_table.is_stim_list.any()

            session_summary.prob_first_recall /= float(len(fr4_table_by_session_list))

            fr4_stim_list_table = fr4_session_table[fr4_session_table.is_stim_list]
            fr4_non_stim_list_table = fr4_session_table[~fr4_session_table.is_stim_list & (fr4_session_table['list']>=4)]

            session_summary.n_correct_stim = fr4_stim_list_table.recalled.sum()
            session_summary.n_total_stim = len(fr4_stim_list_table)
            session_summary.pc_from_stim = 100 * session_summary.n_correct_stim / float(session_summary.n_total_stim)

            session_summary.n_correct_nonstim = fr4_non_stim_list_table.recalled.sum()
            session_summary.n_total_nonstim = len(fr4_non_stim_list_table)
            session_summary.pc_from_nonstim = 100 * session_summary.n_correct_nonstim / float(session_summary.n_total_nonstim)

            session_summary.chisqr, session_summary.pvalue, _ = proportions_chisquare([session_summary.n_correct_stim, session_summary.n_correct_nonstim], [session_summary.n_total_stim, session_summary.n_total_nonstim])

            stim_lists = fr4_stim_list_table['list'].unique()
            non_stim_lists = fr4_non_stim_list_table['list'].unique()

            session_summary.n_stim_intr = 0
            session_summary.n_nonstim_intr = 0
            for ev in sess_intr_events:
                if ev.intrusion in stim_lists:
                    session_summary.n_stim_intr += 1
                if ev.intrusion in non_stim_lists:
                    session_summary.n_nonstim_intr += 1
            session_summary.pc_from_stim_intr = 100*session_summary.n_stim_intr / float(session_summary.n_total_stim)
            session_summary.pc_from_nonstim_intr = 100*session_summary.n_nonstim_intr / float(session_summary.n_total_nonstim)

            session_summary_array.append(session_summary)

        self.pass_object('session_summary_array', session_summary_array)
