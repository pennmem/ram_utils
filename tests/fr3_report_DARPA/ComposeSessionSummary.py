from ram_utils.RamPipeline import *
from SessionSummary import SessionSummary

import numpy as np
import time

from statsmodels.stats.proportion import proportions_chisquare


class ComposeSessionSummary(RamTask):
    def __init__(self, mark_as_completed=True):
        RamTask.__init__(self, mark_as_completed)

    def run(self):
        subject = self.pipeline.subject
        task = self.pipeline.task

        events = self.get_passed_object(task + '_events')
        math_events = self.get_passed_object(task + '_math_events')
        intr_events = self.get_passed_object(task + '_intr_events')
        rec_events = self.get_passed_object(task + '_rec_events')
        all_events = self.get_passed_object(task + '_all_events')
        channels = self.get_passed_object('monopolar_channels')
        tal_info = self.get_passed_object('bipolar_pairs')

        sessions = np.unique(events.session)

        self.pass_object('NUMBER_OF_SESSIONS', len(sessions))
        self.pass_object('NUMBER_OF_ELECTRODES', len(channels))

        session_data = []
        session_summary_array = []

        positions = np.unique(events.serialpos)
        first_recall_counter = np.zeros(positions.size, dtype=int)
        total_list_counter = 0

        irt_within_cat = []
        irt_between_cat = []

        cumulative_n_items_from_stim = 0
        cumulative_n_recalls_from_stim = 0
        cumulative_n_intr_from_stim = 0

        cumulative_n_items_from_nonstim = 0
        cumulative_n_recalls_from_nonstim = 0
        cumulative_n_intr_from_nonstim = 0

        for session in sessions:
            session_summary = SessionSummary()

            session_events = events[events.session == session]
            n_sess_events = len(session_events)

            session_rec_events = rec_events[rec_events.session == session]

            session_all_events = all_events[all_events.session == session]
            timestamps = session_all_events.mstime
            first_time_stamp = np.min(timestamps)
            last_time_stamp = np.max(timestamps)
            session_length = '%.2f' % ((last_time_stamp - first_time_stamp) / 60000.0)
            session_date = time.strftime('%d-%b-%Y', time.localtime(last_time_stamp/1000))

            session_data.append([session, session_date, session_length])

            session_name = 'Sess%02d' % session

            print 'Session =', session_name

            session_summary.number = session
            session_summary.name = session_name
            session_summary.length = session_length
            session_summary.date = session_date
            session_summary.n_words = len(session_events)
            session_summary.n_correct_words = np.sum(session_events.recalled)
            session_summary.pc_correct_words = 100*session_summary.n_correct_words / float(session_summary.n_words)

            positions = np.unique(session_events.serialpos)
            prob_recall = np.empty_like(positions, dtype=float)
            for i,pos in enumerate(positions):
                pos_events = session_events[session_events.serialpos == pos]
                prob_recall[i] = np.sum(pos_events.recalled) / float(len(pos_events))

            session_summary.prob_recall = prob_recall

            lists = np.unique(session_events.list)
            n_lists = len(lists)
            prob_first_recall = np.zeros(len(positions), dtype=float)
            session_irt_within_cat = []
            session_irt_between_cat = []
            session_summary.n_recalls_per_list = np.zeros(n_lists, dtype=np.int)
            session_summary.n_intr_per_list = np.zeros(n_lists, dtype=np.int)
            session_summary.n_stims_per_list = np.zeros(n_lists, dtype=np.int)
            session_summary.is_stim_list = np.zeros(n_lists, dtype=np.bool)
            items_per_list = np.zeros(n_lists, dtype=np.int)

            for lst in lists:
                list_events = session_all_events[session_all_events.list == lst]
                list_rec_events = session_rec_events[(session_rec_events.list == lst) & (session_rec_events.intrusion == 0)]
                list_intr_events = session_rec_events[(session_rec_events.list == lst) & (session_rec_events.intrusion >= 4)]
                list_word_events = list_events[list_events.type=='WORD']

                #session_summary.n_recalls_per_list[lst-1] = len(list_rec_events)
                #session_summary.n_intr_per_list[lst-1] = len(list_intr_events)
                session_summary.n_recalls_per_list[lst-1] = np.sum(list_word_events.recalled)
                session_summary.n_stims_per_list[lst-1] = np.sum(list_events.type=='STIM')
                session_summary.is_stim_list[lst-1] = session_events[session_events.list == lst][0].stimList
                for ie in list_intr_events:
                    session_summary.n_intr_per_list[ie.intrusion-1] += 1

                items_per_list[lst-1] = np.sum(list_events.type=='WORD')

                if list_rec_events.size > 0:
                    list_events = session_events[session_events.list == lst]
                    tmp = np.where(list_events.itemno == list_rec_events[0].itemno)[0]
                    if tmp.size > 0:
                        first_recall_idx = tmp[0]
                        prob_first_recall[first_recall_idx] += 1
                        first_recall_counter[first_recall_idx] += 1
                if task == 'RAM_CatFR3':
                    # list_rec_events = session_rec_events[session_rec_events.list == lst]
                    for i in xrange(1,len(list_rec_events)):
                        cur_ev = list_rec_events[i]
                        prev_ev = list_rec_events[i-1]
                        # if (cur_ev.intrusion == 0) and (prev_ev.intrusion == 0):
                        dt = cur_ev.mstime - prev_ev.mstime
                        if cur_ev.category == prev_ev.category:
                            session_irt_within_cat.append(dt)
                        else:
                            session_irt_between_cat.append(dt)

            prob_first_recall /= float(n_lists)
            total_list_counter += n_lists

            n_items_from_stim = np.sum(items_per_list[session_summary.is_stim_list])
            n_recalls_from_stim = np.sum(session_summary.n_recalls_per_list[session_summary.is_stim_list])
            n_intr_from_stim = np.sum(session_summary.n_intr_per_list[session_summary.is_stim_list])

            cumulative_n_items_from_stim += n_items_from_stim
            cumulative_n_recalls_from_stim += n_recalls_from_stim
            cumulative_n_intr_from_stim += n_intr_from_stim

            nonstim_list_mask = ~session_summary.is_stim_list
            nonstim_list_mask[0:3] = False
            n_items_from_nonstim = np.sum(items_per_list[nonstim_list_mask])
            n_recalls_from_nonstim = np.sum(session_summary.n_recalls_per_list[nonstim_list_mask])
            n_intr_from_nonstim = np.sum(session_summary.n_intr_per_list[nonstim_list_mask])

            cumulative_n_items_from_nonstim += n_items_from_nonstim
            cumulative_n_recalls_from_nonstim += n_recalls_from_nonstim
            cumulative_n_intr_from_nonstim += n_intr_from_nonstim

            session_summary.n_correct_stim = n_recalls_from_stim
            session_summary.n_total_stim = n_items_from_stim
            session_summary.pc_from_stim = 100 * n_recalls_from_stim / float(n_items_from_stim)

            session_summary.n_correct_nonstim = n_recalls_from_nonstim
            session_summary.n_total_nonstim = n_items_from_nonstim
            session_summary.pc_from_nonstim = 100 * n_recalls_from_nonstim / float(n_items_from_nonstim)

            session_summary.n_stim_intr = n_intr_from_stim
            session_summary.pc_from_stim_intr = 100 * n_intr_from_stim / float(n_items_from_stim)

            session_summary.n_nonstim_intr = n_intr_from_nonstim
            session_summary.pc_from_nonstim_intr = 100 * n_intr_from_nonstim / float(n_items_from_nonstim)

            session_summary.chisqr,session_summary.pvalue,_ = proportions_chisquare([n_recalls_from_stim, n_recalls_from_nonstim], [n_items_from_stim, n_items_from_nonstim])
            session_summary.chisqr_intr,session_summary.pvalue_intr,_ = proportions_chisquare([n_intr_from_stim, n_intr_from_nonstim], [n_items_from_stim, n_items_from_nonstim])

            session_summary.irt_within_cat = sum(session_irt_within_cat) / len(session_irt_within_cat) if session_irt_within_cat else 0.0
            session_summary.irt_between_cat = sum(session_irt_between_cat) / len(session_irt_between_cat) if session_irt_between_cat else 0.0

            irt_within_cat += session_irt_within_cat
            irt_between_cat += session_irt_between_cat

            session_summary.prob_first_recall = prob_first_recall

            if math_events is not None:
                session_math_events = math_events[math_events.session == session]
                session_summary.n_math = len(session_math_events)
                session_summary.n_correct_math = np.sum(session_math_events.iscorrect)
                session_summary.pc_correct_math = 100*session_summary.n_correct_math / float(session_summary.n_math)
                session_summary.math_per_list = session_summary.n_math / float(n_lists)

            session_intr_events = intr_events[intr_events.session == session]

            session_summary.n_pli = np.sum(session_intr_events.intrusion > 0)
            session_summary.pc_pli = 100*session_summary.n_pli / float(n_sess_events)
            session_summary.n_eli = np.sum(session_intr_events.intrusion == -1)
            session_summary.pc_eli = 100*session_summary.n_eli / float(n_sess_events)

            session_summary_array.append(session_summary)


        self.pass_object('SESSION_DATA', session_data)
        self.pass_object('session_summary_array', session_summary_array)

        cumulative_summary = SessionSummary()
        cumulative_summary.n_words = len(events)
        cumulative_summary.n_correct_words = np.sum(events.recalled)
        cumulative_summary.pc_correct_words = 100*cumulative_summary.n_correct_words / float(cumulative_summary.n_words)

        cumulative_summary.irt_within_cat = sum(irt_within_cat) / len(irt_within_cat) if irt_within_cat else 0.0
        cumulative_summary.irt_between_cat = sum(irt_between_cat) / len(irt_between_cat) if irt_between_cat else 0.0

        positions = np.unique(events.serialpos)
        prob_recall = np.empty_like(positions, dtype=float)
        for i,pos in enumerate(positions):
            pos_events = events[events.serialpos == pos]
            prob_recall[i] = np.sum(pos_events.recalled) / float(len(pos_events))
        cumulative_summary.prob_recall = prob_recall

        prob_first_recall = first_recall_counter / float(total_list_counter)
        cumulative_summary.prob_first_recall = prob_first_recall

        cumulative_summary.n_correct_stim = cumulative_n_recalls_from_stim
        cumulative_summary.n_total_stim = cumulative_n_items_from_stim
        cumulative_summary.pc_from_stim = 100 * cumulative_n_recalls_from_stim / float(cumulative_n_items_from_stim)

        cumulative_summary.n_correct_nonstim = cumulative_n_recalls_from_nonstim
        cumulative_summary.n_total_nonstim = cumulative_n_items_from_nonstim
        cumulative_summary.pc_from_nonstim = 100 * cumulative_n_recalls_from_nonstim / float(cumulative_n_items_from_nonstim)

        cumulative_summary.n_stim_intr = cumulative_n_intr_from_stim
        cumulative_summary.pc_from_stim_intr = 100 * cumulative_n_intr_from_stim / float(cumulative_n_items_from_stim)

        cumulative_summary.n_nonstim_intr = cumulative_n_intr_from_nonstim
        cumulative_summary.pc_from_nonstim_intr = 100 * cumulative_n_intr_from_nonstim / float(cumulative_n_items_from_nonstim)

        cumulative_summary.chisqr,cumulative_summary.pvalue,_ = proportions_chisquare([cumulative_n_recalls_from_stim, cumulative_n_recalls_from_nonstim], [cumulative_n_items_from_stim, cumulative_n_items_from_nonstim])
        cumulative_summary.chisqr_intr,cumulative_summary.pvalue_intr,_ = proportions_chisquare([cumulative_n_intr_from_stim, cumulative_n_intr_from_nonstim], [cumulative_n_items_from_stim, cumulative_n_items_from_nonstim])

        if math_events is not None:
            cumulative_summary.n_math = len(math_events)
            cumulative_summary.n_correct_math = np.sum(math_events.iscorrect)
            cumulative_summary.pc_correct_math = 100*cumulative_summary.n_correct_math / float(cumulative_summary.n_math)
            cumulative_summary.math_per_list = cumulative_summary.n_math / float(total_list_counter)

        cumulative_summary.n_pli = np.sum(intr_events.intrusion > 0)
        cumulative_summary.pc_pli = 100*cumulative_summary.n_pli / float(len(events))
        cumulative_summary.n_eli = np.sum(intr_events.intrusion == -1)
        cumulative_summary.pc_eli = 100*cumulative_summary.n_eli / float(len(events))

        self.pass_object('cumulative_summary', cumulative_summary)
