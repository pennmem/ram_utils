import time

import numpy as np
from ...ReportUtils import ReportRamTask
from statsmodels.stats.proportion import proportions_chisquare

from .SessionSummary import SessionSummary

class ComposeSessionSummary(ReportRamTask):
    def __init__(self, params, mark_as_completed=True):
        super(ComposeSessionSummary, self).__init__(mark_as_completed)
        self.params = params

    def run(self):
        subject = self.pipeline.subject
        task = self.pipeline.task

        rec_events = self.get_passed_object(task + '_rec_events')
        all_events = self.get_passed_object(task + '_all_events')
        th_events = self.get_passed_object('th_events')
        monopolar_channels = self.get_passed_object('monopolar_channels')
        xval_output = self.get_passed_object('xval_output')
        thresh = xval_output[-1].jstat_thresh
        print 'thresh =', thresh

        stim_params_to_sess = self.get_passed_object('stim_params_to_sess')
        th_stim_table = self.get_passed_object('th_stim_table')
        th_stim_table['prev_prob'] = th_stim_table['prob'].shift(1)
        th_stim_table['prob_diff'] = th_stim_table['prob'] - th_stim_table['prev_prob']

        sessions = sorted(th_stim_table.session.unique())

        self.pass_object('NUMBER_OF_SESSIONS', len(sessions))
        self.pass_object('NUMBER_OF_ELECTRODES', len(monopolar_channels))

        session_data = []

        th_stim_table_by_session = th_stim_table.groupby(['session'])
        for session,th_stim_session_table in th_stim_table_by_session:            
            session_all_events = all_events[all_events.session == session]
            print 'session types: ',np.unique(session_all_events.type)
            # first_time_stamp = session_all_events[session_all_events.type=='SESS_START'][0].mstime
            timestamps = session_all_events.mstime
            first_time_stamp=np.nanmin(timestamps)
            last_time_stamp = np.nanmax(timestamps)
            session_length = '%.2f' % ((last_time_stamp - first_time_stamp) / 60000.0)
            print 'session length: ',session_length
            session_date = time.strftime('%d-%b-%Y', time.localtime(last_time_stamp/1000))
            n_lists = len(th_stim_session_table.list.unique())
            pc_correct_words = 100.0 * th_stim_session_table.recalled.mean()
            amplitude = th_stim_session_table['Amplitude'].values[-1]

            session_data.append([session, session_date, session_length, n_lists, '$%.2f$\\%%' % pc_correct_words, amplitude])

        self.pass_object('SESSION_DATA', session_data)

        session_summary_array = []

        th_stim_table_by_stim_param = th_stim_table.groupby(['stimAnodeTag','stimCathodeTag','Pulse_Frequency'])
        for stim_param,th_stim_session_table in th_stim_table_by_stim_param:
            session_summary = SessionSummary()

            session_summary.sessions = sorted(th_stim_session_table.session.unique())
            session_summary.stimtag = th_stim_session_table.stimAnodeTag.values[0] + '-' + th_stim_session_table.stimCathodeTag.values[0]
            session_summary.region_of_interest = th_stim_session_table.Region.values[0]
            session_summary.frequency = th_stim_session_table.Pulse_Frequency.values[0]
            session_summary.auc = th_stim_session_table.auc.values[0]
            session_summary.auc_p = th_stim_session_table.auc_perm.values[0]
            session_summary.n_words = len(th_stim_session_table)
            session_summary.n_correct_words = th_stim_session_table.recalled.sum()
            session_summary.pc_correct_words = 100*session_summary.n_correct_words / float(session_summary.n_words)
            session_summary.correct_thresh = np.max([th_events[0].radius_size, np.median(th_events.distErr)])

            sess_sel = np.vectorize(lambda sess: sess in session_summary.sessions)
            sess_rec_events = rec_events[sess_sel(rec_events.session)]
            n_sess_rec_events = len(sess_rec_events)
            #sess_intr_events = intr_events[sess_sel(intr_events.session)]
            #sess_math_events = math_events[sess_sel(math_events.session)]

            th_stim_table_by_session_list = th_stim_session_table.groupby(['session','list'])
            session_summary.n_lists = len(th_stim_table_by_session_list)

            #session_summary.n_pli = np.sum(sess_intr_events.intrusion > 0)
            #session_summary.pc_pli = 100*session_summary.n_pli / float(n_sess_rec_events)
            #session_summary.n_eli = np.sum(sess_intr_events.intrusion == -1)
            #session_summary.pc_eli = 100*session_summary.n_eli / float(n_sess_rec_events)

            #session_summary.n_math = len(sess_math_events)
            #session_summary.n_correct_math = np.sum(sess_math_events.iscorrect)
            #session_summary.pc_correct_math = 100*session_summary.n_correct_math / float(session_summary.n_math)
            #session_summary.math_per_list = session_summary.n_math / float(session_summary.n_lists)

            th_stim_table_by_pos = th_stim_session_table.groupby('serialpos')
            session_summary.prob_recall = np.empty(len(th_stim_table_by_pos), dtype=float)
            for i, (pos,th_stim_pos_table) in enumerate(th_stim_table_by_pos):
                session_summary.prob_recall[i] = th_stim_pos_table.recalled.sum() / float(len(th_stim_pos_table))

            #session_summary.prob_first_recall = np.zeros(len(th_stim_table_by_pos), dtype=float)
            #first_recall_counter = np.zeros(len(th_stim_table_by_pos), dtype=int)
            session_summary.list_number = np.empty(len(th_stim_table_by_session_list), dtype=int)
            session_summary.n_recalls_per_list = np.empty(len(th_stim_table_by_session_list), dtype=int)
            session_summary.n_stims_per_list = np.zeros(len(th_stim_table_by_session_list), dtype=int)
            session_summary.is_stim_list = np.zeros(len(th_stim_table_by_session_list), dtype=np.bool)
            for list_idx, (sess_list,th_stim_sess_list_table) in enumerate(th_stim_table_by_session_list):
                session = sess_list[0]
                lst = sess_list[1]

                #list_rec_events = rec_events[(rec_events.session==session) & (rec_events['list']==lst) & (rec_events['intrusion']==0)]
                #if list_rec_events.size > 0:
                #    tmp = np.where(th_stim_sess_list_table.itemno.values == list_rec_events[0].itemno)[0]
                #    if tmp.size > 0:
                #        first_recall_idx = tmp[0]
                #        session_summary.prob_first_recall[first_recall_idx] += 1
                #        first_recall_counter[first_recall_idx] += 1

                session_summary.list_number[list_idx] = lst
                session_summary.n_recalls_per_list[list_idx] = th_stim_sess_list_table.recalled.sum()
                session_summary.n_stims_per_list[list_idx] = th_stim_sess_list_table.is_stim_item.sum()
                session_summary.is_stim_list[list_idx] = th_stim_sess_list_table.is_stim_list.any()

            #session_summary.prob_first_recall /= float(len(th_stim_table_by_session_list))

            th_stim_stim_list_table = th_stim_session_table[th_stim_session_table.is_stim_list]
            th_stim_non_stim_list_table = th_stim_session_table[~th_stim_session_table.is_stim_list & (th_stim_session_table['list']>=8)]

            # stim vs non stim LIST analysis
            session_summary.is_stim_list = np.array(th_stim_session_table[th_stim_session_table['list']>=8].is_stim_list.values, dtype=np.bool)
            session_summary.n_correct_stim = th_stim_stim_list_table.recalled.sum()
            session_summary.n_total_stim = len(th_stim_stim_list_table)
            session_summary.pc_from_stim = 100 * session_summary.n_correct_stim / float(session_summary.n_total_stim)
            session_summary.n_correct_nonstim = th_stim_non_stim_list_table.recalled.sum()
            session_summary.n_total_nonstim = len(th_stim_non_stim_list_table)
            session_summary.pc_from_nonstim = 100 * session_summary.n_correct_nonstim / float(session_summary.n_total_nonstim)
            session_summary.chisqr, session_summary.pvalue, _ = proportions_chisquare([session_summary.n_correct_stim, session_summary.n_correct_nonstim], [session_summary.n_total_stim, session_summary.n_total_nonstim])
            stim_lists = th_stim_stim_list_table['list'].unique()
            non_stim_lists = th_stim_non_stim_list_table['list'].unique()

            # stim vs non stim ITEM analysis
            session_summary.is_stim_item = np.array(th_stim_session_table[th_stim_session_table['list']>=8].is_stim_item.values, dtype=np.bool)
            session_summary.all_dist_errs = np.array(th_stim_session_table[th_stim_session_table['list']>=8].distance_err.values, dtype=float)
            th_stim_stim_item_table = th_stim_session_table[th_stim_session_table.is_stim_item]
            th_stim_non_stim_item_table = th_stim_session_table[~th_stim_session_table.is_stim_item & (th_stim_session_table['list']>=8)]
            session_summary.n_correct_stim_item = th_stim_stim_item_table.recalled.sum()
            dist_err_stim_item, _ = np.histogram(th_stim_stim_item_table.distance_err,bins=20,range=(0,100))
            session_summary.dist_err_stim_item = dist_err_stim_item / float(np.sum(dist_err_stim_item))
            session_summary.n_total_stim_item = len(th_stim_stim_item_table)
            session_summary.pc_from_stim_item = 100 * session_summary.n_correct_stim_item / float(session_summary.n_total_stim_item)
            session_summary.n_correct_nonstim_item = th_stim_non_stim_item_table.recalled.sum()
            dist_err_nonstim_item, _ = np.histogram(th_stim_non_stim_item_table.distance_err,bins=20,range=(0,100))
            session_summary.dist_err_nonstim_item = dist_err_nonstim_item / float(np.sum(dist_err_nonstim_item))
            session_summary.n_total_nonstim_item = len(th_stim_non_stim_item_table)
            session_summary.pc_from_nonstim_item = 100 * session_summary.n_correct_nonstim_item / float(session_summary.n_total_nonstim_item)
            session_summary.chisqr_item, session_summary.pvalue_item, _ = proportions_chisquare([session_summary.n_correct_stim_item, session_summary.n_correct_nonstim_item], [session_summary.n_total_stim_item, session_summary.n_total_nonstim_item])

            # post stim vs non stim ITEM analysis
            th_stim_post_stim_item_table = th_stim_session_table[th_stim_session_table.is_post_stim_item]
            th_stim_post_non_stim_item_table = th_stim_session_table[~th_stim_session_table.is_post_stim_item & ~th_stim_session_table.is_stim_item & (th_stim_session_table['list']>=8)]
            session_summary.n_correct_post_stim_item = th_stim_post_stim_item_table.recalled.sum()
            #dist_err_stim_item, _ = np.histogram(th_stim_stim_item_table.distance_err,bins=20,range=(0,100))
            #session_summary.dist_err_stim_item = dist_err_stim_item / float(np.sum(dist_err_stim_item))
            session_summary.n_total_post_stim_item = len(th_stim_post_stim_item_table)
            session_summary.pc_from_post_stim_item = 100 * session_summary.n_correct_post_stim_item / float(session_summary.n_total_post_stim_item)
            session_summary.n_correct_post_nonstim_item = th_stim_post_non_stim_item_table.recalled.sum()
            #dist_err_nonstim_item, _ = np.histogram(th_stim_non_stim_item_table.distance_err,bins=20,range=(0,100))
            #session_summary.dist_err_nonstim_item = dist_err_nonstim_item / float(np.sum(dist_err_nonstim_item))
            session_summary.n_total_post_nonstim_item = len(th_stim_post_non_stim_item_table)
            session_summary.pc_from_post_nonstim_item = 100 * session_summary.n_correct_post_nonstim_item / float(session_summary.n_total_post_nonstim_item)
            session_summary.chisqr_post_item, session_summary.pvalue_post_item, _ = proportions_chisquare([session_summary.n_correct_post_stim_item, session_summary.n_correct_post_nonstim_item], [session_summary.n_total_post_stim_item, session_summary.n_total_post_nonstim_item])


            # stim vs non stim CONFIDENCE ITEM analysis
            session_summary.n_stim_mid_high_conf = np.sum(th_stim_stim_item_table.confidence > 0)
            session_summary.pc_stim_mid_high_conf = 100 * session_summary.n_stim_mid_high_conf / float(session_summary.n_total_stim_item)
            session_summary.n_nonstim_mid_high_conf = np.sum(th_stim_non_stim_item_table.confidence > 0)
            session_summary.pc_nonstim_mid_high_conf = 100 * session_summary.n_nonstim_mid_high_conf / float(session_summary.n_total_nonstim_item)
            session_summary.chisqr_conf, session_summary.pvalue_conf, _ = proportions_chisquare([session_summary.n_stim_mid_high_conf, session_summary.n_nonstim_mid_high_conf], [session_summary.n_total_stim_item, session_summary.n_total_nonstim_item])

            th_stim_stim_list_stim_item_table = th_stim_stim_list_table[th_stim_stim_list_table['is_stim_item']]
            th_stim_stim_list_stim_item_low_table = th_stim_stim_list_stim_item_table[th_stim_stim_list_stim_item_table['prev_prob']<thresh]
            th_stim_stim_list_stim_item_high_table = th_stim_stim_list_stim_item_table[th_stim_stim_list_stim_item_table['prev_prob']>thresh]

            th_stim_stim_list_post_stim_item_table = th_stim_stim_list_table[th_stim_stim_list_table['is_post_stim_item']]
            th_stim_stim_list_post_stim_item_low_table = th_stim_stim_list_post_stim_item_table[th_stim_stim_list_post_stim_item_table['prev_prob']<thresh]
            th_stim_stim_list_post_stim_item_high_table = th_stim_stim_list_post_stim_item_table[th_stim_stim_list_post_stim_item_table['prev_prob']>thresh]

            session_summary.mean_prob_diff_all_stim_item = th_stim_stim_list_stim_item_table['prob_diff'].mean()
            session_summary.sem_prob_diff_all_stim_item = th_stim_stim_list_stim_item_table['prob_diff'].sem()
            session_summary.mean_prob_diff_low_stim_item = th_stim_stim_list_stim_item_low_table['prob_diff'].mean()
            session_summary.sem_prob_diff_low_stim_item = th_stim_stim_list_stim_item_low_table['prob_diff'].sem()

            session_summary.mean_prob_diff_all_post_stim_item = th_stim_stim_list_post_stim_item_table['prob_diff'].mean()
            session_summary.sem_prob_diff_all_post_stim_item = th_stim_stim_list_post_stim_item_table['prob_diff'].sem()
            session_summary.mean_prob_diff_low_post_stim_item = th_stim_stim_list_post_stim_item_low_table['prob_diff'].mean()
            session_summary.sem_prob_diff_low_post_stim_item = th_stim_stim_list_post_stim_item_low_table['prob_diff'].sem()

            th_stim_non_stim_list_table = th_stim_non_stim_list_table[(~th_stim_non_stim_list_table['is_stim_list']) & (th_stim_non_stim_list_table['serialpos']>1)]
            th_stim_non_stim_list_low_table = th_stim_non_stim_list_table[th_stim_non_stim_list_table['prev_prob']<thresh]
            th_stim_non_stim_list_high_table = th_stim_non_stim_list_table[th_stim_non_stim_list_table['prev_prob']>thresh]

            session_summary.control_mean_prob_diff_all = th_stim_non_stim_list_table['prob_diff'].mean()
            session_summary.control_sem_prob_diff_all = th_stim_non_stim_list_table['prob_diff'].sem()
            session_summary.control_mean_prob_diff_low = th_stim_non_stim_list_low_table['prob_diff'].mean()
            session_summary.control_sem_prob_diff_low = th_stim_non_stim_list_low_table['prob_diff'].sem()

            stim_item_recall_rate_low = th_stim_stim_list_stim_item_low_table['recalled'].mean()
            stim_item_recall_rate_high = th_stim_stim_list_stim_item_high_table['recalled'].mean()

            post_stim_item_recall_rate_low = th_stim_stim_list_post_stim_item_low_table['recalled'].mean()
            post_stim_item_recall_rate_high = th_stim_stim_list_post_stim_item_high_table['recalled'].mean()

            non_stim_list_recall_rate_low = th_stim_non_stim_list_low_table['recalled'].mean()
            non_stim_list_recall_rate_high = th_stim_non_stim_list_high_table['recalled'].mean()

            recall_rate = session_summary.n_correct_words / float(session_summary.n_words)

            stim_low_pc_diff_from_mean = 100.0 * (stim_item_recall_rate_low-non_stim_list_recall_rate_low) / recall_rate
            stim_high_pc_diff_from_mean = 100.0 * (stim_item_recall_rate_high-non_stim_list_recall_rate_high) / recall_rate

            post_stim_low_pc_diff_from_mean = 100.0 * (post_stim_item_recall_rate_low-non_stim_list_recall_rate_low) / recall_rate
            post_stim_high_pc_diff_from_mean = 100.0 * (post_stim_item_recall_rate_high-non_stim_list_recall_rate_high) / recall_rate

            session_summary.stim_vs_non_stim_pc_diff_from_mean = (stim_low_pc_diff_from_mean, stim_high_pc_diff_from_mean)

            session_summary.post_stim_vs_non_stim_pc_diff_from_mean = (post_stim_low_pc_diff_from_mean, post_stim_high_pc_diff_from_mean)

            session_summary_array.append(session_summary)

        self.pass_object('session_summary_array', session_summary_array)
