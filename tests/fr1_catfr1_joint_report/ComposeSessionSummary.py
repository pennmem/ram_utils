import os
import time
from operator import itemgetter
import numpy as np

import pandas as pd
from matplotlib import cm
from matplotlib.colors import Normalize

from SessionSummary import SessionSummary
from ReportUtils import ReportRamTask


def make_ttest_table(bp_tal_structs, ttest_results):
    contact_nos = bp_tal_structs.channel_1.str.lstrip('0') + '-' + bp_tal_structs.channel_2.str.lstrip('0')
    ttest_data = [list(a) for a in zip(bp_tal_structs.etype.values, contact_nos.values, bp_tal_structs.index.values, bp_tal_structs.bp_atlas_loc, ttest_results[1], ttest_results[0])]
    return ttest_data


def format_ttest_table(table_data):
    for i,line in enumerate(table_data):
        if abs(line[-1]) < 1.5:
            table_data[:] = table_data[:i]
            return table_data
        line[-2] = '%.3f' % line[-2] if line[-2] >= 0.001 else '\\textless.001'
        color = 'red' if line[-1]>=2.0 else 'blue' if line[-1]<=-2.0 else None
        line[-1] = '%.3f' % line[-1]
        if color is not None:
            if color == 'red':
                line[:] = ['\\textbf{\\textcolor{BrickRed}{%s}}' % s for s in line]
            elif color == 'blue':
                line[:] = ['\\textbf{\\textcolor{blue}{%s}}' % s for s in line]


class ComposeSessionSummary(ReportRamTask):
    def __init__(self, params, mark_as_completed=True):
        super(ComposeSessionSummary,self).__init__(mark_as_completed)
        self.params = params
        if self.dependency_inventory:
            self.dependency_inventory.add_dependent_resource(resource_name='localization',
                                        access_path = ['electrodes','localization'])

    def run(self):
        subject = self.pipeline.subject
        task = self.pipeline.task

        events = self.get_passed_object('events')
        events = events[events.type=='WORD']
        math_events = self.get_passed_object('math_events')
        intr_events = self.get_passed_object('intr_events')
        rec_events = self.get_passed_object('rec_events')
        all_events = self.get_passed_object('all_events')
        monopolar_channels = self.get_passed_object('monopolar_channels')
        bp_tal_structs = self.get_passed_object('bp_tal_structs')
        cat_events = self.get_passed_object('cat_events')

        ttest = self.get_passed_object('ttest')

        xval_output = self.get_passed_object('xval_output')
        perm_test_pvalue = self.get_passed_object('pvalue')
        joint_xval_output = self.get_passed_object('joint_xval_output')
        joint_perm_test_pvalue = self.get_passed_object('joint_pvalue')

        sessions = np.unique(events.session)

        self.pass_object('NUMBER_OF_SESSIONS', len(sessions))
        self.pass_object('NUMBER_OF_ELECTRODES', len(monopolar_channels))

        session_data = []
        session_summary_array = []

        positions = np.unique(events.serialpos)
        first_recall_counter = np.zeros(positions.size, dtype=int)
        total_list_counter = 0

        cat_recalled_events = cat_events[cat_events.recalled == 1]
        irt_within_cat = []
        irt_between_cat = []
        for session in np.unique(cat_events.session):
            cat_sess_recalls = cat_recalled_events[cat_recalled_events.session == session]
            for list in np.unique(cat_sess_recalls.list):
                cat_sess_list_recalls = cat_sess_recalls[cat_sess_recalls.list == list]
                irts = np.diff(cat_sess_list_recalls.mstime)
                within = np.diff(cat_sess_list_recalls.category_num) == 0
                irt_within_cat.extend(irts[within])
                irt_between_cat.extend(irts[within == False])

        for session in sessions:
            session_summary = SessionSummary()

            session_events = events[events.session == session]
            n_sess_events = len(session_events)

            session_rec_events = rec_events[rec_events.session == session]

            session_all_events = all_events[all_events.session == session]
            timestamps = sorted(session_all_events.mstime)
            first_time_stamp = timestamps[0]
            last_time_stamp = timestamps[-3]
            session_length = '%.2f' % ((last_time_stamp - first_time_stamp) / 60000.0)
            session_date = time.strftime('%d-%b-%Y', time.localtime(last_time_stamp/1000))

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

            session_design = 'FR1 $%d$'%session if session<100 else 'CatFR1 $%d$'%(session-100)
            session_data.append([session_design, session_date, session_length, n_lists, '$%.2f$\\%%' % session_summary.pc_correct_words])

            prob_first_recall = np.zeros(len(positions), dtype=float)
            for lst in lists:
                list_rec_events = session_rec_events[(session_rec_events.list == lst) & (session_rec_events.intrusion == 0)]
                if list_rec_events.size > 0:
                    list_events = session_events[session_events.list == lst]
                    tmp = np.where(list_events.item_num == list_rec_events[0].item_num)[0]
                    if tmp.size > 0:
                        first_recall_idx = tmp[0]
                        prob_first_recall[first_recall_idx] += 1
                        first_recall_counter[first_recall_idx] += 1
            prob_first_recall /= float(n_lists)
            total_list_counter += n_lists

            session_summary.prob_first_recall = prob_first_recall

            if math_events is not None:
                session_math_events = math_events[math_events.session == session]
                session_summary.n_math = len(session_math_events)
                session_summary.n_correct_math = np.sum(session_math_events.iscorrect)
                session_summary.pc_correct_math = 100*session_summary.n_correct_math / float(session_summary.n_math)
                session_summary.math_per_list = session_summary.n_math / float(n_lists)

            session_intr_events = intr_events[intr_events.session == session]

            n_sess_rec_events = len(session_rec_events)
            session_summary.n_pli = np.sum(session_intr_events.intrusion > 0)
            session_summary.pc_pli = 100*session_summary.n_pli / float(n_sess_rec_events)
            session_summary.n_eli = np.sum(session_intr_events.intrusion == -1)
            session_summary.pc_eli = 100*session_summary.n_eli / float(n_sess_rec_events)

            session_xval_output = xval_output[session]

            session_summary.auc = '%.2f' % (100*session_xval_output.auc)
            session_summary.fpr = session_xval_output.fpr
            session_summary.tpr = session_xval_output.tpr
            session_summary.pc_diff_from_mean = (session_xval_output.low_pc_diff_from_mean, session_xval_output.mid_pc_diff_from_mean, session_xval_output.high_pc_diff_from_mean)

            session_summary_array.append(session_summary)

        self.pass_object('SESSION_DATA', session_data)
        self.pass_object('session_summary_array', session_summary_array)

        cumulative_summary = SessionSummary()
        cumulative_summary.n_words = len(events)
        cumulative_summary.n_correct_words = np.sum(events.recalled)
        cumulative_summary.pc_correct_words = 100*cumulative_summary.n_correct_words / float(cumulative_summary.n_words)

        cumulative_summary.irt_between_cat = irt_between_cat
        cumulative_summary.irt_within_cat = irt_within_cat

        positions = np.unique(events.serialpos)
        prob_recall = np.empty_like(positions, dtype=float)
        for i,pos in enumerate(positions):
            pos_events = events[events.serialpos == pos]
            prob_recall[i] = np.sum(pos_events.recalled) / float(len(pos_events))
        cumulative_summary.prob_recall = prob_recall

        prob_first_recall = first_recall_counter / float(total_list_counter)
        cumulative_summary.prob_first_recall = prob_first_recall

        if math_events is not None:
            cumulative_summary.n_math = len(math_events)
            cumulative_summary.n_correct_math = np.sum(math_events.iscorrect)
            cumulative_summary.pc_correct_math = 100*cumulative_summary.n_correct_math / float(cumulative_summary.n_math)
            cumulative_summary.math_per_list = cumulative_summary.n_math / float(total_list_counter)

        n_rec_events = len(rec_events)
        cumulative_summary.n_pli = np.sum(intr_events.intrusion > 0)
        cumulative_summary.pc_pli = 100*cumulative_summary.n_pli / float(n_rec_events)
        cumulative_summary.n_eli = np.sum(intr_events.intrusion == -1)
        cumulative_summary.pc_eli = 100*cumulative_summary.n_eli / float(n_rec_events)

        cumulative_xval_output = xval_output[-1]

        cumulative_summary.auc = '%.2f' % (100*cumulative_xval_output.auc)
        cumulative_summary.fpr = cumulative_xval_output.fpr
        cumulative_summary.tpr = cumulative_xval_output.tpr
        cumulative_summary.pc_diff_from_mean = (cumulative_xval_output.low_pc_diff_from_mean, cumulative_xval_output.mid_pc_diff_from_mean, cumulative_xval_output.high_pc_diff_from_mean)
        cumulative_summary.perm_AUCs = self.get_passed_object('perm_AUCs')
        cumulative_summary.perm_test_pvalue = ('= %.3f' % perm_test_pvalue) if perm_test_pvalue>=0.001 else '\leq 0.001'
        cumulative_summary.jstat_thresh = '%.3f' % cumulative_xval_output.jstat_thresh
        cumulative_summary.jstat_percentile = '%.2f' % (100.0*cumulative_xval_output.jstat_quantile)


        cumulative_xval_output = joint_xval_output[-1]

        cumulative_summary.joint_auc = '%.2f' % (100*cumulative_xval_output.auc)
        cumulative_summary.joint_fpr = cumulative_xval_output.fpr
        cumulative_summary.joint_tpr = cumulative_xval_output.tpr
        cumulative_summary.joint_pc_diff_from_mean = (cumulative_xval_output.low_pc_diff_from_mean, cumulative_xval_output.mid_pc_diff_from_mean, cumulative_xval_output.high_pc_diff_from_mean)
        cumulative_summary.joint_perm_AUCs = self.get_passed_object('perm_AUCs')
        cumulative_summary.joint_perm_test_pvalue = ('= %.3f' % joint_perm_test_pvalue) if joint_perm_test_pvalue>=0.001 else '\leq 0.001'
        cumulative_summary.joint_jstat_thresh = '%.3f' % cumulative_xval_output.jstat_thresh
        cumulative_summary.joint_jstat_percentile = '%.2f' % (100.0*cumulative_xval_output.jstat_quantile)



        self.pass_object('cumulative_summary', cumulative_summary)

        # cumulative_ttest_data = [list(a) for a in zip(bp_tal_structs.eType, bp_tal_structs.tagName, ttest[-1][1], ttest[-1][0])]
        cumulative_ttest_data = make_ttest_table(bp_tal_structs, ttest[-1])
        cumulative_ttest_data.sort(key=itemgetter(-2))
        ttest_table = pd.DataFrame(data=cumulative_ttest_data,columns=
        ['Etype','contactno','TagName','atlas_loc','pvalue','t-stat'])
        ttest_table['Group'] = 'Subsequent memory effect (t-stat)'
        norm = Normalize(vmin=-3,vmax=3)
        cmapper = cm.ScalarMappable(norm=norm, cmap=cm.bwr)
        colors = cmapper.to_rgba(ttest_table['t-stat'])[:,:3]
        colors = pd.DataFrame(data=colors,columns = ['R','G','B'])
        ttest_table = pd.concat([ttest_table,colors],axis=1)
        ttest_table[['Group', 'TagName', 't-stat', 'R', 'G', 'B']].to_csv(
            os.path.join(self.workspace_dir, '_'.join([self.pipeline.subject, task, 'SME_ttest.csv'])), )
        self.pass_object('SME_file',os.path.join(self.workspace_dir, '_'.join([self.pipeline.subject, task, 'SME_ttest.csv'])))


        cumulative_ttest_data = format_ttest_table(cumulative_ttest_data)

        self.pass_object('cumulative_ttest_data', cumulative_ttest_data)
