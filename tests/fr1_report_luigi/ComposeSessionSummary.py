
from SessionSummary import SessionSummary
import pandas as pd
from matplotlib import cm
from matplotlib.colors import Normalize
import time
from operator import itemgetter
import os.path
import numpy as np
from RamTaskL import RamTaskL
from FR1EventPreparation import FR1EventPreparation
from MontagePreparation import MontagePreparation
from ComputeFR1Powers import ComputeFR1Powers
from RepetitionRatio import RepetitionRatio
from ComputeClassifier import ComputeClassifier
from ComputeClassifier import ComputeJointClassifier
from ComputeTTest import ComputeTTest

def make_ttest_table(bp_tal_structs, ttest_results):
    contact_nos = bp_tal_structs.channel_1.str.lstrip('0') + '-' + bp_tal_structs.channel_2.str.lstrip('0')
    ttest_data = [list(a) for a in zip(bp_tal_structs.etype.values, contact_nos.values, bp_tal_structs.index.values,
                                       bp_tal_structs.bp_atlas_loc, ttest_results[1], ttest_results[0])]
    return ttest_data


def format_ttest_table(table_data):
    for i, line in enumerate(table_data):
        if abs(line[-1]) < 1.5:
            table_data[:] = table_data[:i]
            return table_data
        line[-2] = '%.3f' % line[-2] if line[-2] >= 0.001 else '\\textless.001'
        color = 'red' if line[-1] >= 2.0 else 'blue' if line[-1] <= -2.0 else None
        line[-1] = '%.3f' % line[-1]
        if color is not None:
            if color == 'red':
                line[:] = ['\\textbf{\\textcolor{BrickRed}{%s}}' % s for s in line]
            elif color == 'blue':
                line[:] = ['\\textbf{\\textcolor{blue}{%s}}' % s for s in line]


class ComposeSessionSummary(RamTaskL):
    params = None

    def define_outputs(self):

        self.add_file_resource('NUMBER_OF_SESSIONS')
        self.add_file_resource('NUMBER_OF_ELECTRODES')
        self.add_file_resource('SESSION_DATA')
        self.add_file_resource('session_summary_array')
        self.add_file_resource('cumulative_summary')
        self.add_file_resource('ttest_table')
        self.add_file_resource('SME_file')
        self.add_file_resource('cumulative_ttest_data')

    def requires(self):
        yield FR1EventPreparation(pipeline=self.pipeline)
        yield ComputeFR1Powers(pipeline=self.pipeline)
        yield MontagePreparation(pipeline=self.pipeline)
        yield RepetitionRatio(pipeline=self.pipeline)
        yield ComputeClassifier(pipeline=self.pipeline)
        yield ComputeJointClassifier(pipeline=self.pipeline)
        yield ComputeTTest(pipeline=self.pipeline)


    def run_impl(self):
        self.params = self.pipeline.params
        task = self.pipeline.task

        events = self.get_passed_object(task + '_events')
        events = events[events.type == 'WORD']
        math_events = self.get_passed_object(task + '_math_events')
        intr_events = self.get_passed_object(task + '_intr_events')
        rec_events = self.get_passed_object(task + '_rec_events')
        all_events = self.get_passed_object(task + '_all_events')
        monopolar_channels = self.get_passed_object('monopolar_channels')
        bp_tal_structs = self.get_passed_object('bp_tal_structs')

        if 'cat' in task:
            repetition_ratios = self.get_passed_object('repetition_ratios')

        ttest = self.get_passed_object('ttest')

        xval_output = self.get_passed_object('xval_output')
        perm_test_pvalue = self.get_passed_object('pvalue')

        joint_xval_output = self.get_passed_object('joint_xval_output')
        joint_perm_test_pvalue = self.get_passed_object('joint_pvalue')

        sessions = np.unique(events.session)

        self.pass_object('NUMBER_OF_SESSIONS', len(sessions))
        self.pass_object('NUMBER_OF_ELECTRODES', len(monopolar_channels))

        session_data = []

        positions = np.unique(events[events.serialpos != -999].serialpos)
        first_recall_counter = np.zeros(positions.size, dtype=int)
        total_list_counter = 0

        irt_within_cat = []
        irt_between_cat = []
        session_summary_array = []

        for session in sessions:
            session_summary = SessionSummary()

            session_events = events[events.session == session]
            n_sess_events = len(session_events)

            session_rec_events = rec_events[rec_events.session == session]
            n_sess_rec_events = len(session_rec_events)

            session_all_events = all_events[all_events.session == session]
            timestamps = sorted(session_all_events.mstime)
            first_time_stamp = timestamps[0]
            last_time_stamp = timestamps[-3]
            session_length = '%.2f' % ((last_time_stamp - first_time_stamp) / 60000.0)
            session_date = time.strftime('%d-%b-%Y', time.localtime(last_time_stamp / 1000))

            session_name = 'Sess%02d' % session

            print 'Session =', session_name

            session_summary.number = session
            session_summary.name = session_name
            session_summary.length = session_length
            session_summary.date = session_date
            session_summary.n_words = len(session_events)
            session_summary.n_correct_words = np.sum(session_events.recalled)
            session_summary.pc_correct_words = 100 * session_summary.n_correct_words / float(session_summary.n_words)

            positions = np.unique(session_events[session_events.serialpos != -999].serialpos)
            prob_recall = np.empty_like(positions, dtype=float)
            for i, pos in enumerate(positions):
                pos_events = session_events[session_events.serialpos == pos]
                prob_recall[i] = np.sum(pos_events.recalled) / float(len(pos_events))

            session_summary.prob_recall = prob_recall

            lists = np.unique(session_events.list)
            n_lists = len(lists)

            session_data.append(
                [session, session_date, session_length, n_lists, '$%.2f$\\%%' % session_summary.pc_correct_words])

            prob_first_recall = np.zeros(len(positions), dtype=float)
            session_irt_within_cat = []
            session_irt_between_cat = []
            for lst in lists:
                list_rec_events = session_rec_events[
                    (session_rec_events.list == lst) & (session_rec_events.intrusion == 0)]
                if list_rec_events.size > 0:
                    list_events = session_events[session_events.list == lst]
                    tmp = np.where(list_events.item_name == list_rec_events[0].item_name)[0]
                    if tmp.size > 0:
                        first_recall_idx = tmp[0]
                        prob_first_recall[first_recall_idx] += 1
                        first_recall_counter[first_recall_idx] += 1
                if task == 'catFR1':
                    # list_rec_events = session_rec_events[session_rec_events.list == lst]
                    for i in xrange(1, len(list_rec_events)):
                        cur_ev = list_rec_events[i]
                        prev_ev = list_rec_events[i - 1]
                        # if (cur_ev.intrusion == 0) and (prev_ev.intrusion == 0):
                        dt = cur_ev.mstime - prev_ev.mstime
                        if cur_ev.category == prev_ev.category:
                            session_irt_within_cat.append(dt)
                        else:
                            session_irt_between_cat.append(dt)
            prob_first_recall /= float(n_lists)
            total_list_counter += n_lists

            session_summary.irt_within_cat = sum(session_irt_within_cat) / len(
                session_irt_within_cat) if session_irt_within_cat else 0.0
            session_summary.irt_between_cat = sum(session_irt_between_cat) / len(
                session_irt_between_cat) if session_irt_between_cat else 0.0

            irt_within_cat += session_irt_within_cat
            irt_between_cat += session_irt_between_cat

            session_summary.prob_first_recall = prob_first_recall

            if math_events is not None:
                session_math_events = math_events[math_events.session == session]
                session_summary.n_math = len(session_math_events)
                session_summary.n_correct_math = np.sum(session_math_events.iscorrect)
                session_summary.pc_correct_math = 100 * session_summary.n_correct_math / float(session_summary.n_math)
                session_summary.math_per_list = session_summary.n_math / float(n_lists)

            session_intr_events = intr_events[intr_events.session == session]

            session_summary.n_pli = np.sum(session_intr_events.intrusion > 0)
            session_summary.pc_pli = 100 * session_summary.n_pli / float(n_sess_rec_events)
            session_summary.n_eli = np.sum(session_intr_events.intrusion == -1)
            session_summary.pc_eli = 100 * session_summary.n_eli / float(n_sess_rec_events)

            session_xval_output = xval_output[session]

            session_summary.auc = '%.2f' % (100 * session_xval_output.auc)
            session_summary.fpr = session_xval_output.fpr
            session_summary.tpr = session_xval_output.tpr
            session_summary.pc_diff_from_mean = (
            session_xval_output.low_pc_diff_from_mean, session_xval_output.mid_pc_diff_from_mean,
            session_xval_output.high_pc_diff_from_mean)

            session_summary_array.append(session_summary)

        self.pass_object('SESSION_DATA', session_data)
        self.pass_object('session_summary_array', session_summary_array)

        cumulative_summary = SessionSummary()
        cumulative_summary.n_words = len(events)
        cumulative_summary.n_correct_words = np.sum(events.recalled)
        cumulative_summary.pc_correct_words = 100 * cumulative_summary.n_correct_words / float(
            cumulative_summary.n_words)

        cumulative_summary.irt_within_cat = sum(irt_within_cat) / len(irt_within_cat) if irt_within_cat else 0.0
        cumulative_summary.irt_between_cat = sum(irt_between_cat) / len(irt_between_cat) if irt_between_cat else 0.0

        positions = np.unique(events[events.serialpos != -999].serialpos)
        prob_recall = np.empty_like(positions, dtype=float)
        for i, pos in enumerate(positions):
            pos_events = events[events.serialpos == pos]
            prob_recall[i] = np.sum(pos_events.recalled) / float(len(pos_events))
        cumulative_summary.prob_recall = prob_recall

        prob_first_recall = first_recall_counter / float(total_list_counter)
        cumulative_summary.prob_first_recall = prob_first_recall

        if math_events is not None:
            cumulative_summary.n_math = len(math_events)
            cumulative_summary.n_correct_math = np.sum(math_events.iscorrect)
            cumulative_summary.pc_correct_math = 100 * cumulative_summary.n_correct_math / float(
                cumulative_summary.n_math)
            cumulative_summary.math_per_list = cumulative_summary.n_math / float(total_list_counter)

        n_rec_events = len(rec_events)
        cumulative_summary.n_pli = np.sum(intr_events.intrusion > 0)
        cumulative_summary.pc_pli = 100 * cumulative_summary.n_pli / float(n_rec_events)
        cumulative_summary.n_eli = np.sum(intr_events.intrusion == -1)
        cumulative_summary.pc_eli = 100 * cumulative_summary.n_eli / float(n_rec_events)

        cumulative_xval_output = xval_output[-1]

        if 'cat' in task:
            cumulative_summary.repetition_ratio = repetition_ratios

        cumulative_summary.auc = '%.2f' % (100 * cumulative_xval_output.auc)
        cumulative_summary.fpr = cumulative_xval_output.fpr
        cumulative_summary.tpr = cumulative_xval_output.tpr
        cumulative_summary.pc_diff_from_mean = (
        cumulative_xval_output.low_pc_diff_from_mean, cumulative_xval_output.mid_pc_diff_from_mean,
        cumulative_xval_output.high_pc_diff_from_mean)
        cumulative_summary.perm_AUCs = self.get_passed_object('perm_AUCs')
        cumulative_summary.perm_test_pvalue = (
        '= %.3f' % perm_test_pvalue) if perm_test_pvalue >= 0.001 else '\leq 0.001'
        cumulative_summary.jstat_thresh = '%.3f' % cumulative_xval_output.jstat_thresh
        cumulative_summary.jstat_percentile = '%.2f' % (100.0 * cumulative_xval_output.jstat_quantile)

        cumulative_xval_output = joint_xval_output[-1]

        cumulative_summary.joint_auc = '%.2f' % (100 * cumulative_xval_output.auc)
        cumulative_summary.joint_fpr = cumulative_xval_output.fpr
        cumulative_summary.joint_tpr = cumulative_xval_output.tpr
        cumulative_summary.joint_pc_diff_from_mean = (
        cumulative_xval_output.low_pc_diff_from_mean, cumulative_xval_output.mid_pc_diff_from_mean,
        cumulative_xval_output.high_pc_diff_from_mean)
        cumulative_summary.joint_perm_AUCs = self.get_passed_object('perm_AUCs')
        cumulative_summary.joint_perm_test_pvalue = (
        '= %.3f' % joint_perm_test_pvalue) if joint_perm_test_pvalue >= 0.001 else '\leq 0.001'
        cumulative_summary.joint_jstat_thresh = '%.3f' % cumulative_xval_output.jstat_thresh
        cumulative_summary.joint_jstat_percentile = '%.2f' % (100.0 * cumulative_xval_output.jstat_quantile)

        self.pass_object('cumulative_summary', cumulative_summary)

        # cumulative_ttest_data = [list(a) for a in zip(bp_tal_structs.eType, bp_tal_structs.tagName, ttest[-1][1], ttest[-1][0])]
        cumulative_ttest_data = make_ttest_table(bp_tal_structs, ttest[-1])
        cumulative_ttest_data.sort(key=itemgetter(-2))
        ttest_table = pd.DataFrame(data=cumulative_ttest_data, columns=
        ['Etype', 'contactno', 'TagName', 'atlas_loc', 'pvalue', 't-stat'])
        ttest_table['Group'] = 'Subsequent memory effect (t-stat)'
        norm = Normalize(vmin=-3, vmax=3)
        cmapper = cm.ScalarMappable(norm=norm, cmap=cm.bwr)
        colors = cmapper.to_rgba(ttest_table['t-stat'])[:, :3]
        colors = pd.DataFrame(data=colors, columns=['R', 'G', 'B'])
        ttest_table = pd.concat([ttest_table, colors], axis=1)
        self.pass_object('ttest_table', ttest_table)

        ttest_table[['Group', 'TagName', 't-stat', 'R', 'G', 'B']].to_csv(
            os.path.join(self.pipeline.workspace_dir, '_'.join([self.pipeline.subject, task, 'SME_ttest.csv'])), )
        self.pass_object('SME_file',
                         os.path.join(self.pipeline.workspace_dir, '_'.join([self.pipeline.subject, task, 'SME_ttest.csv'])))
        cumulative_ttest_data = format_ttest_table(cumulative_ttest_data)

        self.pass_object('cumulative_ttest_data', cumulative_ttest_data)
