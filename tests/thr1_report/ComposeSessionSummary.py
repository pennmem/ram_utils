from RamPipeline import *
from SessionSummary import SessionSummary
import pandas as pd
from matplotlib import cm
from matplotlib.colors import Normalize

import numpy as np
import time
from operator import itemgetter

from ReportUtils import ReportRamTask
from scipy import stats

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

        events = self.get_passed_object(task + '_events')
        all_events = self.get_passed_object(task + '_all_events')
        monopolar_channels = self.get_passed_object('monopolar_channels')
        bp_tal_structs = self.get_passed_object('bp_tal_structs')

        ttest = self.get_passed_object('ttest')

        xval_output = self.get_passed_object('xval_output')
        perm_test_pvalue = self.get_passed_object('pvalue')

        sessions = np.unique(events.session)

        self.pass_object('NUMBER_OF_SESSIONS', len(sessions))
        self.pass_object('NUMBER_OF_ELECTRODES', len(monopolar_channels))

        session_data = []
        total_list_counter = 0
        session_summary_array = []

        for session in sessions:
            session_summary = SessionSummary()

            session_events = events[events.session == session]
            n_sess_events = len(session_events)

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

            probe_positions = np.unique(session_events.probepos)
            prob_probe_recall = np.empty_like(probe_positions, dtype=float)
            for i,pos in enumerate(probe_positions):
                pos_events = session_events[session_events.probepos == pos]
                prob_probe_recall[i] = np.sum(pos_events.recalled) / float(len(pos_events))
            session_summary.prob_probe_recall = prob_probe_recall

            lists = np.unique(session_events.trial)
            n_lists = len(lists)

            session_data.append([session, session_date, session_length, n_lists, '$%.2f$\\%%' % session_summary.pc_correct_words])
            total_list_counter += n_lists

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

        positions = np.unique(events.serialpos)
        prob_recall = np.empty_like(positions, dtype=float)
        for i,pos in enumerate(positions):
            pos_events = events[events.serialpos == pos]
            prob_recall[i] = np.sum(pos_events.recalled) / float(len(pos_events))
        cumulative_summary.prob_recall = prob_recall

        probe_positions = np.unique(events.probepos)
        prob_probe_recall = np.empty_like(probe_positions, dtype=float)
        for i,pos in enumerate(probe_positions):
            pos_events = events[events.probepos == pos]
            prob_probe_recall[i] = np.sum(pos_events.recalled) / float(len(pos_events))
        cumulative_summary.prob_probe_recall = prob_probe_recall

        cumulative_xval_output = xval_output[-1]
        cumulative_summary.auc = '%.2f' % (100*cumulative_xval_output.auc)
        cumulative_summary.fpr = cumulative_xval_output.fpr
        cumulative_summary.tpr = cumulative_xval_output.tpr
        cumulative_summary.pc_diff_from_mean = (cumulative_xval_output.low_pc_diff_from_mean, cumulative_xval_output.mid_pc_diff_from_mean, cumulative_xval_output.high_pc_diff_from_mean)
        cumulative_summary.perm_AUCs = self.get_passed_object('perm_AUCs')
        cumulative_summary.perm_test_pvalue = ('= %.3f' % perm_test_pvalue) if perm_test_pvalue>=0.001 else '\leq 0.001'
        cumulative_summary.jstat_thresh = '%.3f' % cumulative_xval_output.jstat_thresh
        cumulative_summary.jstat_percentile = '%.2f' % (100.0*cumulative_xval_output.jstat_quantile)

        self.pass_object('cumulative_summary', cumulative_summary)

        # cumulative_ttest_data = [list(a) for a in zip(bp_tal_structs.eType, bp_tal_structs.tagName, ttest[-1][1], ttest[-1][0])]
        ttest_tables = []
        cumulative_ttest_datas = []
        sme_files = []

        for i, freq_name in enumerate(self.params.ttest_names):

            ttest_res = (ttest[-1][0][:, i], ttest[-1][1][:, i])
            cumulative_ttest_data = make_ttest_table(bp_tal_structs, ttest_res)
            # cumulative_ttest_data = make_ttest_table(bp_tal_structs, ttest[-1])
            cumulative_ttest_data.sort(key=itemgetter(-2))
            ttest_table = pd.DataFrame(data=cumulative_ttest_data,columns=
            ['Etype','contactno','TagName','atlas_loc','pvalue','t-stat'])
            ttest_table['Group'] = 'Subsequent memory effect (t-stat)'
            norm = Normalize(vmin=-3,vmax=3)
            cmapper = cm.ScalarMappable(norm=norm, cmap=cm.bwr)
            colors = cmapper.to_rgba(ttest_table['t-stat'])[:,:3]
            colors = pd.DataFrame(data=colors,columns = ['R','G','B'])
            ttest_table = pd.concat([ttest_table,colors],axis=1)
            ttest_table[['Group','TagName','t-stat','R','G','B']].to_csv(
                os.path.join(self.workspace_dir,'_'.join([self.pipeline.subject,task,'SME_ttest_%s.csv' % freq_name])),)
            cumulative_ttest_data = format_ttest_table(cumulative_ttest_data)

            ttest_tables.append(ttest_table)
            cumulative_ttest_datas.append(cumulative_ttest_data)
            sme_files.append(os.path.join(self.workspace_dir,'_'.join([self.pipeline.subject,task,'SME_ttest_%s.csv' % freq_name])))

        self.pass_object('ttest_table',ttest_tables)
        self.pass_object('SME_file',sme_files)
        self.pass_object('cumulative_ttest_data', cumulative_ttest_datas)
