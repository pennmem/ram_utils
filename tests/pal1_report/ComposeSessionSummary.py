from RamPipeline import *
from SessionSummary import SessionSummary

import numpy as np
import pandas as pd
import time
from operator import itemgetter

from statsmodels.stats.proportion import proportion_confint


def make_atlas_loc(tag, atlas_loc, comments):

    def colon_connect(s1, s2):
        if isinstance(s1, pd.Series):
            s1 = s1.values[0]
        if isinstance(s2, pd.Series):
            s2 = s2.values[0]
        return s1 if (s2 is None or s2=='' or s2 is np.nan) else s2 if (s1 is None or s1=='' or s1 is np.nan) else s1 + ': ' + s2

    e1, e2 = tag.split('-')
    if (e1 in atlas_loc.index) and (e2 in atlas_loc.index):
        return colon_connect(atlas_loc.ix[e1], comments.ix[e1] if comments is not None else None), colon_connect(atlas_loc.ix[e2], comments.ix[e2] if comments is not None else None)
    elif tag in atlas_loc.index:
        return colon_connect(atlas_loc.ix[tag], comments.ix[tag] if comments is not None else None), colon_connect(atlas_loc.ix[tag], comments.ix[tag] if comments is not None else None)
    else:
        return '--', '--'


def make_ttest_table(bipolar_pairs, loc_info, ttest_results):
    ttest_data = None
    has_depth = ('Das Volumetric Atlas Location' in loc_info)
    has_surface_only = ('Freesurfer Desikan Killiany Surface Atlas Location' in loc_info)
    if has_depth or has_surface_only:
        atlas_loc = loc_info['Das Volumetric Atlas Location' if has_depth else 'Freesurfer Desikan Killiany Surface Atlas Location']
        comments = loc_info['Comments'] if ('Comments' in loc_info) else None
        n = len(bipolar_pairs)
        ttest_data = [list(a) for a in zip(bipolar_pairs.eType, bipolar_pairs.tagName, [None] * n, [None] * n, ttest_results[1], ttest_results[0])]
        for i, tag in enumerate(bipolar_pairs.tagName):
            ttest_data[i][2], ttest_data[i][3] = make_atlas_loc(tag, atlas_loc, comments)
    else:
        ttest_data = [list(a) for a in zip(bipolar_pairs.eType, bipolar_pairs.tagName, ttest_results[1], ttest_results[0])]

    return ttest_data

def make_ttest_table_header(loc_info):
    table_format = table_header = None
    if ('Das Volumetric Atlas Location' in loc_info) or ('Freesurfer Desikan Killiany Surface Atlas Location' in loc_info):
        table_format = 'C{.75cm} C{2.5cm} C{4cm} C{4cm} C{1.25cm} C{1.25cm}'
        table_header = r'Type & Electrode Pair & Atlas Loc1 & Atlas Loc2 & \textit{p} & \textit{t}-stat'
    else:
        table_format = 'C{.75cm} C{2.5cm} C{1.25cm} C{1.25cm}'
        table_header = r'Type & Electrode Pair & \textit{p} & \textit{t}-stat'
    return table_format, table_header

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

def count_nonvoc_passes(events):
    n_nonvoc_pass = 0
    for i,ev in enumerate(events):
        if ev.type == 'REC_START':
            j = i+1
            while events[j].resp_word == '<>':
                j += 1
            if events[j].type == 'REC_END':
                n_nonvoc_pass += 1
    return n_nonvoc_pass


class ComposeSessionSummary(RamTask):
    def __init__(self, params, mark_as_completed=True):
        RamTask.__init__(self, mark_as_completed)
        self.params = params
        if self.dependency_inventory:
            self.dependency_inventory.add_dependent_resource(resource_name='localization',
                                        access_path = ['electrodes','localization'])

    def run(self):
        subject = self.pipeline.subject
        task = self.pipeline.task

        events = self.get_passed_object(task + '_events')
        math_events = self.get_passed_object(task + '_math_events')
        intr_events = self.get_passed_object(task + '_intr_events')
        rec_events = self.get_passed_object(task + '_rec_events')
        test_probe_events = self.get_passed_object(task + '_test_probe_events')
        all_events = self.get_passed_object(task + '_all_events')
        monopolar_channels = self.get_passed_object('monopolar_channels')
        bipolar_pairs = self.get_passed_object('bipolar_pairs')
        loc_info = self.get_passed_object('loc_info')

        ttest = self.get_passed_object('ttest')

        xval_output = self.get_passed_object('xval_output')
        perm_test_pvalue = self.get_passed_object('pvalue')

        sessions = np.unique(events.session)

        self.pass_object('NUMBER_OF_SESSIONS', len(sessions))
        self.pass_object('NUMBER_OF_ELECTRODES', len(monopolar_channels))

        session_data = []
        session_summary_array = []
        session_ttest_data = []

        total_list_counter = 0
        n_success_per_lag = np.zeros(11, dtype=np.int)
        n_events_per_lag = np.zeros(11, dtype=np.int)

        for session in sessions:
            session_summary = SessionSummary()

            session_events = events[events.session == session]
            n_sess_events = len(session_events)

            session_rec_events = rec_events[rec_events.session == session]

            session_test_probe_events = test_probe_events[test_probe_events.session == session]

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
            session_summary.n_pairs = len(session_events)
            session_summary.n_correct_pairs = np.sum(session_events.correct)
            session_summary.pc_correct_pairs = 100*session_summary.n_correct_pairs / float(session_summary.n_pairs)

            positions = np.unique(session_events.serialpos)
            session_summary.prob_recall = np.empty_like(positions, dtype=float)
            for i,pos in enumerate(positions):
                pos_events = session_events[session_events.serialpos == pos]
                session_summary.prob_recall[i] = np.sum(pos_events.correct) / float(len(pos_events))

            lists = np.unique(session_events.list)
            n_lists = len(lists)
            total_list_counter += n_lists

            session_data.append([session, session_date, session_length, n_lists, '$%.2f$\\%%' % session_summary.pc_correct_pairs])

            for lst in lists:
                list_rec_events = session_rec_events[session_rec_events.list==lst]
                for ev in list_rec_events:
                    lag = 5 + ev.probepos - ev.serialpos
                    if ev.correct==1:
                        n_success_per_lag[lag] += 1

                list_test_probe_events = session_test_probe_events[session_test_probe_events.list==lst]
                for ev in list_test_probe_events:
                    lag = 5 + ev.probepos - ev.serialpos
                    n_events_per_lag[lag] += 1

            if math_events is not None:
                session_math_events = math_events[math_events.session == session]
                session_summary.n_math = len(session_math_events)
                session_summary.n_correct_math = np.sum(session_math_events.iscorrect)
                session_summary.pc_correct_math = 100*session_summary.n_correct_math / float(session_summary.n_math)
                session_summary.math_per_list = session_summary.n_math / float(n_lists)

            session_intr_events = intr_events[intr_events.session == session]

            session_summary.n_pli = np.sum(session_intr_events.intrusion >= 0)
            session_summary.pc_pli = 100*session_summary.n_pli / float(n_sess_events)
            session_summary.n_eli = np.sum(session_intr_events.intrusion == -1)
            session_summary.pc_eli = 100*session_summary.n_eli / float(n_sess_events)

            session_xval_output = xval_output[session]

            session_summary.auc = '%.2f' % (100*session_xval_output.auc)
            session_summary.fpr = session_xval_output.fpr
            session_summary.tpr = session_xval_output.tpr
            session_summary.pc_diff_from_mean = (session_xval_output.low_pc_diff_from_mean, session_xval_output.mid_pc_diff_from_mean, session_xval_output.high_pc_diff_from_mean)

            session_summary_array.append(session_summary)

            session_ttest = ttest[session]
            if isinstance(session_ttest,tuple):
                if ('Das Volumetric Atlas Location' in loc_info) or ('Freesurfer Desikan Killiany Surface Atlas Location' in loc_info):
                    session_ttest_data.append([[None, None, None, None, np.nan, np.nan]])
                else:
                    session_ttest_data.append([[None, None, np.nan, np.nan]])
            else:
                ttest_data = make_ttest_table(bipolar_pairs, loc_info, session_ttest)
                ttest_data.sort(key=itemgetter(-2))
                ttest_data = format_ttest_table(ttest_data)
                session_ttest_data.append(ttest_data)

        self.pass_object('SESSION_DATA', session_data)
        self.pass_object('session_summary_array', session_summary_array)
        self.pass_object('session_ttest_data', session_ttest_data)

        cumulative_summary = SessionSummary()
        cumulative_summary.n_pairs = len(events)
        cumulative_summary.n_correct_pairs = np.sum(events.correct)
        cumulative_summary.pc_correct_pairs = 100*cumulative_summary.n_correct_pairs / float(cumulative_summary.n_pairs)
        cumulative_summary.wilson1, cumulative_summary.wilson2 = proportion_confint(cumulative_summary.n_correct_pairs, cumulative_summary.n_pairs, alpha=0.05, method='wilson')
        cumulative_summary.wilson1 *= 100.0
        cumulative_summary.wilson2 *= 100.0

        cumulative_summary.n_voc_pass = np.sum(events['pass']==1)
        cumulative_summary.pc_voc_pass = 100*cumulative_summary.n_voc_pass / float(cumulative_summary.n_pairs)

        cumulative_summary.n_nonvoc_pass = count_nonvoc_passes(all_events)
        cumulative_summary.pc_nonvoc_pass = 100*cumulative_summary.n_nonvoc_pass / float(cumulative_summary.n_pairs)

        positions = np.unique(events.serialpos)
        prob_recall = np.empty_like(positions, dtype=float)
        for i,pos in enumerate(positions):
            pos_events = events[events.serialpos == pos]
            prob_recall[i] = np.sum(pos_events.correct) / float(len(pos_events))
        cumulative_summary.prob_recall = prob_recall

        cumulative_summary.study_lag_values = []
        cumulative_summary.prob_study_lag = []
        for i,n_ev in enumerate(n_events_per_lag):
            if n_ev>0:
                cumulative_summary.study_lag_values.append(i+1)
                cumulative_summary.prob_study_lag.append(n_success_per_lag[i]/float(n_ev))

        if math_events is not None:
            cumulative_summary.n_math = len(math_events)
            cumulative_summary.n_correct_math = np.sum(math_events.iscorrect)
            cumulative_summary.pc_correct_math = 100*cumulative_summary.n_correct_math / float(cumulative_summary.n_math)
            cumulative_summary.math_per_list = cumulative_summary.n_math / float(total_list_counter)

        cumulative_summary.n_pli = np.sum(intr_events.intrusion >= 0)
        cumulative_summary.pc_pli = 100*cumulative_summary.n_pli / float(len(events))
        cumulative_summary.n_eli = np.sum(intr_events.intrusion == -1)
        cumulative_summary.pc_eli = 100*cumulative_summary.n_eli / float(len(events))

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

        cumulative_ttest_data = make_ttest_table(bipolar_pairs, loc_info, ttest[-1])
        cumulative_ttest_data.sort(key=itemgetter(-2))
        cumulative_ttest_data = format_ttest_table(cumulative_ttest_data)

        self.pass_object('cumulative_ttest_data', cumulative_ttest_data)

        ttable_format, ttable_header = make_ttest_table_header(loc_info)
        self.pass_object('ttable_format', ttable_format)
        self.pass_object('ttable_header', ttable_header)
