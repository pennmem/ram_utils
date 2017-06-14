from ram_utils.RamPipeline import *
from .SessionSummary import SessionSummary

import numpy as np
from scipy.stats import sem
import pandas as pd
import time
from operator import itemgetter

from ...ReportUtils import  ReportRamTask


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
        chest_events = self.get_passed_object(task + '_chest_events')
        rec_events = self.get_passed_object(task + '_rec_events')
        all_events = self.get_passed_object(task + '_all_events')
        scores = self.get_passed_object('scores')
        # time_events = self.get_passed_object(task + '_time_events')
        monopolar_channels = self.get_passed_object('monopolar_channels')
        bp_tal_structs = self.get_passed_object('bp_tal_structs')

        ttest = self.get_passed_object('ttest')

        xval_output = self.get_passed_object('xval_output')
        perm_test_pvalue = self.get_passed_object('pvalue')

        if self.params.doConf_classification & self.get_passed_object('conf_decode_success'):
            xval_output_conf = self.get_passed_object('xval_output_conf')
            perm_test_pvalue_conf = self.get_passed_object('pvalue_conf')        

        if self.params.doDist_classification:        
            xval_output_thresh = self.get_passed_object('model_output_thresh')

        if self.params.doClass_wTranspose:
            xval_output_transpose = self.get_passed_object('xval_output_transpose')
            perm_test_pvalue_transpose = self.get_passed_object('pvalue_transpose') 

        sessions = np.unique(events.session)
        self.pass_object('NUMBER_OF_SESSIONS', len(sessions))
        self.pass_object('NUMBER_OF_ELECTRODES', len(monopolar_channels))

        session_data = []

        # create session table info
        for i,session in enumerate(sessions):
            # filter to just current session
            session_events = events[events.session == session]
            n_sess_events = len(session_events)

            session_rec_events = rec_events[rec_events.session == session]
            session_all_events = all_events[all_events.session == session]
            # try:
            #     session_score = score_events['sessionScore'][i]
            # except IndexError:
            #     session_score = score_events['sessionScore']
            # try:
            #     session_score = session_score[0,...]
            # except (TypeError,ValueError,IndexError):
            #     pass
            session_score=scores[i]
            # info about session
            timestamps = session_all_events.mstime
            first_time_stamp = np.min(timestamps)
            last_time_stamp = np.max(timestamps)
            session_length = '%.2f' % ((last_time_stamp - first_time_stamp) / 60000.0)
            session_date = time.strftime('%d-%b-%Y', time.localtime(last_time_stamp/1000))
            session_name = 'Sess%02d' % session
            session_completed = True if n_sess_events==100 else False
            n_items = len(session_events)
            n_correct_items = np.sum(session_events.recalled)
            pc_correct_items = 100*n_correct_items / float(n_items)  
            
            n_correct_items_transpose   = np.sum((session_events.recalled==True)|(session_events.recalled_ifFlipped==True))
            pc_correct_items_transpose = 100*n_correct_items_transpose / float(n_items)             
                   
            lists = np.unique(session_events.trial)
            n_lists = len(lists)          
            print 'Session =', session_name, 'Score = ', session_score 

            session_data.append([session, session_date, session_length, n_lists, '$%.2f$\\%%' % pc_correct_items,session_score])
        print '\n\n session_data: {} \n\n'.format(session_data)

        self.pass_object('SESSION_DATA', session_data)
    

        # cumulative results across all sessions
        cumulative_summary = SessionSummary()
        cumulative_summary.n_items = len(events)
        cumulative_summary.n_correct_items = np.sum(events.recalled)
        cumulative_summary.pc_correct_items = 100*cumulative_summary.n_correct_items / float(cumulative_summary.n_items)
        cumulative_summary.n_transposed_items = np.sum((events.recalled==True)|(events.recalled_ifFlipped==True))
        cumulative_summary.pc_transposed_items = 100*cumulative_summary.n_transposed_items / float(cumulative_summary.n_items)
        cumulative_summary.mean_norm_err = np.mean(events.norm_err)



        # analyses 1: probability correct as a function of confidence
        # analysis 2: probability histogram of of distance error for each confidence level
        prob_by_conf = np.zeros(3,dtype=float)
        dist_hist = np.zeros((3,20),dtype=float)
        percent_conf = np.zeros(3,dtype=float)        
        for conf in xrange(3):
            prob_by_conf[conf] = events.recalled[events.confidence==conf].mean()
            percent_conf[conf] = np.mean(events.confidence==conf)            
            [hist,b]=np.histogram(events.distErr[events.confidence==conf],bins=20,range=(0,100))
            hist = hist / float(np.sum(hist))
            dist_hist[conf] = hist

        cumulative_summary.prob_by_conf = prob_by_conf
        cumulative_summary.percent_conf = percent_conf        
        cumulative_summary.dist_hist = dist_hist
        
        # analysis 3: distance error by block number
        err_by_block = np.zeros(5,dtype=float)
        err_by_block_sem = np.zeros(5,dtype=float)
        for block in xrange(5):
            err_by_block[block] = events.distErr[events.block==block].mean() 
            err_by_block_sem[block] = sem(events.distErr[events.block==block])
            
        cumulative_summary.err_by_block = err_by_block 
        cumulative_summary.err_by_block_sem = err_by_block_sem         
        if self.params.doDist_classification:
            cumulative_summary.aucs_by_thresh = xval_output_thresh.aucs_by_thresh
            cumulative_summary.pval_by_thresh = xval_output_thresh.pval_by_thresh
            cumulative_summary.pCorr_by_thresh = xval_output_thresh.pCorr_by_thresh
            cumulative_summary.thresholds = xval_output_thresh.thresholds        

        # classification results
        cumulative_xval_output = xval_output[-1]
        cumulative_summary.auc = '%.2f' % (100*cumulative_xval_output.auc)
        cumulative_summary.fpr = cumulative_xval_output.fpr
        cumulative_summary.tpr = cumulative_xval_output.tpr
        cumulative_summary.pc_diff_from_mean = (cumulative_xval_output.low_pc_diff_from_mean, cumulative_xval_output.mid_pc_diff_from_mean, cumulative_xval_output.high_pc_diff_from_mean)
        cumulative_summary.perm_AUCs = self.get_passed_object('perm_AUCs')
        cumulative_summary.perm_test_pvalue = ('= %.3f' % perm_test_pvalue) if perm_test_pvalue>=0.001 else '\leq 0.001'
        cumulative_summary.jstat_thresh = '%.3f' % cumulative_xval_output.jstat_thresh
        cumulative_summary.jstat_percentile = '%.2f' % (100.0*cumulative_xval_output.jstat_quantile)

        if self.params.doConf_classification & self.get_passed_object('conf_decode_success'):
            cumulative_xval_output_conf = xval_output_conf[-1]
            cumulative_summary.auc_conf = '%.2f' % (100*cumulative_xval_output_conf.auc)
            cumulative_summary.fpr_conf = cumulative_xval_output_conf.fpr
            cumulative_summary.tpr_conf = cumulative_xval_output_conf.tpr
            cumulative_summary.pc_diff_from_mean_conf = (cumulative_xval_output_conf.low_pc_diff_from_mean, cumulative_xval_output_conf.mid_pc_diff_from_mean, cumulative_xval_output_conf.high_pc_diff_from_mean)
            cumulative_summary.perm_AUCs_conf = self.get_passed_object('perm_AUCs_conf')
            cumulative_summary.perm_test_pvalue_conf = ('= %.3f' % perm_test_pvalue_conf) if perm_test_pvalue_conf>=0.001 else '\leq 0.001'
            cumulative_summary.jstat_thresh_conf = '%.3f' % cumulative_xval_output_conf.jstat_thresh
            cumulative_summary.jstat_percentile_conf = '%.2f' % (100.0*cumulative_xval_output_conf.jstat_quantile)


        if self.params.doClass_wTranspose:
            cumulative_xval_output_transpose = xval_output_transpose[-1]
            cumulative_summary.auc_transpose = '%.2f' % (100*cumulative_xval_output_transpose.auc)
            cumulative_summary.fpr_transpose = cumulative_xval_output_transpose.fpr
            cumulative_summary.tpr_transpose = cumulative_xval_output_transpose.tpr
            cumulative_summary.pc_diff_from_mean_transpose = (cumulative_xval_output_transpose.low_pc_diff_from_mean, cumulative_xval_output_transpose.mid_pc_diff_from_mean, cumulative_xval_output_transpose.high_pc_diff_from_mean)
            cumulative_summary.perm_AUCs_transpose = self.get_passed_object('perm_AUCs_transpose')
            cumulative_summary.perm_test_pvalue_transpose = ('= %.3f' % perm_test_pvalue_transpose) if perm_test_pvalue_transpose>=0.001 else '\leq 0.001'
            cumulative_summary.jstat_thresh_transpose = '%.3f' % cumulative_xval_output_transpose.jstat_thresh
            cumulative_summary.jstat_percentile_transpose = '%.2f' % (100.0*cumulative_xval_output_transpose.jstat_quantile)
            
            
        self.pass_object('cumulative_summary', cumulative_summary)

        # electrode tttest data for each freq_bin
        cumulative_ttest_data_LTA = make_ttest_table(bp_tal_structs, ttest[0])
        cumulative_ttest_data_LTA.sort(key=itemgetter(-2))
        cumulative_ttest_data_LTA = format_ttest_table(cumulative_ttest_data_LTA)
        self.pass_object('cumulative_ttest_data_LTA', cumulative_ttest_data_LTA)
        
        cumulative_ttest_data_HTA = make_ttest_table(bp_tal_structs, ttest[1])
        cumulative_ttest_data_HTA.sort(key=itemgetter(-2))
        cumulative_ttest_data_HTA = format_ttest_table(cumulative_ttest_data_HTA)  
        self.pass_object('cumulative_ttest_data_HTA', cumulative_ttest_data_HTA)      
        
        cumulative_ttest_data_G = make_ttest_table(bp_tal_structs, ttest[2])
        cumulative_ttest_data_G.sort(key=itemgetter(-2))
        cumulative_ttest_data_G = format_ttest_table(cumulative_ttest_data_G)      
        self.pass_object('cumulative_ttest_data_G', cumulative_ttest_data_G)              
        
        cumulative_ttest_data_HFA = make_ttest_table(bp_tal_structs, ttest[3])
        cumulative_ttest_data_HFA.sort(key=itemgetter(-2))
        cumulative_ttest_data_HFA = format_ttest_table(cumulative_ttest_data_HFA)  
        self.pass_object('cumulative_ttest_data_HFA', cumulative_ttest_data_HFA)      
