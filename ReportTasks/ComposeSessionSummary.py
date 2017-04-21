import SessionSummary
from ReportUtils import  ReportRamTask
import numpy as np
import datetime,time
from sklearn.metrics import roc_curve
import pandas as pd
from sklearn.externals import joblib

SESSION_SUMMARY={
    'catFR1':SessionSummary.catFR1SessionSummary,
    'FR1' :  SessionSummary.FR1SessionSummary,
    'PAL1':  SessionSummary.PAL1SessionSummary,
    'TH1' :  SessionSummary.TH1SessionSummary
}

class ComposeSessionSummary(ReportRamTask):
    '''
    There are 3 ways that data goes into a report: Plots, tables, and literals.
    The first two can be handled by DataFrames; the last needs no handling.
    '''
    def __init__(self):
        super(ComposeSessionSummary,self).__init__(mark_as_completed=False)
        self.event_table = pd.DataFrame.from_records([e for e in self.events],columns = self.events.dtype.names)
        # Entries should be in the form 'TITLE':DataFrame() for tables and plots
        # and 'NAME':value for raw numbers or strings
        self.tables = {}


    @property
    def events(self):
        return self.get_passed_object('events')


    @property
    def xval_results(self):
        return self.get_passed_object('xval_results')
    
    
    @staticmethod
    def prob_recall(events):
        events_by_serialpos = events.loc[events['type']=='WORD'].groupby('serialpos')
        return events_by_serialpos.aggregate(lambda x: np.mean(ComposeSessionSummary.recalls(x)))
    
    @staticmethod
    def rec_events(events):
        return events[events.type=='REC_WORD']

    @staticmethod
    def recalls(events):
        return events.recalled

    def prob_first_recall(self,events):
        rec_events = self.rec_events(events)
        lists = np.unique(events.list)
        pfr  = np.zeros(lists.shape)
        for lst in lists:
            list_rec_events = rec_events[(rec_events.list == lst) & (rec_events.intrusion == 0)]
            if list_rec_events.size > 0:
                list_events = events[events.list == lst]
                tmp = np.where(list_events.item_num == list_rec_events[0].item_num)[0]
                if tmp.size > 0:
                    first_recall_idx = tmp[0]
                    pfr[first_recall_idx] += 1
        pfr /= len(lists)
        return pfr
    
    def irt_over_categories(self,events):
        irt_within_cat = []
        irt_between_cat = []
        lists = np.unique(events.list)
        rec_events = self.rec_events(events)
        for lst in lists:
            list_rec_events = rec_events[(rec_events.list == lst) & (rec_events.intrusion == 0)]
            for i in xrange(1, len(list_rec_events)):
                cur_ev = list_rec_events[i]
                prev_ev = list_rec_events[i - 1]
                # if (cur_ev.intrusion == 0) and (prev_ev.intrusion == 0):
                dt = cur_ev.mstime - prev_ev.mstime
                if cur_ev.category == prev_ev.category:
                    irt_within_cat.append(dt)
                else:
                    irt_between_cat.append(dt)
        return np.mean(irt_within_cat),np.mean(irt_between_cat)

    def trials(self,events):
        raise NotImplementedError

    def fill_summary(self,events,session=-1):
        summary=self.summaries[session]
        summary.name = session
        summary.date = datetime.date.fromtimestamp(events.mstime[0]).strftime('%m-%d-%Y')
        summary.length = time.ctime(events.mstime[-1] - events.mstime[0])
        summary.n_trials = len(self.trials(events))
        summary.n_correct_trials = sum(self.recalls(self.trials(events)))
        summary.pc_correct_trials = 100* summary.n_correct_trials / float(summary.n_trials)

        summary.auc = self.xval_results[session].auc
        summary.perm_AUCs = self.xval_results[session].perm_AUCs
        (fpr, tpr) = roc_curve(self.recalls(events), self.xval_results[session].probs)
        summary.fpr = fpr
        summary.tpr = tpr
        summary.jstat_thresh = self.xval_results[session].jstat_thresh
        summary.jstat_percentile = self.xval_results[session].jstat_quantile

    def run(self):
        events = self.events
        sessions  = np.unique(events.session)
        sess_info_table = pd.DataFrame(index=sessions)
        sess_info_table.index.name='Session'
        sess_info_table['Date'] = [datetime.date.fromtimestamp(events[events.session==sess].mstime[0]).strftime('%m-%d-%Y')
                                   for sess in sessions]
        sess_info_table['Length (min)'] = [time.ctime(events[events.session==sess].mstime[-1] - events[events.session==sess].mstime[0])
            for sess in sessions]
        sess_info_table['# lists'] = [events[events.session==sess].list.max() for sess in sessions]
        sess_info_table['Perf'] = ['{:2.2}%}'.format(self.recalls(events[events.session==sess]).mean()) for sess in sessions]
        self.tables['sess_info_table'] = sess_info_table

        self.tables['n_electrodes'] = len(self.get_passed_object('monopolar_channels'))

        
class ComposeFR1Summary(ComposeSessionSummary):
    def __init__(self):
        super(ComposeFR1Summary, self).__init__()

    def run(self):
        super(ComposeFR1Summary, self).run()

        # PERFORMANCE TABLE
        words = self.events.loc[self.events.type == 'WORD']
        n_words = '%d words'%(len(words))
        n_correct  = '{} correct ({:2.2}%)'.format(int(words.recalled.sum()),words.recalled.mean())
        n_pli = '{} PLI ({:2.2}%)'.format(int((words.intrusion>0).sum()),(words.intrusion>0).mean())
        n_eli = '{} ELI ({:2.2}%)'.format(int((words.intrusion==-1).sum()),(words.intrusion==-1).mean())
        self.tables['perf_table'] = pd.DataFrame(data=[n_words,n_correct,n_pli,n_eli],)

        # MATH DISTRACTOR TABLE
        math = self.events.loc[self.events.type=='PROB']
        n_math = '%d math problems'%len(math)
        n_correct = '{} correct ({:2.2}%'.format(int(math.iscorrect.sum()),math.iscorrect.mean())
        n_correct_per_list = '{:.2} per list'.format(math.groupby('list').iscorrect.mean())
        self.tables['math_table'] = pd.DataFrame(data=[n_math,n_correct,n_correct_per_list])

        #PREC
        self.tables['p_rec'] = words.groupby('serialpos').recalled.mean()
        self.tables['p_rec'].columns = ['Probability of recall'] # TODO: CONFIRM THIS WORKS

        #PFR
        recalls = self.events.loc[self.events.type=='REC_WORD']
        pfr = np.zeros((len(self.events.list.unique()),len(self.events.serialpos.unique())))
        for lst,list_recalls in recalls.groupby('list'):
            first_rec_pos = list_recalls[0].serialpos
            lst_loc = 0 if lst==-1 else lst
            pfr[lst_loc,first_rec_pos] = 1
        self.tables['pfr'] = pd.DataFrame(data=pfr.mean(0),index=self.events.serialpos.unique())

        #t-test results
        self.tables['ttest'] = self.get_passed_object('ttest_table')


        # ROC CURVE AND TERCILE PLOT
        full_output= self.get_passed_object('xval_output')[-1]

        # ROC CURVE
        self.tables['roc'] = pd.DataFrame(data=full_output.tpr,index=pd.Index(data=full_output.fpr,name='False Alarm Rate'),
                                          columns= ['Hit Rate'])

        # TERCILE PLOT
        self.tables['terc'] = pd.DataFrame(data=[full_output.low_pc_diff_from_mean,full_output.mid_pc_diff_from_mean,
                                                 full_output.high_pc_diff_from_mean],
                                           index=pd.Index(['Low','Mid','High'],name='Tercile of Classifier Estimate'),
                                           columns=['Recall Change from Mean (%)'])

        # AUC NUMBERS
        self.tables['auc'] = full_output.auc
        self.tables['auc_pval'] = full_output.pvalue
        self.tables['classifier_median'] = full_output.jstat_thresh

        self.pass_object('tables',self.tables)
        joblib.dump(self.tables,self.get_path_to_resource_in_workspace('session_tables_dict.pkl'))










