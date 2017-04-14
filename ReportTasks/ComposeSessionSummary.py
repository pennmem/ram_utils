import SessionSummary
from ReportUtils import  ReportRamTask
import numpy as np
import datetime,time
from sklearn.metrics import roc_curve
import pandas as pd

SESSION_SUMMARY={
    'catFR1':SessionSummary.catFR1SessionSummary,
    'FR1' :  SessionSummary.FR1SessionSummary,
    'PAL1':  SessionSummary.PAL1SessionSummary,
    'TH1' :  SessionSummary.TH1SessionSummary
}

class ComposeSessionSummary(ReportRamTask):
    def __init__(self):
        super(ComposeSessionSummary,self).__init__(mark_as_completed=False)
        self.summaries = {}
        self.event_table = pd.DataFrame.from_records([e for e in self.events],columns = self.events.dtype.names)

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
        for session in sessions:
            sess_events = events[events.session==session]
            self.summaries[session] = SESSION_SUMMARY[self.pipeline.task]()
            self.fill_summary(sess_events,session)
        self.summaries[-1] =  SESSION_SUMMARY[self.pipeline.task]()
        self.fill_summary(events)

        
class ComposeFR1Summary(ComposeSessionSummary):
    @staticmethod
    def intr_events(events):
        return events[(events.type=='REC_WORD') & (events.intrusion !=0)]

    def trials(self,events):
        return events[events.type=='WORD']


    @property
    def events(self):
        return self.get_passed_object('all_events')

    def fill_summary(self,events,session=-1):
        super(ComposeFR1Summary,self).fill_summary(events,session)
        summary = self.summaries[session]
        n_rec_events = len(self.rec_events(events))
        intr_events = self.intr_events(events)

        summary.n_pli = np.sum(intr_events.intrusion > 0)
        summary.pc_pli = 100 * summary.n_pli / float(n_rec_events)
        summary.n_eli = np.sum(intr_events.intrusion == -1)
        summary.pc_eli = 100 * summary.n_eli / float(n_rec_events)





