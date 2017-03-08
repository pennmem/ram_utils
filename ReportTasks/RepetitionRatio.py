from ReportUtils import ReportRamTask
from ptsa.data.readers.IndexReader import JsonIndexReader
import os.path
import numpy as np
from ptsa.data.readers import BaseEventReader
from sklearn.externals import joblib



class RepetitionRatio(ReportRamTask):
    def __init__(self,mark_as_completed=True):
        super(RepetitionRatio,self).__init__(mark_as_completed)

    @property
    def events(self):
        return self.get_passed_object('events')

    def run(self):
        subject=self.pipeline.subject
        all_repetition_ratios = self.load_repetition_ratios()
        repetition_ratios = self.repetition_ratios(self.events)

        self.pass_object('repetition_ratios',repetition_ratios)
        self.pass_object('all_repetition_ratios',all_repetition_ratios)

        joblib.dump(all_repetition_ratios,self.get_path_to_resource_in_workspace('all_repetition_ratios.pkl'))
        joblib.dump(repetition_ratios,self.get_path_to_resource_in_workspace(subject+'-repetition_ratios.pkl'))

    def restore(self):
        all_repetition_ratios = joblib.load(self.get_path_to_resource_in_workspace('all_repetition_ratios.pkl'))
        repetition_ratios = joblib.load(self.get_path_to_resource_in_workspace(self.pipeline.subject+'-repetition_ratios.pkl'))

        self.pass_object('repetition_ratios',repetition_ratios)
        self.pass_object('all_repetition_ratios',all_repetition_ratios)


    def load_repetition_ratios(self):
            jr  = JsonIndexReader(os.path.join(self.pipeline.mountpoint,'protocols','r1.json'))
            subjects = jr.aggregate_values('subject',experiment='catFR1') | jr.aggregate_values('subject',experiment='catFR3')
            events1=None; events3=None
            repetition_ratios = np.empty(len(subjects))
            for i,subject in enumerate(subjects):
                if 'catFR1' in jr.experiments(subject=subject):
                    events1 = np.concatenate([BaseEventReader(filename=f).read()
                                          for f in jr.aggregate_values('task_events',subject=subject,experiment='catFR1')]).view(np.recarray)
                if 'catFR3' in jr.experiments(subject=subject):
                    events3 = np.concatenate([BaseEventReader(filename=f).read()
                                              for f in jr.aggregate_values('task_events',subject=subject,experiment='catFR3') if f]).view(np.recarray)
                if events1 is not None and events3 is not None:
                    events = np.concatenate([events1,events3[events1.dtype.names]]).view(np.recarray)
                else:
                    events = events1 if events1 is not None else events3
                repetition_ratios[i] = self.repetition_ratios(events)
            return repetition_ratios

    @staticmethod
    def repetition_ratios(events):
        recalls = events[events.recalled==1]
        sessions = np.unique(recalls.session)
        lists = np.unique(recalls.list)
        recall_lists = [recalls[(recalls.session==sess) & (recalls.list==lst)] for sess in sessions for lst in lists]
        return np.nanmean([repetition_ratio(lst) for lst in recall_lists])


def repetition_ratio(recall_list):
    is_repetition = np.diff(recall_list.category_num)==0
    return np.sum(is_repetition)/float(len(recall_list)-1)

