from RamPipeline import RamTask
from os import path
import os
import numpy as np
import cPickle
from sklearn.externals import joblib
from ptsa.data.readers.IndexReader import JsonIndexReader
from ptsa.data.readers import BaseEventReader
import hashlib


class RepetitionRatio(RamTask):
    def __init__(self,recompute_all_ratios=False,mark_as_completed=False):
        super(RepetitionRatio,self).__init__(mark_as_completed)
        self.repetition_ratios = None
        self.repetition_percentiles = None
        self.recompute_all_ratios = recompute_all_ratios

    def input_hashsum(self):
        subject = self.pipeline.subject
        task = self.pipeline.task
        tmp = subject.split('_')
        subj_code = tmp[0]
        montage = 0 if len(tmp)==1 else int(tmp[1])

        json_reader = JsonIndexReader(path.join(self.pipeline.mount_point, 'protocols/r1.json'))

        hash_md5 = hashlib.md5()

        event_files = sorted(list(json_reader.aggregate_values('all_events', subject=subj_code, montage=montage, experiment=task)))
        for fname in event_files:
            with open(fname,'rb') as f: hash_md5.update(f.read())

        return hash_md5.digest()

    def restore(self):
        subject= self.pipeline.subject
        self.repetition_ratios = joblib.load(path.join(self.pipeline.mount_point,self.workspace_dir,subject+'-repetition-ratios.pkl'))
        # self.repetition_percentiles = joblib.load(path.join(self.pipeline.mount_point,self.workspace_dir,subject+'-repetition-percentiles.pkl'))
        all_recall_ratios_dict = joblib.load(path.join(path.dirname(self.get_workspace_dir()),'all_repetition_ratios_dict'))
        all_recall_ratios = np.array([np.nanmean(x) for x in all_recall_ratios_dict.itervalues()])
        np.save(path.join(path.dirname(self.get_workspace_dir()),'all_repetition_ratios'),all_recall_ratios)

        self.pass_object('all_repetition_ratios',all_recall_ratios)
        self.pass_object('repetition_ratios',self.repetition_ratios)
        # self.pass_object('repetition_percentiles',self.repetition_percentiles)

    def run(self):
        subject = self.pipeline.subject.split('_')[0]
        task = self.pipeline.task
        events = self.get_passed_object('cat_events')
        recalls = events[events.recalled==1]
        sessions = np.unique(recalls.session)
        print '%d sessions'%len(sessions)
        if self.recompute_all_ratios:
            all_recall_ratios_dict = self.initialize_repetition_ratio()
        else:
            try:
                all_recall_ratios_dict = joblib.load(path.join(path.basename(self.workspace_dir),'all_repetition_ratios_dict'))
            except IOError:
                all_recall_ratios_dict = self.initialize_repetition_ratio()
        self.repetition_ratios = all_recall_ratios_dict[subject]
        all_recall_ratios = np.hstack([np.nanmean(x) for x in all_recall_ratios_dict.itervalues()])
        # all_recall_ratios.sort()
        # self.get_percentiles(all_recall_ratios)

        self.pass_object('all_repetition_ratios',all_recall_ratios)
        self.pass_object('repetition_ratios', self.repetition_ratios)
        # self.pass_object('repetition_percentiles', self.repetition_percentiles)

        joblib.dump(all_recall_ratios_dict, path.join(self.get_workspace_dir(), 'all_recall_ratios_dict'))
        joblib.dump(self.repetition_ratios,
                    path.join(self.pipeline.mount_point, self.workspace_dir, subject + '-repetition-ratios.pkl'))

        # joblib.dump(self.repetition_percentiles,path.join(self.pipeline.mount_point,self.workspace_dir,subject+'-repetition-percentiles.pkl'))


    def get_percentiles(self,all_recall_ratios):
        self.repetition_percentiles =self.repetition_ratios.copy()
        for i,ratio in enumerate(self.repetition_percentiles.flat):
              self.repetition_percentiles.flat[i] = len(np.where(all_recall_ratios < ratio)[0]) / float(len(all_recall_ratios.flat))

    def initialize_repetition_ratio(self):
        task = self.pipeline.task
        j_reader = JsonIndexReader('/protocols/r1.json')
        subjects = j_reader.subjects(experiment='catFR1')
        all_repetition_rates = {}
    
        for subject in subjects:
            try:
                print 'Repetition ratios for subject: ',subject
    
                evs_field_list = ['item_num', 'serialpos', 'session', 'subject', 'rectime', 'experiment', 'mstime', 'type',
                                  'eegoffset', 'iscorrect', 'answer', 'recalled', 'item_name', 'intrusion', 'montage', 'list',
                                  'eegfile', 'msoffset']
                evs_field_list += ['category', 'category_num']
    
                tmp = subject.split('_')
                subj_code = tmp[0]
                montage = 0 if len(tmp) == 1 else int(tmp[1])
    
                json_reader = JsonIndexReader(path.join('/','protocols/r1.json'))
    
                event_files = sorted(
                    list(json_reader.aggregate_values('all_events', subject=subj_code, montage=montage, experiment='catFR1')))
                events = None
                for sess_file in event_files:
                    e_path = path.join(str(sess_file))
                    e_reader = BaseEventReader(filename=e_path, eliminate_events_with_no_eeg=True)
    
                    sess_events = e_reader.read()[evs_field_list]
    
                    if events is None:
                        events = sess_events
                    else:
                        events = np.hstack((events, sess_events))
    
                events = events.view(np.recarray)
                print len(events),' events found'
                recalls = events[events.recalled==1]
                sessions = np.unique(recalls.session)
                lists=np.unique(recalls.list)
                repetition_rates = np.empty([len(sessions),len(lists)])
    
                for i,r in enumerate(repetition_rates.flat):
                    repetition_rates.flat[i] = np.nan
                for session in sessions:
                    sess_recalls = recalls[recalls.session == session]
                    lists = np.unique(sess_recalls.list)
                    repetition_rates[session][:len(lists)] = [repetition_ratio(sess_recalls[sess_recalls.list == l])
                                                 for l in lists]
                all_repetition_rates[subject] = repetition_rates.copy()
            except Exception as e:
                print 'Subject ',subject,'failed:'
                print e
        joblib.dump(all_repetition_rates,path.join(path.dirname(self.pipeline.workspace_dir),'all_repetition_ratios_dict'))
        return all_repetition_rates

def repetition_ratio(recall_list):
    is_repetition = np.diff(recall_list.category_num)==0
    return np.sum(is_repetition)/float(len(recall_list)-1)
