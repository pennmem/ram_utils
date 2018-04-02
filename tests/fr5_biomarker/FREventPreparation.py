import os
import os.path
import numpy as np
import hashlib
from sklearn.externals import joblib

from ptsa.data.readers import BaseEventReader
from ptsa.data.readers  import JsonIndexReader

from ramutils.pipeline import *

from ReportTasks.RamTaskMethods import create_baseline_events


class FREventPreparation(RamTask):
    def __init__(self, mark_as_completed=True):
        super(FREventPreparation, self).__init__(mark_as_completed)

    def input_hashsum(self):
        subject = self.pipeline.subject
        tmp = subject.split('_')
        subj_code = tmp[0]
        montage = 0 if len(tmp) == 1 else int(tmp[1])

        json_reader = JsonIndexReader(os.path.join(self.pipeline.mount_point, 'protocols/r1.json'))

        hash_md5 = hashlib.md5()

        fr1_event_files = sorted(
            list(json_reader.aggregate_values('task_events', subject=subj_code, montage=montage, experiment='FR1')))
        for fname in fr1_event_files:
            with open(fname, 'rb') as f: hash_md5.update(f.read())

        catfr1_event_files = sorted(
            list(json_reader.aggregate_values('task_events', subject=subj_code, montage=montage, experiment='catFR1')))
        for fname in catfr1_event_files:
            with open(fname, 'rb') as f: hash_md5.update(f.read())

        fr3_event_files = sorted(
            list(json_reader.aggregate_values('task_events', subject=subj_code, montage=montage, experiment='FR3')))
        for fname in fr3_event_files:
            with open(fname, 'rb') as f: hash_md5.update(f.read())

        catfr3_event_files = sorted(
            list(json_reader.aggregate_values('task_events', subject=subj_code, montage=montage, experiment='catFR3')))
        for fname in catfr3_event_files:
            with open(fname, 'rb') as f: hash_md5.update(f.read())

        return hash_md5.digest()

    def run(self):
        subject = self.pipeline.subject
        tmp = subject.split('_')
        subj_code = tmp[0]
        montage = 0 if len(tmp) == 1 else int(tmp[1])
        json_reader = JsonIndexReader(os.path.join(self.pipeline.mount_point, 'protocols/r1.json'))

        if self.pipeline.args.sessions:
            fr1_sessions = [s for s in self.pipeline.args.sessions if s < 100]
            catfr1_sessions = [s - 100 for s in self.pipeline.args.sessions if s >= 100]

            event_files = [json_reader.get_value('task_events',subject=subj_code,montage=montage,experiment='FR1',session=s)
                             for s in sorted(fr1_sessions)]
            fr1_events = [filter_session(BaseEventReader(filename=event_path).read()) for event_path in event_files]
            event_files = [json_reader.get_value('task_events',subject=subj_code,montage=montage,experiment='catFR1',session=s)
                           for s in sorted(catfr1_sessions)]
            catfr1_events = [BaseEventReader(filename=event_path).read() for event_path in event_files]

        else:
            event_files = sorted(
               list(json_reader.aggregate_values('task_events', subject=subj_code, montage=montage, experiment='FR1')))
            fr1_events = [filter_session(BaseEventReader(filename=event_path).read()) for event_path in event_files]

            catfr1_events = [filter_session(BaseEventReader(filename=event_path).read()) for event_path in
                                             json_reader.aggregate_values('task_events',subject=subj_code,experiment='catFR1',
                                                                          montage = montage)]

        if len(fr1_events):
            fr1_events = np.concatenate(fr1_events).view(np.recarray)
        if len(catfr1_events):
            catfr1_events = np.concatenate(catfr1_events).view(np.recarray)
            catfr1_events=catfr1_events[['item_num', 'serialpos', 'session', 'subject', 'rectime', 'experiment',
                                         'mstime', 'type', 'eegoffset',  'recalled', 'item_name',
                                         'intrusion', 'montage', 'list', 'eegfile', 'msoffset']].copy()
            catfr1_events.session += 100
            if len(fr1_events):
                fr1_events = fr1_events[['item_num', 'serialpos', 'session', 'subject', 'rectime', 'experiment', 'mstime',
                                         'type', 'eegoffset', 'recalled', 'item_name', 'intrusion',
                                         'montage', 'list', 'eegfile', 'msoffset']].copy()

        if len(fr1_events) and len(catfr1_events):
            events = np.append(fr1_events,catfr1_events).view(np.recarray)
        else:
            events = fr1_events if len(fr1_events) else catfr1_events
        events = events[events.list>-1]

        events = create_baseline_events(events,1000,29000)
        irts = np.append([0],np.diff(events.mstime))

        encoding_events_mask = events.type == 'WORD'
        retrieval_events_mask = (events.type == 'REC_WORD') | (events.type == 'REC_BASE')
        retrieval_events_mask_0s = retrieval_events_mask & (events.type == 'REC_BASE')
        retrieval_events_mask_1s = retrieval_events_mask & (events.type == 'REC_WORD') & (events.intrusion == 0)  & (irts > 1000)

        filtered_events = events[encoding_events_mask | retrieval_events_mask_0s | retrieval_events_mask_1s]

        events = filtered_events.view(np.recarray)

        print len(events), 'WORD events'

        joblib.dump(events,self.get_path_to_resource_in_workspace(subject+'-FR_events.pkl'))
        self.pass_object('FR_events', events)

    def restore(self):
        subject=self.pipeline.subject
        events = joblib.load(self.get_path_to_resource_in_workspace(subject+'-FR_events.pkl'))
        self.pass_object('FR_events',events)


def filter_session(sess_events):
    last_list = sess_events[sess_events.type == 'REC_END'][-1]['list']  # drop any incomplete lists
    return sess_events[sess_events.list <= last_list]

