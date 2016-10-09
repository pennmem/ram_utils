__author__ = 'm'

import numpy as np
import pandas as pd
from sklearn.externals import joblib

from ptsa.data.readers import BaseEventReader
from ptsa.data.readers.IndexReader import JsonIndexReader

from RamPipeline import *

from ReportUtils import ReportRamTask

import hashlib


class PSEventPreparation(ReportRamTask):
    def __init__(self, mark_as_completed=True):
        super(PSEventPreparation,self).__init__(mark_as_completed)

    def input_hashsum(self):
        subject = self.pipeline.subject
        task = self.pipeline.task
        tmp = subject.split('_')
        subj_code = tmp[0]
        montage = 0 if len(tmp)==1 else int(tmp[1])

        json_reader = JsonIndexReader(os.path.join(self.pipeline.mount_point, 'data/eeg/protocols/r1.json'))

        hash_md5 = hashlib.md5()

        event_files = sorted(list(json_reader.aggregate_values('task_events', subject=subj_code, montage=montage, experiment=task)))
        for fname in event_files:
            hash_md5.update(open(fname,'rb').read())

        return hash_md5.digest()

    def restore(self):
        subject = self.pipeline.subject
        task = self.pipeline.task

        events = joblib.load(self.get_path_to_resource_in_workspace(subject+'-'+task+'-ps_events.pkl'))
        self.pass_object(task+'_events', events)

        control_events = joblib.load(self.get_path_to_resource_in_workspace('control_events.pkl'))
        self.pass_object('control_events', control_events)

    def run(self):
        subject = self.pipeline.subject
        task = self.pipeline.task
        tmp = subject.split('_')
        subj_code = tmp[0]
        montage = 0 if len(tmp)==1 else int(tmp[1])

        json_reader = JsonIndexReader(os.path.join(self.pipeline.mount_point, 'data/eeg/protocols/r1.json'))

        event_files = sorted(list(json_reader.aggregate_values('task_events', subject=subj_code, experiment=task)))
        events = None
        for sess_file in event_files:
            e_path = os.path.join(self.pipeline.mount_point, str(sess_file))
            print e_path
            e_reader = BaseEventReader(filename=e_path, eliminate_events_with_no_eeg=True)

            sess_events = e_reader.read()

            if events is None:
                events = sess_events
            else:
                events = np.hstack((events,sess_events))

        events = events.view(np.recarray)

        control_events = events[events.type=='SHAM']
        joblib.dump(control_events, self.get_path_to_resource_in_workspace('control_events.pkl'))
        self.pass_object('control_events', control_events)

        sessions = np.unique(events.session)
        print task, 'sessions:', sessions

        stim_params = pd.DataFrame.from_records(events.stim_params)
        events = pd.DataFrame.from_records(events)
        del events['stim_params']
        events = pd.concat([events, stim_params], axis=1)

        events = compute_isi(events)

        is_stim_event_type_vec = np.vectorize(is_stim_event_type)
        stim_mask = is_stim_event_type_vec(events.type)
        if task == 'PS3':
            # stim_inds = np.where(stim_mask)[0]
            # stim_events = pd.DataFrame(events[stim_mask])
            # last_burst_inds = stim_inds + stim_events['nBursts'].values
            # last_bursts = events.ix[last_burst_inds]
            # events = stim_events
            # events['train_duration'] = last_bursts['mstime'].values - events['mstime'].values + last_bursts['pulse_duration'].values
            Exception('PS3 not supported')
        else:
            events = events[stim_mask]

        events = events.to_records(index=False)

        print len(events), 'stim', task, 'events'

        joblib.dump(events, self.get_path_to_resource_in_workspace(subject+'-'+task+'-ps_events.pkl'))
        self.pass_object(task+'_events', events)

def is_stim_event_type(event_type):
    return event_type == 'STIM_ON'

def compute_isi(events):
    print 'Computing ISI'

    events['isi'] = np.nan

    for i in xrange(1,len(events)):
        curr_ev = events.ix[i]
        if is_stim_event_type(curr_ev.type):
            prev_ev = events.ix[i-1]
            if curr_ev.session == prev_ev.session:
                if is_stim_event_type(prev_ev.type) or prev_ev.type == 'BURST':
                    prev_mstime = prev_ev.mstime
                    if prev_ev.pulse_duration > 0:
                        prev_mstime += prev_ev.pulse_duration
                    dt = curr_ev.mstime - prev_mstime
                    if dt < 7000.0:
                        events.isi.values[i] = dt

    return events
