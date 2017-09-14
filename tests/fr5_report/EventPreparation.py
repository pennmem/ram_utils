__author__ = 'm'

import os.path
import numpy as np

from ptsa.data.readers import BaseEventReader
from ptsa.data.readers.IndexReader import JsonIndexReader

from RamPipeline import *
from ReportUtils import ReportRamTask

import hashlib
from copy import deepcopy
from ReportTasks.RamTaskMethods import create_baseline_events
import pandas as pd
from ReportTasks.RamTaskMethods import filter_session

class FR1EventPreparation(ReportRamTask):
    def __init__(self, mark_as_completed=False):
        super(FR1EventPreparation,self).__init__(mark_as_completed)

    def input_hashsum(self):
        subject = self.pipeline.subject
        tmp = subject.split('_')
        subj_code = tmp[0]
        montage = 0 if len(tmp)==1 else int(tmp[1])

        json_reader = JsonIndexReader(os.path.join(self.pipeline.mount_point, 'protocols/r1.json'))

        hash_md5 = hashlib.md5()

        fr1_event_files = sorted(list(json_reader.aggregate_values('all_events', subject=subj_code, montage=montage, experiment='FR1')))
        for fname in fr1_event_files:
            with open(fname,'rb') as f: hash_md5.update(f.read())

        return hash_md5.digest()

    def run(self):
        subject = self.pipeline.subject
        tmp = subject.split('_')
        subj_code = tmp[0]
        montage = 0 if len(tmp)==1 else int(tmp[1])

        json_reader = JsonIndexReader(os.path.join(self.pipeline.mount_point, 'protocols/r1.json'))

        if self.pipeline.sessions:
            print "Sessions: ", self.pipeline.sessions
            fr1_sessions = [s for s in self.pipeline.sessions if s<100]
            event_files = [json_reader.get_value('task_events', subject=subj_code,
                                                                   montage=montage, experiment='FR1',session=s)
                           for s in sorted(fr1_sessions)]

            catfr1_sessions = [s-100 for s in self.pipeline.sessions if 100<s<200]

            catfr1_event_files = [json_reader.get_value('task_events',
                                                               subject=subj_code, montage=montage, experiment='catFR1',session=s)
                                  for s in catfr1_sessions]

        else:
            print 'All sessions'
            event_files = json_reader.aggregate_values('task_events',subject=subj_code,montage=montage,experiment='FR1')
            catfr1_event_files = json_reader.aggregate_values('task_events',subject=subj_code,montage=montage,experiment='catFR1')

        fr1_events = np.concatenate(
            filter_session([BaseEventReader(filename=f, eliminate_events_with_no_eeg=True).read() for f in event_files])
        ).view(np.recarray)

        if any(catfr1_event_files):

            catfr1_events = np.concatenate([filter_session(BaseEventReader(filename=f,eliminate_events_with_no_eeg=True).read())
                                            for f in catfr1_event_files]
                                           ).view(np.recarray)
            catfr1_events =catfr1_events[['item_num', 'serialpos', 'session', 'subject', 'rectime', 'experiment', 'mstime', 'type', 'eegoffset', 'recalled', 'item_name', 'intrusion', 'montage', 'list', 'eegfile', 'msoffset']]
            catfr1_events.session+=100
            fr1_events = fr1_events[['item_num', 'serialpos', 'session', 'subject', 'rectime', 'experiment', 'mstime', 'type', 'eegoffset', 'recalled', 'item_name', 'intrusion', 'montage', 'list', 'eegfile', 'msoffset']]

            fr1_events=np.concatenate([fr1_events,catfr1_events]).view(np.recarray)

        fr1_events = fr1_events[fr1_events.list>-1]

        if not (fr1_events.type == 'REC_BASE').any():
            fr1_events = create_baseline_events(fr1_events,1000,29000)

        encoding_events_mask = fr1_events.type == 'WORD'
        retrieval_events_mask = (fr1_events.type == 'REC_WORD') | (fr1_events.type == 'REC_BASE')
        irts = np.append([0], np.diff(fr1_events.mstime))
        retrieval_events_mask_0s = retrieval_events_mask & (fr1_events.type == 'REC_BASE')
        retrieval_events_mask_1s = retrieval_events_mask & (fr1_events.type == 'REC_WORD') & (
        fr1_events.intrusion == 0) & (irts > 1000)
        intr_events = fr1_events[(fr1_events.intrusion!=-999) & (fr1_events.intrusion !=0)]

        filtered_events = fr1_events[encoding_events_mask | retrieval_events_mask_0s | retrieval_events_mask_1s]

        fr1_events = filtered_events.view(np.recarray)

        print len(fr1_events), 'sample events'

        self.pass_object('FR1_events', fr1_events)
        self.pass_object('FR1_intr_events',intr_events)


class MissingEventError(Exception):
    pass


class FR5EventPreparation(ReportRamTask):
    def __init__(self):
        super(FR5EventPreparation,self).__init__(mark_as_completed=False)

    def run(self):
        jr  = JsonIndexReader(os.path.join(self.pipeline.mount_point,'protocols','r1.json'))
        temp=self.pipeline.subject.split('_')
        subject= temp[0]
        montage = 0 if len(temp)==1 else temp[1]
        task = self.pipeline.task

        fr5_sessions = [s for s in jr.sessions(subject=subject,montage=montage,experiment=task)
                        if not jr.aggregate_values('ps4_events',subject=subject,montage=montage,experiment=task,session=s)
                        ]
        if montage:
            events = np.concatenate([ filter_session(BaseEventReader(
                filename=jr.get_value('task_events',subject=subject,montage=montage,experiment=task,session=s)).read())
                                      for s in fr5_sessions]).view(np.recarray)
        else:
            events = np.concatenate([ filter_session(BaseEventReader(
                filename=jr.get_value('task_events',subject=subject,experiment=task,session=s)).read())
                                      for s in fr5_sessions]).view(np.recarray)

        events = events[events.list>-1]

        self.pass_object('all_events', events)

        if not (events.type=='WORD').any():
            raise MissingEventError('No events found that are valid for analysis')

        math_events = np.concatenate([BaseEventReader(filename=f).read() for f in
                jr.aggregate_values('math_events',subject=subject,experiment=task,
                                                                                 montage=montage)]
                                     ).view(np.recarray)


        math_events = math_events[math_events.type=='PROB']


        rec_events = events[events.type == 'REC_WORD']

        ps_events = [BaseEventReader(filename=event_path).read() for event_path in
                                    jr.aggregate_values('ps4_events',subject=subject,experiment=task,montage=montage)]
        if ps_events:
            ps_events = np.concatenate(ps_events).view(np.recarray)
            self.pass_object('ps_events',ps_events)


        intr_events = rec_events[(rec_events.intrusion!=-999) & (rec_events.intrusion!=0)]

        events = events[events.type=='WORD']

        print len(events), 'WORD events'


        self.pass_object('FR_math_events', math_events)
        self.pass_object('FR_intr_events', intr_events)
        self.pass_object('FR_rec_events', rec_events)

        self.pass_object(task+'_events',events)

def free_epochs(times, duration, pre, post, start=None, end=None):
    # (list(vector(int))*int*int*int) -> list(vector(int))
    """
    Given a list of event times, find epochs between them when nothing is happening

    Parameters:
    -----------

    times:
        An iterable of 1-d numpy arrays, each of which indicates event times

    duration: int
        The length of the desired empty epochs

    pre: int
        the time before each event to exclude

    post: int
        The time after each event to exclude

    """
    n_trials = len(times)
    epoch_times = []
    for i in range(n_trials):
        ext_times = times[i]
        if start is not None:
            ext_times = np.append([start[i]], ext_times)
        if end is not None:
            ext_times = np.append(ext_times, [end[i]])
        pre_times = ext_times - pre
        post_times = ext_times + post
        interval_durations = pre_times[1:] - post_times[:-1]
        free_intervals = np.where(interval_durations > duration)[0]
        trial_epoch_times = []
        for interval in free_intervals:
            begin = post_times[interval]
            finish = pre_times[interval + 1] - duration
            interval_epoch_times = range(begin, finish, duration)
            trial_epoch_times.extend(interval_epoch_times)
        epoch_times.append(np.array(trial_epoch_times))
    epoch_array = np.empty((n_trials, max([len(x) for x in epoch_times])))
    epoch_array[...] = -np.inf
    for i, epoch in enumerate(epoch_times):
        epoch_array[i, :len(epoch)] = epoch
    return epoch_array




