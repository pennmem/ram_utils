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

        event_files = sorted(list(json_reader.aggregate_values('task_events', subject=subj_code, montage=montage, experiment='FR1')))
        fr1_events = [BaseEventReader(filename=f,eliminate_events_with_no_eeg=True).read() for f in event_files]

        fr1_events=np.concatenate(fr1_events).view(np.recarray)
        if not (fr1_events.type == 'REC_BASE').any():
            fr1_events = create_baseline_events(fr1_events)

        encoding_events_mask = fr1_events.type == 'WORD'
        retrieval_events_mask = (fr1_events.type == 'REC_WORD') | (fr1_events.type == 'REC_BASE')
        irts = np.append([0], np.diff(fr1_events.mstime))
        retrieval_events_mask_0s = retrieval_events_mask & (fr1_events.type == 'REC_BASE')
        retrieval_events_mask_1s = retrieval_events_mask & (fr1_events.type == 'REC_WORD') & (
        fr1_events.intrusion == 0) & (irts > 1000)

        filtered_events = fr1_events[encoding_events_mask | retrieval_events_mask_0s | retrieval_events_mask_1s]

        fr1_events = filtered_events.view(np.recarray)

        print len(fr1_events), 'WORD events'

        self.pass_object('FR1_events', fr1_events)




class FR5EventPreparation(ReportRamTask):
    def __init__(self):
        super(FR5EventPreparation,self).__init__(mark_as_completed=False)
    def run(self):
        jr  = JsonIndexReader(os.path.join(self.pipeline.mount_point,'protocols','r1.json'))
        temp=self.pipeline.subject.split('_')
        subject= temp[0]
        montage = 0 if len(temp)==1 else temp[1]
        task = self.pipeline.task

        events = [ BaseEventReader(filename=event_path).read() for event_path in
                                jr.aggregate_values('task_events',subject=subject,montage=montage,experiment=task)]


        if events:
            events = np.concatenate(events).view(np.recarray)


        if not (events.type=='REC_BASE').any():
            events = create_baseline_events(events)


        self.pass_object('all_events', events)

        math_events = BaseEventReader(
            filename=jr.get_value('math_events',subject=subject,experiment=task,
                                                                                 montage=montage)
        ).read()#
        math_events = math_events[math_events.type=='PROB']

        ps_events = [BaseEventReader(filename=event_path,eliminate_events_with_no_eeg=False).read()
                     for event_path in jr.aggregate_values('ps4_events',subject=subject,experiment=task,montage=montage)]

        if ps_events:
            ps_events = np.concatenate(ps_events).view(np.recarray)

        rec_events = events[events.type == 'REC_WORD']

        base_events= events[events.type=='REC_BASE']

        intr_events = rec_events[(rec_events.intrusion!=-999) & (rec_events.intrusion!=0)]


        encoding_events_mask = events.type == 'WORD'
        retrieval_events_mask = (events.type == 'REC_WORD') | (events.type == 'REC_BASE')
        irts = np.append([0], np.diff(events.mstime))
        retrieval_events_mask_0s = retrieval_events_mask & (events.type == 'REC_BASE')
        retrieval_events_mask_1s = retrieval_events_mask & (events.type == 'REC_WORD') & (events.intrusion == 0) & (irts > 1000)


        filtered_events = events[encoding_events_mask | retrieval_events_mask_0s | retrieval_events_mask_1s].view(np.recarray)

        print len(events), 'sample events'


        self.pass_object('FR_baseline_events',base_events)
        self.pass_object('FR_events', events)
        self.pass_object('FR_math_events', math_events)
        self.pass_object('FR_intr_events', intr_events)
        self.pass_object('FR_rec_events', rec_events)
        self.pass_object('ps_events',ps_events)

        self.pass_object(task+'_events',filtered_events)

def modify_recalls(events):
    """ assigns recalls at random, and inserts rec_word events to match
    For testing purposes only"""

    encoding_mask = events.type=='WORD'
    word_events = events[encoding_mask]
    word_events.recalled = np.random.randint(2,size=word_events.shape)
    rec_words = []
    for word in word_events:
        if word.recalled:
            rec_start = events[(events.list==word.list) & (events.type=='REC_START')].mstime
            rec_end = events[(events.list==word.list)& (events.type=='REC_END')].mstime
            rec_eeg_start = events[(events.list==word.list) & (events.type=='REC_START')].eegoffset
            rec_word = word.copy().view(np.recarray)
            rec_word.type='REC_WORD'
            rec_word.mstime = np.random.randint(rec_start,rec_end)
            rec_word.eegoffset = rec_eeg_start + (rec_word.mstime-rec_start)
            rec_words.append(rec_word)
    events[encoding_mask] = word_events
    events = np.concatenate([events,rec_words]).view(np.recarray)
    events.sort(order='mstime')
    return events


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




