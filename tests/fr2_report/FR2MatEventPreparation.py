__author__ = 'm'

import os
import os.path
import numpy as np

from ptsa.data.readers import BaseEventReader
from ptsa.data.readers.IndexReader import JsonIndexReader

from RamPipeline import *
from ReportUtils import ReportRamTask

import hashlib


class FR2EventPreparation(ReportRamTask):
    def __init__(self, params, mark_as_completed=True):
        super(FR2EventPreparation, self).__init__(mark_as_completed)
        self.params = params

    def input_hashsum(self):
        subject = self.pipeline.subject
        task = self.pipeline.task
        tmp = subject.split('_')
        subj_code = tmp[0]
        montage = 0 if len(tmp)==1 else int(tmp[1])

        json_reader = JsonIndexReader(os.path.join(self.pipeline.mount_point, 'protocols/r1.json'))

        hash_md5 = hashlib.md5()

        event_files = sorted(list(json_reader.aggregate_values('all_events', subject=subj_code, montage=montage, experiment=task)))
        for fname in event_files:
            with open(fname,'rb') as f: hash_md5.update(f.read())

        return hash_md5.digest()

    def run(self):
        subject = self.pipeline.subject
        task = self.pipeline.task
        ram_task = task

        evs_field_list = ['itemno', 'serialpos', 'session', 'subject', 'rectime', 'expVersion', 'mstime', 'type', 'eegoffset', 'recalled', 'item', 'intrusion', 'list', 'eegfile', 'msoffset','isStim', 'stimList']
        new_field_list = ['item_num', 'serialpos', 'session', 'subject', 'rectime', 'experiment','mstime', 'type', 'eegoffset', 'recalled', 'item_name', 'intrusion', 'list', 'eegfile', 'msoffset','is_stim', 'stim_list']

        if 'cat' in task:
            evs_field_list += ['category', 'category_num']
            new_field_list += ['category', 'category_num']
            ram_task = 'Cat'+ram_task.split('cat')[1]

        event_files = ['/data/events/RAM_%s/%s_events.mat'%(ram_task,subject)]
        events = None
        for sess_file in event_files:
            e_path = os.path.join(self.pipeline.mount_point, str(sess_file))
            e_reader = BaseEventReader(filename=e_path, eliminate_events_with_no_eeg=True)

            sess_events = e_reader.read()[evs_field_list]

            if events is None:
                events = sess_events
            else:
                events = np.hstack((events,sess_events))

        events = events.view(np.recarray)

        print events.dtype.names
        events.dtype.names  = tuple(new_field_list)
        print 'new names: ',events.dtype.names

        if self.params.stim is True:
            events = events[events.stim_list!=0]
        elif self.params.stim is False:
            events = events[events.stim_list!=1]


        self.pass_object(task+'_all_events', events)

        math_events = BaseEventReader(filename='/data/events/RAM_%s/%s_math.mat'%(ram_task,subject)).read()

        rec_events = events[events.type == 'REC_WORD']

        intr_events = rec_events[(rec_events.intrusion!=-999) & (rec_events.intrusion!=0)]

        events = events[events.type == 'WORD']

        print len(events), task, 'WORD events'

        self.pass_object(task+'_events', events)
        self.pass_object(task+'_math_events', math_events)
        self.pass_object(task+'_intr_events', intr_events)
        self.pass_object(task+'_rec_events', rec_events)
