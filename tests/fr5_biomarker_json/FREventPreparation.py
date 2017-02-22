__author__ = 'm'

import os
import os.path
import numpy as np

from ptsa.data.readers import BaseEventReader
from ptsa.data.readers.IndexReader import JsonIndexReader

from RamPipeline import *
from ReportUtils import RamTask

import hashlib


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

        fr1_events_fname = os.path.abspath(
            os.path.join(self.pipeline.mount_point, 'data/events', 'RAM_FR5_TRAINING', subj_code + '_events.mat'))

        json_reader = JsonIndexReader(os.path.join(self.pipeline.mount_point, 'protocols/r1.json'))

        event_files = sorted(
            list(json_reader.aggregate_values('task_events', subject=subj_code, montage=montage, experiment='FR1')))

        e_reader = BaseEventReader(filename=fr1_events_fname, eliminate_events_with_no_eeg=True)
        fr1_events = e_reader.read()
        print

        encoding_events_mask = fr1_events.type == 'WORD'
        retrieval_events_mask = (fr1_events.type == 'REC_WORD') | (fr1_events.type == 'REC_BASE')

        retrieval_events_mask_0s = retrieval_events_mask & (fr1_events.type == 'REC_BASE')
        retrieval_events_mask_1s = retrieval_events_mask & (fr1_events.type == 'REC_WORD') & \
                                   (fr1_events['repeat'] == 0) & (fr1_events.pirt > 1000) & (fr1_events.intrusion == 0)

        filtered_events = fr1_events[encoding_events_mask | retrieval_events_mask_0s | retrieval_events_mask_1s]

        print

        # events = None
        # for sess_file in event_files:
        #     e_path = os.path.join(self.pipeline.mount_point, str(sess_file))
        #     print e_path
        #     e_reader = BaseEventReader(filename=e_path, eliminate_events_with_no_eeg=True)
        #
        #     sess_events = e_reader.read()[['item_num', 'serialpos', 'session', 'subject', 'rectime', 'mstime', 'type', 'eegoffset', 'recalled', 'item_name', 'intrusion', 'montage', 'list', 'eegfile', 'msoffset']]
        #     sess_events = sess_events[sess_events.type=='WORD']
        #
        #     if events is None:
        #         events = sess_events
        #     else:
        #         events = np.hstack((events,sess_events))
        #
        # event_files = sorted(list(json_reader.aggregate_values('task_events', subject=subj_code, montage=montage, experiment='catFR1')))
        # for sess_file in event_files:
        #     e_path = os.path.join(self.pipeline.mount_point, str(sess_file))
        #     print e_path
        #     e_reader = BaseEventReader(filename=e_path, eliminate_events_with_no_eeg=True)
        #
        #     sess_events = e_reader.read()
        #     sess_events.session += 100
        #     sess_events = sess_events[['item_num', 'serialpos', 'session', 'subject', 'rectime', 'mstime', 'type', 'eegoffset', 'recalled', 'item_name', 'intrusion', 'montage', 'list', 'eegfile', 'msoffset']]
        #     sess_events = sess_events[sess_events.type=='WORD']
        #
        #     if events is None:
        #         events = sess_events
        #     else:
        #         events = np.hstack((events,sess_events))
        #
        # event_files = sorted(list(json_reader.aggregate_values('task_events', subject=subj_code, montage=montage, experiment='FR3')))
        # for sess_file in event_files:
        #     e_path = os.path.join(self.pipeline.mount_point, str(sess_file))
        #     print e_path
        #     e_reader = BaseEventReader(filename=e_path, eliminate_events_with_no_eeg=True)
        #
        #     sess_events = e_reader.read()
        #     sess_events = sess_events[(sess_events.stim_list==0) & (sess_events.type=='WORD')]
        #     sess_events.session += 200
        #     sess_events = sess_events[['item_num', 'serialpos', 'session', 'subject', 'rectime', 'mstime', 'type', 'eegoffset', 'recalled', 'item_name', 'intrusion', 'montage', 'list', 'eegfile', 'msoffset']]
        #
        #     if events is None:
        #         events = sess_events
        #     else:
        #         events = np.hstack((events,sess_events))
        #
        # event_files = sorted(list(json_reader.aggregate_values('task_events', subject=subj_code, montage=montage, experiment='catFR3')))
        # for sess_file in event_files:
        #     e_path = os.path.join(self.pipeline.mount_point, str(sess_file))
        #     print e_path
        #     e_reader = BaseEventReader(filename=e_path, eliminate_events_with_no_eeg=True)
        #
        #     sess_events = e_reader.read()
        #     sess_events = sess_events[(sess_events.stim_list==0) & (sess_events.type=='WORD')]
        #     sess_events.session += 300
        #     sess_events = sess_events[['item_num', 'serialpos', 'session', 'subject', 'rectime', 'mstime', 'type', 'eegoffset', 'recalled', 'item_name', 'intrusion', 'montage', 'list', 'eegfile', 'msoffset']]
        #
        #     if events is None:
        #         events = sess_events
        #     else:
        #         events = np.hstack((events,sess_events))

        # events = events.view(np.recarray)
        events = filtered_events.view(np.recarray)

        print len(events), 'WORD events'

        self.pass_object('FR_events', events)
        # self.pass_object('encoding_events_mask',encoding_events_mask)
        # self.pass_object('retrieval_events_mask_0s',retrieval_events_mask_0s)
        # self.pass_object('retrieval_events_mask_1s',retrieval_events_mask_1s)

# class FREventPreparation(RamTask):
#     def __init__(self, mark_as_completed=True):
#         super(FREventPreparation,self).__init__(mark_as_completed)
#
#     def input_hashsum(self):
#         subject = self.pipeline.subject
#         tmp = subject.split('_')
#         subj_code = tmp[0]
#         montage = 0 if len(tmp)==1 else int(tmp[1])
#
#         json_reader = JsonIndexReader(os.path.join(self.pipeline.mount_point, 'protocols/r1.json'))
#
#         hash_md5 = hashlib.md5()
#
#         fr1_event_files = sorted(list(json_reader.aggregate_values('task_events', subject=subj_code, montage=montage, experiment='FR1')))
#         for fname in fr1_event_files:
#             with open(fname,'rb') as f: hash_md5.update(f.read())
#
#         catfr1_event_files = sorted(list(json_reader.aggregate_values('task_events', subject=subj_code, montage=montage, experiment='catFR1')))
#         for fname in catfr1_event_files:
#             with open(fname,'rb') as f: hash_md5.update(f.read())
#
#         fr3_event_files = sorted(list(json_reader.aggregate_values('task_events', subject=subj_code, montage=montage, experiment='FR3')))
#         for fname in fr3_event_files:
#             with open(fname,'rb') as f: hash_md5.update(f.read())
#
#         catfr3_event_files = sorted(list(json_reader.aggregate_values('task_events', subject=subj_code, montage=montage, experiment='catFR3')))
#         for fname in catfr3_event_files:
#             with open(fname,'rb') as f: hash_md5.update(f.read())
#
#         return hash_md5.digest()
#
#     def run(self):
#         subject = self.pipeline.subject
#         tmp = subject.split('_')
#         subj_code = tmp[0]
#         montage = 0 if len(tmp)==1 else int(tmp[1])
#
#         json_reader = JsonIndexReader(os.path.join(self.pipeline.mount_point, 'protocols/r1.json'))
#
#         event_files = sorted(list(json_reader.aggregate_values('task_events', subject=subj_code, montage=montage, experiment='FR1')))
#         events = None
#         for sess_file in event_files:
#             e_path = os.path.join(self.pipeline.mount_point, str(sess_file))
#             print e_path
#             e_reader = BaseEventReader(filename=e_path, eliminate_events_with_no_eeg=True)
#
#             sess_events = e_reader.read()[['item_num', 'serialpos', 'session', 'subject', 'rectime', 'mstime', 'type', 'eegoffset', 'recalled', 'item_name', 'intrusion', 'montage', 'list', 'eegfile', 'msoffset']]
#             sess_events = sess_events[sess_events.type=='WORD']
#
#             if events is None:
#                 events = sess_events
#             else:
#                 events = np.hstack((events,sess_events))
#
#         event_files = sorted(list(json_reader.aggregate_values('task_events', subject=subj_code, montage=montage, experiment='catFR1')))
#         for sess_file in event_files:
#             e_path = os.path.join(self.pipeline.mount_point, str(sess_file))
#             print e_path
#             e_reader = BaseEventReader(filename=e_path, eliminate_events_with_no_eeg=True)
#
#             sess_events = e_reader.read()
#             sess_events.session += 100
#             sess_events = sess_events[['item_num', 'serialpos', 'session', 'subject', 'rectime', 'mstime', 'type', 'eegoffset', 'recalled', 'item_name', 'intrusion', 'montage', 'list', 'eegfile', 'msoffset']]
#             sess_events = sess_events[sess_events.type=='WORD']
#
#             if events is None:
#                 events = sess_events
#             else:
#                 events = np.hstack((events,sess_events))
#
#         event_files = sorted(list(json_reader.aggregate_values('task_events', subject=subj_code, montage=montage, experiment='FR3')))
#         for sess_file in event_files:
#             e_path = os.path.join(self.pipeline.mount_point, str(sess_file))
#             print e_path
#             e_reader = BaseEventReader(filename=e_path, eliminate_events_with_no_eeg=True)
#
#             sess_events = e_reader.read()
#             sess_events = sess_events[(sess_events.stim_list==0) & (sess_events.type=='WORD')]
#             sess_events.session += 200
#             sess_events = sess_events[['item_num', 'serialpos', 'session', 'subject', 'rectime', 'mstime', 'type', 'eegoffset', 'recalled', 'item_name', 'intrusion', 'montage', 'list', 'eegfile', 'msoffset']]
#
#             if events is None:
#                 events = sess_events
#             else:
#                 events = np.hstack((events,sess_events))
#
#         event_files = sorted(list(json_reader.aggregate_values('task_events', subject=subj_code, montage=montage, experiment='catFR3')))
#         for sess_file in event_files:
#             e_path = os.path.join(self.pipeline.mount_point, str(sess_file))
#             print e_path
#             e_reader = BaseEventReader(filename=e_path, eliminate_events_with_no_eeg=True)
#
#             sess_events = e_reader.read()
#             sess_events = sess_events[(sess_events.stim_list==0) & (sess_events.type=='WORD')]
#             sess_events.session += 300
#             sess_events = sess_events[['item_num', 'serialpos', 'session', 'subject', 'rectime', 'mstime', 'type', 'eegoffset', 'recalled', 'item_name', 'intrusion', 'montage', 'list', 'eegfile', 'msoffset']]
#
#             if events is None:
#                 events = sess_events
#             else:
#                 events = np.hstack((events,sess_events))
#
#         events = events.view(np.recarray)
#
#         print len(events), 'WORD events'
#
#         self.pass_object('FR_events', events)
