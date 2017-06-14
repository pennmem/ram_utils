__author__ = 'm'

import hashlib
import os.path

import numpy as np
from ...ReportTasks.RamTaskMethods import create_baseline_events
from ...ReportUtils import RamTask
from ptsa.data.readers import BaseEventReader
from ptsa.data.readers.IndexReader import JsonIndexReader

from ram_utils.RamPipeline import *


class FREventPreparationWithRecall(RamTask):
    def __init__(self, mark_as_completed=True):
        super(FREventPreparationWithRecall, self).__init__(mark_as_completed)

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

        return hash_md5.digest()

    def run(self):
        subject = self.pipeline.subject
        tmp = subject.split('_')
        subj_code = tmp[0]
        montage = 0 if len(tmp) == 1 else int(tmp[1])

        json_reader = JsonIndexReader(os.path.join(self.pipeline.mount_point, 'protocols/r1.json'))

        event_files = sorted(
            list(json_reader.aggregate_values('task_events', subject=subj_code, montage=montage, experiment='FR1')))

        catfr1_event_files = sorted(
            list(json_reader.aggregate_values('task_events', subject=subj_code, montage=montage, experiment='CatFR1')))

        # have to convert path to str explicitly to avoid exception in TypedUtils from PTSA.
        # when path comes from the console it is a unicode str.
        fr1_evs_list = [BaseEventReader(filename=str(event_path)).read() for event_path in event_files]

        catfr1_evs_list = [BaseEventReader(filename=str(event_path)).read() for event_path in catfr1_event_files]

        processed_fr1_events = self.process_events(fr1_evs_list)
        processed_catfr1_events = self.process_events(catfr1_evs_list)

        self.pass_object('FR_events_with_recall', processed_fr1_events)
        self.pass_object('CatFR_events_with_recall', processed_catfr1_events)

    def process_events(self, evs_list):
        """
        Generates recall events based on the list of raw events read from the disk

        :param fr1_evs_list: list of raw events read from disk (list of FR1 events or list of CatFR1)
        :return: {recarray} processed events for a given task (FR1 or CatFR1)
        """
        if not len(evs_list):
            return None

        events = np.concatenate(
            evs_list
        ).view(np.recarray)
        events = create_baseline_events(events, 1000, 29000)

        encoding_events_mask = events.type == 'WORD'
        retrieval_events_mask = (events.type == 'REC_WORD') | (events.type == 'REC_BASE')
        irts = np.append([0], np.diff(events.mstime))
        retrieval_events_mask_0s = retrieval_events_mask & (events.type == 'REC_BASE')
        retrieval_events_mask_1s = retrieval_events_mask & (events.type == 'REC_WORD') & (
            events.intrusion == 0) & (irts > 1000)

        filtered_events = events[encoding_events_mask | retrieval_events_mask_0s | retrieval_events_mask_1s]

        processed_events = filtered_events.view(np.recarray)

        return processed_events
