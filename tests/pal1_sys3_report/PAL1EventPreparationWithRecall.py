from ReportTasks.RamTaskMethods import filter_session
import random
import os
import os.path
import numpy as np
from numpy.lib.recfunctions import append_fields
from ptsa.data.readers import BaseEventReader
from ptsa.data.readers.IndexReader import JsonIndexReader

from ReportUtils import RamTask

import hashlib
import warnings
from ReportTasks.RamTaskMethods import create_baseline_events_pal
from ptsa.data.readers import EEGReader


class PAL1EventPreparationWithRecall(RamTask):
    def __init__(self, mark_as_completed=True):
        super(PAL1EventPreparationWithRecall, self).__init__(mark_as_completed)
        self.samplerate = None

    def input_hashsum(self):
        subject = self.pipeline.subject
        tmp = subject.split('_')
        subj_code = tmp[0]
        montage = 0 if len(tmp) == 1 else int(tmp[1])

        json_reader = JsonIndexReader(os.path.join(self.pipeline.mount_point, 'protocols/r1.json'))

        hash_md5 = hashlib.md5()

        pal1_event_files = sorted(
            list(json_reader.aggregate_values('task_events', subject=subj_code, montage=montage, experiment='PAL1')))
        for fname in pal1_event_files:
            with open(fname, 'rb') as f: hash_md5.update(f.read())

        pal3_event_files = sorted(
            list(json_reader.aggregate_values('task_events', subject=subj_code, montage=montage, experiment='PAL3')))
        for fname in pal3_event_files:
            with open(fname, 'rb') as f: hash_md5.update(f.read())

        return hash_md5.digest()


    def process_session_rec_events_nih(self, evs):
        """
        Filters out events based on PAL5 design doc

        :param evs: session events
        :return: filtered event recarray
        """

        rec_evs = evs[(evs.type == 'TEST_PROBE') | (evs.type=='PROBE_START')]

        incorrect_has_response_mask = (rec_evs.RT != -999) & (rec_evs.correct == 0)
        incorrect_no_response_mask = rec_evs.RT == -999

        incorrect_has_response = rec_evs[incorrect_has_response_mask]
        incorrect_no_response = rec_evs[incorrect_no_response_mask]
        correct_mask = rec_evs.correct == 1

        # test
        tot_events = sum(incorrect_no_response_mask)+sum(incorrect_has_response_mask) + sum(rec_evs.correct==1)
        np.testing.assert_equal(tot_events,len(rec_evs))



        # correct_response_times = rec_evs[incorrect_has_response_mask | correct_mask].RT

        correct_response_times = rec_evs[ correct_mask].RT # todo fixed based on Jim's suggestion

        response_time_rand_indices = np.random.randint(0, len(correct_response_times), sum(incorrect_no_response_mask))

        rec_evs.RT[incorrect_no_response_mask] = correct_response_times[response_time_rand_indices]
        # rec_evs.RT[incorrect_no_response_mask] = 2500 # todo remove from production code

        rec_evs.type = 'REC_EVENT'

        # rec_evs.eegoffset = rec_evs.eegoffset + rec_evs.RT

        rec_evs.eegoffset = rec_evs.eegoffset + (rec_evs.RT*self.samplerate/1000.0).astype(np.int64)



        return rec_evs




    def get_sample_rate(self,evs):
        """
        This fcn reads short segmenty of eeg data. We do this to get samplerate

        :param evs: events
        :return: samplerate
        """


        monopolar_channels = self.get_passed_object('monopolar_channels')

        eeg_reader = EEGReader(events=evs, channels=monopolar_channels, start_time=0.0,
                               end_time=1.0)
        eeg = eeg_reader.read()
        self.samplerate = float(eeg['samplerate'])

        return self.samplerate

    def run(self):

        evs_field_list = ['session', 'list', 'serialpos', 'type', 'probepos', 'study_1',
                          'study_2', 'cue_direction', 'probe_word', 'expecting_word',
                          'resp_word', 'correct', 'intrusion', 'resp_pass', 'vocalization',
                          'RT', 'mstime', 'msoffset', 'eegoffset', 'eegfile'
                          ]

        subject = self.pipeline.subject
        tmp = subject.split('_')
        subj_code = tmp[0]
        montage = 0 if len(tmp) == 1 else int(tmp[1])

        json_reader = JsonIndexReader(os.path.join(self.pipeline.mount_point, 'protocols/r1.json'))

        event_files = sorted(
            list(json_reader.aggregate_values('task_events', subject=subj_code, montage=montage, experiment='PAL1')))
        events = None

        trivial_rec_events = None

        for sess_file in event_files:
            e_path = os.path.join(self.pipeline.mount_point, str(sess_file))

            e_reader = BaseEventReader(filename=e_path, eliminate_events_with_no_eeg=True)
            try:
                sess_events = filter_session(e_reader.read())[evs_field_list]
            except IOError:
                warnings.warn('Could not process %s. Please make sure that the event file exist' % e_path,
                              RuntimeWarning)
                continue

            if self.samplerate is None:
                self.samplerate = self.get_sample_rate(sess_events[:2])
                self.pass_object('samplerate', self.samplerate)

            rec_events = self.process_session_rec_events_nih(evs=sess_events)


            study_pair_events = sess_events[(sess_events.type == 'STUDY_PAIR')]


            merged_events = np.hstack((study_pair_events, rec_events)).view(np.recarray)

            # sorting according to eegoffset
            merged_events = merged_events[np.argsort(merged_events.eegoffset)]

            if events is None:
                events = merged_events
            else:
                # events = np.hstack((events, np.hstack((study_pair_events, rec_events)).view(np.recarray)))
                events = np.hstack((events, merged_events))

            events = events.view(np.recarray)

        self.pass_object('PAL1_events_with_recall', events)

        rec_start_events = events.copy()

        rec_start_events = rec_start_events[rec_start_events.type == 'REC_EVENT']

        # rec_start_events.eegoffset  = rec_start_events.eegoffset - rec_start_events.RT
        rec_start_events.eegoffset  = rec_start_events.eegoffset - (rec_start_events.RT*self.samplerate/1000.0).astype(np.int64)



        # rec_start_events.eegoffset = rec_start_events.rec_start # todo - original code

        self.pass_object('rec_start_events', rec_start_events)
