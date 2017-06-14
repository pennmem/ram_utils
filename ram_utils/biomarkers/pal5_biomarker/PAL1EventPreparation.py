__author__ = 'm'



import random
import os
import os.path
import numpy as np
from numpy.lib.recfunctions import append_fields
from ptsa.data.readers import BaseEventReader
from ptsa.data.readers.IndexReader import JsonIndexReader

from ram_utils.RamPipeline import *
from ReportUtils import RamTask

import hashlib
import warnings
from ReportTasks.RamTaskMethods import create_baseline_events_pal
from ptsa.data.readers import EEGReader


class PAL1EventPreparation(RamTask):
    def __init__(self, mark_as_completed=True):
        super(PAL1EventPreparation, self).__init__(mark_as_completed)
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

    # def process_session_rec_events(self, evs):
    #     """
    #     Filters out events based on PAL5 design doc
    #
    #     :param evs: session events
    #     :return: filtered event recarray
    #     """
    #
    #     ends = evs[evs.type == 'REC_END'].eegoffset
    #     starts = evs[evs.type == 'REC_START'].eegoffset
    #     rec_evs = evs[(evs.type == 'REC_EVENT')]
    #
    #     # first let's get rid of all REC_EVENT's that are outside rec_start-rec_end interval
    #
    #     starts_outside = ends[:-1]
    #     ends_outside = starts[1:]
    #
    #     outside_event_offsets = None
    #
    #     # rec_evs[(rec_evs.eegoffset > ends[7] ) & (rec_evs.eegoffset < starts[8])]
    #     # 725182 726099
    #     for start, end in zip(starts_outside, ends_outside):
    #
    #         mask = (rec_evs.eegoffset > start) & (rec_evs.eegoffset < end)
    #         voc_evs = rec_evs[mask]
    #         num_voc_events = len(voc_evs)
    #         if num_voc_events > 0:
    #             rec_evs.keep_event[np.nonzero(mask)[0][:]] = 0
    #
    #             # if outside_event_offsets is None:
    #             #     outside_event_offsets = voc_evs.eegoffset
    #             # else:
    #             #     outside_event_offsets = np.hstack((outside_event_offsets,voc_evs.eegoffset))
    #
    #     # outside_event_offsets = outside_event_offsets.view(np.recarray)
    #     # print 'outside_event_offsets=',outside_event_offsets
    #
    #     rec_evs = rec_evs[rec_evs.keep_event == 1]
    #
    #     # removing multiple rec_events during single probe epoch
    #
    #     counter = 0
    #     first_response_time = []
    #     for start, end in zip(starts, ends):
    #
    #         mask = (rec_evs.eegoffset >= start) & (rec_evs.eegoffset <= end)
    #         voc_evs = rec_evs[mask]
    #
    #         print 'number of rec_events so far=', len(rec_evs[rec_evs.eegoffset <= end])
    #
    #         num_voc_events = len(voc_evs)
    #         # print 'num_voc_events=', num_voc_events
    #         # if num_voc_events == 1:
    #         #     first_response_time.append(voc_evs[0].eegoffset - start)
    #
    #
    #         if num_voc_events > 1:
    #             # getting rid of all but first rec event
    #
    #             rec_evs.keep_event[np.nonzero(mask)[0][1:]] = 0
    #
    #         if num_voc_events >= 1:
    #             rec_evs.rec_start[np.nonzero(mask)[0][0]] = start
    #
    #             # print 'GOT MULTI REC'
    #             # print voc_evs
    #
    #         if num_voc_events != 0:
    #             first_response_time.append(voc_evs[0].eegoffset - start)
    #
    #         if num_voc_events == 0:
    #             print 'NO REC_EVENT'
    #
    #         counter += num_voc_events
    #         print 'counter=', counter
    #         print
    #
    #     print 'counter=', counter
    #
    #     rec_evs = rec_evs[rec_evs.keep_event == 1]
    #
    #     row_temmplate = rec_evs[0].copy()
    #
    #     # will add surrogate events
    #     new_rows = []
    #
    #     for start, end in zip(starts, ends):
    #
    #         voc_evs = rec_evs[(rec_evs.eegoffset >= start) & (rec_evs.eegoffset <= end)]
    #         num_voc_events = len(voc_evs)
    #
    #         if num_voc_events == 0:
    #             # print 'NO REC_EVENT'
    #             new_row = row_temmplate.copy()
    #             new_row.type = 'REC_EVENT'
    #             new_row.rec_start = start
    #             new_row.eegoffset = start + random.choice(first_response_time)
    #             new_row.correct = 0
    #             new_row.study_1 = 'FAKE'
    #             new_row.study_2 = 'NEWS'
    #             new_row.probe_word = 'CROOKED'
    #             new_row.expecting_word = 'HILLARY'
    #
    #             new_rows.append(new_row)
    #
    #     if len(new_rows):
    #         rec_evs = np.append(rec_evs, new_rows).view(np.recarray)
    #
    #     return rec_evs

    def process_session_rec_events_nih(self, evs):
        """
        Filters out events based on PAL5 design doc

        :param evs: session events
        :return: filtered event recarray
        """

        rec_evs = evs[evs.type == 'TEST_PROBE']

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

        # rec_evs.RT[incorrect_no_response_mask] = correct_response_times[response_time_rand_indices]
        rec_evs.RT[incorrect_no_response_mask] = 2500 # todo remove from production code

        rec_evs.type = 'REC_EVENT'

        # rec_evs.eegoffset = rec_evs.eegoffset + rec_evs.RT

        rec_evs.eegoffset = rec_evs.eegoffset + (rec_evs.RT*self.samplerate/1000.0).astype(np.int64)



        return rec_evs


        #         new_rows.append(new_row)
        #
        # rec_evs

        # starts_shift = ends[:-1]
        # ends_shift = starts[1:]
        #
        # counter_outside_voc_evs = 0
        # for start, end in zip(starts_shift, ends_shift):
        #     outside_voc_evs = rec_evs[(rec_evs.eegoffset > start) & (rec_evs.eegoffset < end)]
        #
        #     num_outside_voc_evs = len(outside_voc_evs)
        #     counter_outside_voc_evs += num_outside_voc_evs
        # print 'counter_outside_voc_evs=', counter_outside_voc_evs

    # def process_trivial_session_rec_events(self, evs):
    #     """
    #
    #     :param evs:
    #     :return:
    #     """
    #
    #     ends = evs[evs.type == 'REC_END'].eegoffset
    #     starts = evs[evs.type == 'REC_START'].eegoffset
    #     rec_evs = evs[(evs.type == 'REC_EVENT')]
    #
    #     first_response_time = []
    #     counter = 0
    #     for start, end in zip(starts, ends):
    #
    #         mask = (rec_evs.eegoffset >= start) & (rec_evs.eegoffset <= end)
    #         voc_evs = rec_evs[mask]
    #
    #         # print 'number of rec_events so far=', len(rec_evs[rec_evs.eegoffset <= end])
    #
    #         num_voc_events = len(voc_evs)
    #         # print 'num_voc_events=', num_voc_events
    #         # if num_voc_events == 1:
    #         #     first_response_time.append(voc_evs[0].eegoffset - start)
    #
    #         if num_voc_events > 1:
    #             # getting rid of all but first rec event
    #
    #             rec_evs.keep_event[np.nonzero(mask)[0][1:]] = 0
    #
    #             # print 'GOT MULTI REC'
    #             # print voc_evs
    #
    #         if num_voc_events != 0:
    #             first_response_time.append(voc_evs[0].eegoffset - start)
    #
    #         # if num_voc_events == 0:
    #         #     print 'NO REC_EVENT'
    #
    #         counter += num_voc_events
    #         # print 'counter=', counter
    #         # print
    #
    #     # print 'counter=', counter
    #
    #     rec_evs = rec_evs[rec_evs.keep_event == 1]
    #
    #     return rec_evs


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
                sess_events = e_reader.read()[evs_field_list]
            except IOError:
                warnings.warn('Could not process %s. Please make sure that the event file exist' % e_path,
                              RuntimeWarning)
                continue

            if self.samplerate is None:
                self.samplerate = self.get_sample_rate(sess_events[:2])
                self.pass_object('samplerate', self.samplerate)


            sess_events = append_fields(sess_events, 'keep_event', sess_events.correct,
                                        dtypes=sess_events.correct.dtype, usemask=False,
                                        asrecarray=True)

            sess_events.keep_event = 1

            sess_events = append_fields(sess_events, 'rec_start', sess_events.eegoffset,
                                        dtypes=sess_events.eegoffset.dtype, usemask=False,
                                        asrecarray=True)

            sess_events.rec_start = -1

            # rec_events = self.process_session_rec_events(evs=sess_events)

            rec_events = self.process_session_rec_events_nih(evs=sess_events)

            # rec_events_orig = self.process_session_rec_events(evs=sess_events)

            # # ----------------------------------------------------------------------------------------
            # nih_path = 'd:/data/events/RAM_PAL1/R1250N_sess_0.mat'
            # nih_e_reader = BaseEventReader(filename=nih_path, eliminate_events_with_no_eeg=False)
            # nih_sess_events = nih_e_reader.read()
            #
            #
            # rec_evs_test = self.process_session_rec_events_nih(evs=sess_events)
            #
            # # ----------------------------------------------------------------------------------------


            # study_pair_events = sess_events[(sess_events.type == 'STUDY_PAIR') | (sess_events.type == 'PRACTICE_PAIR')]
            study_pair_events = sess_events[(sess_events.type == 'STUDY_PAIR')]

            # rec_rvs_trivial_sess = self.process_trivial_session_rec_events(sess_events)


            # if trivial_rec_events is None:
            #     trivial_rec_events = rec_rvs_trivial_sess
            #
            # else:
            #     trivial_rec_events = np.hstack((trivial_rec_events, rec_rvs_trivial_sess))
            #
            # trivial_rec_events = trivial_rec_events.view(np.recarray)
            merged_events = np.hstack((study_pair_events, rec_events)).view(np.recarray)

            # sorting according to eegoffset
            merged_events = merged_events[np.argsort(merged_events.eegoffset)]

            if events is None:
                events = merged_events
            else:
                # events = np.hstack((events, np.hstack((study_pair_events, rec_events)).view(np.recarray)))
                events = np.hstack((events, merged_events))

            events = events.view(np.recarray)

        self.pass_object('PAL1_events', events)

        rec_start_events = events.copy()

        rec_start_events = rec_start_events[rec_start_events.type == 'REC_EVENT']

        # rec_start_events.eegoffset  = rec_start_events.eegoffset - rec_start_events.RT
        rec_start_events.eegoffset  = rec_start_events.eegoffset - (rec_start_events.RT*self.samplerate/1000.0).astype(np.int64)



        # rec_start_events.eegoffset = rec_start_events.rec_start # todo - original code

        self.pass_object('rec_start_events', rec_start_events)
