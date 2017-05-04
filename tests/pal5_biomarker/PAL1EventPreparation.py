__author__ = 'm'

import random
import os
import os.path
import numpy as np
from numpy.lib.recfunctions import append_fields
from ptsa.data.readers import BaseEventReader
from ptsa.data.readers.IndexReader import JsonIndexReader

from RamPipeline import *
from ReportUtils import RamTask

import hashlib
import warnings
from ReportTasks.RamTaskMethods import create_baseline_events_pal


class PAL1EventPreparation(RamTask):
    def __init__(self, mark_as_completed=True):
        super(PAL1EventPreparation, self).__init__(mark_as_completed)

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

    def process_session_rec_events(self, evs):
        """
        Filters out events based on PAL5 design doc

        :param evs: session events
        :return: filtered event recarray
        """

        ends = evs[evs.type == 'REC_END'].eegoffset
        starts = evs[evs.type == 'REC_START'].eegoffset
        rec_evs = evs[(evs.type == 'REC_EVENT')]

        # first let's get rid of all REC_EVENT's that are outside rec_start-rec_end interval

        starts_outside = ends[:-1]
        ends_outside = starts[1:]

        outside_event_offsets = None

        # rec_evs[(rec_evs.eegoffset > ends[7] ) & (rec_evs.eegoffset < starts[8])]
        # 725182 726099
        for start, end in zip(starts_outside, ends_outside):

            mask = (rec_evs.eegoffset > start) & (rec_evs.eegoffset < end)
            voc_evs = rec_evs[mask]
            num_voc_events = len(voc_evs)
            if num_voc_events > 0:

                rec_evs.keep_event[np.nonzero(mask)[0][:]] = 0

                # if outside_event_offsets is None:
                #     outside_event_offsets = voc_evs.eegoffset
                # else:
                #     outside_event_offsets = np.hstack((outside_event_offsets,voc_evs.eegoffset))

        # outside_event_offsets = outside_event_offsets.view(np.recarray)
        # print 'outside_event_offsets=',outside_event_offsets

        rec_evs = rec_evs[rec_evs.keep_event == 1]

        # removing multiple rec_events during single probe epoch

        counter = 0
        first_response_time = []
        for start, end in zip(starts, ends):

            mask = (rec_evs.eegoffset >= start) & (rec_evs.eegoffset <= end)
            voc_evs = rec_evs[mask]

            print 'number of rec_events so far=', len(rec_evs[rec_evs.eegoffset <= end])

            num_voc_events = len(voc_evs)
            # print 'num_voc_events=', num_voc_events
            # if num_voc_events == 1:
            #     first_response_time.append(voc_evs[0].eegoffset - start)

            if num_voc_events > 1:
                # getting rid of all but first rec event

                rec_evs.keep_event[np.nonzero(mask)[0][1:]] = 0

                # print 'GOT MULTI REC'
                # print voc_evs

            if num_voc_events != 0:
                first_response_time.append(voc_evs[0].eegoffset - start)

            if num_voc_events == 0:
                print 'NO REC_EVENT'

            counter += num_voc_events
            print 'counter=', counter
            print

        print 'counter=', counter

        rec_evs = rec_evs[rec_evs.keep_event == 1]

        row_temmplate = rec_evs[0].copy()

        # will add surrogate events
        new_rows = []

        for start, end in zip(starts, ends):

            voc_evs = rec_evs[(rec_evs.eegoffset >= start) & (rec_evs.eegoffset <= end)]
            num_voc_events = len(voc_evs)

            if num_voc_events == 0:
                # print 'NO REC_EVENT'
                new_row = row_temmplate.copy()
                new_row.type = 'REC_EVENT'
                new_row.eegoffset = start + random.choice(first_response_time)
                new_row.correct = 0
                new_row.study_1 = 'FAKE'
                new_row.study_2 = 'NEWS'
                new_row.probe_word = 'CROOKED'
                new_row.expecting_word = 'HILLARY'

                new_rows.append(new_row)

        if len(new_rows):
            rec_evs = np.append(rec_evs, new_rows).view(np.recarray)

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

    def process_trivial_session_rec_events(self, evs):

        ends = evs[evs.type == 'REC_END'].eegoffset
        starts = evs[evs.type == 'REC_START'].eegoffset
        rec_evs = evs[(evs.type == 'REC_EVENT')]

        first_response_time = []
        counter = 0
        for start, end in zip(starts, ends):

            mask = (rec_evs.eegoffset >= start) & (rec_evs.eegoffset <= end)
            voc_evs = rec_evs[mask]

            print 'number of rec_events so far=', len(rec_evs[rec_evs.eegoffset <= end])

            num_voc_events = len(voc_evs)
            # print 'num_voc_events=', num_voc_events
            # if num_voc_events == 1:
            #     first_response_time.append(voc_evs[0].eegoffset - start)

            if num_voc_events > 1:
                # getting rid of all but first rec event

                rec_evs.keep_event[np.nonzero(mask)[0][1:]] = 0

                # print 'GOT MULTI REC'
                # print voc_evs

            if num_voc_events != 0:
                first_response_time.append(voc_evs[0].eegoffset - start)

            if num_voc_events == 0:
                print 'NO REC_EVENT'

            counter += num_voc_events
            print 'counter=', counter
            print

        print 'counter=', counter

        rec_evs = rec_evs[rec_evs.keep_event == 1]

        return rec_evs


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

            sess_events = append_fields(sess_events, 'keep_event', sess_events.correct,
                                        dtypes=sess_events.correct.dtype, usemask=False,
                                        asrecarray=True)

            sess_events.keep_event = 1

            rec_events = self.process_session_rec_events(evs=sess_events)

            study_pair_events = sess_events[(sess_events.type == 'STUDY_PAIR') | (sess_events.type == 'PRACTICE_PAIR')]


            rec_rvs_trivial_sess = self.process_trivial_session_rec_events(sess_events)


            if trivial_rec_events is None:
                trivial_rec_events = rec_rvs_trivial_sess

            else:
                trivial_rec_events = np.hstack((trivial_rec_events, rec_rvs_trivial_sess))

            trivial_rec_events = trivial_rec_events.view(np.recarray)

            if events is None:
                events = np.hstack((study_pair_events, rec_events))
            else:
                events = np.hstack((events, np.hstack((study_pair_events, rec_events)).view(np.recarray)))
            events = events.view(np.recarray)

        print
        # # events = events.view(np.recarray)
        #
        # ######################################################################################
        #
        #
        # evs = events[events.session == 0]
        #
        # ######################################################################################
        #
        # print
        #
        # # processed_events = create_baseline_events_pal(events)
        #
        # encoding_events_mask = processed_events.type == 'STUDY_PAIR'
        # retrieval_events_mask = (processed_events.type == 'REC_EVENT') | (processed_events.type == 'REC_BASE')
        # irts = np.append([0], np.diff(processed_events.mstime))
        # retrieval_events_mask_0s = retrieval_events_mask & (processed_events.type == 'REC_BASE')
        # retrieval_events_mask_1s = retrieval_events_mask & (processed_events.type == 'REC_EVENT') & (
        #     processed_events.intrusion == 0) & (irts > 1000)
        #
        # # final filtering
        # processed_events = processed_events[encoding_events_mask | retrieval_events_mask_0s | retrieval_events_mask_1s]
        #
        # processed_events = processed_events.view(np.recarray)
        #
        # print len(processed_events), 'STUDY_PAIR events'

        self.pass_object('PAL1_events', events)



        # old code - based on FR5
        # processed_events = create_baseline_events_pal(events)
        #
        # encoding_events_mask = processed_events.type == 'STUDY_PAIR'
        # retrieval_events_mask = (processed_events.type == 'REC_EVENT') | (processed_events.type == 'REC_BASE')
        # irts = np.append([0], np.diff(processed_events.mstime))
        # retrieval_events_mask_0s = retrieval_events_mask & (processed_events.type == 'REC_BASE')
        # retrieval_events_mask_1s = retrieval_events_mask & (processed_events.type == 'REC_EVENT') & (
        #     processed_events.intrusion == 0) & (irts > 1000)
        #
        # # final filtering
        # processed_events = processed_events[encoding_events_mask | retrieval_events_mask_0s | retrieval_events_mask_1s]
        #
        # processed_events = processed_events.view(np.recarray)
        #
        # print len(processed_events), 'STUDY_PAIR events'
        #
        # self.pass_object('PAL1_events', processed_events)
