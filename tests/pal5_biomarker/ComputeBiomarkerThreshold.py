__author__ = 'm'

from RamPipeline import *

import numpy as np
# from morlet import MorletWaveletTransform
from ptsa.extensions.morlet.morlet import MorletWaveletTransform
from sklearn.externals import joblib

from ptsa.data.readers import EEGReader
from ptsa.data.readers.IndexReader import JsonIndexReader
from ReportTasks.RamTaskMethods import compute_wavelets_powers

import hashlib
import warnings


class ComputeBiomarkerThreshold(RamTask):
    def __init__(self, params, mark_as_completed=True):
        RamTask.__init__(self, mark_as_completed)

        self.params = params
        self.retrieval_wavelet_pow_mat = None
        self.samplerate = None

    def input_hashsum(self):
        subject = self.pipeline.subject
        tmp = subject.split('_')
        subj_code = tmp[0]
        montage = 0 if len(tmp) == 1 else int(tmp[1])

        json_reader = JsonIndexReader(os.path.join(self.pipeline.mount_point, 'protocols/r1.json'))

        hash_md5 = hashlib.md5()

        bp_paths = json_reader.aggregate_values('pairs', subject=subj_code, montage=montage)
        for fname in bp_paths:
            with open(fname, 'rb') as f: hash_md5.update(f.read())

        pal1_event_files = sorted(
            list(json_reader.aggregate_values('task_events', subject=subj_code, montage=montage, experiment='PAL1')))
        for fname in pal1_event_files:
            try:
                with open(fname, 'rb') as f:
                    hash_md5.update(f.read())
            except IOError:
                warnings.warn('Could not process %s. Please make sure that the event file exist' % fname,
                              RuntimeWarning)

        # pal3_event_files = sorted(list(json_reader.aggregate_values('task_events', subject=subj_code, montage=montage, experiment='PAL3')))
        # for fname in pal3_event_files:
        #     with open(fname,'rb') as f: hash_md5.update(f.read())

        return hash_md5.digest()

    def filter_pow_mat(self, pow_mat):
        """
        This function filters power matrix to exclude certain bipolar pairs - here the ones that "touch" stimulated
        electrodes
        :return: None
        """
        bipolar_pairs = self.get_passed_object('bipolar_pairs')
        reduced_pairs = self.get_passed_object('reduced_pairs')
        to_include = np.array([bp in reduced_pairs for bp in bipolar_pairs])
        pow_mat = pow_mat[:,to_include,...]

        return pow_mat

    def run(self):

        self.pipeline.subject = self.pipeline.subject.split('_')[0]
        subject = self.pipeline.subject

        monopolar_channels = self.get_passed_object('monopolar_channels')
        bipolar_pairs = self.get_passed_object('bipolar_pairs')


        events = self.get_passed_object('PAL1_events')

        events = events[events.type == 'REC_EVENT']

        rec_start_events = self.get_passed_object('rec_start_events')
        rec_start_events = rec_start_events[rec_start_events.type=='REC_EVENT']


        mean_dict = self.get_passed_object('features_mean_dict')
        std_dict = self.get_passed_object('features_std_dict')

        lr_classifier = self.get_passed_object('lr_classifier') # this si classifier trained on ALL sessions

        # but we need to evaluate outsample classifiers:
        xval_output = self.get_passed_object('xval_output')


        sessions = np.sort(np.unique(rec_start_events.session))

        samplerate = self.get_passed_object('samplerate')


        if samplerate is None:
            monopolar_channels = self.get_passed_object('monopolar_channels')

            eeg_reader = EEGReader(events=events[:2], channels=monopolar_channels, start_time=0.0,
                                   end_time=1.0)
            samplerate = float(eeg_reader.read().samplerate.data)

        params = self.params

        sliding_window_interval_delta = int(params.sliding_window_interval * samplerate)
        sliding_window_start_offset = int(params.sliding_window_start_offset * samplerate)
        sliding_window_length = int(
            (abs(params.pal1_retrieval_start_time) - abs(params.pal1_retrieval_end_time)) * samplerate)

        pal1_retrieval_start_offset = int(params.pal1_retrieval_start_time * samplerate)

        min_biomarker_pool = []

        total_window_evals = 0

        for sess in sessions:
            print 'Session %s'%sess

            outsample_classifier = xval_output[sess].classifier

            m = mean_dict[sess]
            s = std_dict[sess]



            sess_rec_start_events = rec_start_events [rec_start_events.session == sess]
            sess_events = events[events.session == sess]

            retrieval_wavelet_pow_mat = compute_wavelets_powers(sess_rec_start_events, monopolar_channels,
                                                                  bipolar_pairs,
                                                                  0.0,
                                                                  params.recall_period, params.pal1_buf,
                                                                  params.freqs)



            # retrieval_wavelet_pow_mat_1 = \
            #     joblib.load(
            #         self.get_path_to_resource_in_workspace(subject + '-retrieval_wavelet_pow_mat_sess_%d.pkl' % sess))

            retrieval_wavelet_pow_mat = self.filter_pow_mat(retrieval_wavelet_pow_mat)  # removing excluded electrodes

            # transposing so that event axis is first, and time axis is last

            retrieval_wavelet_pow_mat = retrieval_wavelet_pow_mat.transpose(2, 1, 0, 3)
            retrieval_wavelet_pow_mat += np.finfo(float).eps
            np.log10(retrieval_wavelet_pow_mat, out=retrieval_wavelet_pow_mat)

            sess_rec_start_events = rec_start_events[rec_start_events.session == sess]
            # sess_events = events[events.session == sess]

            retrieval_classifiers = []
            for ev_num, (rec_start_ev, ev) in enumerate(zip(sess_rec_start_events, sess_events)):
                rec_ev_offset = ev.eegoffset - rec_start_ev.eegoffset
                rec_ev_epoch_start = rec_ev_offset-625
                rec_ev_epoch_end =  rec_ev_offset-100

                if rec_ev_epoch_start<0:
                    retrieval_classifiers.append(-1.0)
                    continue

                ev_wavelet_pow = retrieval_wavelet_pow_mat[ev_num, :, :,
                                 rec_ev_epoch_start:rec_ev_epoch_end]

                mean_powers = np.nanmean(ev_wavelet_pow, -1)
                mean_powers = mean_powers.reshape(1,-1)
                features = (mean_powers - m)/s
                biomarker = outsample_classifier.predict_proba(features)[:, 1][0]
                retrieval_classifiers.append(biomarker)

            print retrieval_classifiers



            for ev_num, (rec_start_ev, ev) in enumerate(zip(sess_rec_start_events, sess_events)):
                # print 'processing event=', ev_num
                before_rec_event_window_length = ev.eegoffset - rec_start_ev.eegoffset

                number_of_classifier_evals = int(((
                                                  ev.eegoffset - rec_start_ev.eegoffset) + pal1_retrieval_start_offset - sliding_window_start_offset)) / int(sliding_window_interval_delta)

                if number_of_classifier_evals < 1:
                    continue

                total_window_evals += number_of_classifier_evals

                start_offsets = sliding_window_start_offset + np.arange(
                    number_of_classifier_evals) * sliding_window_interval_delta

                event_biomarkers = []

                # if ev_num==148:
                #     print

                for start_offset in start_offsets:
                    # print 'start_offset=',start_offset
                    ev_wavelet_pow = retrieval_wavelet_pow_mat[ev_num, :, :,
                                     start_offset:start_offset + sliding_window_length]

                    mean_powers = np.nanmean(ev_wavelet_pow, -1)
                    mean_powers = mean_powers.reshape(1,-1)
                    features = (mean_powers - m)/s

                    # mean_powers_flat = mean_powers.flatten()
                    #
                    # features = (mean_powers_flat - m)/s


                    biomarker = outsample_classifier.predict_proba(features)[:, 1][0]

                    event_biomarkers.append(biomarker)

                min_biomarker = np.min(event_biomarkers)
                min_biomarker_pool.append(min_biomarker)

            print 'session complete - median min biomarker = ', np.median(min_biomarker_pool)

        print 'total_window_evals=',total_window_evals

        retrieval_biomarker_threshold = np.median(min_biomarker_pool)

        print "retrieval_biomarker_threshold=",retrieval_biomarker_threshold

        self.pass_object('retrieval_biomarker_threshold', retrieval_biomarker_threshold)
        np.save(self.get_path_to_resource_in_workspace('retrieval_biomarker_threshold.npy'),retrieval_biomarker_threshold)

    def restore(self):
        retrieval_biomarker_threshold = np.load(self.get_path_to_resource_in_workspace('retrieval_biomarker_threshold.npy'))
        self.pass_object('retrieval_biomarker_threshold', retrieval_biomarker_threshold)
