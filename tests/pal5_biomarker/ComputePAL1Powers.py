__author__ = 'm'

from RamPipeline import *

import numpy as np
# from morlet import MorletWaveletTransform
from ptsa.extensions.morlet.morlet import MorletWaveletTransform
from sklearn.externals import joblib

from ptsa.data.readers import EEGReader
from ptsa.data.readers.IndexReader import JsonIndexReader

import hashlib
import warnings
from ReportTasks.RamTaskMethods import get_reduced_pairs,get_excluded_dict
import json

try:
    from ReportTasks.RamTaskMethods import compute_powers
    from ReportTasks.RamTaskMethods import compute_wavelets_powers
except ImportError as ie:
    if 'MorletWaveletFilterCpp' in ie.message:
        print 'Update PTSA for better perfomance'
        compute_powers = None
    else:
        raise ie

class ComputePAL1Powers(RamTask):
    def __init__(self, params, mark_as_completed=True):
        RamTask.__init__(self, mark_as_completed)

        self.params = params
        self.pow_mat = None
        self.samplerate = None
        self.wavelet_transform = MorletWaveletTransform()
        self.wavelet_transform_retrieval = MorletWaveletTransform()
        self.bipolar_pairs= None

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

    def restore(self):
        subject = self.pipeline.subject

        self.pow_mat = joblib.load(self.get_path_to_resource_in_workspace(subject + '-pow_mat.pkl'))

        self.samplerate = joblib.load(self.get_path_to_resource_in_workspace(subject + '-samplerate.pkl'))
        events = self.get_passed_object('PAL1_events')
        if not len(events) == len(self.pow_mat):
            print 'Restored matrix of different length than events. Recomputing powers.'
            self.run()
        else:
            self.pass_object('pow_mat', self.pow_mat)

            self.pass_object('samplerate', self.samplerate)

    def run(self):

        self.pipeline.subject = self.pipeline.subject.split('_')[0]
        subject = self.pipeline.subject

        events = self.get_passed_object('PAL1_events')

        is_encoding_event = (events.type == 'PRACTICE_PAIR') | (events.type == 'STUDY_PAIR')

        sessions = np.unique(events.session)
        print 'sessions:', sessions

        # channels = self.get_passed_object('channels')
        # tal_info = self.get_passed_object('tal_info')
        monopolar_channels = self.get_passed_object('monopolar_channels')
        bipolar_pairs = self.get_passed_object('bipolar_pairs')
        params = self.params

        print 'Computing powers during encoding'
        encoding_pow_mat, encoding_events = compute_powers(events[is_encoding_event], monopolar_channels,bipolar_pairs,
                                                           params.pal1_start_time, params.pal1_end_time, params.pal1_buf,
                                                           params.freqs, params.log_powers,ComputePowers=self,
                                                           )

        print 'Computing powers during retrieval'

        retrieval_pow_mat, retrieval_events = compute_powers(events[~is_encoding_event], monopolar_channels,bipolar_pairs,
                                                             params.pal1_retrieval_start_time,
                                                             params.pal1_retrieval_end_time, params.pal1_retrieval_buf,
                                                             params.freqs, params.log_powers,ComputePowers=self,
                                                             )

        if self.bipolar_pairs is not None:
            # recording was in bipolar mode; re-compute excluded pairs
            reduced_pairs = get_reduced_pairs(self,self.bipolar_pairs)
            config_pairs_dict  = self.get_passed_object('config_pairs_dict')[subject]['pairs']
            excluded_pairs = get_excluded_dict(config_pairs_dict, reduced_pairs)
            joblib.dump(reduced_pairs,self.get_path_to_resource_in_workspace(subject+'-reduced_pairs.pkl'))
            with open(self.get_path_to_resource_in_workspace('excluded_pairs.json'),'w') as excluded_file:
                json.dump({subject:{'pairs':excluded_pairs}},excluded_file)
            self.pass_object('reduced_pairs',reduced_pairs)
            # replace bipolar_pairs_path with config_pairs_path
            self.pass_object('bipolar_pairs_path',self.get_passed_object('config_pairs_path'))
            self.pass_object('bipolar_pairs',self.bipolar_pairs)
            joblib.dump(self.bipolar_pairs,self.get_path_to_resource_in_workspace(subject+'-bipolar_pairs.pkl'))

        events =  np.concatenate([encoding_events,retrieval_events]).view(np.recarray)
        events.sort(order=['session','list','mstime'])
        is_encoding_event  = (events.type=='PRACTICE_PAIR') | (events.type=='STUDY_PAIR')
        self.pass_object('PAL1_events',events)

        self.pow_mat = np.zeros((len(events),encoding_pow_mat.shape[-1]))
        self.pow_mat[is_encoding_event, ...] = encoding_pow_mat
        self.pow_mat[~is_encoding_event, ...] = retrieval_pow_mat

        # self.compute_powers(events, sessions, monopolar_channels, bipolar_pairs)

        self.pass_object('pow_mat', self.pow_mat)
        self.pass_object('samplerate', self.samplerate)

        joblib.dump(self.pow_mat, self.get_path_to_resource_in_workspace(subject + '-pow_mat.pkl'))
        joblib.dump(self.samplerate, self.get_path_to_resource_in_workspace(subject + '-samplerate.pkl'))




