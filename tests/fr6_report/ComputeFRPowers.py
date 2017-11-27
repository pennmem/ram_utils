import hashlib
import os

import numpy as np
from ptsa.data.readers.IndexReader import JsonIndexReader
from ptsa.extensions.morlet.morlet import MorletWaveletTransform
from sklearn.externals import joblib

from ramutils.pipeline import RamTask
from ramutils.powers import compute_powers


class ComputeFRPowers(RamTask):
    def __init__(self, params, mark_as_completed=True):
        RamTask.__init__(self, mark_as_completed)
        self.params = params
        self.pow_mat = None
        self.samplerate = None
        self.wavelet_transform = MorletWaveletTransform()
        self.wavelet_transform_retrieval = MorletWaveletTransform()

    def input_hashsum(self):
        subject = self.pipeline.subject
        tmp = subject.split('_')
        subj_code = tmp[0]
        montage = 0 if len(tmp)==1 else int(tmp[1])
        json_reader = JsonIndexReader(os.path.join(self.pipeline.mount_point, 'protocols/r1.json'))
        hash_md5 = hashlib.md5()
        bp_paths = json_reader.aggregate_values('pairs', subject=subj_code, montage=montage)
        for fname in bp_paths:
            with open(fname,'rb') as f: hash_md5.update(f.read())
        return hash_md5.digest()

    def restore(self):
        subject = self.pipeline.subject
        try:
            self.pow_mat = joblib.load(self.get_path_to_resource_in_workspace(subject + '-pow_mat.pkl'))
            self.samplerate = joblib.load(self.get_path_to_resource_in_workspace(subject + '-samplerate.pkl'))
            events = joblib.load(self.get_path_to_resource_in_workspace(subject + '-fr1_events.pkl'))
            new_events = self.get_passed_object('FR1_events')
            assert (np.unique(events.session) == np.unique(new_events.session)).all()
        except (IOError,AssertionError):
            self.run()
        else:
            self.pass_object('FR1_events',events)
            self.pass_object('pow_mat', self.pow_mat)
            self.pass_object('samplerate', self.samplerate)

    def run(self):
        self.pipeline.subject = self.pipeline.subject.split('_')[0]
        subject = self.pipeline.subject
        events = self.get_passed_object('FR1_events')
        
        is_encoding_event = events.type=='WORD'
        sessions = np.unique(events.session)
        print('sessions:', sessions)

        params=self.params

        print('Computing powers during encoding')
        encoding_pow_mat, encoding_events = compute_powers(events[is_encoding_event], 
                                                           params.fr1_start_time,
                                                           params.fr1_end_time,
                                                           params.fr1_buf,
                                                           params.freqs,
                                                           params.log_powers)

        print('Computing powers during retrieval')
        retrieval_pow_mat, retrieval_events = compute_powers(events[~is_encoding_event],
                                                             params.fr1_retrieval_start_time,
                                                             params.fr1_retrieval_end_time,
                                                             params.fr1_retrieval_buf,
                                                             params.freqs,
                                                             params.log_powers)

        events = np.concatenate([encoding_events,retrieval_events]).view(np.recarray)
        events.sort(order=['session','list','mstime'])

        is_encoding_event = events.type=='WORD'
        self.pass_object('FR1_events',events)

        self.pow_mat = np.zeros((len(events),encoding_pow_mat.shape[-1]))
        self.pow_mat[is_encoding_event,...] = encoding_pow_mat
        self.pow_mat[~is_encoding_event,...] = retrieval_pow_mat

        self.pass_object('pow_mat', self.pow_mat)
        self.pass_object('samplerate', self.samplerate)

        joblib.dump(events,self.get_path_to_resource_in_workspace(subject+'-fr1_events.pkl'))
        joblib.dump(self.pow_mat, self.get_path_to_resource_in_workspace(subject + '-pow_mat.pkl'))
        joblib.dump(self.samplerate, self.get_path_to_resource_in_workspace(subject + '-samplerate.pkl'))

        return
