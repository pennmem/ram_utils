import os
import numpy as np
from ptsa.extensions.morlet.morlet import MorletWaveletTransform
from sklearn.externals import joblib

from ptsa.data.readers import EEGReader
from ptsa.data.readers  import JsonIndexReader

import hashlib
import warnings

try:
    from ReportTasks.RamTaskMethods import compute_powers
    from ReportTasks.RamTaskMethods import compute_wavelets_powers
except ImportError as ie:
    if 'MorletWaveletFilterCpp' in ie.message:
        print 'Update PTSA for better perfomance'
        compute_powers = None
    else:
        raise ie

from ramutils.pipeline import RamTask


class ComputePowersWithRecall(RamTask):
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

        fr11_event_files = sorted(
            list(json_reader.aggregate_values('task_events', subject=subj_code, montage=montage, experiment='FR1')))
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
        params = self.params

        freq_min = int(round(params.freqs[0]))
        freq_max = int(round(params.freqs[-1]))

        self.pow_mat = joblib.load(self.get_path_to_resource_in_workspace(
            subject + '-combined_pow_mat_%d_%d.pkl' % (freq_min, freq_max)))

        self.pass_object('pow_mat_with_recall', self.pow_mat)

    def run(self):

        self.pipeline.subject = self.pipeline.subject.split('_')[0]
        subject = self.pipeline.subject

        # events = self.get_passed_object('PAL1_events')

        evs = self.get_passed_object('combined_evs')

        fr1_encoding_mask = (evs.type == 'WORD') & (evs.exp_name == 'FR1')
        fr1_retrieval_mask = (evs.type == 'REC_EVENT') & (evs.exp_name == 'FR1')

        pal1_encoding_mask = (evs.type == 'WORD') & (evs.exp_name == 'PAL1')
        pal1_retrieval_mask = (evs.type == 'REC_EVENT') & (evs.exp_name == 'PAL1')

        sessions = np.unique(evs.session)
        print 'sessions:', sessions

        # channels = self.get_passed_object('channels')
        # tal_info = self.get_passed_object('tal_info')
        monopolar_channels = self.get_passed_object('monopolar_channels')
        bipolar_pairs = self.get_passed_object('bipolar_pairs')
        params = self.params

        fr_session_present = np.sum(fr1_encoding_mask.astype(np.int)) != 0

        if fr_session_present:
            print 'Computing powers during FR encoding'
            encoding_fr1_pow_mat, encoding_fr1_events = compute_powers(evs[fr1_encoding_mask], monopolar_channels,bipolar_pairs,
                                                                       params.fr1_start_time, params.fr1_end_time,
                                                                       params.fr1_buf,
                                                                       params.freqs, params.log_powers,

                                                                       )

            print 'Computing powers during FR retrieval'
            retrieval_fr1_pow_mat, retrieval_fr1_events = compute_powers(evs[fr1_retrieval_mask], monopolar_channels,bipolar_pairs,
                                                                         params.fr1_retrieval_start_time,
                                                                         params.fr1_retrieval_end_time,
                                                                         params.fr1_retrieval_buf,
                                                                         params.freqs, params.log_powers,

                                                                         )


        print 'Computing powers during PAL encoding'
        encoding_pal1_pow_mat, encoding_pal1_events = compute_powers(evs[pal1_encoding_mask], monopolar_channels,bipolar_pairs,
                                                                     params.pal1_start_time, params.pal1_end_time,
                                                                     params.pal1_buf,
                                                                     params.freqs, params.log_powers,

                                                                     )

        print 'Computing powers during PAL retrieval'

        retrieval_pal1_pow_mat, retrieval_pal1_events = compute_powers(evs[pal1_retrieval_mask], monopolar_channels,bipolar_pairs,
                                                                       params.pal1_retrieval_start_time,
                                                                       params.pal1_retrieval_end_time,
                                                                       params.pal1_retrieval_buf,
                                                                       params.freqs, params.log_powers,
                                                                       )
        self.pow_mat = np.zeros((len(evs), encoding_pal1_pow_mat.shape[-1]))


        if fr_session_present:
            self.pow_mat[fr1_encoding_mask, ...] = encoding_fr1_pow_mat
            self.pow_mat[fr1_retrieval_mask, ...] = retrieval_fr1_pow_mat

        self.pow_mat[pal1_encoding_mask, ...] = encoding_pal1_pow_mat
        self.pow_mat[pal1_retrieval_mask, ...] = retrieval_pal1_pow_mat

        self.pass_object('pow_mat_with_recall', self.pow_mat)

        freq_min = int(round(params.freqs[0]))
        freq_max = int(round(params.freqs[-1]))

        joblib.dump(self.pow_mat, self.get_path_to_resource_in_workspace(
            subject + '-combined_pow_mat_%d_%d.pkl' % (freq_min, freq_max)))

