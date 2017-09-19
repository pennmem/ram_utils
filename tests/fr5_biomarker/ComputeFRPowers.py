__author__ = 'm'

from RamPipeline import *

import numpy as np
from .MontagePreparation import get_reduced_pairs,get_excluded_dict
from ptsa.extensions.morlet.morlet import MorletWaveletTransform
from sklearn.externals import joblib
import json

from ptsa.data.readers import EEGReader
from ptsa.data.readers.IndexReader import JsonIndexReader

import hashlib
from ReportTasks.RamTaskMethods import compute_powers


class ComputeFRPowers(RamTask):
    def __init__(self, params, mark_as_completed=True):
        RamTask.__init__(self, mark_as_completed)
        self.params = params
        self.pow_mat = None
        self.samplerate = None
        self.wavelet_transform = MorletWaveletTransform()
        self.wavelet_transform_retrieval = MorletWaveletTransform()
        self.bipolar_pairs = None

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

        self.pow_mat = joblib.load(self.get_path_to_resource_in_workspace(subject + '-pow_mat.pkl'))
        self.samplerate = joblib.load(self.get_path_to_resource_in_workspace(subject + '-samplerate.pkl'))
        events =self.get_passed_object('FR_events')
        if not len(events)==len(self.pow_mat):
            print 'Restored matrix of different length than events. Recomputing powers.'
            self.run()
        else:
            self.pass_object('pow_mat', self.pow_mat)
            self.pass_object('samplerate', self.samplerate)

    def run(self):
        self.pipeline.subject = self.pipeline.subject.split('_')[0]
        subject = self.pipeline.subject


        events = self.get_passed_object('FR_events')
        is_encoding_event = events.type=='WORD'

        sessions = np.unique(events.session)
        print 'sessions:', sessions

        # channels = self.get_passed_object('channels')
        # tal_info = self.get_passed_object('tal_info')
        monopolar_channels = self.get_passed_object('monopolar_channels')
        bipolar_pairs = self.get_passed_object('bipolar_pairs')
        params=self.params

        print 'Computing powers during encoding'
        encoding_pow_mat, encoding_events = compute_powers(events[is_encoding_event], monopolar_channels,
                                              params.fr1_start_time, params.fr1_end_time, params.fr1_buf,
                                              params.freqs, params.log_powers,
                                                           bipolar_pairs=bipolar_pairs,ComputePowers=self)

        print 'Computing powers during retrieval'
        retrieval_pow_mat, retrieval_events = compute_powers(events[~is_encoding_event], monopolar_channels,
                                              params.fr1_retrieval_start_time, params.fr1_retrieval_end_time, params.fr1_retrieval_buf,
                                              params.freqs, params.log_powers, bipolar_pairs=bipolar_pairs,
                                                             ComputePowers=self)
        if self.bipolar_pairs is not None:
            # recording was in bipolar mode; re-compute excluded pairs
            reduced_pairs = get_reduced_pairs(self,self.bipolar_pairs)
            reduced_pair_dict = get_excluded_dict(self.get_passed_object('bipolar_dict'), reduced_pairs)
            self.pass_object('reduced_pairs',reduced_pairs)
            self.pass_object('bipolar_pairs',self.bipolar_pairs)
            with open(self.get_path_to_resource_in_workspace('excluded_pairs.json'),'w') as excluded_file:
                json.dump({subject:{'pairs':reduced_pair_dict}},excluded_file)

        events = np.concatenate([encoding_events,retrieval_events]).view(np.recarray)
        events.sort(order=['session','list','mstime'])
        self.pass_object('FR_events',events)
        joblib.dump(events,self.get_path_to_resource_in_workspace('%s-FR_events.pkl'%subject))
        is_encoding_event = events.type=='WORD'

        self.pow_mat = np.zeros((len(events),encoding_pow_mat.shape[-1]))



        self.pow_mat[is_encoding_event,...] = encoding_pow_mat
        self.pow_mat[~is_encoding_event,...] = retrieval_pow_mat
        # self.compute_powers(events,events.session,monopolar_channels,bipolar_pairs)
        # if subject == 'R1302M':
        #     # Exclude some known bad events for this subject
        #     try:
        #         short_events = np.concatenate(events[:499],events[621:]).view(np.recarray)
        #         self.pow_mat = self.pow_mat[np.in1d(events,short_events)]
        #         self.pass_object('FR_events',events)
        #     except Exception:
        #         print('Using all events as specified')
        #         pass

        # self.compute_powers(events, sessions, monopolar_channels, bipolar_pairs)

        self.pass_object('pow_mat', self.pow_mat)
        self.pass_object('samplerate', self.samplerate)



        joblib.dump(self.pow_mat, self.get_path_to_resource_in_workspace(subject + '-pow_mat.pkl'))
        joblib.dump(self.samplerate, self.get_path_to_resource_in_workspace(subject + '-samplerate.pkl'))

    def compute_powers(self, events, sessions, monopolar_channels , bipolar_pairs):

        retrieval_events_mask = (events.type == 'REC_WORD') | (events.type == 'REC_BASE')
        encoding_events_mask = (events.type == 'WORD')


        n_freqs = len(self.params.freqs)
        n_bps = len(bipolar_pairs)

        self.pow_mat = None

        pow_ev = None
        winsize = bufsize = None
        for sess in np.unique(sessions):
            sess_events = events[events.session == sess]
            n_events = len(sess_events)

            sess_encoding_events_mask = (sess_events.type == 'WORD')


            print 'Loading EEG for', n_events, 'events of session', sess

            # eegs = Events(sess_events).get_data(channels=channels, start_time=self.params.fr1_start_time, end_time=self.params.fr1_end_time,
            #                             buffer_time=self.params.fr1_buf, eoffset='eegoffset', keep_buffer=True, eoffset_in_time=False)

            # from ptsa.data.readers import TimeSeriesEEGReader
            # time_series_reader = TimeSeriesEEGReader(events=sess_events, start_time=self.params.fr1_start_time,
            #                                  end_time=self.params.fr1_end_time, buffer_time=self.params.fr1_buf, keep_buffer=True)
            #
            # eegs = time_series_reader.read(monopolar_channels)

            eeg_reader = EEGReader(events=sess_events, channels=monopolar_channels,
                                   start_time=self.params.fr1_start_time,
                                   end_time=self.params.fr1_end_time, buffer_time=0.0)

            eegs = eeg_reader.read().add_mirror_buffer(duration=self.params.fr1_buf)


            eeg_retrieval_reader = EEGReader(events=sess_events, channels=monopolar_channels,
                                   start_time=self.params.fr1_retrieval_start_time,
                                   end_time=self.params.fr1_retrieval_end_time, buffer_time=0.0)

            eegs_retrieval = eeg_retrieval_reader.read().add_mirror_buffer(duration=self.params.fr1_retrieval_buf)




            # mirroring
            # eegs[...,:1365] = eegs[...,2730:1365:-1]
            # eegs[...,2731:4096] = eegs[...,2729:1364:-1]



            if self.samplerate is None:
                self.samplerate = float(eegs.samplerate)

                # encoding
                winsize = int(round(self.samplerate*(self.params.fr1_end_time-self.params.fr1_start_time+2*self.params.fr1_buf)))
                bufsize = int(round(self.samplerate*self.params.fr1_buf))
                print 'samplerate =', self.samplerate, 'winsize =', winsize, 'bufsize =', bufsize
                pow_ev = np.empty(shape=n_freqs*winsize, dtype=float)
                self.wavelet_transform.init(self.params.width, self.params.freqs[0], self.params.freqs[-1], n_freqs, self.samplerate, winsize)

                # retrieval
                winsize_retrieval = int(round(self.samplerate*(self.params.fr1_retrieval_end_time-self.params.fr1_retrieval_start_time+2*self.params.fr1_retrieval_buf)))
                bufsize_retrieval = int(round(self.samplerate*self.params.fr1_retrieval_buf))
                print 'samplerate =', self.samplerate, 'winsize_retrieval =', winsize_retrieval, 'bufsize_retrieval =', bufsize_retrieval
                pow_ev_retrieval = np.empty(shape=n_freqs*winsize_retrieval, dtype=float)
                self.wavelet_transform_retrieval.init(self.params.width, self.params.freqs[0], self.params.freqs[-1], n_freqs, self.samplerate, winsize_retrieval)




            print 'Computing FR1/catFR1 powers'

            sess_pow_mat = np.empty(shape=(n_events, n_bps, n_freqs), dtype=np.float)

            #monopolar_channels_np = np.array(monopolar_channels)
            for i,bp in enumerate(bipolar_pairs):

                print 'Computing powers for bipolar pair', bp
                elec1 = np.where(monopolar_channels == bp[0])[0][0]
                elec2 = np.where(monopolar_channels == bp[1])[0][0]

                bp_data = np.subtract(eegs[elec1],eegs[elec2])
                bp_data.attrs['samplerate'] = self.samplerate


                bp_data_retrieval = np.subtract(eegs_retrieval[elec1],eegs_retrieval[elec2])
                bp_data_retrieval.attrs['samplerate'] = self.samplerate



                # bp_data = eegs[elec1] - eegs[elec2]
                # bp_data = eegs[elec1] - eegs[elec2]
                # bp_data = eegs.values[elec1] - eegs.values[elec2]

                bp_data = bp_data.filtered([58,62], filt_type='stop', order=self.params.filt_order)
                bp_data_retrieval = bp_data_retrieval.filtered([58,62], filt_type='stop', order=self.params.filt_order)

                n_enc=0
                n_retr=0
                for ev in xrange(n_events):
                    # if encoding_events_mask[ev]:

                    if sess_encoding_events_mask[ev]:
                        self.wavelet_transform.multiphasevec(bp_data[n_enc][0:winsize], pow_ev)
                        pow_ev_stripped = np.reshape(pow_ev, (n_freqs,winsize))[:,bufsize:winsize-bufsize]
                        n_enc +=1
                    else:
                        self.wavelet_transform_retrieval.multiphasevec(bp_data_retrieval[n_retr][0:winsize_retrieval], pow_ev_retrieval)
                        pow_ev_stripped = np.reshape(pow_ev_retrieval, (n_freqs,winsize_retrieval))[:,bufsize_retrieval:winsize_retrieval-bufsize_retrieval]
                        n_retr+=1


                    if self.params.log_powers:
                        np.log10(pow_ev_stripped, out=pow_ev_stripped)
                    sess_pow_mat[ev,i,:] = np.nanmean(pow_ev_stripped, axis=1)

            self.pow_mat = np.concatenate((self.pow_mat,sess_pow_mat), axis=0) if self.pow_mat is not None else sess_pow_mat

        self.pow_mat = np.reshape(self.pow_mat, (len(events), n_bps*n_freqs))
