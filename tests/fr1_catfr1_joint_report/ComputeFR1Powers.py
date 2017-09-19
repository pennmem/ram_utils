__author__ = 'm'

from RamPipeline import *

import numpy as np
#from morlet import MorletWaveletTransform
from ptsa.extensions.morlet.morlet import MorletWaveletTransform
from sklearn.externals import joblib

from ptsa.data.readers import EEGReader
from ptsa.data.readers.IndexReader import JsonIndexReader
try:
    from ReportTasks.RamTaskMethods import compute_powers
except ImportError as ie:
    if 'MorletWaveletFilterCpp' in ie.message:
        print 'Update PTSA for better perfomance'
        compute_powers = None
    else:
        raise ie

from ReportUtils import ReportRamTask

import hashlib


class ComputeFR1Powers(ReportRamTask):
    def __init__(self, params, mark_as_completed=True):
        super(ComputeFR1Powers, self).__init__(mark_as_completed)
        self.params = params
        self.pow_mat = None
        self.samplerate = None
        self.wavelet_transform = MorletWaveletTransform()

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

        fr1_event_files = sorted(list(json_reader.aggregate_values('all_events', subject=subj_code, montage=montage, experiment='FR1')))
        for fname in fr1_event_files:
            with open(fname,'rb') as f: hash_md5.update(f.read())

        catfr1_event_files = sorted(list(json_reader.aggregate_values('all_events', subject=subj_code, montage=montage, experiment='catFR1')))
        for fname in catfr1_event_files:
            with open(fname,'rb') as f: hash_md5.update(f.read())

        return hash_md5.digest()

    def restore(self):
        subject = self.pipeline.subject
        task = self.pipeline.task

        try:
            self.pow_mat = joblib.load(self.get_path_to_resource_in_workspace(subject + '-pow_mat.pkl'))
        except IOError:
            self.pow_mat = joblib.load(self.get_path_to_resource_in_workspace('-'.join([subject,task,'pow_mat.pkl'])))
        try:
            self.samplerate = joblib.load(self.get_path_to_resource_in_workspace(subject + '-samplerate.pkl'))
        except IOError:
            self.samplerate = joblib.load(self.get_path_to_resource_in_workspace('-'.join([subject,task,'samplerate.pkl'])))

        try:
            events=joblib.load(self.get_path_to_resource_in_workspace(subject + '-fr1_events.pkl'))
            self.pass_object('events',events)
        except IOError:
            events = self.get_passed_object('events')
            if len(events)>len(self.pow_mat):
                self.run()
            else:
                pass

        self.pass_object('pow_mat', self.pow_mat)
        self.pass_object('samplerate', self.samplerate)

    def run(self):
        subject = self.pipeline.subject
        task = self.pipeline.task

        events = self.get_passed_object('events')

        sessions = np.unique(events.session)
        print 'sessions:', sessions

        # channels = self.get_passed_object('channels')
        # tal_info = self.get_passed_object('tal_info')
        monopolar_channels = self.get_passed_object('monopolar_channels')
        bipolar_pairs = self.get_passed_object('bipolar_pairs')

        params = self.params
        if compute_powers is None:
            self.compute_powers(events,sessions,monopolar_channels,bipolar_pairs)
        else:
            if compute_powers is None:
                self.compute_powers(events, sessions, monopolar_channels, bipolar_pairs)
            else:
                encoding_mask = events.type == 'WORD'

                encoding_pow_mat, encoding_events = compute_powers(events[encoding_mask], monopolar_channels,

                                                                   params.fr1_start_time, params.fr1_end_time,
                                                                   params.fr1_buf,
                                                                   params.freqs, params.log_powers,
                                                                   bipolar_pairs=bipolar_pairs,)
                retrieval_pow_mat, retrieval_events = compute_powers(events[~encoding_mask], monopolar_channels,

                                                                     params.fr1_retrieval_start_time,
                                                                     params.fr1_retrieval_end_time,
                                                                     params.fr1_retrieval_buf,
                                                                     params.freqs, params.log_powers,
                                                                     bipolar_pairs=bipolar_pairs,)

                events = np.concatenate([encoding_events, retrieval_events]).view(np.recarray)
                events.sort(order=['session', 'list', 'mstime'])
                encoding_mask = events.type == 'WORD'
                self.pow_mat = np.zeros((len(events), len(bipolar_pairs) * len(params.freqs)))
                self.pow_mat[encoding_mask] = encoding_pow_mat
                self.pow_mat[~encoding_mask] = retrieval_pow_mat

                joblib.dump(events,self.get_path_to_resource_in_workspace(subject+'-fr1_events.pkl'))
                self.pass_object('events', events)

            self.pass_object('pow_mat', self.pow_mat)
            self.pass_object('samplerate', self.samplerate)

        joblib.dump(self.pow_mat, self.get_path_to_resource_in_workspace(subject + '-pow_mat.pkl'))
        joblib.dump(self.samplerate, self.get_path_to_resource_in_workspace(subject + '-samplerate.pkl'))

    def compute_powers(self, events, sessions,monopolar_channels , bipolar_pairs ):
        n_freqs = len(self.params.freqs)
        n_bps = len(bipolar_pairs)

        self.pow_mat = None

        pow_ev = None
        winsize = bufsize = None
        for sess in sessions:
            sess_events = events[events.session == sess]
            n_events = len(sess_events)

            print 'Loading EEG for', n_events, 'events of session', sess

            # eegs = Events(sess_events).get_data(channels=channels, start_time=self.params.fr1_start_time, end_time=self.params.fr1_end_time,
            #                             buffer_time=self.params.fr1_buf, eoffset='eegoffset', keep_buffer=True, eoffset_in_time=False)

            # from ptsa.data.readers import TimeSeriesEEGReader
            # time_series_reader = TimeSeriesEEGReader(events=sess_events, start_time=self.params.fr1_start_time,
            #                                  end_time=self.params.fr1_end_time, buffer_time=self.params.fr1_buf, keep_buffer=True)
            #
            # eegs = time_series_reader.read(monopolar_channels)

            # VERSION 2/22/2016
            # eeg_reader = EEGReader(events=sess_events, channels=monopolar_channels,
            #                        start_time=self.params.fr1_start_time,
            #                        end_time=self.params.fr1_end_time, buffer_time=self.params.fr1_buf)

            # VERSION WITH MIRRORING
            eeg_reader = EEGReader(events=sess_events, channels=monopolar_channels,
                                   start_time=self.params.fr1_start_time,
                                   end_time=self.params.fr1_end_time, buffer_time=0.0)


            eegs = eeg_reader.read()
            if eeg_reader.removed_bad_data():
                print 'REMOVED SOME BAD EVENTS !!!'
                sess_events = eegs['events'].values.view(np.recarray)
                n_events = len(sess_events)
                events = np.hstack((events[events.session!=sess],sess_events)).view(np.recarray)
                ev_order = np.argsort(events, order=('session','list','mstime'))
                events = events[ev_order]
                self.pass_object('events', events)


            # mirroring
            #eegs[...,:1365] = eegs[...,2730:1365:-1]
            #eegs[...,2731:4096] = eegs[...,2729:1364:-1]

            eegs = eegs.add_mirror_buffer(duration=self.params.fr1_buf)


            if self.samplerate is None:
                self.samplerate = float(eegs.samplerate)
                winsize = int(round(self.samplerate*(self.params.fr1_end_time-self.params.fr1_start_time+2*self.params.fr1_buf)))
                bufsize = int(round(self.samplerate*self.params.fr1_buf))
                print 'samplerate =', self.samplerate, 'winsize =', winsize, 'bufsize =', bufsize
                pow_ev = np.empty(shape=n_freqs*winsize, dtype=float)
                self.wavelet_transform.init(self.params.width, self.params.freqs[0], self.params.freqs[-1], n_freqs, self.samplerate, winsize)

            print 'Computing FR1 powers'

            sess_pow_mat = np.empty(shape=(n_events, n_bps, n_freqs), dtype=np.float)

            #monopolar_channels_np = np.array(monopolar_channels)
            for i,bp in enumerate(bipolar_pairs):

                print 'Computing powers for bipolar pair', bp
                elec1 = np.where(monopolar_channels == bp[0])[0][0]
                elec2 = np.where(monopolar_channels == bp[1])[0][0]

                bp_data = eegs[elec1] - eegs[elec2]
                bp_data.attrs['samplerate'] = self.samplerate


                bp_data = bp_data.filtered([58,62], filt_type='stop', order=self.params.filt_order)
                for ev in xrange(n_events):
                    self.wavelet_transform.multiphasevec(bp_data[ev][0:winsize], pow_ev)

                    pow_ev_stripped = np.reshape(pow_ev, (n_freqs,winsize))[:,bufsize:winsize-bufsize]
                    pow_zeros = np.where(pow_ev_stripped==0.0)[0]
                    if len(pow_zeros)>0:
                        print bp, ev
                        print sess_events[ev].eegfile, sess_events[ev].eegoffset
                        if len(pow_zeros)>0:
                            print bp, ev
                            print sess_events[ev].eegfile, sess_events[ev].eegoffset
                            self.raise_and_log_report_exception(
                                                    exception_type='NumericalError',
                                                    exception_message='Corrupt EEG File'
                                                    )
                    if self.params.log_powers:
                        np.log10(pow_ev_stripped, out=pow_ev_stripped)
                    sess_pow_mat[ev,i,:] = np.nanmean(pow_ev_stripped, axis=1)

            self.pow_mat = np.concatenate((self.pow_mat,sess_pow_mat), axis=0) if self.pow_mat is not None else sess_pow_mat

        self.pow_mat = np.reshape(self.pow_mat, (len(events), n_bps*n_freqs))
