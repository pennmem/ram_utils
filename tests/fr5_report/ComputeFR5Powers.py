import os
import numpy as np
from scipy.stats.mstats import zscore
from ptsa.extensions.morlet.morlet import MorletWaveletTransform
from sklearn.externals import joblib

from ReportTasks.RamTaskMethods import compute_powers

from ptsa.data.readers import EEGReader
from ptsa.data.readers.IndexReader import JsonIndexReader
from ReportUtils import ReportRamTask

import hashlib


class ComputeFR5Powers(ReportRamTask):
    def __init__(self, params, mark_as_completed=True):
        super(ComputeFR5Powers, self).__init__(mark_as_completed)
        self.params = params
        self.pow_mat = None
        self.samplerate = None
        self.wavelet_transform = MorletWaveletTransform()

    def input_hashsum(self):
        subject = self.pipeline.subject
        task = self.pipeline.task
        tmp = subject.split('_')
        subj_code = tmp[0]
        montage = 0 if len(tmp)==1 else int(tmp[1])

        json_reader = JsonIndexReader(os.path.join(self.pipeline.mount_point, 'protocols/r1.json'))

        hash_md5 = hashlib.md5()
        hash_md5.update(__file__)

        bp_paths = json_reader.aggregate_values('pairs', subject=subj_code, montage=montage)
        for fname in bp_paths:
            with open(fname,'rb') as f: hash_md5.update(f.read())

        event_files = sorted(list(json_reader.aggregate_values('all_events', subject=subj_code, montage=montage, experiment=task)))
        for fname in event_files:
            with open(fname,'rb') as f: hash_md5.update(f.read())

        return hash_md5.digest()

    def restore(self):
        subject = self.pipeline.subject
        task = self.pipeline.task


        events = joblib.load(self.get_path_to_resource_in_workspace(subject+'-events.pkl'))
        self.pass_object(task+'_events',events)

        self.pow_mat = joblib.load(self.get_path_to_resource_in_workspace(subject + '-' + task + '-fr_stim_pow_mat.pkl'))
        self.samplerate = joblib.load(self.get_path_to_resource_in_workspace(subject + '-samplerate.pkl'))

        self.pass_object('fr_stim_pow_mat', self.pow_mat)
        self.pass_object('samplerate', self.samplerate)

        post_stim_pow_mat = joblib.load(self.get_path_to_resource_in_workspace(subject+'-post_stim_powers.pkl'))
        stim_events = joblib.load(self.get_path_to_resource_in_workspace(subject+'-stim_off_events.pkl'))
        self.pass_object('post_stim_pow_mat',post_stim_pow_mat)
        self.pass_object('stim_off_events',stim_events)


    def run(self):
        subject = self.pipeline.subject
        task = self.pipeline.task

        events = self.get_passed_object(task+'_events')
        all_events = self.get_passed_object('all_events')
        stim_off_events = all_events[all_events.type=='STIM_OFF']



        sessions = np.unique(events.session)
        print 'sessions:', sessions

        monopolar_channels = self.get_passed_object('monopolar_channels')
        bipolar_pairs = self.get_passed_object('bipolar_pairs')

        self.pow_mat,events = compute_powers(events, monopolar_channels, bipolar_pairs,
                                             self.params.fr1_start_time,self.params.fr1_end_time,self.params.fr1_buf,
                                             self.params.freqs,self.params.log_powers)
        self.pass_object(task+'_events',events)
        print 'self.pow_mat.shape:',self.pow_mat.shape

        post_stim_powers, stim_off_events = compute_powers(stim_off_events,monopolar_channels,bipolar_pairs,
                                                            self.params.post_stim_start_time,self.params.post_stim_end_time,
                                                            self.params.post_stim_buf,self.params.freqs,self.params.log_powers)

        for session in sessions:
            sess_stims = stim_off_events.session==session
            post_stim_powers[sess_stims] = zscore(post_stim_powers[sess_stims])

        self.pass_object('fr_stim_pow_mat', self.pow_mat)
        self.pass_object('samplerate', self.samplerate)
        self.pass_object('post_stim_pow_mat',post_stim_powers)
        self.pass_object('stim_off_events',stim_off_events)

        events = self.get_passed_object(task+'_events')
        joblib.dump(events,self.get_path_to_resource_in_workspace(subject+'-events.pkl'))
        joblib.dump(self.pow_mat, self.get_path_to_resource_in_workspace(subject + '-' + task + '-fr_stim_pow_mat.pkl'))
        joblib.dump(self.samplerate, self.get_path_to_resource_in_workspace(subject + '-samplerate.pkl'))

        joblib.dump(stim_off_events,self.get_path_to_resource_in_workspace(subject+'-stim_off_events.pkl'))
        joblib.dump(post_stim_powers,self.get_path_to_resource_in_workspace(subject+'-post_stim_powers.pkl'))

    def compute_powers(self, events, sessions,monopolar_channels, bipolar_pairs):
        n_freqs = len(self.params.freqs)
        n_bps = len(bipolar_pairs)

        self.pow_mat = None

        pow_ev = None
        winsize = bufsize = None
        for sess in sessions:
            sess_events = events[events.session == sess]
            n_events = len(sess_events)

            print 'Loading EEG for', n_events, 'events of session', sess

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
                self.pass_object(self.pipeline.task+'_events', events)
            print '%d events remaining'%len(sess_events)
            print 'eeg.shape:',eegs.shape
            assert len(sess_events) == eegs.shape[1]


            eegs = eegs.add_mirror_buffer(duration=self.params.fr1_buf)

            if self.samplerate is None:
                self.samplerate = float(eegs.samplerate)
                winsize = int(round(self.samplerate*(self.params.fr1_end_time-self.params.fr1_start_time+2*self.params.fr1_buf)))
                bufsize = int(round(self.samplerate*self.params.fr1_buf))
                print 'samplerate =', self.samplerate, 'winsize =', winsize, 'bufsize =', bufsize
                pow_ev = np.empty(shape=n_freqs*winsize, dtype=float)
                self.wavelet_transform.init(self.params.width, self.params.freqs[0], self.params.freqs[-1], n_freqs, self.samplerate, winsize)

            print 'Computing FR5 powers'

            sess_pow_mat = np.empty(shape=(n_events, n_bps, n_freqs), dtype=np.float)

            for i,bp in enumerate(bipolar_pairs):
                print 'Computing powers for bipolar pair', bp
                elec1 = np.where(monopolar_channels == bp[0])[0][0]
                elec2 = np.where(monopolar_channels == bp[1])[0][0]

                bp_data = np.subtract(eegs[elec1],eegs[elec2])
                bp_data.attrs['samplerate'] = self.samplerate

                bp_data = bp_data.filtered([58,62], filt_type='stop', order=self.params.filt_order)
                for ev in xrange(n_events):
                    self.wavelet_transform.multiphasevec(bp_data[ev][0:winsize], pow_ev)
                    pow_ev_stripped = np.reshape(pow_ev, (n_freqs,winsize))[:,bufsize:winsize-bufsize]+np.finfo(np.float).eps/2
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

            sess_pow_mat = zscore(sess_pow_mat, axis=0, ddof=1)

            self.pow_mat = np.concatenate((self.pow_mat,sess_pow_mat), axis=0) if self.pow_mat is not None else sess_pow_mat
            print 'self.pow_mat.shape:',self.pow_mat.shape
            assert len(sess_pow_mat)==len(sess_events)

        self.pow_mat = np.reshape(self.pow_mat, (len(events), n_bps*n_freqs))
