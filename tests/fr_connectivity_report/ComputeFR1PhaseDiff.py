__author__ = 'm'

from RamPipeline import *

import numpy as np
from morlet import MorletWaveletTransform
from circular_stat import circ_diff_time_bins
from sklearn.externals import joblib

from ptsa.data.readers import EEGReader
from ptsa.data.readers.IndexReader import JsonIndexReader
from ReportUtils import ReportRamTask

import hashlib


class ComputeFR1PhaseDiff(ReportRamTask):
    def __init__(self, params, mark_as_completed=True):
        super(ComputeFR1PhaseDiff,self).__init__(mark_as_completed)
        self.params = params
        self.wavelets = None
        self.phase_diff_mat = None
        self.samplerate = None
        self.wavelet_transform = MorletWaveletTransform()

    def input_hashsum(self):
        subject = self.pipeline.subject
        tmp = subject.split('_')
        subj_code = tmp[0]
        montage = 0 if len(tmp)==1 else int(tmp[1])

        json_reader = JsonIndexReader(os.path.join(self.pipeline.mount_point, 'data/eeg/protocols/r1.json'))

        hash_md5 = hashlib.md5()

        bp_paths = json_reader.aggregate_values('pairs', subject=subj_code, montage=montage)
        for fname in bp_paths:
            hash_md5.update(open(fname,'rb').read())

        fr1_event_files = sorted(list(json_reader.aggregate_values('all_events', subject=subj_code, montage=montage, experiment='FR1')))
        for fname in fr1_event_files:
            hash_md5.update(open(fname,'rb').read())

        catfr1_event_files = sorted(list(json_reader.aggregate_values('all_events', subject=subj_code, montage=montage, experiment='catFR1')))
        for fname in catfr1_event_files:
            hash_md5.update(open(fname,'rb').read())

        return hash_md5.digest()

    def restore(self):
        subject = self.pipeline.subject

        self.phase_diff_mat = joblib.load(self.get_path_to_resource_in_workspace(subject + '-phase_diff_mat.pkl'))
        self.samplerate = joblib.load(self.get_path_to_resource_in_workspace(subject + '-samplerate.pkl'))

        self.pass_object('phase_diff_mat', self.phase_diff_mat)
        self.pass_object('samplerate', self.samplerate)

    def run(self):
        subject = self.pipeline.subject

        events = self.get_passed_object('events')

        sessions = np.unique(events.session)
        print 'sessions:', sessions

        monopolar_channels = self.get_passed_object('monopolar_channels')
        bipolar_pairs = self.get_passed_object('bipolar_pairs')
        bipolar_pair_pairs = self.get_passed_object('bipolar_pair_pairs')

        self.compute_wavelets(events, sessions, monopolar_channels, bipolar_pairs)
        self.compute_phase_differences(bipolar_pair_pairs)

        del self.wavelets
        self.wavelets = None

        self.pass_object('phase_diff_mat', self.phase_diff_mat)
        self.pass_object('samplerate', self.samplerate)

        joblib.dump(self.phase_diff_mat, self.get_path_to_resource_in_workspace(subject + '-phase_diff_mat.pkl'))
        joblib.dump(self.samplerate, self.get_path_to_resource_in_workspace(subject + '-samplerate.pkl'))

    def compute_wavelets(self, events, sessions, monopolar_channels, bipolar_pairs):
        n_events = len(events)
        n_freqs = len(self.params.freqs)
        n_bps = len(bipolar_pairs)

        self.wavelets = None
        cur_ev = 0

        wav_ev = None
        winsize = bufsize = tsize = None
        for sess in sessions:
            sess_events = events[events.session == sess]
            n_sess_events = len(sess_events)

            print 'Loading EEG for', n_sess_events, 'events of session', sess

            eeg_reader = EEGReader(events=sess_events, channels=monopolar_channels,
                                   start_time=self.params.fr1_start_time,
                                   end_time=self.params.fr1_end_time, buffer_time=self.params.fr1_buf)

            eegs = eeg_reader.read()
            #eegs = eegs.resampled(resampled_rate=256.0)
            #print 'New samplerate =', eegs.samplerate
            if eeg_reader.removed_bad_data():
                # NB: this is not supported yet in this pipeline
                print 'REMOVED SOME BAD EVENTS !!!'
                import sys
                sys.exit(0)
                # sess_events = eegs['events'].values.view(np.recarray)
                # n_sess_events = len(sess_events)
                # events = np.hstack((events[events.session!=sess],sess_events)).view(np.recarray)
                # ev_order = np.argsort(events, order=('session','list','mstime'))
                # events = events[ev_order]
                # self.pass_object(self.pipeline.task+'_events', events)


            #eegs = eegs.add_mirror_buffer(duration=self.params.fr1_buf)

            if self.samplerate is None:
                self.samplerate = float(eegs.samplerate)
                #self.samplerate = 256.0
                winsize = int(round(self.samplerate*(self.params.fr1_end_time-self.params.fr1_start_time+2*self.params.fr1_buf)))
                bufsize = int(round(self.samplerate*self.params.fr1_buf))
                tsize = winsize - 2*bufsize
                print 'samplerate =', self.samplerate, 'winsize =', winsize, 'bufsize =', bufsize
                self.wavelet_transform.init(self.params.width, self.params.freqs[0], self.params.freqs[-1], n_freqs, self.samplerate, winsize)
                wav_ev = np.empty(shape=n_freqs*winsize, dtype=np.complex)
                self.wavelets = np.empty(shape=(n_events, n_bps, n_freqs, tsize), dtype=np.complex)

            print 'Computing FR1 wavelets'

            eegs = eegs.filtered([58,62], filt_type='stop', order=self.params.filt_order)

            for i,bp in enumerate(bipolar_pairs):
                print 'Computing wavelets for bipolar pair', bp
                elec1 = np.where(monopolar_channels == bp[0])[0][0]
                elec2 = np.where(monopolar_channels == bp[1])[0][0]

                #print 'Shape =', eegs[elec1].values.shape
                # bp_data = TimeSeriesX(resample(eegs[elec1].values,winsize,axis=1) - resample(eegs[elec2].values,winsize,axis=1), dims=['events','time'])
                # bp_data.attrs['samplerate'] = self.samplerate

                bp_data = eegs[elec1] - eegs[elec2]
                #bp_data.attrs['samplerate'] = self.samplerate
                #bp_data = bp_data.filtered([58,62], filt_type='stop', order=self.params.filt_order)

                for ev in xrange(n_sess_events):
                    self.wavelet_transform.multiphasevec_complex(bp_data[ev][0:winsize], wav_ev)
                    self.wavelets[cur_ev+ev,i,...] = np.reshape(wav_ev, (n_freqs,winsize))[:,bufsize:winsize-bufsize]

            cur_ev += n_sess_events

    def compute_phase_differences(self, bipolar_pair_pairs):
        n_bp_pairs = len(bipolar_pair_pairs)
        n_events,n_bps,n_freqs,tsize = self.wavelets.shape
        n_bins = self.params.fr1_n_bins

        self.phase_diff_mat = np.empty(shape=(n_bp_pairs, n_freqs, n_bins, n_events), dtype=np.complex)
        phase_diff = np.empty(tsize, dtype=np.complex)
        phase_diff_mat_tmp = np.empty(n_bins, dtype=np.complex)

        for j,bpp in enumerate(bipolar_pair_pairs):
            print "Computing phase differences for bp pair", bpp
            for i in xrange(n_events):
                bp1,bp2 = bpp
                for f in xrange(n_freqs):
                    circ_diff_time_bins(self.wavelets[i,bp1,f,:], self.wavelets[i,bp2,f,:], phase_diff, phase_diff_mat_tmp)
                    self.phase_diff_mat[j,f,:,i] = phase_diff_mat_tmp
