__author__ = 'm'

from RamPipeline import *

import numpy as np
from morlet import MorletWaveletTransform
from sklearn.externals import joblib

from ptsa.data.events import Events
from ptsa.data.readers import EEGReader

class ComputeFRPowers(RamTask):
    def __init__(self, params, mark_as_completed=True):
        RamTask.__init__(self, mark_as_completed)
        self.params = params
        self.pow_mat = None
        self.samplerate = None
        self.wavelet_transform = MorletWaveletTransform()

    def restore(self):
        subject = self.pipeline.subject

        self.pow_mat = joblib.load(self.get_path_to_resource_in_workspace(subject + '-pow_mat.pkl'))
        self.samplerate = joblib.load(self.get_path_to_resource_in_workspace(subject + '-samplerate.pkl'))

        self.pass_object('pow_mat', self.pow_mat)
        self.pass_object('samplerate', self.samplerate)

    def run(self):
        subject = self.pipeline.subject

        events = self.get_passed_object('FR_events')

        sessions = np.unique(events.session)
        print 'sessions:', sessions

        # channels = self.get_passed_object('channels')
        # tal_info = self.get_passed_object('tal_info')
        monopolar_channels = self.get_passed_object('monopolar_channels')
        bipolar_pairs = self.get_passed_object('bipolar_pairs')

        self.compute_powers(events, sessions, monopolar_channels, bipolar_pairs)

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

            eeg_reader = EEGReader(events=sess_events, channels=monopolar_channels,
                                   start_time=self.params.fr1_start_time,
                                   end_time=self.params.fr1_end_time, buffer_time=self.params.fr1_buf)

            eegs = eeg_reader.read()
            if eeg_reader.removed_bad_data():
                print 'REMOVED SOME BAD EVENTS !!!'
                sess_events = eegs['events'].values.view(np.recarray)
                n_events = len(sess_events)
                events = np.hstack((events[events.session!=sess],sess_events)).view(np.recarray)
                ev_order = np.argsort(events, order=('session','list','mstime'))
                events = events[ev_order]
                self.pass_object('FR_events', events)

            # print 'eegs=',eegs.values[0,0,:2],eegs.values[0,0,-2:]
            # sys.exit()
            #
            # a = eegs[0]-eegs[1]

            # mirroring
            #eegs[...,:1365] = eegs[...,2730:1365:-1]
            #eegs[...,2731:4096] = eegs[...,2729:1364:-1]

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
            for i,ti in enumerate(bipolar_pairs):
                # print bp
                # print monopolar_channels

                # print np.where(monopolar_channels == bp[0])
                # print np.where(monopolar_channels == bp[1])
                bp = ti['channel_str']
                print 'Computing powers for bipolar pair', bp
                elec1 = np.where(monopolar_channels == bp[0])[0][0]
                elec2 = np.where(monopolar_channels == bp[1])[0][0]
                # print 'elec1=',elec1
                # print 'elec2=',elec2
                # eegs_elec1 = eegs[elec1]
                # eegs_elec2 = eegs[elec2]
                # print 'eegs_elec1=',eegs_elec1
                # print 'eegs_elec2=',eegs_elec2
                # eegs_elec1.reset_coords('channels')
                # eegs_elec2.reset_coords('channels')

                bp_data = eegs[elec1] - eegs[elec2]
                bp_data.attrs['samplerate'] = self.samplerate

                # bp_data = eegs[elec1] - eegs[elec2]
                # bp_data = eegs[elec1] - eegs[elec2]
                # bp_data = eegs.values[elec1] - eegs.values[elec2]

                bp_data = bp_data.filtered([58,62], filt_type='stop', order=self.params.filt_order)
                for ev in xrange(n_events):
                    self.wavelet_transform.multiphasevec(bp_data[ev][0:winsize], pow_ev)
                    #if np.min(pow_ev) < 0.0:
                    #    print ev, events[ev]
                    #    joblib.dump(bp_data[ev], 'bad_bp_ev%d'%ev)
                    #    joblib.dump(eegs[elec1][ev], 'bad_elec1_ev%d'%ev)
                    #    joblib.dump(eegs[elec2][ev], 'bad_elec2_ev%d'%ev)
                    #    print 'Negative powers detected'
                    #    import sys
                    #    sys.exit(1)
                    pow_ev_stripped = np.reshape(pow_ev, (n_freqs,winsize))[:,bufsize:winsize-bufsize]
                    if self.params.log_powers:
                        np.log10(pow_ev_stripped, out=pow_ev_stripped)
                    sess_pow_mat[ev,i,:] = np.nanmean(pow_ev_stripped, axis=1)

            self.pow_mat = np.concatenate((self.pow_mat,sess_pow_mat), axis=0) if self.pow_mat is not None else sess_pow_mat

        self.pow_mat = np.reshape(self.pow_mat, (len(events), n_bps*n_freqs))
