__author__ = 'm'

import os.path

from RamPipeline import *

import numpy as np
from morlet import MorletWaveletTransform
from sklearn.externals import joblib

from ptsa.data.readers import EEGReader

from ReportUtils import ReportRamTask

from scipy.io import loadmat


class ComputeFR1Powers(ReportRamTask):
    def __init__(self, params, mark_as_completed=True):
        super(ComputeFR1Powers,self).__init__(mark_as_completed)
        self.params = params
        self.pow_mat = None
        self.samplerate = None
        self.wavelet_transform = MorletWaveletTransform()

    def initialize(self):
        pass
        # if self.dependency_inventory:
        #     self.dependency_inventory.add_dependent_resource(resource_name='fr1_events',
        #                                 access_path = ['experiments','fr1','events'])
        #     self.dependency_inventory.add_dependent_resource(resource_name='catfr1_events',
        #                                 access_path = ['experiments','catfr1','events'])
        #     self.dependency_inventory.add_dependent_resource(resource_name='bipolar',
        #                                 access_path = ['electrodes','bipolar'])

    def restore(self):
        subject = self.pipeline.subject
        task = 'RAM_FR1_CatFR1_joint'

        try:
            self.pow_mat = joblib.load(self.get_path_to_resource_in_workspace(subject + '-RAM_FR1_CatFR1_joint-pow_mat.pkl'))
        except:
            self.pow_mat = joblib.load(self.get_path_to_resource_in_workspace(subject + '-RAM_FR1-pow_mat.pkl'))

        self.samplerate = joblib.load(self.get_path_to_resource_in_workspace(subject + '-samplerate.pkl'))

        self.pass_object('pow_mat', self.pow_mat)
        self.pass_object('samplerate', self.samplerate)

    def run(self):
        subject = self.pipeline.subject
        task = 'RAM_FR1_CatFR1_joint'

        events = self.get_passed_object('events')

        sessions = np.unique(events.session)
        print 'sessions:', sessions

        # channels = self.get_passed_object('channels')
        # tal_info = self.get_passed_object('tal_info')
        monopolar_channels = self.get_passed_object('monopolar_channels')
        bipolar_pairs = self.get_passed_object('bipolar_pairs')
        bp_map = self.get_passed_object('bp_map')

        self.compute_powers(events, sessions, monopolar_channels, bipolar_pairs, bp_map)

        self.pass_object('pow_mat', self.pow_mat)
        self.pass_object('samplerate', self.samplerate)

        joblib.dump(self.pow_mat, self.get_path_to_resource_in_workspace(subject + '-' + task + '-pow_mat.pkl'))
        joblib.dump(self.samplerate, self.get_path_to_resource_in_workspace(subject + '-samplerate.pkl'))

    def compute_powers(self, events, sessions,monopolar_channels , bipolar_pairs, bp_map):
        n_freqs = len(self.params.freqs)
        n_bps = len(bipolar_pairs)

        self.pow_mat = None

        pow_ev = None
        winsize = bufsize = None
        for sess in sessions:
            sess_events = events[events.session == sess]
            n_events = len(sess_events)

            print 'Loading EEG for', n_events, 'events of session', sess

            windex_out = None
            ied_count = None
            ied_start = None
            ied_finish = None
            hfo_count = None
            hfo_start = None
            hfo_finish = None
            if self.params.windex_ied_cleanup:
                wfile = os.path.join(self.pipeline.mount_point,'home1/shennan.weiss/stas_0922/%s-sess%d_out_r.mat'%(self.pipeline.subject,sess))
                if os.path.isfile(wfile):
                    windex_out = loadmat(wfile, squeeze_me=True, variable_names=['IED','IED_start','IED_finish','total_hfo','hfo_start','hfo_finish'])
                    ied_count = windex_out['IED']
                    ied_start = windex_out['IED_start']
                    ied_finish = windex_out['IED_finish']
                    hfo_count = windex_out['total_hfo']
                    hfo_start = windex_out['hfo_start']
                    hfo_finish = windex_out['hfo_finish']

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

            # VERSION WITH NO MIRRORING
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
                self.pass_object(self.pipeline.task+'_events', events)


            # mirroring
            #eegs[...,:1365] = eegs[...,2730:1365:-1]
            #eegs[...,2731:4096] = eegs[...,2729:1364:-1]

            #eegs = eegs.add_mirror_buffer(duration=self.params.fr1_buf)


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

                    if windex_out is not None:
                        sess_pow_mask = np.ones(pow_ev_stripped.shape[1], dtype=np.bool)
                        bp_idx = bp_map[i]

                        if ied_count[ev,bp_idx] > 0:
                            start_points = ied_start[ev,bp_idx]-1
                            end_points = ied_finish[ev,bp_idx]-1
                            if isinstance(start_points, (int,long,float)):
                                print 'IED: Excluding t =', start_points, 'to', end_points, 'for (%d,%d)'%(ev,bp_idx)
                                sess_pow_mask[int(start_points):int(end_points)] = False
                            else:
                                for j in xrange(len(start_points)):
                                    t1 = start_points[j]
                                    t2 = end_points[j]
                                    print 'IED: Excluding t =', t1, 'to', t2, 'for (%d,%d)'%(ev,bp_idx)
                                    sess_pow_mask[t1:t2] = False

                        if hfo_count[ev,bp_idx] > 0:
                            start_points = hfo_start[ev,bp_idx]-1
                            end_points = hfo_finish[ev,bp_idx]-1
                            if isinstance(start_points, (int,long)):
                                print 'HFO: Excluding t =', start_points, 'to', end_points, 'for (%d,%d)'%(ev,bp_idx)
                                sess_pow_mask[start_points:end_points] = False
                            else:
                                for j in xrange(len(start_points)):
                                    t1 = start_points[j]
                                    t2 = end_points[j]
                                    print 'HFO: Excluding t =', t1, 'to', t2, 'for (%d,%d)'%(ev,bp_idx)
                                    sess_pow_mask[t1:t2] = False

                        if not np.any(sess_pow_mask):
                            sess_pow_mask[:] = True
                        for f_idx in xrange(n_freqs):
                            pow_ev_stripped_f = pow_ev_stripped[f_idx]
                            if f_idx>=5:
                                sess_pow_mat[ev,i,f_idx] = np.nanmean(pow_ev_stripped_f[sess_pow_mask])
                            else:
                                sess_pow_mat[ev,i,f_idx] = np.nanmean(pow_ev_stripped_f)

                    else:
                        sess_pow_mat[ev,i,:] = np.nanmean(pow_ev_stripped, axis=1)

            self.pow_mat = np.concatenate((self.pow_mat,sess_pow_mat), axis=0) if self.pow_mat is not None else sess_pow_mat

        self.pow_mat = np.reshape(self.pow_mat, (len(events), n_bps*n_freqs))
