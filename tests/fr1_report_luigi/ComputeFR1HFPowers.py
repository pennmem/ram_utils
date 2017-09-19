__author__ = 'm'

from scipy.stats.mstats import zscore
from ptsa.extensions.morlet.morlet import MorletWaveletTransform
import os.path
import numpy as np
from RamPipeline import *
from RamTaskL import RamTaskL
from FR1EventPreparation import FR1EventPreparation
from MontagePreparation import MontagePreparation
from ComputeFR1Powers import ComputeFR1Powers
from ptsa.data.readers import EEGReader
from ptsa.data.readers.IndexReader import JsonIndexReader
import hashlib



try:
    from ReportTasks.RamTaskMethods import compute_powers
except ImportError as ie:
    if 'MorletWaveletFilterCpp' in ie.message:
        print 'Update PTSA for better perfomance'
        compute_powers = None
    else:
        raise ie



class ComputeFR1HFPowers(RamTaskL):

    pow_mat = None
    samplerate = None
    wavelet_transform = MorletWaveletTransform()

    def requires(self):
        yield MontagePreparation(pipeline=self.pipeline)
        yield FR1EventPreparation(pipeline=self.pipeline)
        yield ComputeFR1Powers(pipeline=self.pipeline)


    def define_outputs(self):
        self.add_file_resource('hf_pow_mat')
        self.add_file_resource('hf_events')


    def input_hashsum(self):
        subject = self.pipeline.subject
        task = self.pipeline.task
        tmp = subject.split('_')
        subj_code = tmp[0]
        montage = 0 if len(tmp)==1 else int(tmp[1])

        json_reader = JsonIndexReader(os.path.join(self.pipeline.mount_point, 'protocols','r1.json'))

        hash_md5 = hashlib.md5()

        bp_paths = json_reader.aggregate_values('pairs', subject=subj_code, montage=montage)
        for fname in bp_paths:
            with open(fname,'rb') as f: hash_md5.update(f.read())

        event_files = sorted(list(json_reader.aggregate_values('all_events', subject=subj_code, montage=montage, experiment=task)))
        for fname in event_files:
            with open(fname,'rb') as f: hash_md5.update(f.read())

        return hash_md5.digest()


    def run_impl(self):

        task = self.pipeline.task
        params = self.pipeline.params

        events = self.get_passed_object(task+'_events_compute_powers')


        sessions = np.unique(events.session)
        print 'sessions:', sessions

        monopolar_channels = self.get_passed_object('monopolar_channels')
        bipolar_pairs = self.get_passed_object('bipolar_pairs')


        if compute_powers is None:
            self.compute_powers(events,sessions,monopolar_channels,bipolar_pairs)
        else:
            self.pow_mat,events=compute_powers(events,monopolar_channels, bipolar_pairs,
                                                   params.hfs_start_time,params.hfs_end_time,params.hfs_buf,
                                                   params.hfs,params.log_powers)
            self.pow_mat = self.pow_mat.reshape((len(events),len(bipolar_pairs),-1))
            for session in sessions:
                self.pow_mat[events.session==session] = zscore(self.pow_mat[events.session==session],axis=0,ddof=1)
            self.pow_mat = np.nanmean(self.pow_mat,axis=-1)
            self.pass_object('hf_events',events)

        self.pass_object('hf_pow_mat', self.pow_mat)

    def compute_powers(self, events, sessions,monopolar_channels , bipolar_pairs ):
        n_hfs = len(self.params.hfs)
        n_bps = len(bipolar_pairs)

        self.pow_mat = None

        pow_ev = None
        winsize = bufsize = None
        for sess in sessions:
            sess_events = events[events.session == sess]
            n_events = len(sess_events)

            print 'Loading EEG for', n_events, 'events of session', sess

            eeg_reader = EEGReader(events=sess_events, channels=monopolar_channels,
                                   start_time=self.params.hfs_start_time,
                                   end_time=self.params.hfs_end_time, buffer_time=0.)

            eegs = eeg_reader.read()
            if eeg_reader.removed_bad_data():
                print 'REMOVED SOME BAD EVENTS !!!'
                sess_events = eegs['events'].values.view(np.recarray)
                n_events = len(sess_events)
                events = np.hstack((events[events.session!=sess],sess_events)).view(np.recarray)
                ev_order = np.argsort(events, order=('session','list','mstime'))
                events = events[ev_order]
                self.pass_object(self.pipeline.task+'_events', events)

            #eegs['events'] = np.arange(eegs.events.shape[0])

            eegs = eegs.add_mirror_buffer(duration=self.params.hfs_buf)

            if self.samplerate is None:
                self.samplerate = float(eegs.samplerate)
                winsize = int(round(self.samplerate*(self.params.hfs_end_time-self.params.hfs_start_time+2*self.params.hfs_buf)))
                bufsize = int(round(self.samplerate*self.params.hfs_buf))
                print 'samplerate =', self.samplerate, 'winsize =', winsize, 'bufsize =', bufsize
                pow_ev = np.empty(shape=n_hfs*winsize, dtype=float)
                self.wavelet_transform.init(self.params.width, self.params.hfs[0], self.params.hfs[-1], n_hfs, self.samplerate, winsize)

            print 'Computing FR1 powers'

            sess_pow_mat = np.empty(shape=(n_events, n_bps, n_hfs), dtype=np.float)

            # for i,ti in enumerate(bipolar_pairs):
            #     bp = ti['channel_str']
            #     print 'Computing powers for bipolar pair', bp
            #     elec1 = np.where(monopolar_channels == bp[0])[0][0]
            #     elec2 = np.where(monopolar_channels == bp[1])[0][0]

            for i,bp in enumerate(bipolar_pairs):

                print 'Computing powers for bipolar pair', bp
                elec1 = np.where(monopolar_channels == bp[0])[0][0]
                elec2 = np.where(monopolar_channels == bp[1])[0][0]

                bp_data = np.subtract(eegs[elec1],eegs[elec2])
                bp_data.attrs['samplerate'] = self.samplerate

                bp_data = bp_data.filtered([58,62], filt_type='stop', order=self.params.filt_order)
                for ev in xrange(n_events):
                    self.wavelet_transform.multiphasevec(bp_data[ev][0:winsize], pow_ev)
                    pow_ev_stripped = np.reshape(pow_ev, (n_hfs,winsize))[:,bufsize:winsize-bufsize]
                    if self.params.log_powers:
                        np.log10(pow_ev_stripped, out=pow_ev_stripped)
                    sess_pow_mat[ev,i,:] = np.nanmean(pow_ev_stripped, axis=1)

            sess_pow_mat = zscore(sess_pow_mat, axis=0, ddof=1)
            sess_pow_mat = np.nanmean(sess_pow_mat, axis=2)

            self.pow_mat = np.vstack((self.pow_mat,sess_pow_mat)) if self.pow_mat is not None else sess_pow_mat
