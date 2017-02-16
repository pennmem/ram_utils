__author__ = 'm'

from RamPipeline import *

import numpy as np
#from morlet import MorletWaveletTransform
from ptsa.extensions.morlet.morlet import MorletWaveletTransform
from sklearn.externals import joblib

from ptsa.data.events import Events
from ptsa.data.readers import EEGReader
from ReportUtils import MissingDataError
from ReportUtils import ReportRamTask
from ptsa.data.readers.IndexReader import JsonIndexReader

import hashlib
from ReportTasks.RamTaskMethods import compute_powers

class ComputeFRPowers(ReportRamTask):
    def __init__(self, params, mark_as_completed=True):
        super(ComputeFRPowers, self).__init__(mark_as_completed)
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

        has_fr1 = (sessions<100).any()
        has_catfr1 = (sessions>=100).any()

        try:
            fr1_powers = joblib.load(self.get_path_to_resource_in_workspace(subject + '-FR1-pow_mat.pkl'))
        except IOError:
            fr1_powers = None
        try:
            catfr1_powers = joblib.load(self.get_path_to_resource_in_workspace(subject + '-catFR1-pow_mat.pkl'))
        except IOError:
            catfr1_powers = None

        if (has_fr1 and fr1_powers is None) or (has_catfr1 and catfr1_powers is None):
            params = self.params
            self.pow_mat, events = compute_powers(events, monopolar_channels, bipolar_pairs,
                                                  params.fr1_start_time, params.fr1_end_time, params.fr1_buf,
                                                  params.freqs, params.log_powers)

            self.pass_object('FR_events', events)
        elif fr1_powers is not None and catfr1_powers is not None:
            self.pow_mat = np.vstack((fr1_powers,catfr1_powers))
        else:
            self.pow_mat = fr1_powers if catfr1_powers is None else catfr1_powers

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
            try:
                eegs = eeg_reader.read()

            except IOError as err:
                self.raise_and_log_report_exception(
                                                    exception_type='MissingDataError',
                                                    exception_message='Could not read EEG file for subject %s'%(self.pipeline.subject)
                                                    )

                # raise MissingDataError('Could not read EEG file for subject %s'%(self.pipeline.subject))

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
            for i,bp in enumerate(bipolar_pairs):
                # print bp
                # print monopolar_channels

                # print np.where(monopolar_channels == bp[0])
                # print np.where(monopolar_channels == bp[1])

                # bp = ti['channel_str']

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

                bp_data = np.subtract(eegs[elec1],eegs[elec2])
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
