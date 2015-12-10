__author__ = 'm'

from RamPipeline import *

import numpy as np
from scipy.signal import resample
from ptsa.wavelet import phase_pow_multi
from sklearn.externals import joblib


class ComputeFR1Powers(RamTask):
    def __init__(self, params, mark_as_completed=True):
        RamTask.__init__(self, mark_as_completed)
        self.params = params
        self.pow_mat = None

    def restore(self):
        subject = self.pipeline.subject
        task = self.pipeline.task
        self.pow_mat = joblib.load(self.get_path_to_resource_in_workspace(subject + '-' + task + '-pow_mat.pkl'))
        self.pass_object('pow_mat', self.pow_mat)

    def run(self):
        subject = self.pipeline.subject
        task = self.pipeline.task

        events = self.get_passed_object(task+'_events')

        sessions = np.unique(events.session)
        print 'sessions:', sessions

        channels = self.get_passed_object('channels')
        tal_info = self.get_passed_object('tal_info')
        self.compute_powers(events, sessions, channels, tal_info)

        joblib.dump(self.pow_mat, self.get_path_to_resource_in_workspace(subject + '-' + task + '-pow_mat.pkl'))
        self.pass_object('pow_mat', self.pow_mat)

    def compute_powers(self, events, sessions, channels, tal_info):
        n_freqs = len(self.params.freqs)
        n_bps = len(tal_info)
        nt = int((self.params.fr1_end_time-self.params.fr1_start_time+2*self.params.fr1_buf+1e-5) * 50)
        nb = int((self.params.fr1_buf+1e-5) * 50)
        n_times = nt - 2*nb
# ['eegfile', 'eegoffset', 'expVersion', 'intrusion', 'isStim', 'item', 'itemno', 'list', 'msoffset', 'mstime', 'recalled', 'rectime', 'serialpos', 'session', 'stimAmp', 'stimAnode', 'stimAnodeTag', 'stimCathode', 'stimCathodeTag', 'stimList', 'stimLoc', 'subject', 'type', 'esrc']


        self.pow_mat = None

        for sess in sessions:
            sess_events = events[events.session == sess]


            # sess_events = sess_events[0:300]

            n_events = len(sess_events)

            print 'Loading EEG for', n_events, 'events of session', sess

            evs=sess_events[0:5]

            from TimeSeriesEEGReader import TimeSeriesEEGReader

            time_series_reader = TimeSeriesEEGReader(evs, data_dir_prefix=self.pipeline.mount_point)

            time_series_reader.start_time = self.params.fr1_start_time
            time_series_reader.end_time = self.params.fr1_end_time
            time_series_reader.buffer_time = self.params.fr1_buf
            time_series_reader.keep_buffer = True

            time_series_reader.read(channels)


            from ButterworthFilter import ButterworthFiler
            b_filter = ButterworthFiler()
            b_filter.samplerate = time_series_reader.samplerate


            # time_series_reader.read(channels=channels, start_time=self.params.fr1_start_time, end_time=self.params.fr1_end_time,
            #                             buffer_time=self.params.fr1_buf, keep_buffer=True)



            eegs= evs.get_data(channels=channels, start_time=self.params.fr1_start_time, end_time=self.params.fr1_end_time,
                                        buffer_time=self.params.fr1_buf, eoffset='eegoffset', keep_buffer=True, eoffset_in_time=False,verbose=True)




            eegs_new = time_series_reader.get_output()

            print 'Computing FR1 powers'

            sess_pow_mat = np.empty(shape=(n_events, n_bps, n_freqs, n_times), dtype=np.float)



            for i,ti in enumerate(tal_info):
                bp = ti['channel_str']
                print 'Computing powers for bipolar pair', bp
                elec1 = np.where(channels == bp[0])[0][0]
                elec2 = np.where(channels == bp[1])[0][0]
                bp_data = eegs[elec1] - eegs[elec2]
                bp_data = bp_data.filtered([58,62], filt_type='stop', order=self.params.filt_order)



                bp_data_new = eegs_new[elec1] - eegs_new[elec2]
                b_filter.set_input(bp_data_new)
                b_filter.filter()

                bp_data_new_filtered = b_filter.get_output()



                for ev in xrange(5):
                    pow_ev = phase_pow_multi(self.params.freqs, bp_data[ev], to_return='power')
                    #
                    # if not np.all(pow_ev > 0.):
                    #     raise RuntimeError('BEFORE LOG: negative power for electrodes'+str(bp)+' event=%s'%str((ev.session,ev.list,ev.serialpos)))

                    if np.min(pow_ev) < 0.0:
                        print 'Got negative powers'

                    if self.params.log_powers:
                        # pow_ev[pow_ev<1.0] = 1.0 # we do not allow power values to be less than 1 to avoid potential logarithmic instability
                        np.log10(pow_ev, out=pow_ev)

                    # if not np.all(np.isfinite(pow_ev)):
                    #     raise RuntimeError('AFTER LOG: NAN power for electrodes'+str(bp)+' event=%s'%str((ev.session,ev.list,ev.serialpos)))

                    pow_ev = resample(pow_ev, num=nt, axis=1)

                    # if not np.all(np.isfinite(pow_ev)):
                    #     raise RuntimeError('AFTER RESAMPLE: NAN power for electrodes'+str(bp)+' event=%s'%str((ev.session,ev.list,ev.serialpos)))

                    sess_pow_mat[ev,i,:,:] = pow_ev[:,nb:-nb]

            self.pow_mat = np.concatenate((self.pow_mat,sess_pow_mat), axis=0) if self.pow_mat is not None else sess_pow_mat


            # from TimeSeriesEEGReader import TimeSeriesEEGReader
            # time_series_reader = TimeSeriesEEGReader(events, data_dir_prefix=self.pipeline.mount_point)
            # time_series_reader.read(channels=channels, start_time=self.params.fr1_start_time, end_time=self.params.fr1_end_time,
            #                             buffer_time=self.params.fr1_buf, keep_buffer=True)
            #
            #
            #
            # eegs = sess_events.get_data(channels=channels, start_time=self.params.fr1_start_time, end_time=self.params.fr1_end_time,
            #                             buffer_time=self.params.fr1_buf, eoffset='eegoffset', keep_buffer=True, eoffset_in_time=False,verbose=True)

            # print 'Computing FR1 powers'
            #
            # sess_pow_mat = np.empty(shape=(n_events, n_bps, n_freqs, n_times), dtype=np.float)
            #
            # for i,ti in enumerate(tal_info):
            #     bp = ti['channel_str']
            #     print 'Computing powers for bipolar pair', bp
            #     elec1 = np.where(channels == bp[0])[0][0]
            #     elec2 = np.where(channels == bp[1])[0][0]
            #     bp_data = eegs[elec1] - eegs[elec2]
            #     bp_data = bp_data.filtered([58,62], filt_type='stop', order=self.params.filt_order)
            #     for ev in xrange(n_events):
            #         pow_ev = phase_pow_multi(self.params.freqs, bp_data[ev], to_return='power')
            #         #
            #         # if not np.all(pow_ev > 0.):
            #         #     raise RuntimeError('BEFORE LOG: negative power for electrodes'+str(bp)+' event=%s'%str((ev.session,ev.list,ev.serialpos)))
            #
            #         if np.min(pow_ev) < 0.0:
            #             print 'Got negative powers'
            #
            #         if self.params.log_powers:
            #             # pow_ev[pow_ev<1.0] = 1.0 # we do not allow power values to be less than 1 to avoid potential logarithmic instability
            #             np.log10(pow_ev, out=pow_ev)
            #
            #         # if not np.all(np.isfinite(pow_ev)):
            #         #     raise RuntimeError('AFTER LOG: NAN power for electrodes'+str(bp)+' event=%s'%str((ev.session,ev.list,ev.serialpos)))
            #
            #         pow_ev = resample(pow_ev, num=nt, axis=1)
            #
            #         # if not np.all(np.isfinite(pow_ev)):
            #         #     raise RuntimeError('AFTER RESAMPLE: NAN power for electrodes'+str(bp)+' event=%s'%str((ev.session,ev.list,ev.serialpos)))
            #
            #         sess_pow_mat[ev,i,:,:] = pow_ev[:,nb:-nb]
            #
            # self.pow_mat = np.concatenate((self.pow_mat,sess_pow_mat), axis=0) if self.pow_mat is not None else sess_pow_mat
