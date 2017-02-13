__author__ = 'm'


from RamPipeline import *

import numpy as np
from scipy.stats.mstats import zscore
#from morlet import MorletWaveletTransform
from ptsa.extensions.morlet.morlet import MorletWaveletTransform
from sklearn.externals import joblib
from ptsa.data.readers.IndexReader import JsonIndexReader
from ptsa.data.events import Events
from ptsa.data.readers import EEGReader,BaseRawReader
from ReportUtils import ReportRamTask

import hashlib
from ReportTasks.RamTaskMethods import compute_powers


class ComputePSPowers(ReportRamTask):
    def __init__(self, params, mark_as_completed=True):
        super(ComputePSPowers,self).__init__( mark_as_completed)
        self.params = params
        self.wavelet_transform = MorletWaveletTransform()

    def input_hashsum(self):
        subject = self.pipeline.subject
        task = self.pipeline.task
        tmp = subject.split('_')
        subj_code = tmp[0]
        montage = 0 if len(tmp)==1 else int(tmp[1])

        json_reader = JsonIndexReader(os.path.join(self.pipeline.mount_point, 'protocols/r1.json'))

        hash_md5 = hashlib.md5()

        bp_paths = json_reader.aggregate_values('pairs', subject=subj_code, montage=montage)
        for fname in bp_paths:
            with open(fname,'rb') as f: hash_md5.update(f.read())

        event_files = sorted(list(json_reader.aggregate_values('task_events', subject=subj_code, montage=montage, experiment=task)))
        for fname in event_files:
            with open(fname,'rb') as f: hash_md5.update(f.read())

        return hash_md5.digest()

    def restore(self):
        subject = self.pipeline.subject
        task = self.pipeline.task

        ps_pow_mat_pre = joblib.load(self.get_path_to_resource_in_workspace(subject+'-'+task+'-ps_pow_mat_pre.pkl'))
        ps_pow_mat_post = joblib.load(self.get_path_to_resource_in_workspace(subject+'-'+task+'-ps_pow_mat_post.pkl'))

        self.pass_object('ps_pow_mat_pre',ps_pow_mat_pre)
        self.pass_object('ps_pow_mat_post',ps_pow_mat_post)


    def run(self):
        subject = self.pipeline.subject
        task = self.pipeline.task

        #fetching objects from other tasks
        events = self.get_passed_object(self.pipeline.task+'_events')
        # channels = self.get_passed_object('channels')
        # tal_info = self.get_passed_object('tal_info')
        monopolar_channels = self.get_passed_object('monopolar_channels')
        bipolar_pairs = self.get_passed_object('bipolar_pairs')


        sessions = np.unique(events.session)
        print task, 'sessions:', sessions

        params = self.params
        pre_start_time = self.params.ps_start_time - self.params.ps_offset
        pre_end_time = self.params.ps_end_time - self.params.ps_offset
        post_start_time = self.params.ps_offset
        post_end_time = self.params.ps_offset + (self.params.ps_end_time - self.params.ps_start_time)

        ps_pow_mat_pre, pre_events = compute_powers(events, monopolar_channels, bipolar_pairs,
                                              pre_start_time, pre_end_time, params.ps_buf,
                                              params.freqs, params.log_powers)
        ps_pow_mat_post, post_events = compute_powers(events, monopolar_channels, bipolar_pairs,
                                              post_start_time, post_end_time, params.ps_buf,
                                              params.freqs, params.log_powers)

        for session in sessions:
            joint_powers = zscore(np.concatenate((ps_pow_mat_pre[events.session==session],ps_pow_mat_post[events.session==session])),
                                  axis=0,ddof=1)
            n_events = (events.session==session).astype(np.int).sum()
            ps_pow_mat_pre[events.session==session] = joint_powers[:n_events,...]
            ps_pow_mat_post[events.session==session] = joint_powers[n_events:,...]


        pre_events = pre_events.tolist()
        post_events = post_events.tolist()

        all_events = np.rec.array([event for event in pre_events if event in post_events],dtype=events.dtype)

        self.pass_object(task+'_events', all_events)

        # ps_pow_mat_pre, ps_pow_mat_post = self.compute_ps_powers(events, sessions, monopolar_channels, bipolar_pairs, task)

        joblib.dump(ps_pow_mat_pre, self.get_path_to_resource_in_workspace(subject+'-'+task+'-ps_pow_mat_pre.pkl'))
        joblib.dump(ps_pow_mat_post, self.get_path_to_resource_in_workspace(subject+'-'+task+'-ps_pow_mat_post.pkl'))

        self.pass_object('ps_pow_mat_pre',ps_pow_mat_pre)
        self.pass_object('ps_pow_mat_post',ps_pow_mat_post)

    def compute_ps_powers(self, events, sessions, monopolar_channels, bipolar_pairs, task):
        subject = self.pipeline.subject

        n_freqs = len(self.params.freqs)
        n_bps = len(bipolar_pairs)

        pow_mat_pre = pow_mat_post = None

        pow_ev = None
        samplerate = winsize = bufsize = None

        monopolar_channels_list = list(monopolar_channels)
        for sess in sessions:

            sess_events = events[events.session == sess]
            # print type(sess_events)

            n_events = len(sess_events)

            print 'Loading EEG for', n_events, 'events of session', sess

            pre_start_time = self.params.ps_start_time - self.params.ps_offset
            pre_end_time = self.params.ps_end_time - self.params.ps_offset

            # eegs_pre = Events(sess_events).get_data(channels=channels, start_time=pre_start_time, end_time=pre_end_time,
            #             buffer_time=self.params.ps_buf, eoffset='eegoffset', keep_buffer=True, eoffset_in_time=False)


            eeg_pre_reader = EEGReader(events=sess_events, channels=np.array(monopolar_channels_list),
                                   start_time=pre_start_time,
                                   end_time=pre_end_time, buffer_time=self.params.ps_buf)

            eegs_pre = eeg_pre_reader.read()

            if eeg_pre_reader.removed_bad_data():
                print 'REMOVED SOME BAD EVENTS !!!'
                sess_events = eegs_pre['events'].values.view(np.recarray)
                n_events = len(sess_events)
                events = np.hstack((events[events.session!=sess],sess_events)).view(np.recarray)
                ev_order = np.argsort(events, order=('session','mstime'))
                events = events[ev_order]
                joblib.dump(events, self.get_path_to_resource_in_workspace(subject+'-'+task+'-ps_events.pkl'))
                self.pass_object(self.pipeline.task+'_events', events)

            if samplerate is None:
                # samplerate = round(eegs_pre.samplerate)
                # samplerate = eegs_pre.attrs['samplerate']

                samplerate = float(eegs_pre['samplerate'])

                winsize = int(round(samplerate*(pre_end_time-pre_start_time+2*self.params.ps_buf)))
                bufsize = int(round(samplerate*self.params.ps_buf))
                print 'samplerate =', samplerate, 'winsize =', winsize, 'bufsize =', bufsize
                pow_ev = np.empty(shape=n_freqs*winsize, dtype=float)
                self.wavelet_transform.init(self.params.width, self.params.freqs[0], self.params.freqs[-1], n_freqs, samplerate, winsize)

            # mirroring
            nb_ = int(round(samplerate*(self.params.ps_buf)))
            eegs_pre[...,-nb_:] = eegs_pre[...,-nb_-2:-2*nb_-2:-1]

            dim3_pre = eegs_pre.shape[2]  # because post-stim time inreval does not align for all stim events (stims have different duration)
                                          # we have to take care of aligning eegs_post ourselves time dim to dim3

            # eegs_post = np.zeros_like(eegs_pre)

            from ptsa.data.TimeSeriesX import TimeSeriesX
            eegs_post = TimeSeriesX(np.zeros_like(eegs_pre),dims=eegs_pre.dims,coords=eegs_pre.coords)


            post_start_time = self.params.ps_offset
            post_end_time = self.params.ps_offset + (self.params.ps_end_time - self.params.ps_start_time)

            post_start_offsets = np.copy(sess_events.eegoffset)


            for i_ev in xrange(n_events):
                #ev_offset = sess_events[i_ev].stim_duration if task!='PS3' else sess_events[i_ev].train_duration
                ev_offset = sess_events[i_ev].stim_duration
                if ev_offset > 0:
                    ev_offset *= 0.001
                else:
                    ev_offset = 0.0

                post_start_offsets[i_ev] += (ev_offset + post_start_time - self.params.ps_buf)*samplerate

            read_size = eegs_pre['time'].shape[0]
            dataroot = sess_events[0].eegfile
            brr = BaseRawReader(dataroot = dataroot, start_offsets=post_start_offsets, channels=np.array(monopolar_channels_list),read_size = read_size)

            eegs_post , read_ok_mask= brr.read()


            # #removing bad events from both pre and post eegs
            if np.any(~read_ok_mask):
                # print 'YES'
                read_mask_ok_events = np.all(read_ok_mask,axis=0)
                eegs_post = eegs_post[:, read_mask_ok_events, :]
                # sess_events = sess_events[read_mask_ok_events]
                eegs_pre = eegs_pre [:, read_mask_ok_events, :]

                # FIXING ARRAY ALL EVENTS - MAKE IT A FUNCTION!
                sess_events = eegs_pre['events'].values.view(np.recarray)
                n_events = len(sess_events)
                events = np.hstack((events[events.session!=sess],sess_events)).view(np.recarray)
                ev_order = np.argsort(events, order=('session','mstime'))
                events = events[ev_order]
                joblib.dump(events, self.get_path_to_resource_in_workspace(subject+'-'+task+'-ps_events.pkl'))
                self.pass_object(self.pipeline.task+'_events', events)

            eegs_post = eegs_post.rename({'offsets':'time','start_offsets':'events'})
            eegs_post['events'] = sess_events
            eegs_post['time'] = eegs_pre['time'].data
            eegs_post = TimeSeriesX(eegs_post)

            # mirroring
            eegs_post[...,:nb_] = eegs_post[...,2*nb_:nb_:-1]

            print 'Computing', task, 'powers'

            sess_pow_mat_pre = np.empty(shape=(n_events, n_bps, n_freqs), dtype=np.float)
            sess_pow_mat_post = np.empty_like(sess_pow_mat_pre)

            for i,bp in enumerate(bipolar_pairs):
                # bp = ti['channel_str']
                print 'Computing powers for bipolar pair', bp
                elec1 = np.where(monopolar_channels == bp[0])[0][0]
                elec2 = np.where(monopolar_channels == bp[1])[0][0]

            #
            # for i,ti in enumerate(tal_info):
            #     bp = ti['channel_str']
            #     print 'Computing powers for bipolar pair', bp
            #     elec1 = np.where(channels == bp[0])[0][0]
            #     elec2 = np.where(channels == bp[1])[0][0]

                bp_data_pre = np.subtract(eegs_pre[elec1],eegs_pre[elec2])
                # bp_data_pre.attrs['samplerate'] = samplerate

                bp_data_pre = bp_data_pre.filtered([58,62], filt_type='stop', order=self.params.filt_order)
                for ev in xrange(n_events):
                    #pow_pre_ev = phase_pow_multi(self.params.freqs, bp_data_pre[ev], to_return='power')
                    self.wavelet_transform.multiphasevec(bp_data_pre[ev][0:winsize], pow_ev)
                    #sess_pow_mat_pre[ev,i,:] = np.mean(pow_pre_ev[:,nb_:-nb_], axis=1)
                    pow_ev_stripped = np.reshape(pow_ev, (n_freqs,winsize))[:,bufsize:winsize-bufsize]
                    pow_zeros = np.where(pow_ev_stripped==0.0)[0]

                    if len(pow_zeros)>0:
                        print 'pre', bp, ev
                        print sess_events[ev].eegfile, sess_events[ev].eegoffset
                        self.raise_and_log_report_exception(
                                                exception_type='NumericalError',
                                                exception_message='Corrupt EEG File'
                                                )

                    if self.params.log_powers:
                        np.log10(pow_ev_stripped, out=pow_ev_stripped)
                    sess_pow_mat_pre[ev,i,:] = np.nanmean(pow_ev_stripped, axis=1)

                bp_data_post = np.subtract(eegs_post[elec1],eegs_post[elec2])
                # bp_data_post.attrs['samplerate'] = samplerate

                bp_data_post = bp_data_post.filtered([58,62], filt_type='stop', order=self.params.filt_order)
                for ev in xrange(n_events):
                    #pow_post_ev = phase_pow_multi(self.params.freqs, bp_data_post[ev], to_return='power')
                    self.wavelet_transform.multiphasevec(bp_data_post[ev][0:winsize], pow_ev)
                    #sess_pow_mat_post[ev,i,:] = np.mean(pow_post_ev[:,nb_:-nb_], axis=1)
                    pow_ev_stripped = np.reshape(pow_ev, (n_freqs,winsize))[:,bufsize:winsize-bufsize]
                    pow_zeros = np.where(pow_ev_stripped==0.0)[0]

                    if len(pow_zeros)>0:
                        print 'pre', bp, ev
                        print sess_events[ev].eegfile, sess_events[ev].eegoffset
                        self.raise_and_log_report_exception(
                                                exception_type='NumericalError',
                                                exception_message='Corrupt EEG File'
                                                )

                    if self.params.log_powers:
                        np.log10(pow_ev_stripped, out=pow_ev_stripped)
                    sess_pow_mat_post[ev,i,:] = np.nanmean(pow_ev_stripped, axis=1)

            sess_pow_mat_pre = sess_pow_mat_pre.reshape((n_events, n_bps*n_freqs))
            #sess_pow_mat_pre = zscore(sess_pow_mat_pre, axis=0, ddof=1)

            sess_pow_mat_post = sess_pow_mat_post.reshape((n_events, n_bps*n_freqs))
            #sess_pow_mat_post = zscore(sess_pow_mat_post, axis=0, ddof=1)

            sess_pow_mat_joint = zscore(np.vstack((sess_pow_mat_pre,sess_pow_mat_post)), axis=0, ddof=1)
            sess_pow_mat_pre = sess_pow_mat_joint[:n_events,...]
            sess_pow_mat_post = sess_pow_mat_joint[n_events:,...]

            pow_mat_pre = np.vstack((pow_mat_pre,sess_pow_mat_pre)) if pow_mat_pre is not None else sess_pow_mat_pre
            pow_mat_post = np.vstack((pow_mat_post,sess_pow_mat_post)) if pow_mat_post is not None else sess_pow_mat_post

        return pow_mat_pre, pow_mat_post
