import os
import hashlib
import numpy as np

from scipy.stats.mstats import zscore
from sklearn.externals import joblib

from ReportUtils import ReportRamTask
from ramutils.eeg.powers import compute_powers
from ptsa.data.readers import EEGReader
from ptsa.data.readers.IndexReader import JsonIndexReader



class ComputeFR6Powers(ReportRamTask):
    def __init__(self, params, mark_as_completed=True):
        super(ComputeFR6Powers, self).__init__(mark_as_completed)
        self.params = params
        self.pow_mat = None
        self.samplerate = None

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
        print('sessions:', sessions)

        self.pow_mat,events = compute_powers(events,
                                             self.params.fr1_start_time,
                                             self.params.fr1_end_time,
                                             self.params.fr1_buf,
                                             self.params.freqs,
                                             self.params.log_powers)

        self.pass_object(task+'_events', events)
        print('self.pow_mat.shape:', self.pow_mat.shape)
        post_stim_powers, stim_off_events = compute_powers(stim_off_events,
                                                           self.params.post_stim_start_time,
                                                           self.params.post_stim_end_time,
                                                           self.params.post_stim_buf,
                                                           self.params.freqs,
                                                           self.params.log_powers)

        # Normalize the power matrices based on the within-session mean and standard deviation
        # TODO: Normalization scheme should really be trying to match ramulator. This amounts
        # to a map between events and mean/std deviation used for normalization during the
        # experiment
        for session in sessions:
            stim_off_sess_stims = (stim_off_events.session == session)
            post_stim_powers[stim_off_sess_stims] = zscore(post_stim_powers[stim_off_sess_stims])
            sess_stims = (events.session == session)
            self.pow_mat[sess_stims] = zscore(self.pow_mat[sess_stims])

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

        return