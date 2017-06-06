from ReportUtils import ReportRamTask
from ptsa.data.readers.IndexReader import JsonIndexReader
import os
import hashlib
from ReportTasks.RamTaskMethods import compute_powers
from sklearn.externals import joblib


class ComputePAL5Powers(ReportRamTask):
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

        pal1_event_files = sorted(list(json_reader.aggregate_values('all_events', subject=subj_code, montage=montage, experiment='PAL1')))
        for fname in pal1_event_files:
            with open(fname,'rb') as f: hash_md5.update(f.read())

        pal_stim_event_files = sorted(list(json_reader.aggregate_values('all_events', subject=subj_code, montage=montage, experiment=task)))
        for fname in pal_stim_event_files:
            with open(fname,'rb') as f: hash_md5.update(f.read())

        return hash_md5.digest()

    def __init__(self,params,mark_as_completed):
        self.params=params
        super(ComputePAL5Powers,self).__init__(mark_as_completed=mark_as_completed)
        self.samplerate = 1000


    def run(self):
        events = self.get_passed_object('events')

        monopolar_channels = self.get_passed_object('monopolar_channels')
        bipolar_pairs = self.get_passed_object('bipolar_pairs')

        pow_mat,events = compute_powers(events,monopolar_channels,bipolar_pairs,
                                        self.params.pal1_start_time,self.params.pal1_end_time, self.params.pal1_buf_time,
                                        self.params.pal1_freqs,log_powers=True,ComputePowers=self)

        joblib.dump(events,self.get_path_to_resource_in_workspace('pal5_events.pkl'))
        joblib.dump(pow_mat,self.get_path_to_resource_in_workspace('pal5_pow_mat.pkl'))
        joblib.dump(self.samplerate,self.get_path_to_resource_in_workspace('samplerate.pkl'))

        self.pass_object('events',events)
        self.pass_object('pal_stim_pow_mat',pow_mat)
        self.pass_object('pal_stim_samplerate',self.samplerate)


