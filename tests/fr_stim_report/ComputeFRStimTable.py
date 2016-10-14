import os.path

import numpy as np
import pandas as pd

from sklearn.externals import joblib

from ReportUtils import ReportRamTask
from ptsa.data.readers.IndexReader import JsonIndexReader
from parse_biomarker_output import parse_biomarker_output

import hashlib


class StimParams(object):
    def __init__(self):
        self.amplitude = None
        self.pulse_frequency = None
        self.burst_frequency = None
        self.pulse_duration = None
        self.stimAnodeTag = None
        self.stimCathodeTag = None

    def __hash__(self):
        return hash(repr(self.stimAnodeTag)+repr(self.stimCathodeTag)+repr(self.pulse_frequency))


class ComputeFRStimTable(ReportRamTask):
    def __init__(self, params, mark_as_completed=True):
        super(ComputeFRStimTable,self).__init__(mark_as_completed)
        self.params = params
        self.stim_params_to_sess = None
        self.sess_to_thresh = None
        self.fr_stim_table = None

    def input_hashsum(self):
        subject = self.pipeline.subject
        task = self.pipeline.task
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

        fr_stim_event_files = sorted(list(json_reader.aggregate_values('all_events', subject=subj_code, montage=montage, experiment=task)))
        for fname in fr_stim_event_files:
            hash_md5.update(open(fname,'rb').read())

        return hash_md5.digest()

    def restore(self):
        subject = self.pipeline.subject
        self.stim_params_to_sess = joblib.load(self.get_path_to_resource_in_workspace(subject+'-stim_params_to_sess.pkl'))
        self.fr_stim_table = pd.read_pickle(self.get_path_to_resource_in_workspace(subject+'-fr_stim_table.pkl'))
        self.pass_object('stim_params_to_sess', self.stim_params_to_sess)
        self.pass_object('fr_stim_table', self.fr_stim_table)

    def run(self):
        bp_tal_structs = self.get_passed_object('bp_tal_structs')

        all_events = self.get_passed_object(self.pipeline.task+'_all_events')
        events = self.get_passed_object(self.pipeline.task+'_events')

        n_events = len(events)

        lr_classifier = self.get_passed_object('lr_classifier')

        xval_output = self.get_passed_object('xval_output')
        class_thresh = xval_output[-1].jstat_thresh

        fr_stim_pow_mat = self.get_passed_object('fr_stim_pow_mat')
        fr_stim_prob = lr_classifier.predict_proba(fr_stim_pow_mat)[:,1]

        is_stim_item = np.zeros(n_events, dtype=np.bool)
        is_post_stim_item = np.zeros(n_events, dtype=np.bool)
        j = 0
        for i,ev in enumerate(all_events):
            if ev.type=='WORD':
                if (all_events[i+1].type=='STIM_ON') or (all_events[i+1].type=='WORD_OFF' and (all_events[i+2].type=='STIM_ON' or (all_events[i+2].type=='DISTRACT_START' and all_events[i+3].type=='STIM_ON'))):
                    is_stim_item[j] = True
                if (all_events[i-1].type=='STIM_OFF') or (all_events[i+1].type=='STIM_OFF'):
                    is_post_stim_item[j] = True
                j += 1

        self.fr_stim_table = pd.DataFrame()
        self.fr_stim_table['item'] = events.word
        self.fr_stim_table['session'] = events.session
        self.fr_stim_table['list'] = events.list
        self.fr_stim_table['serialpos'] = events.serialpos
        self.fr_stim_table['itemno'] = events.wordno
        self.fr_stim_table['is_stim_list'] = [(s==1) for s in events.stim_list]
        self.fr_stim_table['is_stim_item'] = is_stim_item
        self.fr_stim_table['is_post_stim_item'] = is_post_stim_item
        self.fr_stim_table['recalled'] = events.recalled
        self.fr_stim_table['thresh'] = class_thresh

        self.stim_params_to_sess = dict()
        self.sess_to_thresh = dict()

        sessions = np.unique(events.session)
        for sess in sessions:
            if self.pipeline.task=='FR3' and self.pipeline.subject!='R1124J_1':
                sess_mask = (events.session==sess)
                sess_prob, thresh = parse_biomarker_output(os.path.join(self.pipeline.mount_point, 'data/eeg', self.pipeline.subject, 'raw/FR3_%d'%sess, 'commandOutput.txt'))
                sess_prob = sess_prob[12:]  # discard practice list
                if len(sess_prob) == np.sum(sess_mask):
                    fr_stim_prob[sess_mask] = sess_prob  # plug biomarker output
                self.fr_stim_table.loc[sess_mask,'thresh'] = thresh

            sess_stim_events = all_events[(all_events.session==sess) & (all_events.type=='STIM_ON')]
            sess_stim_event = sess_stim_events[-1]

            ch1 = '%03d' % sess_stim_event.stim_params.anode_number
            ch2 = '%03d' % sess_stim_event.stim_params.cathode_number

            stim_tag = bp_tal_structs.index[((bp_tal_structs.channel_1==ch1) & (bp_tal_structs.channel_2==ch2)) | ((bp_tal_structs.channel_1==ch2) & (bp_tal_structs.channel_2==ch1))].values[0]
            stim_anode_tag, stim_cathode_tag = stim_tag.split('-')

            sess_stim_params = StimParams()
            sess_stim_params.stimAnodeTag = stim_anode_tag
            sess_stim_params.stimCathodeTag = stim_cathode_tag
            sess_stim_params.pulse_frequency = sess_stim_event.stim_params.pulse_freq
            sess_stim_params.amplitude = sess_stim_event.stim_params.amplitude / 1000.0
            sess_stim_params.pulse_duration = 500
            sess_stim_params.burst_frequency = -999

            if sess_stim_params in self.stim_params_to_sess:
                self.stim_params_to_sess[sess_stim_params].append(sess)
            else:
                self.stim_params_to_sess[sess_stim_params] = [sess]

        self.fr_stim_table['prob'] = fr_stim_prob

        stim_anode_tag = np.empty(n_events, dtype='|S16')
        stim_cathode_tag = np.empty(n_events, dtype='|S16')
        region = np.empty(n_events, dtype='|S64')
        pulse_frequency = np.empty(n_events, dtype=int)
        amplitude = np.empty(n_events, dtype=float)
        pulse_duration = np.empty(n_events, dtype=int)
        burst_frequency = np.empty(n_events, dtype=int)

        for stim_params,sessions in self.stim_params_to_sess.iteritems():
            sessions_mask = np.array([(ev.session in sessions) for ev in events], dtype=np.bool)
            stim_anode_tag[sessions_mask] = stim_params.stimAnodeTag
            stim_cathode_tag[sessions_mask] = stim_params.stimCathodeTag
            region[sessions_mask] = bp_tal_structs.bp_atlas_loc.ix[stim_params.stimAnodeTag+'-'+stim_params.stimCathodeTag]
            pulse_frequency[sessions_mask] = stim_params.pulse_frequency
            amplitude[sessions_mask] = stim_params.amplitude
            pulse_duration[sessions_mask] = stim_params.pulse_duration
            burst_frequency[sessions_mask] = stim_params.burst_frequency

        self.fr_stim_table['stimAnodeTag'] = stim_anode_tag
        self.fr_stim_table['stimCathodeTag'] = stim_cathode_tag
        self.fr_stim_table['Region'] = region
        self.fr_stim_table['Pulse_Frequency'] = pulse_frequency
        self.fr_stim_table['Amplitude'] = amplitude
        self.fr_stim_table['Duration'] = pulse_duration
        self.fr_stim_table['Burst_Frequency'] = burst_frequency

        self.pass_object('stim_params_to_sess', self.stim_params_to_sess)
        joblib.dump(self.stim_params_to_sess, self.get_path_to_resource_in_workspace(self.pipeline.subject+'-stim_params_to_sess.pkl'))

        self.pass_object('fr_stim_table', self.fr_stim_table)
        self.fr_stim_table.to_pickle(self.get_path_to_resource_in_workspace(self.pipeline.subject+'-fr_stim_table.pkl'))
