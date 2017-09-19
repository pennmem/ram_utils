import os.path

import numpy as np
import pandas as pd

from sklearn.externals import joblib
from sklearn.metrics import roc_auc_score
from random import random
from ReportUtils import ReportRamTask

from parse_biomarker_output import parse_biomarker_output

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


class ComputeTHStimTable(ReportRamTask):
    def __init__(self, params, mark_as_completed=True):
        super(ComputeTHStimTable,self).__init__(mark_as_completed)
        self.params = params
        self.stim_params_to_sess = None
        self.sess_to_thresh = None
        self.th_stim_table = None

    def initialize(self):
        if self.dependency_inventory:
            self.dependency_inventory.add_dependent_resource(resource_name='th1_events',
                                        access_path = ['experiments','th1','events'])
            self.dependency_inventory.add_dependent_resource(resource_name='th3_events',
                                        access_path = ['experiments','th3','events'])
            self.dependency_inventory.add_dependent_resource(resource_name='th4_events',
                                        access_path = ['experiments','th4','events'])
            self.dependency_inventory.add_dependent_resource(resource_name='bipolar',
                                        access_path = ['electrodes','bipolar'])
            self.dependency_inventory.add_dependent_resource(resource_name='bipolar_json',
                                        access_path = ['electrodes','bipolar_json'])

    def restore(self):
        subject = self.pipeline.subject
        self.stim_params_to_sess = joblib.load(self.get_path_to_resource_in_workspace(subject+'-stim_params_to_sess.pkl'))
        self.th_stim_table = pd.read_pickle(self.get_path_to_resource_in_workspace(subject+'-th_stim_table.pkl'))
        self.pass_object('stim_params_to_sess', self.stim_params_to_sess)
        self.pass_object('th_stim_table', self.th_stim_table)

    def run(self):
        bp_tal_structs = self.get_passed_object('bp_tal_structs')

        all_events = self.get_passed_object(self.pipeline.task+'_all_events')
        events = self.get_passed_object(self.pipeline.task+'_events')

        # need to th1 events to figure out the distance threshold or correct/incorrect
        th_events = self.get_passed_object('th_events')
        correct_thresh = np.max([th_events[0].radius_size, np.median(th_events.distErr)])
        recalls = events.distErr <= correct_thresh
        recalls[events.confidence==0]=0

        n_events = len(events)

        lr_classifier = self.get_passed_object('lr_classifier')

        xval_output = self.get_passed_object('xval_output')
        class_thresh = xval_output[-1].jstat_thresh

        th_stim_pow_mat = self.get_passed_object('th_stim_pow_mat')
        th_stim_prob = lr_classifier.predict_proba(th_stim_pow_mat)[:,1]

        is_stim_item = np.zeros(n_events, dtype=np.bool)
        is_post_stim_item = np.zeros(n_events, dtype=np.bool)
        j = 0
        for i,ev in enumerate(all_events):
            if (ev.type=='CHEST') and (ev.confidence >=0):
                if all_events[i+1].type=='STIM':
                    is_stim_item[j] = True
                # >=8 so we don't include the pre-session stims
                if (all_events[i-1].type=='STIM_OFF') and all_events[i-1].trial >= 8:# or (all_events[i+1].type=='STIM_OFF'):
                    is_post_stim_item[j] = True
                j += 1

        self.th_stim_table = pd.DataFrame()
        self.th_stim_table['item'] = events['item_name']
        self.th_stim_table['session'] = events.session
        self.th_stim_table['list'] = events.trial
        self.th_stim_table['serialpos'] = events.chestNum
        #self.th_stim_table['itemno'] = events.itemno
        self.th_stim_table['is_stim_list'] = [(s==1) for s in events.stimList]
        self.th_stim_table['is_stim_item'] = is_stim_item
        self.th_stim_table['is_post_stim_item'] = is_post_stim_item
        self.th_stim_table['recalled'] = recalls
        # self.th_stim_table['prob'] = th_stim_prob
        self.th_stim_table['distance_err'] = events.distErr
        self.th_stim_table['confidence'] = events.confidence
        self.th_stim_table['thresh'] = class_thresh

        self.stim_params_to_sess = dict()
        self.sess_to_thresh = dict()

        sessions = np.unique(events.session)
        for sess in sessions:
            if self.pipeline.task=='RAM_TH3':
                sess_mask = (events.session==sess)
                sess_prob, thresh = parse_biomarker_output(os.path.join(self.pipeline.mount_point, 'data/eeg', self.pipeline.subject, 'raw/TH3_%d'%sess, 'commandOutput.txt'))                
                if len(sess_prob) == np.sum(sess_mask):                    
                    th_stim_prob[sess_mask] = sess_prob  # plug biomarker output
                self.th_stim_table.loc[sess_mask,'thresh'] = thresh

            sess_stim_events = all_events[(all_events.session==sess) & (all_events.type=='STIM')]
            sess_stim_event = sess_stim_events[-1]

            ch1 = '%03d' % sess_stim_event.stimParams.elec1
            ch2 = '%03d' % sess_stim_event.stimParams.elec2

            stim_tag = bp_tal_structs.index[((bp_tal_structs.channel_1==ch1) & (bp_tal_structs.channel_2==ch2)) | ((bp_tal_structs.channel_1==ch2) & (bp_tal_structs.channel_2==ch1))].values[0]
            stim_anode_tag, stim_cathode_tag = stim_tag.split('-')

            sess_stim_params = StimParams()
            sess_stim_params.stimAnodeTag = stim_anode_tag
            sess_stim_params.stimCathodeTag = stim_cathode_tag
            sess_stim_params.pulse_frequency = sess_stim_event.stimParams.pulseFreq
            sess_stim_params.amplitude = sess_stim_event.stimParams.amplitude / 1000.0
            sess_stim_params.pulse_duration = 500
            sess_stim_params.burst_frequency = -999

            if sess_stim_params in self.stim_params_to_sess:
                self.stim_params_to_sess[sess_stim_params].append(sess)
            else:
                self.stim_params_to_sess[sess_stim_params] = [sess]

        self.th_stim_table['prob'] = th_stim_prob

        stim_anode_tag = np.empty(n_events, dtype='|S16')
        stim_cathode_tag = np.empty(n_events, dtype='|S16')
        region = np.empty(n_events, dtype='|S64')
        pulse_frequency = np.empty(n_events, dtype=int)
        amplitude = np.empty(n_events, dtype=float)
        pulse_duration = np.empty(n_events, dtype=int)
        burst_frequency = np.empty(n_events, dtype=int)

        # JFM: added in auc when predicting non-stim items based on TH1 classifier,
        # as well as permutation test (shuffling recalls)
        auc = np.empty(n_events, dtype=float)
        auc_p = np.empty(n_events, dtype=float)

        for stim_params,sessions in self.stim_params_to_sess.iteritems():
            sessions_mask = np.array([(ev.session in sessions) for ev in events], dtype=np.bool)
            stim_anode_tag[sessions_mask] = stim_params.stimAnodeTag
            stim_cathode_tag[sessions_mask] = stim_params.stimCathodeTag
            region[sessions_mask] = bp_tal_structs.bp_atlas_loc.ix[stim_params.stimAnodeTag+'-'+stim_params.stimCathodeTag]
            pulse_frequency[sessions_mask] = stim_params.pulse_frequency
            amplitude[sessions_mask] = stim_params.amplitude
            pulse_duration[sessions_mask] = stim_params.pulse_duration
            burst_frequency[sessions_mask] = stim_params.burst_frequency

            # classifying TH3 non-stim
            is_stim_list = np.array(events.stimList, dtype=np.bool)
            stim_site_probs = lr_classifier.predict_proba(th_stim_pow_mat[sessions_mask & ~is_stim_list])[:,1]
            stim_site_recalls = recalls[sessions_mask & ~is_stim_list]
            stim_site_auc = roc_auc_score(stim_site_recalls, stim_site_probs)
            auc[sessions_mask] = stim_site_auc
            sess_auc_perm = np.empty(200, dtype=float)
            for perm in xrange(200):
                perm_recalls = sorted(stim_site_recalls, key=lambda *args: random())
                sess_auc_perm[perm] = roc_auc_score(perm_recalls, stim_site_probs)
            stim_site_auc_p = np.mean(stim_site_auc < sess_auc_perm)
            auc_p[sessions_mask] = stim_site_auc_p

        self.th_stim_table['stimAnodeTag'] = stim_anode_tag
        self.th_stim_table['stimCathodeTag'] = stim_cathode_tag
        self.th_stim_table['Region'] = region
        self.th_stim_table['Pulse_Frequency'] = pulse_frequency
        self.th_stim_table['Amplitude'] = amplitude
        self.th_stim_table['Duration'] = pulse_duration
        self.th_stim_table['Burst_Frequency'] = burst_frequency
        self.th_stim_table['auc'] = auc
        self.th_stim_table['auc_perm'] = auc_p

        self.pass_object('stim_params_to_sess', self.stim_params_to_sess)
        joblib.dump(self.stim_params_to_sess, self.get_path_to_resource_in_workspace(self.pipeline.subject+'-stim_params_to_sess.pkl'))

        self.pass_object('th_stim_table', self.th_stim_table)
        self.th_stim_table.to_pickle(self.get_path_to_resource_in_workspace(self.pipeline.subject+'-th_stim_table.pkl'))
