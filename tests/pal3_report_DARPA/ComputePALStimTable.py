import numpy as np
import pandas as pd

from sklearn.externals import joblib

from ReportUtils import ReportRamTask


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


def bipolar_label_to_loc_tag(bp, loc_tags):
    if bp=='' or bp=='[]':
        return 'Undetermined'
    label = bp[0]+'-'+bp[1]
    if label in loc_tags:
        lt = loc_tags[label]
        return lt if lt!='' and lt!='[]' else 'Undetermined'
    label = bp[1]+'-'+bp[0]
    if label in loc_tags:
        lt = loc_tags[label]
        return lt if lt!='' and lt!='[]' else 'Undetermined'
    else:
        return 'Undetermined'


class ComputePALStimTable(ReportRamTask):
    def __init__(self, params, mark_as_completed=True):
        super(ComputePALStimTable,self).__init__(mark_as_completed)
        self.params = params
        self.stim_params_to_sess = None
        self.pal_stim_table = None

    def initialize(self):
        if self.dependency_inventory:

            self.dependency_inventory.add_dependent_resource(resource_name='pal1_events',
                                        access_path = ['experiments','pal1','events'])

            self.dependency_inventory.add_dependent_resource(resource_name='pal3_events',
                                        access_path = ['experiments','pal3','events'])

            self.dependency_inventory.add_dependent_resource(resource_name='bipolar',
                                        access_path = ['electrodes','bipolar'])

    def restore(self):
        subject = self.pipeline.subject
        self.stim_params_to_sess = joblib.load(self.get_path_to_resource_in_workspace(subject+'-stim_params_to_sess.pkl'))
        self.pal_stim_table = pd.read_pickle(self.get_path_to_resource_in_workspace(subject+'-pal_stim_table.pkl'))
        self.pass_object('stim_params_to_sess', self.stim_params_to_sess)
        self.pass_object('pal_stim_table', self.pal_stim_table)

    def run(self):
        channel_to_label_map = self.get_passed_object('channel_to_label_map')
        loc_tag = self.get_passed_object('loc_tag')

        all_events = self.get_passed_object(self.pipeline.task+'_all_events')
        events = self.get_passed_object(self.pipeline.task+'_events')

        n_events = len(events)

        lr_classifier = self.get_passed_object('lr_classifier')

        pal_stim_pow_mat = self.get_passed_object('pal_stim_pow_mat')
        pal_stim_prob = lr_classifier.predict_proba(pal_stim_pow_mat)[:,1]

        is_stim_item = np.zeros(n_events, dtype=np.bool)
        j = 0
        for i,ev in enumerate(all_events):
            if ev.type=='STUDY_PAIR':
                if all_events[i+1].type=='STIM':
                    is_stim_item[j] = True
                j += 1

        self.pal_stim_table = pd.DataFrame()
        self.pal_stim_table['study_1'] = events['study_1']
        self.pal_stim_table['study_2'] = events['study_2']
        self.pal_stim_table['session'] = events.session
        self.pal_stim_table['list'] = events.list
        self.pal_stim_table['serialpos'] = events.serialpos
        self.pal_stim_table['is_stim_list'] = [(s==1) for s in events.stimList]
        self.pal_stim_table['is_stim_item'] = is_stim_item
        self.pal_stim_table['recalled'] = events.correct
        self.pal_stim_table['prob'] = pal_stim_prob

        self.stim_params_to_sess = dict()

        sessions = np.unique(events.session)
        for sess in sessions:
            sess_stim_events = all_events[(all_events.session==sess) & (all_events.type=='STIM')]
            sess_stim_event = sess_stim_events[-1]

            stim_pair = (sess_stim_event.stimParams.elec1,sess_stim_event.stimParams.elec2)
            stim_tag = channel_to_label_map[stim_pair if stim_pair in channel_to_label_map else (sess_stim_event.stimParams.elec1,sess_stim_event.stimParams.elec2)].upper()
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
            region[sessions_mask] = bipolar_label_to_loc_tag((stim_params.stimAnodeTag,stim_params.stimCathodeTag), loc_tag)
            pulse_frequency[sessions_mask] = stim_params.pulse_frequency
            amplitude[sessions_mask] = stim_params.amplitude
            pulse_duration[sessions_mask] = stim_params.pulse_duration
            burst_frequency[sessions_mask] = stim_params.burst_frequency

        self.pal_stim_table['stimAnodeTag'] = stim_anode_tag
        self.pal_stim_table['stimCathodeTag'] = stim_cathode_tag
        self.pal_stim_table['Region'] = region
        self.pal_stim_table['Pulse_Frequency'] = pulse_frequency
        self.pal_stim_table['Amplitude'] = amplitude
        self.pal_stim_table['Duration'] = pulse_duration
        self.pal_stim_table['Burst_Frequency'] = burst_frequency

        self.pass_object('stim_params_to_sess', self.stim_params_to_sess)
        joblib.dump(self.stim_params_to_sess, self.get_path_to_resource_in_workspace(self.pipeline.subject+'-stim_params_to_sess.pkl'))

        self.pass_object('pal_stim_table', self.pal_stim_table)
        self.pal_stim_table.to_pickle(self.get_path_to_resource_in_workspace(self.pipeline.subject+'-pal_stim_table.pkl'))
