import numpy as np
import pandas as pd

from ReportUtils import ReportRamTask


class ComputeFR4Table(ReportRamTask):
    def __init__(self, params, mark_as_completed=True):
        super(ComputeFR4Table,self).__init__(mark_as_completed)
        self.params = params
        self.fr4_table = None

    def initialize(self):
        if self.dependency_inventory:
            self.dependency_inventory.add_dependent_resource(resource_name='bipolar',
                                        access_path = ['electrodes','bipolar'])

    def restore(self):
        subject = self.pipeline.subject
        self.fr4_table = pd.read_pickle(self.get_path_to_resource_in_workspace(subject+'-fr4_table.pkl'))
        self.pass_object('fr4_table', self.fr4_table)

    def run(self):
        subject = self.pipeline.subject
        task = self.pipeline.task

        all_events = self.get_passed_object(self.pipeline.task+'_all_events')
        events = self.get_passed_object(self.pipeline.task+'_events')

        stim_events = all_events[all_events.type=='STIM']
        sessions = np.unique(stim_events.session)

        for sess in sessions:
            sess_stim_events = stim_events[stim_events.session==sess]

            elec1 = sess_stim_events[0].stimParams.elect1
            elec2 = sess_stim_events[0].stimParams.elect2
            pulse_freq = sess_stim_events[0].stimParams.pulseFreq
            amplitude = sess_stim_events[0].stimParams.amplitude / 1000.0
            duration = 500
            burst_freq = -999
