import numpy as np
from ptsa.data.readers import EEGReader

from ReportUtils import ReportRamTask
from scipy.io import savemat


class SaveEEG(ReportRamTask):
    def __init__(self, params, mark_as_completed=True):
        super(SaveEEG,self).__init__(mark_as_completed)
        self.params = params
        self.samplerate = None

    def initialize(self):
        if self.dependency_inventory:
            self.dependency_inventory.add_dependent_resource(resource_name='fr1_events',
                                        access_path = ['experiments','fr1','events'])
            self.dependency_inventory.add_dependent_resource(resource_name='catfr1_events',
                                        access_path = ['experiments','catfr1','events'])
            self.dependency_inventory.add_dependent_resource(resource_name='bipolar',
                                        access_path = ['electrodes','bipolar'])

    def run(self):
        subject = self.pipeline.subject

        events = self.get_passed_object('events')

        sessions = np.unique(events.session)
        print 'sessions:', sessions

        monopolar_channels = self.get_passed_object('monopolar_channels')
        bipolar_pairs = self.get_passed_object('bipolar_pairs')

        n_bps = len(bipolar_pairs)

        for sess in sessions:
            sess_events = events[events.session == sess]
            n_events = len(sess_events)

            print 'Loading EEG for', n_events, 'events of session', sess

            eeg_reader = EEGReader(events=sess_events, channels=monopolar_channels,
                                   start_time=self.params.fr1_start_time,
                                   end_time=self.params.fr1_end_time, buffer_time=0.0)

            eegs = eeg_reader.read()

            if self.samplerate is None:
                self.samplerate = float(eegs.samplerate)

            bp_eegs = np.empty(shape=(n_events, n_bps, eegs.values.shape[2]), dtype=float)
            for i,bp in enumerate(bipolar_pairs):
                print 'Computing EEG for bipolar pair', bp
                elec1 = np.where(monopolar_channels == bp[0])[0][0]
                elec2 = np.where(monopolar_channels == bp[1])[0][0]

                bp_data = eegs[elec1] - eegs[elec2]
                bp_data.attrs['samplerate'] = self.samplerate
                bp_eegs[:,i,:] = bp_data.values

            savemat('/scratch/busygin/bp_eegs/%s-sess%d.mat'%(subject,sess), {'eeg':bp_eegs, 'samplerate':self.samplerate})
