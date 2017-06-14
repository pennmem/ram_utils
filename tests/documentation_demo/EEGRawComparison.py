__author__ = 'm'

import sys
sys.path.append('/Users/m/PTSA_NEW_GIT')



from ram_utils.RamPipeline import *

from ptsa.data.readers import BaseEventReader
from ptsa.data.readers import PTSAEventReader
from ptsa.data.events import Events

import numpy as np


class EEGRawComparison(RamTask):
    def __init__(self, mark_as_completed=False):
        RamTask.__init__(self, mark_as_completed)

    def run(self):


        # PTSA EVENT  READER

        # e_path = join(self.pipeline.mount_point, 'data/events', self.pipeline.task,self.pipeline.subject+'_events.mat')
        e_path = '/Users/m/data/events/RAM_FR1/R1060M_events.mat'
        # e_path = '/Users/m/data/events/RAM_FR1/R1056M_events.mat'
        e_reader = PTSAEventReader(event_file=e_path, eliminate_events_with_no_eeg=True)

        e_reader.read()

        events = e_reader.get_output()

        events = events[events.type == 'WORD']

        events = events[0:30]

        ev_order = np.argsort(events, order=('session','list','mstime'))
        events = events[ev_order]



        events = Events(events) # necessary workaround for new numpy
        print 'events=',events

        eegs= events.get_data(channels=['002','003'], start_time=0.0, end_time=1.6,
                                        buffer_time=1.0, eoffset='eegoffset', keep_buffer=False, eoffset_in_time=False,verbose=True)

        print eegs

        # BASE READER


        base_e_reader = BaseEventReader(event_file=e_path, eliminate_events_with_no_eeg=True, use_ptsa_events_class=False)



        base_e_reader.read()

        base_events = base_e_reader.get_output()

        base_events = base_events[base_events.type == 'WORD']

        base_ev_order = np.argsort(base_events, order=('session','list','mstime'))
        base_events = base_events[base_ev_order]

        base_events = base_events[0:30]


        print 'base_events=',base_events


        from ptsa.data.readers.TimeSeriesEEGReader import TimeSeriesEEGReader

        time_series_reader = TimeSeriesEEGReader(base_events)

        time_series_reader.start_time = 0.0
        time_series_reader.end_time = 1.6
        time_series_reader.buffer_time = 1.0
        time_series_reader.keep_buffer = False

        time_series_reader.read(channels=['002','003'])


        base_eegs = time_series_reader.get_output()
        print






if __name__=='__main__':

    eeg_comparison = EEGRawComparison()

    eeg_comparison.run()

