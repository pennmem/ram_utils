__author__ = 'm'


from RamPipeline import *


class EEGRawPreparation(RamTask):
    def __init__(self, mark_as_completed=False):
        RamTask.__init__(self, mark_as_completed)

    def run(self):
        events = self.get_passed_object('FR1_events')

        sel_events = events[(events.type == 'WORD') & (events.recalled == 1)]

        sel_events = sel_events[0:20]


        from ptsa.data.readers.TimeSeriesEEGReader import TimeSeriesEEGReader

        time_series_reader = TimeSeriesEEGReader(sel_events)

        time_series_reader.start_time = 0.0
        time_series_reader.end_time = 1.6
        time_series_reader.buffer_time = 1.0
        time_series_reader.keep_buffer = True

        time_series_reader.read(channels=['002','003'])


        eegs = time_series_reader.get_output()

        print eegs

