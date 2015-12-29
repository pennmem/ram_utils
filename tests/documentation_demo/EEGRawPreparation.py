__author__ = 'm'


from RamPipeline import *

class EEGRawPreparation(RamTask):
    def __init__(self, mark_as_completed=False):
        RamTask.__init__(self, mark_as_completed)

    def run(self):
        events = self.get_passed_object('FR1_events')

        sel_events = events[(events.type == 'WORD') & (events.recalled == 1)]

        sel_events = sel_events[0:20]

        eegs = sel_events.get_data(channels=['002','003'], start_time=-1.0, end_time=2.0,
                                   buffer_time=1.0, eoffset='eegoffset', keep_buffer=True, eoffset_in_time=False)

        print
