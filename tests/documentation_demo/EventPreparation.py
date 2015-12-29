__author__ = 'm'

import sys

sys.path.append('/Users/m/PTSA_GIT')

from RamPipeline import *

from EventReader import EventReader


class EventPreparation(RamTask):
    def __init__(self, mark_as_completed=True):
        RamTask.__init__(self, mark_as_completed)

    def run(self):
        e_path = '/Volumes/rhino_root/data/events/RAM_FR1/R1060M_events.mat'

        e_reader = EventReader(event_file=e_path, eliminate_events_with_no_eeg=True,
                               data_dir_prefix='/Volumes/rhino_root')

        e_reader.read()

        events = e_reader.get_output()

        events = events[(events.type == 'WORD') & (events.recalled == 1)]

        self.pass_object('FR1_events', events)
