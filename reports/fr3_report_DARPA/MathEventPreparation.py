from RamPipeline import *

from BaseEventReader import BaseEventReader


class MathEventPreparation(RamTask):
    def __init__(self, mark_as_completed=True):
        RamTask.__init__(self, mark_as_completed)

    def run(self):
        #if self.pipeline.task != 'RAM_FR1':
        #    return

        try:
            e_path = join(self.pipeline.mount_point, 'data/events', self.pipeline.task, self.pipeline.subject+'_math.mat')
            e_reader = BaseEventReader(event_file=e_path, eliminate_events_with_no_eeg=False, data_dir_prefix=self.pipeline.mount_point)

            events = e_reader.read()

            events = events[events.type == 'PROB']

            self.pass_object(self.pipeline.task+'_math_events', events)

        except IOError:

            self.pass_object(self.pipeline.task+'_math_events', None)

