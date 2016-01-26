__author__ = 'm'

import os
import os.path
import re
import numpy as np
from scipy.io import loadmat

from ptsa.data.events import Events
from ptsa.data.rawbinwrapper import RawBinWrapper

from RamPipeline import *


class FR1EventPreparation(RamTask):
    def __init__(self, mark_as_completed=True):
        RamTask.__init__(self, mark_as_completed)

    def run(self):
        task = self.pipeline.task

        from ptsa.data.readers import BaseEventReader
        e_path = os.path.join(self.pipeline.mount_point , 'data', 'events', task, self.pipeline.subject + '_events.mat')
        e_reader = BaseEventReader(event_file=e_path, eliminate_events_with_no_eeg=True, use_ptsa_events_class=False)

        events = e_reader.read()


        # events = Events(get_events(subject=self.pipeline.subject, task=task, path_prefix=self.pipeline.mount_point))

        # events = correct_eegfile_field(events)
        # ev_order = np.argsort(events, order=('session','list','mstime'))
        # events = events[ev_order]
        #
        # events = self.attach_raw_bin_wrappers(events)

        events = events[events.type == 'WORD']

        print len(events), task, 'WORD events'

        self.pass_object(task+'_events', events)

#     def attach_raw_bin_wrappers(self, events):
#         eegfiles = np.unique(events.eegfile)
#         events = Events(events).add_fields(esrc=np.dtype(RawBinWrapper))
#         for eegfile in eegfiles:
#             raw_bin_wrapper = RawBinWrapper(self.pipeline.mount_point+eegfile)
#             # events[events.eegfile == eegfile]['esrc'] = raw_bin_wrapper does NOT work!
#             inds = np.where(events.eegfile == eegfile)[0]
#             for i in inds:
#                 events[i]['esrc'] = raw_bin_wrapper
#         return events
#
#
# dtypes = [('subject','|S12'), ('session',np.int), ('experiment','|S12'), ('list',np.int),
#           ('serialpos', np.int), ('type', '|S20'), ('item','|S20'),
#           ('itemno',np.int), ('recalled',np.int),
#           ('amplitude',np.float), ('burst_i',np.int), ('pulse_frequency',np.int),
#           ('burst_frequency',np.int), ('nBursts', np.int), ('pulse_duration', np.int),
#           ('mstime',np.float), ('rectime',np.int), ('intrusion',np.int),
#           ('isStim', np.int), ('category','|S20'), ('categoryNum', np.int),
#           ('stimAnode', np.int), ('stimAnodeTag','|S10'),
#           ('stimCathode', np.int), ('stimCathodeTag', '|S10'),
#           ('eegfile','|S256'), ('eegoffset', np.int)]
#
#
# def get_events(subject, task, path_prefix):
#     event_file = os.path.join(path_prefix + '/data', 'events', task, subject + '_events.mat')
#     events = loadmat(event_file, struct_as_record=True, squeeze_me=True)['events']
#     new_events = np.rec.recarray(len(events), dtype=dtypes)
#     for field in events.dtype.names:
#         try:
#             new_events[field] = events[field]
#         except ValueError:
#             print 'ValueError: field =', field
#     return new_events
#
#
# def correct_eegfile_field(events):
#     events = events[events.eegfile != '[]']  # remove events with no recording
#     data_dir_bad = r'/data.*/' + events[0].subject + r'/eeg'
#     data_dir_good = r'/data/eeg/' + events[0].subject + r'/eeg'
#     for ev in events:
#         ev.eegfile = ev.eegfile.replace('eeg.reref', 'eeg.noreref')
#         ev.eegfile = re.sub(data_dir_bad, data_dir_good, ev.eegfile)
#     return events
#
#
# def attach_raw_bin_wrappers(events):
#     eegfiles = np.unique(events.eegfile)
#     events = events.add_fields(esrc=np.dtype(RawBinWrapper))
#     for eegfile in eegfiles:
#         raw_bin_wrapper = RawBinWrapper(Params.path_prefix+eegfile)
#         # events[events.eegfile == eegfile]['esrc'] = raw_bin_wrapper does NOT work!
#         inds = np.where(events.eegfile == eegfile)[0]
#         for i in inds:
#             events[i]['esrc'] = raw_bin_wrapper
#     return events
