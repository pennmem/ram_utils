__author__ = 'm'

import sys
sys.path.append('/Users/m/PTSA_GIT')

import os
import os.path
import re
import numpy as np
from scipy.io import loadmat

from ptsa.data.events import Events
from ptsa.data.rawbinwrapper import RawBinWrapper

from RamPipeline import *

from EventReader import EventReader


# class EventReader(object):
#     def __init__(self, event_file, **kwds):
#         self.__event_file = event_file
#         self.__events = None
#         self.eliminate_events_with_no_eeg = True
#         self.data_dir_prefix = None
#         self.raw_data_root = None
#         self.subject_path = None
#
#
#         possible_argument_list = ['eliminate_events_with_no_eeg', 'data_dir_prefix']
#
#         for argument in possible_argument_list:
#
#             try:
#                 setattr(self, argument, kwds[argument])
#             except LookupError:
#                 print 'did not find the argument: ', argument
#                 pass
#
#
#     def read(self):
#         from MatlabIO import read_single_matlab_matrix_as_numpy_structured_array
#
#         # extract matlab matrix (called 'events') as numpy structured array
#         struct_array = read_single_matlab_matrix_as_numpy_structured_array(self.__event_file, 'events')
#
#         evs = Events(struct_array)
#
#         if self.eliminate_events_with_no_eeg:
#
#             # eliminating events that have no eeg file
#             indicator = np.empty(len(evs), dtype=bool)
#             indicator[:] = False
#             for i, ev in enumerate(evs):
#                 indicator[i] = type(evs[i].eegfile).__name__.startswith('unicode')
#
#             evs = evs[indicator]
#
#         evs = evs.add_fields(esrc=np.dtype(RawBinWrapper))
#
#         import pathlib
#         for ev in evs:
#             # print 'ev=',ev
#             # print pathlib.Path(str(ev.eegfile))
#             try:
#                 eeg_file_path = join(self.data_dir_prefix, str(pathlib.Path(str(ev.eegfile)).parts[1:]))
#                 ev.esrc = RawBinWrapper(eeg_file_path)
#                 self.raw_data_root=str(eeg_file_path)
#             except TypeError:
#                 print 'skipping event with eegfile=',ev.eegfile
#                 pass
#
#         self.subject_path = str(pathlib.Path(eeg_file_path).parts[:-2])
#
#         # attaching
#         # evs.add_fields(esrc=np.dtype(RawBinWrapper))
#
#         # print evs.esrc
#         self.__events = evs
#
#     def get_subject_path(self):
#         return self.subject_path
#
#     def get_raw_data_root(self):
#         return self.raw_data_root
#
#     def get_output(self):
#         return self.__events
#

# e_path = '/Users/m/data/events/RAM_FR1/R1056M_events.mat'
# e_reader = EventReader(event_file=e_path, eliminate_events_with_no_eeg=True, data_dir_prefix='/Users/m')
#
# e_reader.read()
#
# events = e_reader.get_output()

class EventPreparation(RamTask):
    def __init__(self, mark_as_completed=True):
        RamTask.__init__(self, mark_as_completed)

    def run(self):

        e_path = join(self.pipeline.mount_point, 'data/events', self.pipeline.task,self.pipeline.subject+'_events.mat')
        # e_path = '/Users/m/data/events/RAM_FR1/R1056M_events.mat'
        e_reader = EventReader(event_file=e_path, eliminate_events_with_no_eeg=True, data_dir_prefix=self.pipeline.mount_point)

        e_reader.read()

        events = e_reader.get_output()

        events = events[events.type == 'WORD']

        ev_order = np.argsort(events, order=('session','list','mstime'))
        events = events[ev_order]


        print 'events=',events

        self.pass_object(self.pipeline.task+'_events',events)

    #     subject = self.pipeline.subject
    #     task = self.pipeline.task
    #
    #     events = Events(get_events(subject=subject, task=task, path_prefix=self.pipeline.mount_point))
    #
    #     events = correct_eegfile_field(events)
    #     ev_order = np.argsort(events, order=('session','list','mstime'))
    #     events = events[ev_order]
    #
    #     events = self.attach_raw_bin_wrappers(events)
    #
    #     events = events[events.type == 'WORD']
    #
    #     print len(events), task, 'WORD events'
    #
    #     self.pass_object(task+'_events',events)
    #
    # def attach_raw_bin_wrappers(self, events):
    #     eegfiles = np.unique(events.eegfile)
    #     events = events.add_fields(esrc=np.dtype(RawBinWrapper))
    #     for eegfile in eegfiles:
    #         raw_bin_wrapper = RawBinWrapper(self.pipeline.mount_point+eegfile)
    #         # events[events.eegfile == eegfile]['esrc'] = raw_bin_wrapper does NOT work!
    #         inds = np.where(events.eegfile == eegfile)[0]
    #         for i in inds:
    #             events[i]['esrc'] = raw_bin_wrapper
    #     return events



# class EventPreparation(RamTask):
#     def __init__(self, mark_as_completed=True):
#         RamTask.__init__(self, mark_as_completed)
#
#     def run(self):
#         subject = self.pipeline.subject
#         task = self.pipeline.task
#
#         events = Events(get_events(subject=subject, task=task, path_prefix=self.pipeline.mount_point))
#
#         events = correct_eegfile_field(events)
#         ev_order = np.argsort(events, order=('session','list','mstime'))
#         events = events[ev_order]
#
#         events = self.attach_raw_bin_wrappers(events)
#
#         events = events[events.type == 'WORD']
#
#         print len(events), task, 'WORD events'
#
#         self.pass_object(task+'_events',events)
#
#     def attach_raw_bin_wrappers(self, events):
#         eegfiles = np.unique(events.eegfile)
#         events = events.add_fields(esrc=np.dtype(RawBinWrapper))
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

if __name__=='__main__':

        from BaseEventReader import BaseEventReader
        e_path = join('/Volumes/rhino_root', 'data/events/RAM_FR1/R1060M_math.mat')
        # e_path = '/Users/m/data/events/RAM_FR1/R1056M_events.mat'
        e_reader = BaseEventReader(event_file=e_path, eliminate_events_with_no_eeg=True, data_dir_prefix='/Volumes/rhino_root')

        events = e_reader.read()

        events = e_reader.get_output()

