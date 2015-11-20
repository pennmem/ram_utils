__author__ = 'm'

from sys import argv
import os
import os.path
import re
import numpy as np
from scipy.io import loadmat
from scipy.stats.mstats import zscore

from ptsa.data.events import Events
from ptsa.data.rawbinwrapper import RawBinWrapper

from get_bipolar_subj_elecs import get_bipolar_subj_elecs

from ptsa.wavelet import phase_pow_multi


from RamPipeline import *

class PSEventPreparation(RamTask):
    def __init__(self, task, mark_as_completed=True):
        RamTask.__init__(self, mark_as_completed)
        self.task = task

    def run(self):
        experiment = self.pipeline.experiment
        events = Events(get_events(subject=self.pipeline.subject_id, task='RAM_PS', path_prefix=self.pipeline.mount_point))
        events = events[events.experiment == experiment]

        events = correct_eegfile_field(events)
        events = self.attach_raw_bin_wrappers(events)

        sessions = np.unique(events.session)
        print experiment, 'sessions:', sessions

        events = compute_isi(events)

        is_stim_event_type_vec = np.vectorize(is_stim_event_type)
        events = events[is_stim_event_type_vec(events.type)]
        print len(events), 'stim', experiment, 'events'

        # self.pass_object('task',self.task)
        self.pass_object(experiment+'_events',events)


    def attach_raw_bin_wrappers(self, events):
        eegfiles = np.unique(events.eegfile)
        events = events.add_fields(esrc=np.dtype(RawBinWrapper))
        for eegfile in eegfiles:
            raw_bin_wrapper = RawBinWrapper(self.pipeline.mount_point+eegfile)
            # events[events.eegfile == eegfile]['esrc'] = raw_bin_wrapper does NOT work!
            inds = np.where(events.eegfile == eegfile)[0]
            for i in inds:
                events[i]['esrc'] = raw_bin_wrapper
        return events




dtypes = [('subject','|S12'), ('session',np.int), ('experiment','|S12'), ('list',np.int),
          ('serialpos', np.int), ('type', '|S20'), ('item','|S20'),
          ('itemno',np.int), ('recalled',np.int),
          ('amplitude',np.float), ('burst_i',np.int), ('pulse_frequency',np.int),
          ('burst_frequency',np.int), ('nBursts', np.int), ('pulse_duration', np.int),
          ('mstime',np.float), ('rectime',np.int), ('intrusion',np.int),
          ('isStim', np.int), ('category','|S20'), ('categoryNum', np.int),
          ('stimAnode', np.int), ('stimAnodeTag','|S10'),
          ('stimCathode', np.int), ('stimCathodeTag', '|S10'),
          ('eegfile','|S256'), ('eegoffset', np.int)]


def is_stim_event_type(event_type):
    return event_type in ['STIMULATING', 'BEGIN_BURST', 'STIM_SINGLE_PULSE']


def compute_isi(events):
    print 'Computing ISI'

    events = events.add_fields(isi=np.float)
    events.isi = np.nan

    for i in xrange(1,len(events)):
        curr_ev = events[i]
        if is_stim_event_type(curr_ev.type):
            prev_ev = events[i-1]
            if curr_ev.session == prev_ev.session:
                if is_stim_event_type(prev_ev.type) or prev_ev.type == 'BURST':
                    prev_mstime = prev_ev.mstime
                    if prev_ev.pulse_duration > 0:
                        prev_mstime += prev_ev.pulse_duration
                    curr_ev.isi = curr_ev.mstime - prev_mstime

    return events

def get_events(subject, task, path_prefix):
    event_file = os.path.join(path_prefix + '/data', 'events', task, subject + '_events.mat')
    events = loadmat(event_file, struct_as_record=True, squeeze_me=True)['events']
    new_events = np.rec.recarray(len(events), dtype=dtypes)
    for field in events.dtype.names:
        try:
            new_events[field] = events[field]
        except ValueError:
            print 'ValueError: field =', field
    return new_events

def correct_eegfile_field(events):
    events = events[events.eegfile != '[]']  # remove events with no recording
    data_dir_bad = r'/data.*/' + events[0].subject + r'/eeg'
    data_dir_good = r'/data/eeg/' + events[0].subject + r'/eeg'
    for ev in events:
        ev.eegfile = ev.eegfile.replace('eeg.reref', 'eeg.noreref')
        ev.eegfile = re.sub(data_dir_bad, data_dir_good, ev.eegfile)
    return events


