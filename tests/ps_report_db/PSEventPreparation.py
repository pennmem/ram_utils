__author__ = 'm'

import numpy as np
import pandas as pd
from sklearn.externals import joblib

from ptsa.data.readers import BaseEventReader

from ram_utils.RamPipeline import *
from ReportUtils import MissingExperimentError, MissingDataError

from ReportUtils import ReportRamTask


class PSEventPreparation(ReportRamTask):
    def __init__(self, mark_as_completed=True):
        super(PSEventPreparation,self).__init__(mark_as_completed)

    def initialize(self):
        if self.dependency_inventory:
            self.dependency_inventory.add_dependent_resource(resource_name='ps_events',
                                        access_path = ['experiments','ps','events'])
            self.dependency_inventory.add_dependent_resource(resource_name='bipolar',
                                        access_path = ['electrodes','bipolar'])
            self.dependency_inventory.add_dependent_resource(resource_name='bipolar_json',
                                        access_path = ['electrodes','bipolar_json'])

    def restore(self):
        subject = self.pipeline.subject
        experiment = self.pipeline.experiment

        events = joblib.load(self.get_path_to_resource_in_workspace(subject+'-'+experiment+'-ps_events.pkl'))
        self.pass_object(experiment+'_events', events)

    def run(self):
        subject = self.pipeline.subject
        experiment = self.pipeline.experiment

        e_path = os.path.join(self.pipeline.mount_point, 'data/events/RAM_PS', subject + '_events.mat')
        e_reader = BaseEventReader(filename=e_path, eliminate_events_with_no_eeg=True)

        try:
            events = e_reader.read()
            ev_order = np.argsort(events, order=('session','mstime'))
            events = events[ev_order]

            # try:
            #     events = Events(get_events(subject=subject, task='RAM_PS', path_prefix=self.pipeline.mount_point))
            # except IOError:
            #     raise Exception('No parameter search for subject %s' % subject)
            #

            events = events[events.experiment == experiment]

        except Exception:
            # raise MissingDataError('Missing or Corrupt PS event file')

            self.raise_and_log_report_exception(
                                                exception_type='MissingDataError',
                                                exception_message='Missing or Corrupt PS event file'
                                                )


        if len(events) == 0:
            # raise Exception('No %s events for subject %s' % (experiment,subject))

            # raise MissingExperimentError('No %s events for subject %s' % (experiment,subject))
            self.raise_and_log_report_exception(
                                                exception_type='MissingExperimentError',
                                                exception_message='No %s events for subject %s' % (experiment,subject)
                                                )



        sessions = np.unique(events.session)
        print experiment, 'sessions:', sessions

        events = pd.DataFrame.from_records(events)

        events = compute_isi(events)

        is_stim_event_type_vec = np.vectorize(is_stim_event_type)
        stim_mask = is_stim_event_type_vec(events.type)
        if experiment == 'PS3':
            stim_inds = np.where(stim_mask)[0]
            stim_events = pd.DataFrame(events[stim_mask])
            last_burst_inds = stim_inds + stim_events['nBursts'].values
            last_bursts = events.ix[last_burst_inds]
            events = stim_events
            events['train_duration'] = last_bursts['mstime'].values - events['mstime'].values + last_bursts['pulse_duration'].values
        else:
            events = events[stim_mask]

        events = events.to_records(index=False)

        print len(events), 'stim', experiment, 'events'

        joblib.dump(events, self.get_path_to_resource_in_workspace(subject+'-'+experiment+'-ps_events.pkl'))
        self.pass_object(experiment+'_events', events)


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

def is_stim_event_type(event_type):
    return event_type in ['STIMULATING', 'BEGIN_BURST', 'STIM_SINGLE_PULSE']

def compute_isi(events):
    print 'Computing ISI'

    events['isi'] = np.nan

    for i in xrange(1,len(events)):
        curr_ev = events.ix[i]
        if is_stim_event_type(curr_ev.type):
            prev_ev = events.ix[i-1]
            if curr_ev.session == prev_ev.session:
                if is_stim_event_type(prev_ev.type) or prev_ev.type == 'BURST':
                    prev_mstime = prev_ev.mstime
                    if prev_ev.pulse_duration > 0:
                        prev_mstime += prev_ev.pulse_duration
                    dt = curr_ev.mstime - prev_mstime
                    if dt < 7000.0:
                        events.isi.values[i] = dt

    return events
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
# def correct_eegfile_field(events):
#     events = events[events.eegfile != '[]']  # remove events with no recording
#     data_dir_bad = r'/data.*/' + events[0].subject + r'/eeg'
#     data_dir_good = r'/data/eeg/' + events[0].subject + r'/eeg'
#     for ev in events:
#         ev.eegfile = ev.eegfile.replace('eeg.reref', 'eeg.noreref')
#         ev.eegfile = re.sub(data_dir_bad, data_dir_good, ev.eegfile)
#     return events
