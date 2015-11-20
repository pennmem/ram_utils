__author__ = 'm'

import numpy as np
import time


from RamPipeline import *
from SessionSummary import SessionSummary

class ComposeSessionSummary(RamTask):
    def __init__(self, params, task, mark_as_completed=True):
        RamTask.__init__(self, mark_as_completed)
        self.task = task
        self.params = params

    def run(self):
        experiment = self.pipeline.experiment
        subject = self.pipeline.subject_id

        events = self.get_passed_object(experiment+'_events')

        sessions = np.unique(events.session)

        session_summary_array = []

        all_durs_ev = np.array([])
        all_amps_ev = np.array([])
        all_freqs_ev = np.array([])
        all_burfs_ev = np.array([])

        all_durs = np.array([])
        all_amps = np.array([])
        all_freqs = np.array([])
        all_burfs = np.array([])

        for session in sessions:
            session_summary = SessionSummary()
            session_summary_array.append(session_summary)

            session_events = events[events.session == session]

            timestamps = session_events.mstime
            first_time_stamp = np.min(timestamps)
            last_time_stamp = np.max(timestamps)
            session_length = last_time_stamp - first_time_stamp
            session_date = time.strftime('%Y-%m-%d', time.localtime(last_time_stamp/1000))


            session_name = 'Sess%02d' % session

            stim_tag = session_events[0].stimAnodeTag + '-' + session_events[0].stimCathodeTag

            isi_min = np.nanmin(session_events.isi)
            isi_max = np.nanmax(session_events.isi)
            isi_mid = (isi_max+isi_min) / 2.0
            isi_halfrange = isi_max - isi_mid

            #print 'Session =', SessName, ' StimTag =', StimTag, ' ISI =', ISI_mid, '+/-', ISI_halfrange

            durs_ev = session_events.pulse_duration
            amps_ev = session_events.amplitude
            freqs_ev = session_events.pulse_frequency
            burfs_ev = session_events.burst_frequency

            all_durs_ev = np.hstack((all_durs_ev, durs_ev))
            all_amps_ev = np.hstack((all_amps_ev, amps_ev))
            all_freqs_ev = np.hstack((all_freqs_ev, freqs_ev))
            all_burfs_ev = np.hstack((all_burfs_ev, burfs_ev))

            durs = np.unique(durs_ev)
            amps = np.unique(amps_ev)
            freqs = np.unique(freqs_ev)
            burfs = np.unique(burfs_ev)

            session_summary.name = session_name
            session_summary.length = session_length
            session_summary.date = session_date
            session_summary.stimtag = stim_tag
            session_summary.isi_mid = isi_mid
            session_summary.isi_half_range = isi_halfrange

            if experiment == 'PS1':
                session_summary.constant_name = 'Amplitude'
                session_summary.constant_value = amps[0]
                session_summary.constant_unit = 'mA'
                session_summary.parameter1 = 'Pulse Frequency'
                session_summary.parameter2 = 'Duration'
            elif experiment == 'PS2':
                session_summary.constant_name = 'Duration'
                session_summary.constant_value = durs[-1]
                session_summary.constant_unit = 'ms'
                session_summary.parameter1 = 'Pulse Frequency'
                session_summary.parameter2 = 'Amplitude'
            elif experiment == 'PS3':
                session_summary.constant_name = 'Amplitude'
                session_summary.constant_value = amps[0]
                session_summary.constant_unit = 'mA'
                session_summary.parameter1 = 'Pulse Frequency'
                session_summary.parameter2 = 'Burst Frequency'







