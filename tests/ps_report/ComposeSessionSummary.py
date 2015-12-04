__author__ = 'm'

import numpy as np
import time

from RamPipeline import *
from SessionSummary import SessionSummary

from collections import OrderedDict

from PlotUtils import PlotData

from delta_stat import DeltaStats


class ComposeSessionSummary(RamTask):
    def __init__(self, params, mark_as_completed=True):
        RamTask.__init__(self, mark_as_completed)
        self.params = params

    def restore(self):
        pass

    def run(self):
        experiment = self.pipeline.experiment

        events = self.get_passed_object(experiment+'_events')
        tal_info = self.get_passed_object('tal_info')

        prob_pre = self.get_passed_object('prob_pre')
        prob_diff = self.get_passed_object('prob_diff')

        sessions = np.unique(events.session)

        self.pass_object('NUMBER_OF_SESSIONS', len(sessions))
        self.pass_object('NUMBER_OF_ELECTRODES', len(tal_info))

        session_data = []

        all_durs_ev = np.array([])
        all_amps_ev = np.array([])
        all_freqs_ev = np.array([])
        all_burfs_ev = np.array([])

        all_durs = np.array([])
        all_amps = np.array([])
        all_freqs = np.array([])
        all_burfs = np.array([])

        session_summary_array = []

        prob_array_idx = 0

        for session in sessions:
            session_summary = SessionSummary()

            session_events = events[events.session == session]
            n_sess_events = len(session_events)

            timestamps = session_events.mstime
            first_time_stamp = np.min(timestamps)
            last_time_stamp = np.max(timestamps)
            session_length = (last_time_stamp - first_time_stamp) / 60000.0
            session_date = time.strftime('%Y-%m-%d', time.localtime(last_time_stamp/1000))

            session_data.append([session, session_date, session_length])

            session_name = 'Sess%02d' % session

            stim_tag = session_events[0].stimAnodeTag + '-' + session_events[0].stimCathodeTag

            isi_min = np.nanmin(session_events.isi)
            isi_max = np.nanmax(session_events.isi)
            isi_mid = (isi_max+isi_min) / 2.0
            isi_halfrange = isi_max - isi_mid

            print 'Session =', session_name, ' StimTag =', stim_tag, ' ISI =', isi_mid, '+/-', isi_halfrange

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

            session_prob_pre = prob_pre[prob_array_idx:prob_array_idx+n_sess_events]
            session_prob_diff = prob_diff[prob_array_idx:prob_array_idx+n_sess_events]

            all_durs = np.hstack((all_durs, durs))
            all_amps = np.hstack((all_amps, amps))
            all_freqs = np.hstack((all_freqs, freqs))
            all_burfs = np.hstack((all_burfs, burfs))

            ev_vals = None
            param_grid = None
            if experiment == 'PS1':
                ev_vals = [freqs_ev, durs_ev]
                param_grid = [freqs, durs]
                session_summary.constant_name = 'Amplitude'
                session_summary.constant_value = amps[0]
                session_summary.constant_unit = 'mA'
                session_summary.parameter1 = 'Pulse Frequency'
                session_summary.parameter2 = 'Duration'
            elif experiment == 'PS2':
                ev_vals = [freqs_ev, amps_ev]
                param_grid = [freqs, amps]
                session_summary.constant_name = 'Duration'
                session_summary.constant_value = durs[-1]
                session_summary.constant_unit = 'ms'
                session_summary.parameter1 = 'Pulse Frequency'
                session_summary.parameter2 = 'Amplitude'
            elif experiment == 'PS3':
                ev_vals = [freqs_ev, burfs_ev]
                param_grid = [freqs, burfs]
                session_summary.constant_name = 'Amplitude'
                session_summary.constant_value = amps[0]
                session_summary.constant_unit = 'mA'
                session_summary.parameter1 = 'Pulse Frequency'
                session_summary.parameter2 = 'Burst Frequency'

            delta_stats = DeltaStats(2, ev_vals, param_grid, session_prob_pre, session_prob_diff, 1.0/3.0)
            delta_stats.run()

            data_point_indexes_left = np.arange(1,len(param_grid[0])+1)
            data_point_indexes_right = np.arange(1,len(param_grid[1])+1)

            # computting y axis limits
            min_plot, max_plot = delta_stats.y_range()
            ylim = [min_plot-0.1*(max_plot-min_plot), max_plot+0.1*(max_plot-min_plot)]

            x_tick_labels = [x if x>0 else 'PULSE' for x in param_grid[0]]
            session_summary.plot_data_dict[(0,0)] = PlotData(x=data_point_indexes_left, y=delta_stats.mean_all[0], yerr=delta_stats.stdev_all[0], x_tick_labels=x_tick_labels, ylim=ylim)
            session_summary.plot_data_dict[(1,0)] = PlotData(x=data_point_indexes_left, y=delta_stats.mean_low[0], yerr=delta_stats.stdev_low[0], x_tick_labels=x_tick_labels, ylim=ylim)
            session_summary.plot_data_dict[(2,0)] = PlotData(x=data_point_indexes_left, y=delta_stats.mean_high[0], yerr=delta_stats.stdev_high[0], x_tick_labels=x_tick_labels, ylim=ylim)

            x_tick_labels = param_grid[1]
            session_summary.plot_data_dict[(0,1)] = PlotData(x=data_point_indexes_right, y=delta_stats.mean_all[1], yerr=delta_stats.stdev_all[1], x_tick_labels=x_tick_labels, ylim=ylim)
            session_summary.plot_data_dict[(1,1)] = PlotData(x=data_point_indexes_right, y=delta_stats.mean_low[1], yerr=delta_stats.stdev_low[1], x_tick_labels=x_tick_labels, ylim=ylim)
            session_summary.plot_data_dict[(2,1)] = PlotData(x=data_point_indexes_right, y=delta_stats.mean_high[1], yerr=delta_stats.stdev_high[1], x_tick_labels=x_tick_labels, ylim=ylim)

            session_summary_array.append(session_summary)

            prob_array_idx += len(session_events)

        self.pass_object('SESSION_DATA', session_data)
        self.pass_object('session_summary_array', session_summary_array)

        isi_min = np.nanmin(events.isi)
        isi_max = np.nanmax(events.isi)
        isi_mid = (isi_max+isi_min) / 2.0
        isi_halfrange = isi_max - isi_mid

        print 'ISI =', isi_mid, '+/-', isi_halfrange

        self.pass_object('CUMULATIVE_ISI_MID',isi_mid)
        self.pass_object('CUMULATIVE_ISI_HALF_RANGE',isi_halfrange)

        durs = np.unique(all_durs)
        amps = np.unique(all_amps)
        freqs = np.unique(all_freqs)
        burfs = np.unique(all_burfs)

        ev_vals = None
        param_grid = None
        if experiment == 'PS1':
            ev_vals = [all_freqs_ev, all_durs_ev]
            param_grid = [freqs, durs]
            self.pass_object('CUMULATIVE_PARAMETER1', 'Pulse Frequency')
            self.pass_object('CUMULATIVE_PARAMETER2', 'Duration')
        elif experiment == 'PS2':
            ev_vals = [all_freqs_ev, all_amps_ev]
            param_grid = [freqs, amps]
            self.pass_object('CUMULATIVE_PARAMETER1', 'Pulse Frequency')
            self.pass_object('CUMULATIVE_PARAMETER2', 'Amplitude')
        elif experiment == 'PS3':
            ev_vals = [all_freqs_ev, all_burfs_ev]
            param_grid = [freqs, burfs]
            self.pass_object('CUMULATIVE_PARAMETER1', 'Pulse Frequency')
            self.pass_object('CUMULATIVE_PARAMETER2', 'Burst Frequency')

        delta_stats = DeltaStats(2, ev_vals, param_grid, prob_pre, prob_diff, 1.0/3.0)
        delta_stats.run()

        data_point_indexes_left = np.arange(1,len(param_grid[0])+1)
        data_point_indexes_right = np.arange(1,len(param_grid[1])+1)

        # computing y axis limits
        min_plot, max_plot = delta_stats.y_range()
        ylim = [min_plot-0.1*(max_plot-min_plot), max_plot+0.1*(max_plot-min_plot)]

        cumulative_plot_data_dict = OrderedDict()

        x_tick_labels = [x if x>0 else 'PULSE' for x in param_grid[0]]
        cumulative_plot_data_dict[(0,0)] = PlotData(x=data_point_indexes_left, y=delta_stats.mean_all[0], yerr=delta_stats.stdev_all[0], x_tick_labels=x_tick_labels, ylim=ylim)
        cumulative_plot_data_dict[(1,0)] = PlotData(x=data_point_indexes_left, y=delta_stats.mean_low[0], yerr=delta_stats.stdev_low[0], x_tick_labels=x_tick_labels, ylim=ylim)
        cumulative_plot_data_dict[(2,0)] = PlotData(x=data_point_indexes_left, y=delta_stats.mean_high[0], yerr=delta_stats.stdev_high[0], x_tick_labels=x_tick_labels, ylim=ylim)

        x_tick_labels = param_grid[1]
        cumulative_plot_data_dict[(0,1)] = PlotData(x=data_point_indexes_right, y=delta_stats.mean_all[1], yerr=delta_stats.stdev_all[1], x_tick_labels=x_tick_labels, ylim=ylim)
        cumulative_plot_data_dict[(1,1)] = PlotData(x=data_point_indexes_right, y=delta_stats.mean_low[1], yerr=delta_stats.stdev_low[1], x_tick_labels=x_tick_labels, ylim=ylim)
        cumulative_plot_data_dict[(2,1)] = PlotData(x=data_point_indexes_right, y=delta_stats.mean_high[1], yerr=delta_stats.stdev_high[1], x_tick_labels=x_tick_labels, ylim=ylim)

        self.pass_object('cumulative_plot_data_dict', cumulative_plot_data_dict)
