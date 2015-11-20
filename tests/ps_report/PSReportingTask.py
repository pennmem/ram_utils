from RamPipeline import *
# from MatlabUtils import *
from MatlabIO import *

import numpy as np
import numpy.matlib as npm

import time

from collections import OrderedDict

from PlotUtils import PlotData

from delta_stat import DeltaStats

__author__ = 'm'


def indices(a, func):
    return [i for (i, val) in enumerate(a) if func(val)]


def logit(alpha, betas, x):
    return 1.0 / (1.0 + np.exp(-(alpha+np.dot(betas, x))))


def is_stim_event(event):
    return event.type in ['STIMULATING','BEGIN_BURST','STIM_SINGLE_PULSE']


class SessionSummary(object):
    def __init__(self):
        self.plot_data_dict = OrderedDict() # {panel_plot_coordinate (0,0) : PlotData}
        self.constant_name = self.constant_value = self.constant_unit = None
        self.stimtag = None
        self.parameter1 = self.parameter2 = None
        self.name = None
        self.date = None
        self.length = None
        self.isi_mid = None
        self.isi_half_range = None


class PSReportingTask(RamTask):
    def __init__(self, mark_as_completed=True):
        RamTask.__init__(self, mark_as_completed)

    def run(self):
        subject_id = self.pipeline.subject_id
        experiment = self.pipeline.experiment

        PostStimBuff = 50  # buffer in ms to leave following stim offset

        w_dir = self.get_workspace_dir()

        paramsPS = deserialize_single_object_from_matlab_format(join(w_dir,'paramsPS.mat'),'params')
        bpFull = deserialize_single_object_from_matlab_format(join(w_dir,'bpFull.mat'),'bpFull')
        bp = deserialize_single_object_from_matlab_format(join(w_dir,'bp.mat'),'bp')

        Weights = deserialize_single_object_from_matlab_format(join(w_dir,'Weights.mat'),'Weights')

        ps_events = deserialize_single_object_from_matlab_format(join(w_dir,'PSEvents.mat'),'PSEvents')
        ps_events_size = len(ps_events)

        for ev in ps_events:
            ev.ISI = np.nan

        for i in xrange(1, ps_events_size):
            ev_curr = ps_events[i]
            ev_prev = ps_events[i-1]
            if ev_curr.session == ev_prev.session:
                if is_stim_event(ev_curr) and is_stim_event(ev_prev):
                    ev_curr.ISI = ev_curr.mstime - ev_prev.mstime

        ps_events = [ev for ev in ps_events if is_stim_event(ev)]

        print 'stim events size = ', len(ps_events)

        ps_sessions  = np.unique([e.session for e in ps_events])
        # print 'ps_sessions=',ps_sessions

        self.pass_object('NUMBER_OF_SESSIONS',len(ps_sessions))
        self.pass_object('NUMBER_OF_ELECTRODES',len(bp))

        session_data = []

        all_durs_ev = np.array([])
        all_amps_ev = np.array([])
        all_freqs_ev = np.array([])
        all_burfs_ev = np.array([])

        all_durs = np.array([])
        all_amps = np.array([])
        all_freqs = np.array([])
        all_burfs = np.array([])

        ProbPostAllSessions = np.array([])
        ProbPreAllSessions = np.array([])
        ProbDiffAllSessions = np.array([])

        session_summary_array = []

        for SessNum in ps_sessions:
            session_summary = SessionSummary() # object that contains all the report related information for a given session
            SessEv = [ev for ev in ps_events if ev.session == SessNum]

            timestamps = [ev.mstime for ev in SessEv]
            firstTimestamp = min(timestamps)
            lastTimestamp = max(timestamps)
            SessLength = (lastTimestamp-firstTimestamp)/60000.0
            SessDate = time.strftime('%Y-%m-%d', time.localtime(lastTimestamp/1000))
            session_data.append([SessNum, SessDate, SessLength])

            SessName = 'Sess%02d' % SessNum

            StimTag = SessEv[0].stimAnodeTag + '-' + SessEv[0].stimCathodeTag
            #self.pipeline.add_object_to_pass('STIMTAG',StimTag)

            ISI_min = np.nanmin([ev.ISI for ev in SessEv])
            ISI_max = np.nanmax([ev.ISI for ev in SessEv])
            ISI_mid = (ISI_max+ISI_min) / 2.0
            ISI_halfrange = ISI_max - ISI_mid

            print 'Session =', SessName, ' StimTag =', StimTag, ' ISI =', ISI_mid, '+/-', ISI_halfrange

            durs_ev = [s.pulse_duration for s in SessEv]
            amps_ev = [s.amplitude for s in SessEv]
            freqs_ev = [s.pulse_frequency for s in SessEv]
            burfs_ev = [s.burst_frequency for s in SessEv]

            all_durs_ev = np.hstack((all_durs_ev, durs_ev))
            all_amps_ev = np.hstack((all_amps_ev, amps_ev))
            all_freqs_ev = np.hstack((all_freqs_ev, freqs_ev))
            all_burfs_ev = np.hstack((all_burfs_ev, burfs_ev))

            durs = np.unique(durs_ev)
            amps = np.unique(amps_ev)
            freqs = np.unique(freqs_ev)
            burfs = np.unique(burfs_ev)

            session_summary.name = SessName
            session_summary.length = SessLength
            session_summary.date = SessDate
            session_summary.stimtag = StimTag
            session_summary.isi_mid = ISI_mid
            session_summary.isi_half_range = ISI_halfrange

            all_durs = np.hstack((all_durs, durs))
            all_amps = np.hstack((all_amps, amps))
            all_freqs = np.hstack((all_freqs, freqs))
            all_burfs = np.hstack((all_burfs, burfs))

            StimOnBin = np.ones(len(SessEv), dtype=np.int)*paramsPS.pow.onsetInd

            PreStimInds = npm.repmat(paramsPS.pow.baseBins-1,len(SessEv),1);

            PostStimBin = np.empty_like(StimOnBin, dtype=np.int)

            PostStimInds = np.empty((len(PostStimBin),len(paramsPS.pow.baseBins)), dtype=np.int)

            for iEv in xrange(len(SessEv)):
                dur = SessEv[iEv].pulse_duration
                if dur == -999:
                    dur = durs[-1]

                PostStimBin[iEv] = indices(paramsPS.pow.timeBins[:,0], lambda x: x <= dur+PostStimBuff)[-1]
                PostStimInds[iEv,:] = range(PostStimBin[iEv], PostStimBin[iEv]+len(paramsPS.pow.baseBins))

            DataMat_PostStim = np.empty((50,len(bp),len(SessEv)))
            DataMat_PreStim = np.empty((50,len(bp),len(SessEv)))

            workspace_dir = self.get_workspace_dir()
            for iElec in xrange(len(bp)):
                power_file_name = abspath(join(workspace_dir,'power',subject_id, SessName,'%d-%d_Pow_bin_zs.mat'%(bp[iElec].channel[0],bp[iElec].channel[1])))
                print power_file_name

                bp_session_reader = MatlabIO()
                bp_session_reader.deserialize(power_file_name)

                PowMat = bp_session_reader.PowMat

                pattern_PostStim = np.empty((50,len(SessEv)))
                pattern_PreStim = np.empty((50,len(SessEv)))
                for iEv in xrange(len(SessEv)):
                    pattern_PostStim[:, iEv] = np.nanmean(PowMat[:, PostStimInds[iEv,:],iEv],1)
                    pattern_PreStim[:, iEv] = np.nanmean(PowMat[:, PreStimInds[iEv,:],iEv],1)

                DataMat_PostStim[:,iElec,:] = pattern_PostStim
                DataMat_PreStim[:,iElec,:] = pattern_PreStim

            DataMat_PostStim = DataMat_PostStim.reshape(50*len(bp),len(SessEv), order='F')
            DataMat_PreStim = DataMat_PreStim.reshape(50*len(bp),len(SessEv), order='F')

            W = np.ravel(Weights.MeanCV) # classifier Beta's
            # Beta_0 Weights.MeanIntercept

            ProbPost = logit(Weights.MeanIntercept, W, DataMat_PostStim)
            ProbPre = logit(Weights.MeanIntercept, W, DataMat_PreStim)
            ProbDiff = ProbPost - ProbPre

            ProbPostAllSessions = np.hstack((ProbPostAllSessions, ProbPost))
            ProbPreAllSessions = np.hstack((ProbPreAllSessions, ProbPre))
            ProbDiffAllSessions = np.hstack((ProbDiffAllSessions, ProbDiff))

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

            delta_stats = DeltaStats(2, ev_vals, param_grid, ProbPre, ProbDiff, 1.0/3.0)
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

        self.pass_object('SESSION_DATA',session_data)
        self.pass_object('session_summary_array',session_summary_array)

        ISI_min = np.nanmin([ev.ISI for ev in ps_events])
        ISI_max = np.nanmax([ev.ISI for ev in ps_events])
        ISI_mid = (ISI_max+ISI_min) / 2.0
        ISI_halfrange = ISI_max - ISI_mid

        print 'ISI =', ISI_mid, '+/-', ISI_halfrange

        self.pass_object('CUMULATIVE_ISI_MID',ISI_mid)
        self.pass_object('CUMULATIVE_ISI_HALF_RANGE',ISI_halfrange)

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

        delta_stats = DeltaStats(2, ev_vals, param_grid, ProbPreAllSessions, ProbDiffAllSessions, 1.0/3.0)
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
