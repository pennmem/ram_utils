__author__ = 'm'

from RamPipeline import *
from MatlabUtils import *
from MatlabIO import *

from math import isnan
import numpy as np
import numpy.matlib as npm

import time

from collections import namedtuple


def indices(a, func):
    return [i for (i, val) in enumerate(a) if func(val)]

def logit(alpha, betas, x):
    return 1.0 / (1.0 + np.exp(-(alpha+np.dot(betas, x))))

def calculate_plot(event_vals, param_grid, ProbDiff, low_terc_inds, high_terc_inds):
    n = len(param_grid)

    DeltaMeanAll = np.empty(n)
    DeltaStdevAll = np.empty(n)
    DeltaMeanLow = np.empty(n)
    DeltaStdevLow = np.empty(n)
    DeltaMeanHigh = np.empty(n)
    DeltaStdevHigh = np.empty(n)

    for i,param in enumerate(param_grid):
        param_sel = (event_vals==param)

        ProbDiffAll = ProbDiff[param_sel]

        mean = np.nanmean(ProbDiffAll)
        if isnan(mean): mean = 0.0
        DeltaMeanAll[i] = mean

        stdev = np.nanstd(ProbDiffAll, ddof=1)
        if isnan(stdev): stdev = 0.0
        DeltaStdevAll[i] = stdev

        ProbDiffLow = ProbDiff[param_sel & low_terc_inds]

        mean = np.nanmean(ProbDiffLow)
        if isnan(mean): mean = 0.0
        DeltaMeanLow[i] = mean

        stdev = np.nanstd(ProbDiffLow, ddof=1)
        if isnan(stdev): stdev = 0.0
        DeltaStdevLow[i] = stdev

        ProbDiffHigh = ProbDiff[param_sel & high_terc_inds]

        mean = np.nanmean(ProbDiffHigh)
        if isnan(mean): mean = 0.0
        DeltaMeanHigh[i] = mean

        stdev = np.nanstd(ProbDiffHigh, ddof=1)
        if isnan(stdev): stdev = 0.0
        DeltaStdevHigh[i] = stdev

    return DeltaMeanAll, DeltaStdevAll, DeltaMeanLow, DeltaStdevLow, DeltaMeanHigh, DeltaStdevHigh


class PSReportingTask(RamTask):
    def __init__(self, mark_as_completed=True):
        RamTask.__init__(self, mark_as_completed)

    def run(self):
        PostStimBuff = 50  # buffer in ms to leave following stim offset

        # group_psl = deserialize_single_object_from_matlab_format('GroupPSL.mat','GroupPSL')
        # print group_psl[0].Subject

        w_dir = self.get_workspace_dir()

        paramsPS = deserialize_single_object_from_matlab_format(join(w_dir,'paramsPS.mat'),'params')
        bpFull = deserialize_single_object_from_matlab_format(join(w_dir,'bpFull.mat'),'bpFull')
        bp = deserialize_single_object_from_matlab_format(join(w_dir,'bp.mat'),'bp')

        Weights = deserialize_single_object_from_matlab_format(join(w_dir,'Weights.mat'),'Weights')

        ps2_events = deserialize_single_object_from_matlab_format(join(w_dir,'PS2Events.mat'),'PS2Events')
        ps2_events_size = len(ps2_events)

        for ev in ps2_events:
            ev.ISI = np.nan

        for i in xrange(1, ps2_events_size):
            ev_curr = ps2_events[i]
            ev_prev = ps2_events[i-1]
            #print 'i=',i,' event = ', ev_curr, ' event_type = ', ev_curr.type, ' session = ',ev_curr.session
            if ev_curr.session == ev_prev.session:
                if ev_curr.type=='STIMULATING' and ev_prev.type=='STIMULATING':
                    ev_curr.ISI = ev_curr.mstime - ev_prev.mstime

        indicator = np.ones(ps2_events_size, dtype=bool)

        for i, ind_flag in enumerate(indicator):
            if ps2_events[i].type!='STIMULATING':
                indicator[i]=False

        ps2_events = ps2_events[indicator]

        print 'stimulation_events_size = ',len(ps2_events)

        ps2_sessions  = np.unique([e.session for e in ps2_events])
        # print 'ps2_sessions=',ps2_sessions

        self.pipeline.add_object_to_pass('NUMBER_OF_SESSIONS',str(len(ps2_sessions)))
        self.pipeline.add_object_to_pass('NUMBER_OF_ELECTRODES',str(len(bp)))

        session_data = []

        for SessNum in ps2_sessions:
            SessEv = [ev for ev in ps2_events if ev.session == SessNum]
            timestamps = [ev.mstime for ev in SessEv]

            firstTimestamp = min(timestamps)
            lastTimestamp = max(timestamps)
            SessLength = (lastTimestamp-firstTimestamp)/1000./60.

            SessDate = time.strftime('%Y-%m-%d', time.localtime(lastTimestamp/1000))
            print SessDate
            session_data.append([SessNum,SessDate, SessLength])


        self.pipeline.add_object_to_pass('SESSION_DATA',session_data)

        all_durs_ev = np.array([])
        all_amps_ev = np.array([])
        all_freqs_ev = np.array([])

        all_durs = np.array([])
        all_amps = np.array([])
        all_freqs = np.array([])

        ProbPostAllSessions = np.array([])
        ProbPreAllSessions = np.array([])
        ProbDiffAllSessions = np.array([])

        for SessNum in ps2_sessions:

            SessEv = [ev for ev in ps2_events if ev.session == SessNum]

            SessName = 'Sess%02d' % SessNum

            print SessName

            StimTag = SessEv[0].stimAnodeTag + '-' + SessEv[0].stimCathodeTag
            self.pipeline.add_object_to_pass('STIMTAG',StimTag)


            # for group in group_psl:
            #
            #     if SessNum in np.array(group.Sessions):
            #         StimTag = group.StimElecTag
            #         break

            ISI_min = np.nanmin([ev.ISI for ev in SessEv])
            ISI_max = np.nanmax([ev.ISI for ev in SessEv])
            ISI_mid = (ISI_max+ISI_min) / 2.0
            ISI_halfrange = ISI_max - ISI_mid

            print 'Session =', SessName, ' StimTag =', StimTag, ' ISI =', ISI_mid, '+/-', ISI_halfrange

            durs_ev = [s.pulse_duration for s in SessEv]
            amps_ev = [s.amplitude for s in SessEv]
            freqs_ev = [s.pulse_frequency for s in SessEv]

            all_durs_ev = np.hstack((all_durs_ev, durs_ev))
            all_amps_ev = np.hstack((all_amps_ev, amps_ev))
            all_freqs_ev = np.hstack((all_freqs_ev, freqs_ev))

            durs = np.unique(durs_ev)
            amps = np.unique(amps_ev)
            freqs = np.unique(freqs_ev)


            self.pipeline.add_object_to_pass('DURATION',str(durs[0]))



            all_durs = np.hstack((all_durs, durs))
            all_amps = np.hstack((all_amps, amps))
            all_freqs = np.hstack((all_freqs, freqs))

            StimOnBin = np.ones(len(SessEv), dtype=np.int)*paramsPS.pow.onsetInd

            PreStimInds = npm.repmat(paramsPS.pow.baseBins-1,len(SessEv),1);

            PostStimBin = np.empty_like(StimOnBin, dtype=np.int)

            PostStimInds = np.empty((len(PostStimBin),len(paramsPS.pow.baseBins)), dtype=np.int)

            for iEv in xrange(len(SessEv)):
                PostStimBin[iEv] = indices(paramsPS.pow.timeBins[:,0], lambda x: x <= SessEv[iEv].pulse_duration+PostStimBuff)[-1]
                PostStimInds[iEv,:] = range(PostStimBin[iEv], PostStimBin[iEv]+len(paramsPS.pow.baseBins))

            DataMat_PostStim = np.empty((50,len(bp),len(SessEv)));
            DataMat_PreStim = np.empty((50,len(bp),len(SessEv)));

            # set it up to point to your local directory with report output and make sure
            subject_id = self.pipeline.subject_id
            # workspace_dir = '/Volumes/RHINO/scratch/busygin/PS2_reporting'
            # workspace_dir = '/home1/mswat/scratch/busygin/PS2_reporting'

            workspace_dir = self.get_workspace_dir()
            for iElec in xrange(len(bp)):
                power_file_name = abspath(join(workspace_dir,'power',subject_id, SessName,'%d-%d_Pow_bin_zs.mat'%(bp[iElec].channel[0],bp[iElec].channel[1])))

                bp_session_reader = MatlabIO()
                bp_session_reader.deserialize(power_file_name)

                PowMat = bp_session_reader.PowMat

                pattern_PostStim = np.empty((50,len(SessEv)))
                pattern_PreStim = np.empty((50,len(SessEv)))
                for iEv in xrange(len(SessEv)):
                    pattern_PostStim[:, iEv] = np.nanmean(PowMat[:, PostStimInds[iEv,:],iEv],1)
                    pattern_PreStim[:, iEv] = np.nanmean(PowMat[:, PreStimInds[iEv,:],iEv],1)

                DataMat_PostStim[:,iElec,:] = pattern_PostStim;
                DataMat_PreStim[:,iElec,:] = pattern_PreStim;

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

            low_terc_thresh = np.percentile(ProbPre, 100.0/3.0)
            high_terc_thresh = np.percentile(ProbPre, 2.0*100.0/3.0)

            low_terc_inds = (ProbPre<=low_terc_thresh)
            high_terc_inds = (ProbPre>=high_terc_thresh)

            DeltaMeanAll_amp, DeltaStdevAll_amp, DeltaMeanLow_amp, DeltaStdevLow_amp, DeltaMeanHigh_amp, DeltaStdevHigh_amp = calculate_plot(amps_ev, amps, ProbDiff, low_terc_inds, high_terc_inds)
            DeltaMeanAll_freq, DeltaStdevAll_freq, DeltaMeanLow_freq, DeltaStdevLow_freq, DeltaMeanHigh_freq, DeltaStdevHigh_freq = calculate_plot(freqs_ev, freqs, ProbDiff, low_terc_inds, high_terc_inds)

            data_point_indexes_amp = np.arange(1,len(amps)+1)
            # print 'x_axis_amps_counters=',x_axis_amps_counters
            # print 'amps=',amps
            # sys.exit()

            # self.pipeline.add_object_to_pass('amp_all', [x_axis_amps_counters, DeltaMeanAll, DeltaStdevAll, amps])
            # self.pipeline.add_object_to_pass('amp_low', [x_axis_amps_counters, DeltaMeanLow, DeltaStdevLow, amps])
            # self.pipeline.add_object_to_pass('amp_high', [x_axis_amps_counters, DeltaMeanHigh, DeltaStdevHigh, amps])

            data_point_indexes_freq = np.arange(1,len(freqs)+1)
            # self.pipeline.add_object_to_pass('freq_all',[freqs,DeltaMeanAll_freq,DeltaStdevAll_freq])
            # self.pipeline.add_object_to_pass('freq_low',[freqs,DeltaMeanLow_freq,DeltaStdevLow_freq])
            # self.pipeline.add_object_to_pass('freq_high',[freqs,DeltaMeanHigh_freq,DeltaStdevHigh_freq])

            PlotSpecs = namedtuple(typename='PlotSpecs', field_names='x y yerr x_tick_labels ylim')
            # computting y axis limits
            min_plot = np.min(np.hstack((DeltaMeanAll_freq-DeltaStdevAll_freq, DeltaMeanLow_freq-DeltaStdevLow_freq, DeltaMeanHigh_freq-DeltaStdevHigh_freq, DeltaMeanAll_amp-DeltaStdevAll_amp, DeltaMeanLow_amp-DeltaStdevLow_amp, DeltaMeanHigh_amp-DeltaStdevHigh_amp)))
            max_plot = np.max(np.hstack((DeltaMeanAll_freq+DeltaStdevAll_freq, DeltaMeanLow_freq+DeltaStdevLow_freq, DeltaMeanHigh_freq+DeltaStdevHigh_freq, DeltaMeanAll_amp+DeltaStdevAll_amp, DeltaMeanLow_amp+DeltaStdevLow_amp, DeltaMeanHigh_amp+DeltaStdevHigh_amp)))
            ylim=[min_plot-0.1*(max_plot-min_plot), max_plot+0.1*(max_plot-min_plot)]


            
            # min_amp_array = np.min(np.hstack((DeltaMeanAll-DeltaStdevAll, DeltaMeanLow-DeltaStdevLow, DeltaMeanHigh-DeltaStdevHigh)))
            # max_amp_array = np.max(np.hstack((DeltaMeanAll+DeltaStdevAll, DeltaMeanLow+DeltaStdevLow, DeltaMeanHigh+DeltaStdevHigh)))
            #
            # ylim_amp=[min_amp_array-0.1*(max_amp_array-min_amp_array), max_amp_array+0.1*(max_amp_array-min_amp_array)]


            self.pipeline.add_object_to_pass('amp_all', PlotSpecs(x=data_point_indexes_amp, y=DeltaMeanAll_amp, yerr= DeltaStdevAll_amp, x_tick_labels=amps, ylim=ylim))
            self.pipeline.add_object_to_pass('amp_low', PlotSpecs(x=data_point_indexes_amp, y=DeltaMeanLow_amp, yerr= DeltaStdevLow_amp, x_tick_labels=amps, ylim=ylim))
            self.pipeline.add_object_to_pass('amp_high', PlotSpecs(x=data_point_indexes_amp, y=DeltaMeanHigh_amp, yerr= DeltaStdevHigh_amp, x_tick_labels=amps, ylim=ylim))


            self.pipeline.add_object_to_pass('freq_all', PlotSpecs(x=data_point_indexes_freq, y=DeltaMeanAll_freq, yerr= DeltaStdevAll_freq, x_tick_labels=freqs, ylim=ylim))
            self.pipeline.add_object_to_pass('freq_low', PlotSpecs(x=data_point_indexes_freq, y=DeltaMeanLow_freq, yerr= DeltaStdevLow_freq, x_tick_labels=freqs, ylim=ylim))
            self.pipeline.add_object_to_pass('freq_high', PlotSpecs(x=data_point_indexes_freq, y=DeltaMeanHigh_freq, yerr= DeltaStdevHigh_freq, x_tick_labels=freqs, ylim=ylim))



        ISI_min = np.nanmin([ev.ISI for ev in ps2_events])
        ISI_max = np.nanmax([ev.ISI for ev in ps2_events])
        ISI_mid = (ISI_max+ISI_min) / 2.0
        ISI_halfrange = ISI_max - ISI_mid

        print 'ISI =', ISI_mid, '+/-', ISI_halfrange
        self.pipeline.add_object_to_pass('ISI_MID',str(ISI_mid))
        self.pipeline.add_object_to_pass('ISI_HALF_RANGE',str(ISI_halfrange))

        durs = np.unique(all_durs)
        amps = np.unique(all_amps)
        freqs = np.unique(all_freqs)

        low_terc_thresh = np.percentile(ProbPreAllSessions, 100.0/3.0)
        high_terc_thresh = np.percentile(ProbPreAllSessions, 2.0*100.0/3.0)

        low_terc_inds = (ProbPreAllSessions<=low_terc_thresh)
        high_terc_inds = (ProbPreAllSessions>=high_terc_thresh)

        TotDeltaMeanAll_amp, TotDeltaStdevAll_amp, TotDeltaMeanLow_amp, TotDeltaStdevLow_amp, TotDeltaMeanHigh_amp, TotDeltaStdevHigh_amp = calculate_plot(all_amps_ev, amps, ProbDiffAllSessions, low_terc_inds, high_terc_inds)
        TotDeltaMeanAll_freq, TotDeltaStdevAll_freq, TotDeltaMeanLow_freq, TotDeltaStdevLow_freq, TotDeltaMeanHigh_freq, TotDeltaStdevHigh_freq = calculate_plot(all_freqs_ev, freqs, ProbDiffAllSessions, low_terc_inds, high_terc_inds)

        data_point_indexes_amp = np.arange(1,len(amps)+1)
        data_point_indexes_freq = np.arange(1,len(freqs)+1)

        PlotSpecs = namedtuple(typename='PlotSpecs', field_names='x y yerr x_tick_labels ylim')

        # computting y axis limits
        min_plot = np.min(np.hstack((TotDeltaMeanAll_freq-TotDeltaStdevAll_freq, TotDeltaMeanLow_freq-TotDeltaStdevLow_freq, TotDeltaMeanHigh_freq-TotDeltaStdevHigh_freq, TotDeltaMeanAll_amp-TotDeltaStdevAll_amp, TotDeltaMeanLow_amp-TotDeltaStdevLow_amp, TotDeltaMeanHigh_amp-TotDeltaStdevHigh_amp)))
        max_plot = np.max(np.hstack((TotDeltaMeanAll_freq+TotDeltaStdevAll_freq, TotDeltaMeanLow_freq+TotDeltaStdevLow_freq, TotDeltaMeanHigh_freq+TotDeltaStdevHigh_freq, TotDeltaMeanAll_amp+TotDeltaStdevAll_amp, TotDeltaMeanLow_amp+TotDeltaStdevLow_amp, TotDeltaMeanHigh_amp+TotDeltaStdevHigh_amp)))
        ylim=[min_plot-0.1*(max_plot-min_plot), max_plot+0.1*(max_plot-min_plot)]

        self.pipeline.add_object_to_pass('tot_amp_all', PlotSpecs(x=data_point_indexes_amp, y=TotDeltaMeanAll_amp, yerr= TotDeltaStdevAll_amp, x_tick_labels=amps, ylim=ylim))
        self.pipeline.add_object_to_pass('tot_amp_low', PlotSpecs(x=data_point_indexes_amp, y=TotDeltaMeanLow_amp, yerr= TotDeltaStdevLow_amp, x_tick_labels=amps, ylim=ylim))
        self.pipeline.add_object_to_pass('tot_amp_high', PlotSpecs(x=data_point_indexes_amp, y=TotDeltaMeanHigh_amp, yerr= TotDeltaStdevHigh_amp, x_tick_labels=amps, ylim=ylim))


        self.pipeline.add_object_to_pass('tot_freq_all', PlotSpecs(x=data_point_indexes_freq, y=TotDeltaMeanAll_freq, yerr= TotDeltaStdevAll_freq, x_tick_labels=freqs, ylim=ylim))
        self.pipeline.add_object_to_pass('tot_freq_low', PlotSpecs(x=data_point_indexes_freq, y=TotDeltaMeanLow_freq, yerr= TotDeltaStdevLow_freq, x_tick_labels=freqs, ylim=ylim))
        self.pipeline.add_object_to_pass('tot_freq_high', PlotSpecs(x=data_point_indexes_freq, y=TotDeltaMeanHigh_freq, yerr= TotDeltaStdevHigh_freq, x_tick_labels=freqs, ylim=ylim))
