__author__ = 'm'

from RamPipeline import *
from MatlabUtils import *
from MatlabIO import *

from math import isnan
import numpy as np
import numpy.matlib as npm

import time

def indices(a, func):
    return [i for (i, val) in enumerate(a) if func(val)]

def logit(alpha, betas, x):
    return 1.0 / (1.0 + np.exp(-(alpha+np.dot(betas, x))))


class PSReportingTask(RamTask):
    def __init__(self, mark_as_completed=True):
        RamTask.__init__(self, mark_as_completed)

    def run(self):
        PostStimBuff = 50  # buffer in ms to leave following stim offset

        group_psl = deserialize_single_object_from_matlab_format('GroupPSL.mat','GroupPSL')
        # print group_psl[0].Subject
        paramsPS = deserialize_single_object_from_matlab_format('paramsPS.mat','params')
        bpFull = deserialize_single_object_from_matlab_format('bpFull.mat','bpFull')
        bp = deserialize_single_object_from_matlab_format('bp.mat','bp')

        Weights = deserialize_single_object_from_matlab_format('Weights.mat','Weights')

        ps2_events = deserialize_single_object_from_matlab_format('PS2Events.mat','PS2Events')
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
        print 'ps2_sessions=',ps2_sessions

        for SessNum in ps2_sessions:
            SessEv = [ev for ev in ps2_events if ev.session == SessNum]
            timestamps = [ev.mstime for ev in SessEv]

            firstTimestamp = min(timestamps)
            lastTimestamp = min(timestamps)
            SessLength = (lastTimestamp-firstTimestamp)/1000./60.

            SessDate = time.strftime('%Y-%m-%d', time.localtime(lastTimestamp/1000))
            print SessDate

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

            StimTag = None
            for group in group_psl:

                if SessNum in np.array(group.Sessions):
                    StimTag = group.StimElecTag
                    break

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

            n_durs = len(durs)
            n_amps = len(amps)
            n_freqs = len(freqs)

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
            subject_id = 'R1086M'
            # workspace_dir = '/Volumes/RHINO/scratch/busygin/PS2_reporting'
            workspace_dir = '/home1/mswat/scratch/busygin/PS2_reporting'
            for iElec in xrange(len(bp)):
                power_file_name = abspath(join(workspace_dir,subject_id,'power',subject_id, SessName,'%d-%d_Pow_bin_zs.mat'%(bp[iElec].channel[0],bp[iElec].channel[1])))

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

            DeltaMeanAll = np.empty(n_amps)
            DeltaStdevAll = np.empty(n_amps)
            DeltaMeanLow = np.empty(n_amps)
            DeltaStdevLow = np.empty(n_amps)
            DeltaMeanHigh = np.empty(n_amps)
            DeltaStdevHigh = np.empty(n_amps)

            for i,amp in enumerate(amps):
                amp_sel = (amps_ev==amp)

                ProbDiffAll = ProbDiff[amp_sel]

                mean = np.nanmean(ProbDiffAll)
                if isnan(mean): mean = 0.0
                DeltaMeanAll[i] = mean

                stdev = np.nanstd(ProbDiffAll, ddof=1)
                if isnan(stdev): stdev = 0.0
                DeltaStdevAll[i] = stdev

                ProbDiffLow = ProbDiff[amp_sel & low_terc_inds]

                mean = np.nanmean(ProbDiffLow)
                if isnan(mean): mean = 0.0
                DeltaMeanLow[i] = mean

                stdev = np.nanstd(ProbDiffLow, ddof=1)
                if isnan(stdev): stdev = 0.0
                DeltaStdevLow[i] = stdev

                ProbDiffHigh = ProbDiff[amp_sel & high_terc_inds]

                mean = np.nanmean(ProbDiffHigh)
                if isnan(mean): mean = 0.0
                DeltaMeanHigh[i] = mean

                stdev = np.nanstd(ProbDiffHigh, ddof=1)
                if isnan(stdev): stdev = 0.0
                DeltaStdevHigh[i] = stdev

            # PLOT ALL 6 ARRAYS HERE: DeltaMeanAll, DeltaStdevAll, DeltaMeanLow, DeltaStdevLow, DeltaMeanHigh, DeltaStdevHigh
            data_point_indexes = np.arange(1,len(amps)+1)
            # print 'x_axis_amps_counters=',x_axis_amps_counters
            # print 'amps=',amps
            # sys.exit()

            # self.pipeline.add_object_to_pass('amp_all', [x_axis_amps_counters, DeltaMeanAll, DeltaStdevAll, amps])
            # self.pipeline.add_object_to_pass('amp_low', [x_axis_amps_counters, DeltaMeanLow, DeltaStdevLow, amps])
            # self.pipeline.add_object_to_pass('amp_high', [x_axis_amps_counters, DeltaMeanHigh, DeltaStdevHigh, amps])

            DeltaMeanAll_freq = np.empty(n_freqs)
            DeltaStdevAll_freq = np.empty(n_freqs)
            DeltaMeanLow_freq = np.empty(n_freqs)
            DeltaStdevLow_freq = np.empty(n_freqs)
            DeltaMeanHigh_freq = np.empty(n_freqs)
            DeltaStdevHigh_freq = np.empty(n_freqs)

            for i,freq in enumerate(freqs):
                freq_sel = (freqs_ev==freq)

                ProbDiffAll = ProbDiff[freq_sel]

                mean = np.nanmean(ProbDiffAll)
                if isnan(mean): mean = 0.0
                DeltaMeanAll_freq[i] = mean

                stdev = np.nanstd(ProbDiffAll, ddof=1)
                if isnan(stdev): stdev = 0.0
                DeltaStdevAll_freq[i] = stdev

                ProbDiffLow = ProbDiff[freq_sel & low_terc_inds]

                mean = np.nanmean(ProbDiffLow)
                if isnan(mean): mean = 0.0
                DeltaMeanLow_freq[i] = mean

                stdev = np.nanstd(ProbDiffLow, ddof=1)
                if isnan(stdev): stdev = 0.0
                DeltaStdevLow_freq[i] = stdev

                ProbDiffHigh = ProbDiff[freq_sel & high_terc_inds]

                mean = np.nanmean(ProbDiffHigh)
                if isnan(mean): mean = 0.0
                DeltaMeanHigh_freq[i] = mean

                stdev = np.nanstd(ProbDiffHigh, ddof=1)
                if isnan(stdev): stdev = 0.0
                DeltaStdevHigh_freq[i] = stdev

            # PLOT ALL 6 ARRAYS HERE: DeltaMeanAll, DeltaStdevAll, DeltaMeanLow, DeltaStdevLow, DeltaMeanHigh, DeltaStdevHigh
            data_point_indexes_freq = np.arange(1,len(freqs)+1)
            # self.pipeline.add_object_to_pass('freq_all',[freqs,DeltaMeanAll_freq,DeltaStdevAll_freq])
            # self.pipeline.add_object_to_pass('freq_low',[freqs,DeltaMeanLow_freq,DeltaStdevLow_freq])
            # self.pipeline.add_object_to_pass('freq_high',[freqs,DeltaMeanHigh_freq,DeltaStdevHigh_freq])

            from collections import namedtuple
            PlotSpecs = namedtuple(typename='PlotSpecs', field_names='x y yerr x_tick_labels ylim')
            # computting y axis limits
            min_plot = np.min(np.hstack((DeltaMeanAll_freq-DeltaStdevAll_freq, DeltaMeanLow_freq-DeltaStdevLow_freq, DeltaMeanHigh_freq-DeltaStdevHigh_freq, DeltaMeanAll-DeltaStdevAll, DeltaMeanLow-DeltaStdevLow, DeltaMeanHigh-DeltaStdevHigh)))
            max_plot = np.max(np.hstack((DeltaMeanAll_freq+DeltaStdevAll_freq, DeltaMeanLow_freq+DeltaStdevLow_freq, DeltaMeanHigh_freq+DeltaStdevHigh_freq, DeltaMeanAll+DeltaStdevAll, DeltaMeanLow+DeltaStdevLow, DeltaMeanHigh+DeltaStdevHigh)))
            ylim=[min_plot-0.1*(max_plot-min_plot), max_plot+0.1*(max_plot-min_plot)]


            
            # min_amp_array = np.min(np.hstack((DeltaMeanAll-DeltaStdevAll, DeltaMeanLow-DeltaStdevLow, DeltaMeanHigh-DeltaStdevHigh)))
            # max_amp_array = np.max(np.hstack((DeltaMeanAll+DeltaStdevAll, DeltaMeanLow+DeltaStdevLow, DeltaMeanHigh+DeltaStdevHigh)))
            #
            # ylim_amp=[min_amp_array-0.1*(max_amp_array-min_amp_array), max_amp_array+0.1*(max_amp_array-min_amp_array)]


            self.pipeline.add_object_to_pass('amp_all', PlotSpecs(x=data_point_indexes, y=DeltaMeanAll, yerr= DeltaStdevAll, x_tick_labels=amps, ylim=ylim))
            self.pipeline.add_object_to_pass('amp_low', PlotSpecs(x=data_point_indexes, y=DeltaMeanLow, yerr= DeltaStdevLow, x_tick_labels=amps, ylim=ylim))
            self.pipeline.add_object_to_pass('amp_high', PlotSpecs(x=data_point_indexes, y=DeltaMeanHigh, yerr= DeltaStdevHigh, x_tick_labels=amps, ylim=ylim))





            self.pipeline.add_object_to_pass('freq_all', PlotSpecs(x=data_point_indexes_freq, y=DeltaMeanAll_freq, yerr= DeltaStdevAll_freq, x_tick_labels=freqs, ylim=ylim))
            self.pipeline.add_object_to_pass('freq_low', PlotSpecs(x=data_point_indexes_freq, y=DeltaMeanLow_freq, yerr= DeltaStdevLow_freq, x_tick_labels=freqs, ylim=ylim))
            self.pipeline.add_object_to_pass('freq_high', PlotSpecs(x=data_point_indexes_freq, y=DeltaMeanHigh_freq, yerr= DeltaStdevHigh_freq, x_tick_labels=freqs, ylim=ylim))



        ISI_min = np.nanmin([ev.ISI for ev in ps2_events])
        ISI_max = np.nanmax([ev.ISI for ev in ps2_events])
        ISI_mid = (ISI_max+ISI_min) / 2.0
        ISI_halfrange = ISI_max - ISI_mid

        print 'ISI =', ISI_mid, '+/-', ISI_halfrange

        durs = np.unique(all_durs)
        amps = np.unique(all_amps)
        freqs = np.unique(all_freqs)

        n_durs = len(durs)
        n_amps = len(amps)
        n_freqs = len(freqs)

        low_terc_thresh = np.percentile(ProbPreAllSessions, 100.0/3.0)
        high_terc_thresh = np.percentile(ProbPreAllSessions, 2.0*100.0/3.0)

        low_terc_inds = (ProbPreAllSessions<=low_terc_thresh)
        high_terc_inds = (ProbPreAllSessions>=high_terc_thresh)

        DeltaMeanAll = np.empty(n_amps)
        DeltaStdevAll = np.empty(n_amps)
        DeltaMeanLow = np.empty(n_amps)
        DeltaStdevLow = np.empty(n_amps)
        DeltaMeanHigh = np.empty(n_amps)
        DeltaStdevHigh = np.empty(n_amps)

        for i,amp in enumerate(amps):
            amp_sel = (all_amps_ev==amp)

            ProbDiffAll = ProbDiffAllSessions[amp_sel]

            mean = np.nanmean(ProbDiffAll)
            if isnan(mean): mean = 0.0
            DeltaMeanAll[i] = mean

            stdev = np.nanstd(ProbDiffAll, ddof=1)
            if isnan(stdev): stdev = 0.0
            DeltaStdevAll[i] = stdev

            ProbDiffLow = ProbDiffAllSessions[amp_sel & low_terc_inds]

            mean = np.nanmean(ProbDiffLow)
            if isnan(mean): mean = 0.0
            DeltaMeanLow[i] = mean

            stdev = np.nanstd(ProbDiffLow, ddof=1)
            if isnan(stdev): stdev = 0.0
            DeltaStdevLow[i] = stdev

            ProbDiffHigh = ProbDiffAllSessions[amp_sel & high_terc_inds]

            mean = np.nanmean(ProbDiffHigh)
            if isnan(mean): mean = 0.0
            DeltaMeanHigh[i] = mean

            stdev = np.nanstd(ProbDiffHigh, ddof=1)
            if isnan(stdev): stdev = 0.0
            DeltaStdevHigh[i] = stdev

        # PLOT ALL 6 ARRAYS HERE: DeltaMeanAll, DeltaStdevAll, DeltaMeanLow, DeltaStdevLow, DeltaMeanHigh, DeltaStdevHigh


        DeltaMeanAll = np.empty(n_freqs)
        DeltaStdevAll = np.empty(n_freqs)
        DeltaMeanLow = np.empty(n_freqs)
        DeltaStdevLow = np.empty(n_freqs)
        DeltaMeanHigh = np.empty(n_freqs)
        DeltaStdevHigh = np.empty(n_freqs)

        for i,freq in enumerate(freqs):
            freq_sel = (all_freqs_ev==freq)

            ProbDiffAll = ProbDiffAllSessions[freq_sel]

            mean = np.nanmean(ProbDiffAll)
            if isnan(mean): mean = 0.0
            DeltaMeanAll[i] = mean

            stdev = np.nanstd(ProbDiffAll, ddof=1)
            if isnan(stdev): stdev = 0.0
            DeltaStdevAll[i] = stdev

            ProbDiffLow = ProbDiffAllSessions[freq_sel & low_terc_inds]

            mean = np.nanmean(ProbDiffLow)
            if isnan(mean): mean = 0.0
            DeltaMeanLow[i] = mean

            stdev = np.nanstd(ProbDiffLow, ddof=1)
            if isnan(stdev): stdev = 0.0
            DeltaStdevLow[i] = stdev

            ProbDiffHigh = ProbDiffAllSessions[freq_sel & high_terc_inds]

            mean = np.nanmean(ProbDiffHigh)
            if isnan(mean): mean = 0.0
            DeltaMeanHigh[i] = mean

            stdev = np.nanstd(ProbDiffHigh, ddof=1)
            if isnan(stdev): stdev = 0.0
            DeltaStdevHigh[i] = stdev

        # PLOT ALL 6 ARRAYS HERE: DeltaMeanAll, DeltaStdevAll, DeltaMeanLow, DeltaStdevLow, DeltaMeanHigh, DeltaStdevHigh
