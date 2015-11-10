__author__ = 'm'

import numpy as np
import numpy.matlib as npm

from MatlabIO import *



def indices(a, func):
    return [i for (i, val) in enumerate(a) if func(val)]



if __name__ == "__main__":
    import sys
    from os.path import *



    PostStimBuff = 50  # buffer in ms to leave following stim offset

    group_psl = deserialize_single_object_from_matlab_format('GroupPSL.mat','GroupPSL')
    # print group_psl[0].Subject
    paramsPS = deserialize_single_object_from_matlab_format('paramsPS.mat','params')
    bpFull = deserialize_single_object_from_matlab_format('bpFull.mat','bpFull')
    bp = deserialize_single_object_from_matlab_format('bp.mat','bp')

    ps2_events = deserialize_single_object_from_matlab_format('PS2Events.mat','PS2Events')
    ps2_events_size = len(ps2_events)


    # Weights = deserialize_single_object_from_matlab_format('Weights.mat','Weights')




    for i in xrange(1, ps2_events_size):

        ev_curr = ps2_events[i]
        ev_prev = ps2_events[i-1]
        print 'i=',i,' event = ', ev_curr, ' event_type = ', ev_curr.type, ' session = ',ev_curr.session
        if ev_curr.session == ev_prev.session:
            if ev_curr.type=='STIMULATING' and ev_prev.type=='STIMULATING':
                print 'isi = ', ev_curr.mstime - ev_prev.mstime


    print 'ps2_events type = ', type(ps2_events), ' ', ps2_events.dtype

    indicator = np.ones(ps2_events_size , dtype=bool)



    for i, ind_flag in enumerate(indicator):
        if ps2_events[i].type!='STIMULATING':
            indicator[i]=False

    ps2_events = ps2_events[indicator]

    print 'stimulation_events=',ps2_events
    print 'stimulation_events_size = ',len(ps2_events)

    ps2_sessions  = set([a.session for a in ps2_events])
    print 'ps2_sessions=',ps2_sessions

    # print type(ps2_events[0])

    # filtered_events = ps2_events[np.where(ps2_events=='STIMULATING')]
    #     if strcmp(PS2Events(i-1).type,'STIMULATING') & strcmp(PS2Events(i).type,'STIMULATING') & PS2Events(i-1).session==PS2Events(i).session
    #         PS2Events(i).ISI = PS2Events(i).mstime - PS2Events(i-1).mstime;
    #     else
    #         PS2Events(i).ISI = nan;
    #     end
    # end


    for SessNum in ps2_sessions:
        SessEv = [ev for ev in ps2_events if ev.session == SessNum]
        timestamps = [ev.mstime for ev in SessEv]

        firstTimestamp = min(timestamps)
        lastTimestamp = min(timestamps)
        SessLength = (lastTimestamp-firstTimestamp)/1000./60.

        import time
        SessDate = time.strftime('%Y-%m-%d', time.localtime(lastTimestamp/1000))
        print SessDate

    import string

    for SessNum in ps2_sessions:

        SessEv = [ev for ev in ps2_events if ev.session == SessNum]


        SessName = 'Sess'+string.zfill(str(SessNum),2)

        StimTag = None
        for group in group_psl:

            if SessNum in np.array(group.Sessions):
                StimTag = group.StimElecTag
                break

        print 'StimTag=',StimTag

        #
        # for iPSL=1:length(GroupPSL)
        #     if any(GroupPSL(iPSL).Sessions==SessNum)
        #         StimTag = GroupPSL(iPSL).StimElecTag;
        #         break;
        #     end
        # end


        dur = SessEv[0].pulse_duration
        print dur




        StimOnBin = np.ones(len(SessEv), dtype=np.int)*paramsPS.pow.onsetInd

        PreStimInds = npm.repmat(paramsPS.pow.baseBins,len(SessEv),1);

        PostStimBin = np.empty_like(StimOnBin, dtype=np.int)


        PostStimInds = np.empty((len(PostStimBin),len(paramsPS.pow.baseBins)))

        for iEv in xrange(len(SessEv)):

            inds = indices(paramsPS.pow.timeBins[:,0], lambda x: x <= SessEv[iEv].pulse_duration+PostStimBuff)[-1]
            print inds
            PostStimInds[iEv,:] = range(PostStimBin[iEv], PostStimBin[iEv]+len(paramsPS.pow.baseBins))



        DataMat = None;

        workspace_dir = '/home1/mswat/scratch/py_run_4/'
        subject_id = 'R1086M'

        for iElec in xrange(len(bp)):
            print iElec
            # power_file_name = abspath(join(workspace_dir,subject_id,'power',subject_id, SessName,'%d-%d_Pow_bin_zs.mat'%(bp[iElec].channel[0],bp[iElec].channel[1])))

            power_file_name = abspath(join(SessName,'%d-%d_Pow_bin_zs.mat'%(bp[iElec].channel[0],bp[iElec].channel[1])))
            print 'power_file_name=',power_file_name

            bp_session_reader = MatlabIO()
            bp_session_reader.deserialize(power_file_name)




            print dir(bp_session_reader)

            PowMat = bp_session_reader.PowMat

            pattern_PostStim = np.empty((50,len(SessEv)))
            pattern_PreStim = np.empty((50,len(SessEv)))
            for iEv in xrange(len(SessEv)):
                pattern_PostStim[:, iEv] = np.nanmean(PowMat[: , PreStimInds[iEv,:],iEv],1)
                pattern_PreStim[:, iEv] = np.nanmean(PowMat[:, PreStimInds[iEv,:],iEv],1)

            # DataMat_PostStim
        #     DataMat.PostStim(:,iElec,:) = reshape(pattern.PostStim,50,1,[]);
        #     DataMat.PreStim(:,iElec,:) = reshape(pattern.PreStim,50,1,[]);



        #     pattern.PostStim = nan(50,length(SessEv));
        #     pattern.PreStim = nan(50,length(SessEv));



        # for iElec = 1:length(bp)
        #     cd(fullfile(scratchDir,'power',Subject,SessName))
        #     fname = sprintf('%d-%d_Pow_bin_zs.mat',bp(iElec).channel(1),bp(iElec).channel(2));
        #     load(fname);
        #     pattern.PostStim = nan(50,length(SessEv));
        #     pattern.PreStim = nan(50,length(SessEv));
        #     for iEv = 1:length(SessEv)
        #         pattern.PostStim(:,iEv) = nanmean(PowMat(:,PostStimInds(iEv,:),iEv),2);
        #         pattern.PreStim(:,iEv) = nanmean(PowMat(:,PreStimInds(iEv,:),iEv),2);
        #     end
        #     DataMat.PostStim(:,iElec,:) = reshape(pattern.PostStim,50,1,[]);
        #     DataMat.PreStim(:,iElec,:) = reshape(pattern.PreStim,50,1,[]);
        # end
        # DataMat.PostStim = reshape(DataMat.PostStim,nElecs*length(params.pow.freqs),[]);
        # DataMat.PreStim = reshape(DataMat.PreStim,nElecs*length(params.pow.freqs),[]);


        DataMatExample = np.ones((4,5),dtype=float)

        serialize_objects_in_matlab_format('data_mat_demo.mat',(DataMatExample,'DataMatExample'))


        # class DataMatExampleSerializer(MatlabIO):
        #     def __init__(self,dataMatExample):
        #         self.DataMatExample = dataMatExample
        #
        # dataMatExampleSerializer = DataMatExampleSerializer(DataMatExample)
        #
        # dataMatExampleSerializer.serialize('DataMatExample.mat')





        
        # SessPostProb.Post = glmval([Weights.MeanIntercept;W'],DataMat.PostStim','logit');
        # SessPostProb.Pre = glmval([Weights.MeanIntercept;W'],DataMat.PreStim','logit');

















