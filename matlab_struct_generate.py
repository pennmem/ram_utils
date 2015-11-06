__author__ = 'm'

import numpy as np
import numpy.matlib as npm

from MatlabIO import *

#
# class pow(object):
#
#     def __init__(self):
#         # MatlabIO.__init__(self)
#         self.type = 'fft_slep'
#         self.freqs = np.logspace(np.log10(1), np.log10(200), 50)
#         self.logTrans = 1
#         self.wavenum = np.nan
#         self.bandwidth = 2
#         self.wavenum = 0
#
#         # the following two parameters are specific fft_slep and are different from
#         # the parameters below that specifiy post-power binning.
#         self.winSize = .4  # units in seconds
#         self.winStep = .05  # units in seconds. use nan if computing power for every sample.
#
#         self.timeWin = 100
#         self.timeStep = 50
#         self.freqBins = self.freqs
#         self.zscore = 1
#
# class eeg(object):
#
#     def __init__(self):
#         # MatlabIO.__init__(self)
#
#
#         self.durationMS = 4000
#         self.offsetMS = -1000
#
#         self.bufferMS = 1000
#         self.filtfreq = np.array([58, 62])
#         self.filttype = 'stop'
#         self.filtorder = 1
#         self.sampFreq = 500
#         self.kurtThr = 4
#
#         self.HilbertBands = np.array(
#                             [
#                                 [4, 8],
#                                 [8, 12],
#                                 [30, 50],
#                                 [70, 100]
#                             ]
#                             )
#
#         self.HilbertNames = ['Theta (4-8 Hz)','Alpha (8-12 Hz)','Low Gamma (30-50 Hz)','High Gamma (70-100 Hz)'];
#
#
#
# class params(object):
#     def __init__(self):
#         # MatlabIO.__init__(self)
#         self.eeg = eeg()
#         self.pow = pow()


# class params_new(object):
#     def __init__(self):
#         # MatlabIO.__init__(self)
#         self.eeg = eeg()
#         self.pow = pow()


class pow(object):

    def __init__(self):
        # MatlabIO.__init__(self)
        self.type = 'fft_slep'
        self.freqs = np.logspace(np.log10(1), np.log10(200), 50)
        self.logTrans = 1
        self.wavenum = np.nan
        self.bandwidth = 2
        self.wavenum = 0

        # the following two parameters are specific fft_slep and are different from
        # the parameters below that specifiy post-power binning.
        self.winSize = .4  # units in seconds
        self.winStep = .05  # units in seconds. use nan if computing power for every sample.

        self.timeWin = 100
        self.timeStep = 50
        self.freqBins = self.freqs
        self.zscore = 1

class eeg(object):

    def __init__(self):
        # MatlabIO.__init__(self)


        self.durationMS = 4000
        self.offsetMS = -1000

        self.bufferMS = 1000
        self.filtfreq = np.array([58, 62])
        self.filttype = 'stop'
        self.filtorder = 1
        self.sampFreq = 500
        self.kurtThr = 4

        self.HilbertBands = np.array(
                            [
                                [4, 8],
                                [8, 12],
                                [30, 50],
                                [70, 100]
                            ]
                            )

        self.HilbertNames = ['Theta (4-8 Hz)','Alpha (8-12 Hz)','Low Gamma (30-50 Hz)','High Gamma (70-100 Hz)'];



class params(MatlabIO):
    def __init__(self):
        # MatlabIO.__init__(self)
        self.eeg = eeg()
        self.pow = pow()


class Serializer(MatlabIO):
    params = params()



# class Params(MatlabIO):
#     def __init__(self):
#         self.params = params()


def indices(a, func):
    return [i for (i, val) in enumerate(a) if func(val)]





# dataMatExampleSerializer.serialize('DataMatExample.mat')




if __name__ == "__main__":
    import sys
    from os.path import *


    # eeg = eeg()
    #
    # eeg.serialize('eeg_serialized_new1.mat')
    #
    # eeg_loaded = MatlabIO()
    #
    # eeg_loaded.deserialize('eeg_serialized_new1.mat')
    #
    # print dir(eeg_loaded)
    # print eeg_loaded.filttype

    # sys.exit()
    #
    #
    # print 'dupa'

    DataMatExample = np.ones((4,5),dtype=float)

    serialize_objects_in_matlab_format('data_mat_demo.mat',(DataMatExample,'DataMatExample'))

    object_dict = deserialize_objects_from_matlab_format('data_mat_demo.mat','DataMatExample','my_data')

    print object_dict







    #
    # serializer = Serializer()
    # serializer.serialize('params_serializer.mat')
    #
    #
    # serializer_check = MatlabIO()
    # serializer_check.deserialize('params_serializer.mat')



    # print 'dir()=',dir(serializer_check)
    #
    # print serializer_check.params.eeg.durationMS

    params = params()

    serialize_objects_in_matlab_format('new_serializer_demo.mat',(params,'params'))

    object_dict = deserialize_objects_from_matlab_format('new_serializer_demo.mat','params')

    print "object_dict['params']=", object_dict['params'].eeg.durationMS

    sys.exit()

    # print serializer_check.params['eeg']

    # sys.exit()

    serializer_check.serialize('params_serializer_check.mat')




    PostStimBuff = 50  # buffer in ms to leave following stim offset


    group_psl_reader = MatlabIO()
    group_psl_reader.deserialize('GroupPSL.mat')
    group_psl = group_psl_reader.GroupPSL
    print group_psl[0].Subject

    #
    paramsPS_reader = MatlabIO()
    paramsPS_reader.deserialize('paramsPS.mat')

    paramsPS = paramsPS_reader.params


    bpFull_reader = MatlabIO()
    bpFull_reader.deserialize('bpFull.mat')

    bpFull = bpFull_reader.bpFull


    bp_reader = MatlabIO()
    bp_reader.deserialize('bp.mat')

    bp = bp_reader.bp

    # Weights_reader = MatlabIO()
    #
    # Weights_reader.deserialize('Weights.mat')
    #
    # Weights = Weights_reader.Weights




    ps2_events_reader = MatlabIO()

    ps2_events_reader.deserialize('PS2Events.mat')

    ps2_events = ps2_events_reader.PS2Events


    print 'ps2_events=',ps2_events

    ps2_events_size = len(ps2_events)
    print 'number of ps2 events = ', ps2_events_size


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

















    sys.exit()

    params = params()
    params.serialize('params_serialized_proper_struct.mat')

    params_loaded = MatlabIO_OLD()
    params_loaded.deserialize('params_serialized_proper_struct.mat')
    print 'params_loaded.eeg.durationsMS=',params_loaded.eeg.durationMS

    print 'dir(params_loaded)=',dir(params_loaded)


    #
    #
    #
    print 'params_loaded.Params=',dir(params_loaded)

    params_loaded.serialize('params_serialized_proper_struct_check.mat')







    print ''

    import inspect
    import scipy.io as sio

    res = sio.loadmat('GroupPSL.mat', squeeze_me=True, struct_as_record=False)



    print dir(res)
    print res['GroupPSL'][0].Subject


    group_psl = MatlabIO_OLD()
    group_psl.deserialize('GroupPSL.mat')


    print 'group_psl=',group_psl
    print dir(group_psl)
    print group_psl.items()


    # import scipy.io as sio
    #
    # a_dict = {'eeg':params.eeg, 'pow':params.pow}
    # sio.savemat('saved_struct.mat', {'params': a_dict})




    # print 'dupa - serializing'
    #
    # eeg = Eeg()
    #
    # eeg.serialize('eeg_serialized_new.mat')
    #
    # eeg_loaded = MatlabIO()
    #
    # eeg_loaded.deserialize('eeg_serialized_new.mat')
    #
    # print '*************************************results section'
    # print 'eeg_loaded.durationMS=', eeg_loaded.durationMS
    # print 'eeg_loaded.filtfreq=', eeg_loaded.filtfreq, ' type=',type(eeg_loaded.filtfreq)
    #
    #
    #
    # print '\n\n\n*************************************composite section'
    #
    # params = Params()
    # params.serialize('params_serialized.mat')
    #
    # params_loaded = MatlabIO()
    # params_loaded.deserialize('params_serialized.mat')
    #
    # print 'params_loaded.eeg.durationMS=',params_loaded.eeg.durationMS
    #
    # print 'params_loaded.pow.freqbins=',params_loaded.pow.freqBins
    #
    #
    # params_loaded.serialize('params_serialized_2.mat')
    #
    # print '***************************** checking deserialization again'
    # params_loaded_2 = MatlabIO()
    # params_loaded_2.deserialize('params_serialized.mat')
    #
    # print 'params_loaded_2.eeg.durationMS=',params_loaded_2.eeg.durationMS
    #
    # print 'params_loaded_2.pow.freqbins=',params_loaded_2.pow.freqBins
    #
    # print 'params_loaded_2.items=',params_loaded_2.items






