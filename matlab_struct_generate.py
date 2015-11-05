__author__ = 'm'

import numpy as np

from MatlabIO import MatlabIO_OLD, MatlabIOREADER,MatlabIO

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





if __name__ == "__main__":
    import sys

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


    serializer = Serializer()
    serializer.serialize('params_serializer.mat')


    serializer_check = MatlabIO()
    serializer_check.deserialize('params_serializer.mat')

    print 'dir()=',dir(serializer_check)

    print serializer_check.params.eeg.durationMS


    serializer_check.serialize('params_serializer_check.mat')





    group_psl = MatlabIO()
    group_psl.deserialize('GroupPSL.mat')
    print group_psl.GroupPSL[0].Subject

    ps2_events_reader = MatlabIO()

    ps2_events_reader.deserialize('PS2Events.mat')

    ps2_events = ps2_events_reader.PS2Events

    print 'ps2_events=',ps2_events

    print 'number of ps2 events = ', len(ps2_events)
    for i in xrange(1, len(ps2_events)):
        ev_curr = ps2_events[i]
        ev_prev = ps2_events[i-1]
        print 'i=',i,' event = ', ev_curr, ' event_type = ', ev_curr.type, ' session = ',ev_curr.session
        if ev_curr.session == ev_prev.session:
            if ev_curr.type=='STIMULATING' and ev_prev.type=='STIMULATING':
                print 'isi = ', ev_curr.mstime - ev_prev.mstime

    #     if strcmp(PS2Events(i-1).type,'STIMULATING') & strcmp(PS2Events(i).type,'STIMULATING') & PS2Events(i-1).session==PS2Events(i).session
    #         PS2Events(i).ISI = PS2Events(i).mstime - PS2Events(i-1).mstime;
    #     else
    #         PS2Events(i).ISI = nan;
    #     end
    # end




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






