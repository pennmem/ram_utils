__author__ = 'm'

import numpy as np

from MatlabIO import MatlabIO


class pow(MatlabIO):

    def __init__(self):
        MatlabIO.__init__(self)
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

class eeg(MatlabIO):

    def __init__(self):
        MatlabIO.__init__(self)


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
        MatlabIO.__init__(self)
        self.eeg = eeg()
        self.pow = pow()

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

    params = params()
    params.serialize('params_serialized_proper_struct.mat')

    params_loaded = MatlabIO()
    params_loaded.deserialize('params_serialized_proper_struct.mat')



    #
    #
    #
    print 'params_loaded.Params=',dir(params_loaded)

    params_loaded.serialize('params_serialized_proper_struct_check.mat')


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






