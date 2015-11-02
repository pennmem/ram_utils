__author__ = 'm'
import scipy.io as sio
import numpy as np

import inspect

class MatlabIO(object):
    def __init__(self):pass

    def items(self):
        for class_member in inspect.getmembers(self, lambda a : not(inspect.isroutine(a))):

            class_member_name = class_member[0]
            class_member_val = class_member[1]

            if not(class_member_name.startswith('__') and class_member_name.endswith('__')):
                # print 'class_member_name=', class_member_name
                yield class_member_name, class_member_val

    def serialize(self, name, format='matlab'):
        sio.savemat(name, self)


    def deserialize(self, name, format='matlab'):
        res = sio.loadmat(name,squeeze_me=True, struct_as_record=False)
        # print res
        # print '\n\n\n'

        # name and val are names and values of the attributes read from .mat file
        for name,val in res.items():
            if not(name.startswith('__') and name.endswith('__')):
                # print 'name=',name, ' val=', val, 'type =', type(val)
                setattr(self,name,val)

        pass


class Pow(MatlabIO):
    def __init__(self):
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

class Eeg(MatlabIO):

    def __init__(self):
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



class Params(MatlabIO):
    def __init__(self):
        self.eeg = Eeg()
        self.pow = Pow()


print 'dupa - serializing'

eeg = Eeg()

eeg.serialize('eeg_serialized_new.mat')

eeg_loaded = MatlabIO()

eeg_loaded.deserialize('eeg_serialized_new.mat')

print '*************************************results section'
print 'eeg_loaded.durationMS=', eeg_loaded.durationMS
print 'eeg_loaded.filtfreq=', eeg_loaded.filtfreq, ' type=',type(eeg_loaded.filtfreq)



print '\n\n\n*************************************composite section'

params = Params()
params.serialize('params_serialized.mat')

params_loaded = MatlabIO()
params_loaded.deserialize('params_serialized.mat')

print 'params_loaded.eeg.durationMS=',params_loaded.eeg.durationMS

print 'params_loaded.pow.freqbins=',params_loaded.pow.freqBins


# eeg_loaded.load()

# eeg_loaded  = eeg.load('eeg_serialized.mat')
#
#
#
#
# print 'eeg_loaded=',eeg_loaded
#
#
# # print 'eeg_loaded[durationMS]=', eeg_loaded['durationMS']
# print 'eeg_loaded[durationMS]=', eeg_loaded.durationMS