__author__ = 'm'

import scipy.io as sio
import pandas as pd
import numpy as np

name = 'params.mat'
res = sio.loadmat(name,squeeze_me=True, struct_as_record=False)

print res['params'].eeg.durationMS




res1 = sio.loadmat(name,squeeze_me=False, struct_as_record=True)

print res1['params']['eeg']

print
# print res1['params']['eeg']['durationMS']