 
import sys
import numpy as np
import os
import h5py
# # import cPickle as pickle
# import multiprocessing as mp
# from exceptions import OSError
# #import matplotlib.pyplot as plt

# run_on_rhino = False

# if run_on_rhino:
#     rhino_mount = ''
#     num_mp_procs = 0
# else:
#     rhino_mount = '/home/ctw/fusemounts/rhino'
#     num_mp_procs = 23
    

# # add to python path (for ptsa and other stuff)
# sys.path.append(rhino_mount+'/home1/cweidema/lib/python')
# from get_bipolar_subj_elecs import get_bipolar_subj_elecs

import sys

sys.path.append('/home1/mswat/PTSA_GIT')

from ptsa.data.events import Events
from ptsa.data.rawbinwrapper import RawBinWrapper
from ptsa.wavelet import phase_pow_multi
from scipy.io import loadmat

matlab_events_path = '/data/events/RAM_CatFR1/R1065J_events.mat'

dtypes_stimParams = [('hostTime', np.float), ('elec1', np.float),
                     ('elec2', np.float), ('amplitude', np.float),
                     ('burstFreq', np.float), ('nBursts', np.float),
                     ('pulseFreq', np.float), ('nPulses', np.float),
                     ('pulseWidth', np.float)]


dtypes = [('subject','|S12'),('session',np.int),('list',np.int),
          ('serialpos', np.int), ('type', '|S20'),('item','|S20'),
          ('itemno',np.int),('recalled',np.int),('mstime',np.float),
          ('msoffset',np.int),('rectime',np.int),('intrusion',np.int),
          ('isStim', np.int), ('category','|S20' ), ('categoryNum', np.int),
          ('expVersion','|S32' ), ('stimAnode', np.float),
          ('stimAnodeTag', 'O'),
          ('stimCathode', np.float), ('stimCathodeTag', 'O'),
          ('stimLoc', 'S12'),
          ('stimAmp', np.float), ('stimList', np.int),
          # ('stimParams_hostTime', np.float), ('stimParams_elec1', np.float),
          ('stimParams', 'O'),
          ('eegfile','|S256'), ('eegoffset', np.int)]

matevs = loadmat(matlab_events_path,struct_as_record=True,squeeze_me=True)['events']
newevs = np.rec.recarray(len(matevs),dtype=dtypes)
newevs_stimParams = np.rec.recarray(len(matevs),dtype=dtypes_stimParams)

# sys.exit()

for field in matevs.dtype.names:
    try:
        newevs[field] = matevs[field]
    except ValueError:
        if ((field == 'stimAmp') and
            (np.all(np.array(matevs[field],np.str)=='X'))):
            matevs[field].fill(np.nan)
            newevs[field] = matevs[field]
        else: # not sure what's going on, so raise the same error as above
            newevs[field] = matevs[field]

events = Events(newevs)
events = events.add_fields(esrc=np.dtype(RawBinWrapper))


good_indices = np.ones(len(events),np.bool)
for e,event in enumerate(events):
    try:
        event['esrc'] = RawBinWrapper(event['eegfile'])
    except IOError:
        print('No EEG files for',event['subject'],event['session'],event['eegfile'])
        good_indices[e] = False
        
events = events[good_indices]

start_time = -0.6
end_time = 1.6
buf = 1
baseline = (-.6,-.4)
# eeghz = 500
powhz = 50
freqs = np.logspace(np.log10(3),np.log10(180),12)

chan = 0
dat = events[10:20].get_data(
    channels=chan,start_time=start_time,end_time=end_time,buffer_time=buf,
    eoffset='eegoffset',keep_buffer=True)

dat = phase_pow_multi(freqs,dat[0],to_return='power')
