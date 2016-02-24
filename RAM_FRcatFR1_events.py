 
import numpy as np
# import cPickle
import glob
# import warnings
# import os

from scipy.io import loadmat

# add to python path (for ptsa stuff)
import sys
sys.path.append('/home/ctw/lib/python')
# import celex

event_path = '/home/ctw/fusemounts/rhino/data/events/RAM_CatFR1/' # catFR1
# event_path = '/home/ctw/fusemounts/rhino/data/events/RAM_FR1/' # FR1

bad_subjects = ['R1044J','R1007D'] # catFR1
# bad_subjects = ['R1063C'] # FR1

# outfile = '/home/ctw/Christoph/Analyses/RAM/RAM_catFR/data/RAM_catFR1_events20150923.npy' # catFR1
outfile = '/home/ctw/Christoph/Analyses/RAM/RAM_catFR/data/RAM_catFR1_events20151015.npy' # catFR1
# outfile = '/home/ctw/Christoph/Analyses/RAM/RAM_FR/data/RAM_FR1_events20150929.npy' # FR1
# outfile = '/home/ctw/Christoph/Analyses/RAM/RAM_FR/data/RAM_FR1_events20151015.npy' # FR1




event_struct_list = glob.glob(event_path+'R1*_events.mat')
event_struct_list.extend(glob.glob(event_path+'TJ*_events.mat'))
event_struct_list.extend(glob.glob(event_path+'UT*_events.mat'))
good_events = np.ones(len(event_struct_list),np.bool)
for e,evfile in enumerate(event_struct_list):
    for bs in bad_subjects:
        if bs in evfile:
            good_events[e] = False
event_struct_list = np.array(event_struct_list)[good_events]


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

all_evs = None
for datfile in event_struct_list:
    print datfile
    evs = loadmat(datfile,struct_as_record=True,squeeze_me=True)['events']
    newevs = np.rec.recarray(len(evs),dtype=dtypes)
    newevs_stimParams = np.rec.recarray(len(evs),dtype=dtypes_stimParams)
    for field in evs.dtype.names:
        try:
            newevs[field] = evs[field]
        except ValueError:
            if ((field == 'stimAmp') and
                (np.all(np.array(evs[field],np.str)=='X'))):
                evs[field].fill(np.nan)
                newevs[field] = evs[field]
            else: # not sure what's going on, so raise the same error as above
                newevs[field] = evs[field]
    # append these events to recarray of all events:
    if all_evs is None:
        all_evs = newevs
    else:
        all_evs = np.r_[all_evs,newevs]


# all_evs = None
# for datfile in event_struct_list:
#     print datfile
#     evs = loadmat(datfile,struct_as_record=True,squeeze_me=True)['events']
#     newevs = np.rec.recarray(len(evs),dtype=dtypes)
#     newevs_stimParams = np.rec.recarray(len(evs),dtype=dtypes_stimParams)
#     for field in evs.dtype.names:
#         if field == 'stimParms':
#             for f,field_stimParams in enumerate(evs[0][field].dtype.names):
#                 if field_stimParams == dtypes_stimParams[f][0]:
#                     newevs_stimParams[field_stimParams] =  tuple(np.array(
#                         [x[field_stimParams] for x in evs[field]], dtypes_stimParams[f][1]))
#                 else:
#                     newevs_stimParams[field_stimParams] =  tuple(np.array(
#                         [np.nan for x in evs[field]], dtypes_stimParams[f][1]))
#             newevs[field] = newevs_stimParams
#         else:
#             try:
#                 newevs[field] = evs[field]
#             except ValueError:
#                 if ((field == 'stimAmp') and
#                     (np.all(np.array(evs[field],np.str)=='X'))):
#                     evs[field].fill(np.nan)
#                     newevs[field] = evs[field]
#                 else: # not sure what's going on, so raise the same error as above
#                     newevs[field] = evs[field]
#     # append these events to recarray of all events:
#     if all_evs is None:
#         all_evs = newevs
#     else:
#         all_evs = np.r_[all_evs,newevs]


np.save(outfile,all_evs)
