import sys
sys.path.append('/Users/m/PTSA_NEW_GIT')
sys.path.append('/Users/m/RAM_UTILS_GIT')

import numpy as np
from numpy.testing import *
from ptsa.data.readers import BaseEventReader, BaseRawReader

from ptsa.data.readers.TalReader import TalReader
from ptsa.data.readers import EEGReader
from ptsa.data.filters import MonopolarToBipolarMapper
from ptsa.data.filters import ButterworthFilter
from ptsa.data.filters import DataChopper
from ptsa.data.common import xr

from MatlabIO import serialize_objects_in_matlab_format

from os.path import *
import os
import shutil
from glob import glob
import re


# mount_point  = '/Volumes/rhino_root'
mount_point  = '/'
task = 'RAM_FR1'
experiment = 'FR1'

start_time = 0.0
end_time = 1.6
buf = 1.0




def find_subjects_by_task(task):
    # ev_files = glob('/data/events/%s/R*_events.mat' % task)

    ev_files = glob(mount_point + '/data/events/%s/R*_events.mat' % task)
    subj_list = []

    for f in ev_files:
        search_obj = re.search(r'R1\d\d\d[A-Z](_\d+)?', f)
        if search_obj:

            subj_list.append(search_obj.group())

    return subj_list
    # return [re.search(r'R1\d\d\d[A-Z](_\d+)?', f).group() for f in ev_files]






subjects = find_subjects_by_task(task)
# subjects.append('TJ086')
subjects.sort()

print subjects
# subject = 'R1060M'

# print 'index=',subjects.index('R1138T_1')



for subj_idx, subject in enumerate(subjects):



    # if subj_idx<=59:
    #
    #     continue


    #reading the events
    e_path = join(mount_point, 'data/events/RAM_FR1', subject + '_events.mat')
    e_reader = BaseEventReader(filename=e_path, eliminate_events_with_no_eeg=True)
    events = e_reader.read()

    print subject
    #picking events for e agive nexperiment

    ev_order = np.argsort(events, order=('session','list','mstime'))
    events = events[ev_order]

    events = events[events.type == 'WORD']



    # reading electrode info
    tal_path = join(mount_point,'data/eeg',subject,'tal',subject+'_talLocs_database_bipol.mat')
    tal_reader = TalReader(filename=tal_path)

    bpTalStruct = tal_reader.read()
    monopolar_channels = tal_reader.get_monopolar_channels()
    bipolar_pairs = tal_reader.get_bipolar_pairs()



    # print events


    eeg_pre_reader = EEGReader(
        events=events,
        channels=np.array(monopolar_channels),
        start_time=start_time,
        end_time=end_time, buffer_time=buf
    )


    if not events.shape[0]:
        print 'Skipping subject ', subject,' - no '+experiment+' data'
        continue

    eegs_pre = eeg_pre_reader.read()

    samplerate = float(eegs_pre['samplerate'])


    # print eegs_pre

    m2b = MonopolarToBipolarMapper(time_series=eegs_pre, bipolar_pairs=bipolar_pairs)
    bp_eegs_pre = m2b.filter()
    # bp_eegs_pre


    out_dir = join(mount_point,'scratch/mswat/events_for_tung_FR1/'+experiment,subject)

    try:
        os.makedirs(out_dir)
    except OSError:
        pass


    shutil.copy(tal_path, join(out_dir,'bipolar_tal_structs.mat'))

    for i, ev in enumerate(events):


        ev_file_name = join(out_dir,'event'+str(i).zfill(4))
        print ev_file_name

        serialize_objects_in_matlab_format(ev_file_name,
                                           (bp_eegs_pre.data[:,i,:],'channels_eegs'),
                                           (samplerate,'samplerate'),
                                           (int(ev.recalled),'recalled'),
                                           (int(ev.session),'session'),
                                           (int(ev['list']),'list'),

                                           )



# post_start_time = ps_offset
# post_end_time = ps_offset + (ps_end_time - ps_start_time)
#
#
#
# post_start_offsets = np.copy(events.eegoffset)
#
#
# for i_ev in xrange(len(post_start_offsets)):
#     ev_offset = events[i_ev].pulse_duration if experiment!='PS3' else events[i_ev].train_duration
#     if ev_offset > 0:
#         ev_offset *= 0.001
#     else:
#         ev_offset = 0.0
#
#     post_start_offsets[i_ev] += (ev_offset + post_start_time - ps_buf)*samplerate
#
# read_size = eegs_pre['time'].shape[0]
# dataroot = events[0].eegfile
# brr = BaseRawReader(dataroot = dataroot, start_offsets=post_start_offsets, channels=np.array(monopolar_channels),read_size = read_size)
#
# eegs_post , read_ok_mask= brr.read()
#
# print 'eegs_post'
#
# print eegs_post