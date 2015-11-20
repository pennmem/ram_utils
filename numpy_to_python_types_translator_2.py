__author__ = 'm'

import sys

import numpy as np
import scipy.io as sio



from MatlabIO import *



# --------------------------------- djkhfkdjhfk
events_file = '/Volumes/rhino_root/data/events/RAM_FR1/R1056M_events.mat'

# events_dict = read_matlab_matrices_as_numpy_structured_arrays(events_file, 'events')
#
# events_struct_array = events_dict['events']

events_struct_array = read_single_matlab_matrix_as_numpy_structured_array(file_name=events_file, object_name='events')



print 'GOT HERE'

# sys.exit()

print 'events_struct_array=',events_struct_array

print 'events_struct_array length =', len(events_struct_array)

word_events = events_struct_array[events_struct_array['type']=='WORD']

print 'word_events=',word_events
print 'len(word_events)=',len(word_events)


select_eeg_file = word_events[word_events['eegfile']=='/data/eeg/R1056M/eeg.reref/R1056M_19Jun15_1003']

for eeg_file_name in word_events['eegfile']:
    print 'eeg_file_name=',eeg_file_name,' type=',type(eeg_file_name)



print select_eeg_file[select_eeg_file['item']=='RICE']







