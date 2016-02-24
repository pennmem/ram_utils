__author__ = 'm'

import scipy.io as sio
import pandas as pd
import numpy as np


class Data(object):
    def __init__(self, a, b):
        self.a = a
        self.b = b
        self.repr = 'Value_'+str(self.a)+'_'+str(self.b)


a_array = np.array([Data(i, i*2) for i in xrange(10)])

s = pd.Series(a_array)

print s

# s.where(s['a']<2)


import numpy as np
from scipy.io import loadmat  # this is the SciPy module that loads mat-files
# import matplotlib.pyplot as plt
# from datetime import datetime, date, time
import pandas as pd

params_read = loadmat('params.mat')  # load mat-file

params = params_read['params']
params_type = params.dtype
print 'params_type=',params_type.names
# print params

print params_read['params']['eeg']

print params_read['params']['eeg'].dtype.names


# mdata = mat['measuredData']  # variable in mat file
# mdtype = mdata.dtype  # dtypes of structures are "unsized objects"

params_read = loadmat('params.mat',squeeze_me=True, struct_as_record=False)  # load mat-file

params = params_read['params']
print params
print params.eeg.filtfreq.dtype
print type(params.eeg.durationMS).__name__

print 'array of objects'
print params.__dict__['eeg'].__dict__['filtfreq'].dtype

print np.where(params.__dict__['eeg'].__dict__['filtfreq']<56)



# params_type = params.dtype
# print 'params_type=',params_type.names
# # print params
#
# print params_read['params']['eeg']
#
# print params_read['params']['eeg'].dtype.names

N=10

# TB =     {'names':['n', 'D'], 'formats':[int, int]}
# TA =     {'names':['id', 'B'],'formats':[int, np.dtype((TB, (N)))]}
TB =     {'names':('n', 'D'), 'formats':(int, int)}
TA =     {'names':('id', 'B'),'formats':(int, np.dtype((TB, (N))))}


a = np.empty(10, dtype=TA)
b = np.empty(N, dtype=TB)

print b,' dtype=',b.dtype

print b[0]['n']


print a[0]
print ' dtype=',a.dtype


print a[0]['B']['n']
print np.where(a['B']['n']>4298163000)


print a[a[0]['B']['n']>4298163000]

for name in dir(np):
    obj = getattr(np, name)
    if hasattr(obj, 'dtype'):
        try:
            npn = obj(0)
            nat = npn.item()
            print('%s (%r) -> %s'%(name, npn.dtype.char, type(nat)))
        except:
            pass


import inspect
class EEG(object):
    def __init__(self):
        self.ca = 30L
        self.aa = 10.
        self.ab = 'dupa'


    def to_record(self):
        record_values = []
        for class_member in inspect.getmembers(self, lambda a : not(inspect.isroutine(a))):

            class_member_name = class_member[0]
            class_member_val = class_member[1]
            class_member_type = type(class_member_val).__name__

            if not(class_member_name.startswith('__') and class_member_name.endswith('__')):
                print 'class_member_name=', class_member_name
                print 'class_member_val=', class_member_val
                print 'class_member_type=', class_member_type

                record_values.append(class_member_val)

        return tuple(record_values)

    def get_record_format(self):
        names_list = []
        format_list = []
        for class_member in inspect.getmembers(self, lambda a : not(inspect.isroutine(a))):

            class_member_name = class_member[0]
            class_member_val = class_member[1]
            class_member_type = type(class_member_val)
            class_member_type_name = class_member_type.__name__

            if not(class_member_name.startswith('__') and class_member_name.endswith('__')):
                names_list.append(class_member_name)
                format_list.append(class_member_type)

        return {'names': names_list, 'formats': format_list}


eeg = EEG()
eeg1 = EEG()
eeg1.aa = 21.2



print eeg.__dict__

eeg.to_record()

eeg_record_format =  eeg.get_record_format()

print eeg_record_format

format = {'names': ['aa', 'ab', 'ca'], 'formats': [float, 'S10' , 'l']}

eeg_array = np.empty(10, dtype=format)

# eeg_array = np.empty(10, dtype=eeg_record_format)
#
# print eeg_array


print eeg_array[0]

print eeg_array.dtype

print type(eeg_array[0])

print eeg.to_record()


eeg_array[0] = eeg.to_record()
eeg_array[1] = eeg1.to_record()

eeg_array['aa'] = 12.0

print eeg_array

print np.where(eeg_array['aa']>10.1)

m = eeg_array[eeg_array['aa']>10.1]

print m


i_t = int

print
