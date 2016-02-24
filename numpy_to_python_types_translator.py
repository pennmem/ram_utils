__author__ = 'm'

import sys

import numpy as np
import scipy.io as sio

from collections import defaultdict
# d = defaultdict(set)
d = defaultdict(list)


for name in dir(np):
    obj = getattr(np, name)

    if hasattr(obj, 'dtype'):
        print 'name=',name
        print 'obj=',obj

        try:
            npn = obj(0) # creating object of a type in the dtype list
            print 'npn=',npn
            nat = npn.item()
            print('%s (%r) -> %s'%(name, npn.dtype.char, type(nat)))

            # d[type(nat)].add(npn.dtype.char)
            d[type(nat)].append(npn.dtype.char)

        except:
            pass

print d

import inspect


def determine_numpy_type_abbreviation(inspect_member_info):
    class_member_name = inspect_member_info[0]
    class_member_val = inspect_member_info[1]
    class_member_type = type(class_member_val)

    if class_member_type.__name__ == 'ndarray':
        print 'found array'
        print 'class_member_val=', class_member_val
        print 'class_member_val.dtype=', class_member_val.dtype
        print 'class_member_val.shape=', class_member_val.shape
        print 'class_member_val.dtype.descr=', class_member_val.dtype.descr

        numpy_type_char_abbreviation = (class_member_val.dtype.descr[0][1],class_member_val.shape)
        print 'numpy_type_char_abbreviation=',numpy_type_char_abbreviation
        # sys.exit()
    else:

        # ('f2', '>f8', (2, 3))
        numpy_type_char_abbreviation = d[class_member_type][0]
        if numpy_type_char_abbreviation == 'S':
            numpy_type_char_abbreviation = 'S10'
        elif numpy_type_char_abbreviation == 'U':
            numpy_type_char_abbreviation = 'U10'

    return numpy_type_char_abbreviation

def create_dtype_stub(obj, entry_list=[]):

    for class_member in inspect.getmembers(obj, lambda a : not(inspect.isroutine(a) and not(a.startswith('__') and a.endswith('__')))):

        class_member_name = class_member[0]
        class_member_val = class_member[1]
        class_member_type = type(class_member_val)



        # class_member_type_name = class_member_type.__name__

        # if not(class_member_name.startswith('__') and class_member_name.endswith('__')):
        print 'numpy char symbol for  ', class_member_type, ' = ', d[class_member_type]
        names_list.append(class_member_name)
        # format_list.append(class_member_type)

        # print 'class_member_type=', class_member_type.__name__



        numpy_type_abbreviation = determine_dtype_abbreviation(class_member)
        format_list.append(numpy_type_abbreviation)

    return {'names': names_list, 'formats': format_list}


class Stub(object):
    def __init__(self):
        self.a = 30L
        self.b = 10.
        self.c = 'dupa'



from MatlabIO import *

def get_non_special_class_members(obj):
    for class_member in inspect.getmembers(obj, lambda a: not(inspect.isroutine(a))):
        class_member_name = class_member[0]
        class_member_val = class_member[1]
        class_member_type = type(class_member_val).__name__
        if not(class_member_name.startswith('__') and class_member_name.endswith('__')):
            yield class_member_name,class_member_val,class_member_type


class EEG(object):
    def __init__(self):

        self.ca = 30L
        self.aa = 10.
        self.ab = 'dupa'
        self.m = np.ones((2,2) , dtype=np.float)




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

        # special_member_filter = lambda a: not(inspect.isroutine(a) or a[0].startswith('__') or a[0].endswith('__'))
        #
        #
        # for class_member in inspect.getmembers(self, special_member_filter):
        #
        #     class_member_name = class_member[0]
        #     class_member_val = class_member[1]
        #     class_member_type = type(class_member_val)
        #
        #     if not(class_member_name.startswith('__') and class_member_name.endswith('__')):
        #         print 'numpy char symbol for  ', class_member_type, ' = ', d[class_member_type]
        #         names_list.append(class_member_name)
        #         # format_list.append(class_member_type)
        #
        #         print 'class_member_type=', class_member_type.__name__
        #
        #
        #
        #         numpy_type_abbreviation = determine_numpy_type_abbreviation(class_member)
        #         format_list.append(numpy_type_abbreviation)
        #
        # return {'names': names_list, 'formats': format_list}


        for class_member in inspect.getmembers(self, lambda a : not(inspect.isroutine(a))):


            print 'type'
            class_member_name = class_member[0]
            class_member_val = class_member[1]
            class_member_type = type(class_member_val)



            class_member_type_name = class_member_type.__name__

            if not(class_member_name.startswith('__') and class_member_name.endswith('__')):
                print 'numpy char symbol for  ', class_member_type, ' = ', d[class_member_type]
                names_list.append(class_member_name)
                # format_list.append(class_member_type)

                print 'class_member_type=', class_member_type.__name__



                numpy_type_abbreviation = determine_dtype_abbreviation(class_member)
                format_list.append(numpy_type_abbreviation)

        def special_member_filter(a):
            print 'a=',a.__name__
            return not(inspect.isroutine(a) )
            # or a[0].startswith('__') or a[0].endswith('__'))

        for m_name,m_val, m_type in get_non_special_class_members(self):
            print 'name, val, type=',(m_name,m_val, m_type)


        return {'names': names_list, 'formats': format_list}

eeg = EEG()
eeg1 = EEG()
eeg1.aa = 21.2
eeg1.ab = 'dupa1'
eeg1.ca = 11






print eeg.__dict__

eeg.to_record()


eeg_record_format =  eeg.get_record_format()





print 'eeg_record_format=', eeg_record_format


f =[('ab', 'S10'), ('aa', '<f8'),('ca', '<i8')]

f_array = np.empty(2, dtype=f)

print 'f_array.dtype=',f_array.dtype




# eeg_record_format['formats'][2]='Q'
# eeg_record_format['formats'][1]='S10'

print 'eeg_record_format=',eeg_record_format

# eeg_record_format['names'].append('m')
# eeg_record_format['formats'].append(('<f8',(2,2)))


eeg_array = np.empty(2, dtype=eeg_record_format)


print 'eeg_array=',eeg_array

sys.exit()

print eeg_array[0]



print eeg_array.dtype


# sys.exit()

# sys.exit()
print type(eeg_array[0])

print eeg.to_record()


eeg_array[0] = eeg.to_record()
eeg_array[1] = eeg1.to_record()


# eeg_array[0]['ab'] =  'skjdkdjskljdhjshdjshdljshdl'

print eeg_array



print np.where(eeg_array['aa']>10.1)

m = eeg_array[eeg_array['aa']>10.1]
print m.dtype

print "------------------complex type"

base_format = [('aa', '<f8'), ('ab', 'S10'), ('ca', '<i8'), ('m', '<f8', (2, 2))]
# complex_format = [('com',np.dtype(base_format))]

complex_format = [('com', base_format)]

m_complex = np.empty(2, dtype=complex_format)

print m_complex

print m_complex.dtype

aa_type = np.dtype([('aa', '<f8')])
ab_type = np.dtype([('ab', 'S10')])
ca_type = np.dtype([('ca', '<i8')])

aa_type_1 = np.dtype(aa_type)
# print 'aa_type_1=',aa_type_1

# print np.dtype((aa_type,ab_type,ca_type))



# sys.exit()

serialize_objects_in_matlab_format('eeg_array.mat',(eeg_array,'eeg_array'))
#
eeg_des = deserialize_single_object_from_matlab_format('eeg_array.mat', 'eeg_array')

print type(eeg.aa)

#
eeg_des1 =  sio.loadmat('eeg_array.mat', squeeze_me=True, struct_as_record=True)
#
print 'eeg_des1=',eeg_des1

print 'dtype=', eeg_des1['eeg_array'].dtype



for a  in eeg_des1['eeg_array'].dtype.descr:
    print a

# dt = data.dtype
# dt = dt.descr # this is now a modifiable list, can't modify numpy.dtype


new_format = eeg_des1['eeg_array'].dtype.descr # .descr makes it now a modifiable list, can't modify numpy.dtype


print 'type of eeg_des[0].aa=',type(eeg_des[0].aa)
print 'eeg_des=',type(getattr(eeg_des[0],'aa'))



# for i, record_entry_spec in enumerate(new_format):
#     record_field_name = record_entry_spec[0]
#     print 'record_field_name=',record_field_name
#     print 'getattr(eeg_des[0],record_field_name)', type(getattr(eeg_des[0], record_field_name))
#     python_type_of_record_field = type(getattr(eeg_des[0],record_field_name))
#     print 'record_field_name=',record_field_name, ' type = ', python_type_of_record_field
#     numpy_abbreviated_type_name = d[python_type_of_record_field][0]
#     if numpy_abbreviated_type_name == 'U':
#         numpy_abbreviated_type_name = 'U16'
#
#     print 'd[record_field_name][python_type_of_record_field]=',d[python_type_of_record_field][0]
#     tmp_spec = list(record_entry_spec)
#
#
#     tmp_spec[1] = numpy_abbreviated_type_name
#     new_format[i] = tuple(tmp_spec)
#
    # record_entry_spec[1] = numpy_abbreviated_type_name

for i, record_entry_spec in enumerate(new_format):
    record_field_name = record_entry_spec[0]
    print 'record_field_name=',record_field_name
    print 'getattr(eeg_des[0],record_field_name)', type(getattr(eeg_des[0], record_field_name))
    python_type_of_record_field = type(getattr(eeg_des[0],record_field_name))


    print 'record_field_name=',record_field_name, ' type = ', python_type_of_record_field
    numpy_abbreviated_type_name = d[python_type_of_record_field][0]


    # if class_member_type.__name__ == 'ndarray':
    #     print 'found array'
    #     print 'class_member_val=', class_member_val
    #     print 'class_member_val.dtype=', class_member_val.dtype
    #     print 'class_member_val.shape=', class_member_val.shape
    #     print 'class_member_val.dtype.descr=', class_member_val.dtype.descr
    #
    #     numpy_type_char_abbreviation = (class_member_val.dtype.descr[0][1],class_member_val.shape)
    #     print 'numpy_type_char_abbreviation=',numpy_type_char_abbreviation
    #     # sys.exit()
    # else:
    #
    #     # ('f2', '>f8', (2, 3))
    #     numpy_type_char_abbreviation = d[class_member_type][0]
    #     if numpy_type_char_abbreviation == 'S':
    #         numpy_type_char_abbreviation = 'S10'
    #     elif numpy_type_char_abbreviation == 'U':
    #         numpy_type_char_abbreviation = 'U10'





    if numpy_abbreviated_type_name == 'U':
        numpy_abbreviated_type_name = 'U16'

    print 'd[record_field_name][python_type_of_record_field]=',d[python_type_of_record_field][0]
    tmp_spec = list(record_entry_spec)


    tmp_spec[1] = numpy_abbreviated_type_name
    new_format[i] = tuple(tmp_spec)





eeg_array_1 = np.empty(2, dtype=new_format)

print eeg_array_1.dtype

eeg_array_1['ca'] = 12
eeg_array_1['ab'] = 'dupa12'

print eeg_array_1

print eeg_des1['eeg_array']['aa']


eeg_array_1['aa'] = eeg_des1['eeg_array']['aa']
eeg_array_1['ab'] = eeg_des1['eeg_array']['ab']
eeg_array_1['ca'] = eeg_des1['eeg_array']['ca']

print eeg_array_1
print eeg_array_1.dtype


# eeg_arra

# name = 'params.mat'
# res = sio.loadmat(name,squeeze_me=True, struct_as_record=False)
#
# print res['params'].eeg.durationMS
#
#
#
#
# res1 = sio.loadmat(name,squeeze_me=True, struct_as_record=True)
#
# print res1['params']['eeg']
#
# print res1['params']['eeg']
