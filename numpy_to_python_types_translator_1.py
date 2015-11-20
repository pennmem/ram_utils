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


def determine_numpy_type_abbreviation(inspect_member_info, default_string_length=16):
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
            numpy_type_char_abbreviation = 'S'+str(default_string_length)
        elif numpy_type_char_abbreviation == 'U':
            numpy_type_char_abbreviation = 'U'+str(default_string_length)

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
            yield class_member, class_member_name,class_member_val

def get_record_format(obj):
    names_list = []
    format_list = []


    for class_member, class_member_name, class_member_value in get_non_special_class_members(obj):
        print 'class_member, class_member_name, class_member_value=',(class_member, class_member_name, class_member_value)

        try:
            numpy_type_abbreviation = determine_dtype_abbreviation(class_member)
        except:
            print 'COULD NOT DETERMINE FORMAT FOR:'
            print 'class_member_name=',class_member_value
            print 'class_member_values=',class_member_value
            print 'SKIPPING this'
            continue

        names_list.append(class_member_name)
        format_list.append(numpy_type_abbreviation)


    return {'names': names_list, 'formats': format_list}


class EEG(object):
    def __init__(self):

        self.ca = 30L
        self.aa = 10.
        self.ab = 'dupa'
        self.m = np.ones((2,2) , dtype=np.float)


    def to_record(self):
        record_values = []
        for class_member, class_member_name, class_member_value in get_non_special_class_members(self):
            record_values.append(class_member_value)

        return tuple(record_values)

    def get_record_format(self):
        names_list = []
        format_list = []


        for class_member, class_member_name, class_member_value in get_non_special_class_members(self):
            names_list.append(class_member_name)
            numpy_type_abbreviation = determine_dtype_abbreviation(class_member)
            format_list.append(numpy_type_abbreviation)


        return {'names': names_list, 'formats': format_list}

eeg = EEG()
eeg1 = EEG()
eeg1.aa = 21.2
eeg1.ab = 'dupa1'
eeg1.ca = 11
eeg1.m *= 2

dtype_dict = {'names': ['aa'], 'formats': ['float']}

reconstructed_array = np.recarray(shape=(2,2), dtype=dtype_dict)
print 'reconstructed_array=', reconstructed_array







print eeg.__dict__

eeg.to_record()


eeg_record_format =  get_record_format(eeg)
print 'eeg_record_format=', eeg_record_format

eeg_des = deserialize_single_object_from_matlab_format('eeg_array.mat', 'eeg_array')




eeg_des_demo_element = eeg_des[0]
print 'eeg_des_demo_element =',eeg_des_demo_element

print '------------------extracting new format'

eeg_des_record_format =  get_record_format(eeg_des_demo_element )
print '--------- extracted eeg_des_record_format=',eeg_des_record_format

reconstructed_array = np.recarray(shape=eeg_des.shape, dtype=eeg_des_record_format)

print 'reconstructed_array=',reconstructed_array

eeg_des_1 = deserialize_single_object_from_matlab_format_struct_as_record('eeg_array.mat','eeg_array')
print eeg_des_1


# eeg_des_1 =  sio.loadmat('eeg_array.mat', squeeze_me=True, struct_as_record=True)

print 'names=',eeg_des_record_format['names']

# reconstructed_array['aa'] = eeg_des_1['aa']
# reconstructed_array['ab'] = eeg_des_1['ab']
# reconstructed_array['ca'] = eeg_des_1['ca']
# reconstructed_array['m'] = eeg_des_1['eeg_array']['m']

print '\n\n\n'
print "eeg_des_1['m']=",eeg_des_1['m'][0]
print "eeg_des_1['m']=",eeg_des_1['m'][1]

print 'eeg_des_1.shape =', eeg_des_1['m'].shape

# reconstructed_array['m'][0] = eeg_des_1['eeg_array']['m'][0]
# reconstructed_array['m'][1] = eeg_des_1['eeg_array']['m'][1]


print 'eeg_des_1   ', type(eeg_des_1['m']), ' shape=',eeg_des_1['m'].shape,\
    'dtype=',eeg_des_1['m'].dtype
#
print 'recon    ', type(reconstructed_array['m']), ' shape=',reconstructed_array['m'].shape, ' dtype=',reconstructed_array['m'].dtype





for field_name in eeg_des_record_format['names']:

    print 'field_name = ', field_name
    print 'array_value=',eeg_des_1[field_name]
    field_val = getattr(eeg_des_demo_element ,field_name)
    print field_val, type(field_val).__name__
    field_val_type = type(field_val).__name__

    if field_val_type == 'ndarray':
        for index, x in np.ndenumerate(reconstructed_array):
            print 'index, x = ',index,reconstructed_array[field_name][index]
            print 'index, x = ',index,eeg_des_1['m'][index]
            reconstructed_array[field_name][index] = eeg_des_1[field_name][index]

        pass
    else:
        reconstructed_array[field_name] = eeg_des_1[field_name]

print 'filled reconstructed_array=',reconstructed_array

read_matlab_matrices_as_numpy_structured_arrays('eeg_array.mat', 'eeg_array')

my_new_array = reinterpret_matlab_matrix_as_structured_array(matlab_matrix_as_python_obj =eeg_des, matlab_matrix_structured=eeg_des_1 )

print 'my_new_array=',my_new_array

my_dict = read_matlab_matrices_as_numpy_structured_arrays('eeg_array.mat', 'eeg_array')

print my_dict




struct_array = read_single_matlab_matrix_as_numpy_structured_array('eeg_array.mat', 'eeg_array')

print 'struct_array=',struct_array



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

# complex_format = [('com', base_format)]

complex_format = {'names':['com'],'formats':[eeg_record_format]}

m_complex = np.empty(2, dtype=complex_format)


print 'm_complex=',m_complex

m_complex['com']['aa'] = 10.1
m_complex['com']['ab'] = 'dupa'

print 'm_complex=',m_complex

print 'm_complex.dtype.descr=',m_complex.dtype.descr
print 'm_complex.dtype.fields=',m_complex.dtype.fields

print 'm_complex.dtype.fields=',m_complex.dtype.subdtype

print 'm_complex.dtype.hasobject=',m_complex.dtype.hasobject

# serialize_objects_in_matlab_format('m_complex.mat',(m_complex,'com'))
#

sio.savemat('m_complex.mat',{'com':m_complex})

m_complex_des = deserialize_single_object_from_matlab_format('m_complex.mat', 'com')

m_complex_des_1 = sio.loadmat('m_complex.mat', squeeze_me=True, struct_as_record=True)

print 'type(m_complex_des)=', type(m_complex_des)
print 'type(m_complex_des[0])=', type(m_complex_des[0])

print 'type(m_complex_des[0].m)=', type(m_complex_des[0])

print type(m_complex_des[0].com)

print m_complex_des.dtype
print dir(m_complex_des[0].com)

print '------------',type(m_complex_des)







print 'm_complex_des[0].com=', m_complex_des[0].com, ' type = ',type(m_complex_des[0].com.m)
print ' dir(m_complex_des[0].com)=', dir(m_complex_des[0].com.aa)




# print type(m_complex_des[0].m)
# print m_complex_des[0].m.dtype



print m_complex_des_1['com']
print m_complex_des_1['com'][0][0]['m']
#
# sys.exit()
#
# print eeg_array[0]
#
#
#
# print eeg_array.dtype




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

print eeg_des1['eeg_array']['aa']

print eeg_des[0].aa

print '---------------',type(eeg_des)

sys.exit()
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
