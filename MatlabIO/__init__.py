from MatlabIO import *
import scipy.io as sio
import numpy as np

__author__ = 'm'


def get_numpy_type_dict():
    from collections import defaultdict
    # d = defaultdict(set)
    numpy_type_dict = defaultdict(list)
    
    
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
                numpy_type_dict[type(nat)].append(npn.dtype.char)
    
            except:
                pass
    

    return numpy_type_dict

numpy_type_dict = get_numpy_type_dict()


def serialize_objects_in_matlab_format(file_name, *object_name_pairs):

    class Serializer(MatlabIO):
        def __init__(self):
            pass

    serializer = Serializer()
    for obj,name in object_name_pairs:
        setattr(serializer, name, obj)

    serializer.serialize(file_name)

def deserialize_objects_from_matlab_format(file_name,*object_names):


    # store deserialized objects in the dictionary and return it later
    object_dict = {}

    try:
        deserializer = MatlabIO()
        deserializer.deserialize(file_name)
    except IOError:
        raise IOError('Could not deserialize ' + file_name)

    object_names_not_found = []
    for object_name in object_names:
        try:

            object_dict[object_name] = getattr(deserializer,object_name)

        except AttributeError:
            object_names_not_found.append(object_name)

    if len(object_names_not_found):

        print 'WARNING: Could not retrieve the following objects:'

        for object_name in object_names_not_found:
            print object_name

    return object_dict

def deserialize_single_object_from_matlab_format(file_name,object_name):

    object_dict = deserialize_objects_from_matlab_format(file_name,object_name)
    try:
        return object_dict[object_name]
    except LookupError:
        return None

def deserialize_objects_from_matlab_format_struct_as_record(file_name,*object_names):

    # store deserialized objects in the dictionary and return it later
    object_dict = {}

    try:
        res = sio.loadmat(file_name,squeeze_me=True, struct_as_record=True)

    except IOError:
        raise IOError('Could not deserialize ' + file_name)




    object_names_not_found = []

    for object_name in object_names:
        try:

            # object_dict[object_name] = getattr(res,object_name)
            object_dict[object_name] = res[object_name]

        except AttributeError:
            object_names_not_found.append(object_name)

    if len(object_names_not_found):

        print 'WARNING: Could not retrieve the following objects:'

        for object_name in object_names_not_found:
            print object_name

    return object_dict

def deserialize_single_object_from_matlab_format_struct_as_record(file_name, object_name):
    object_dict = deserialize_objects_from_matlab_format_struct_as_record(file_name,object_name)
    try:
        return object_dict[object_name]
    except LookupError:
        return None

def read_matlab_matrices_as_numpy_structured_arrays(file_name, *object_names):

    var_record_dict = deserialize_objects_from_matlab_format_struct_as_record(file_name,*object_names)
    var_object_dict = deserialize_objects_from_matlab_format(file_name,*object_names)

    print 'var_record_dict=',var_record_dict
    print 'var_object_dict=',var_object_dict

    structured_array_dict = {}

    for (obj_name, obj), (record_rame, record_val),  in zip(var_object_dict.items(), var_record_dict.items() ):
        structured_array_dict[record_rame] = \
            reinterpret_matlab_matrix_as_structured_array(matlab_matrix_as_python_obj=obj, matlab_matrix_structured=record_val)

    return structured_array_dict

def read_single_matlab_matrix_as_numpy_structured_array(file_name, object_name):
    structured_array_dict = read_matlab_matrices_as_numpy_structured_arrays(file_name, object_name)
    try:
        return structured_array_dict[object_name]
    except LookupError:
        return None



def reinterpret_matlab_matrix_as_structured_array(matlab_matrix_as_python_obj, matlab_matrix_structured ):

    template_element = None
    for index, x in np.ndenumerate(matlab_matrix_as_python_obj):
        template_element = x
        break

    import sys
    # print 'template_element=',template_element


    # print '------------------extracting new format'

    template_element_record_format =  get_record_format(template_element )
    # print '--------- extracted template_element_recort_format=',template_element_record_format

    # idx = 1
    # # template_element_record_format_1 = {'names':template_element_record_format['names'][:idx],'formats':template_element_record_format['formats'][:idx]}
    #
    # template_element_record_format_1 = {'names':['aa'],'formats':[('<f8',())]}
    # # print '--------- extracted new_template=',template_element_record_format_1
    #
    # reconstructed_array = np.recarray(shape=matlab_matrix_as_python_obj.shape, dtype=template_element_record_format_1)
    #
    # # reconstructed_array = np.recarray(shape=matlab_matrix_as_python_obj.shape, dtype=template_element_record_format)
    # print 'reconstructed_array=',reconstructed_array
    # return reconstructed_array
    # # sys.exit()

    reconstructed_array = np.recarray(shape=matlab_matrix_as_python_obj.shape, dtype=template_element_record_format)

    for field_name in template_element_record_format['names']:

        # print 'field_name = ', field_name
        # print 'array_value=',matlab_matrix_structured [field_name]
        field_val = getattr(template_element ,field_name)
        print field_val, type(field_val).__name__
        field_val_type = type(field_val).__name__

        if field_val_type == 'ndarray':
            for index, x in np.ndenumerate(reconstructed_array):
                # print 'index, x = ',index,reconstructed_array[field_name][index]
                # print 'index, x = ',index,matlab_matrix_structured ['m'][index]
                reconstructed_array[field_name][index] = matlab_matrix_structured [field_name][index]

            pass
        else:
            reconstructed_array[field_name] = matlab_matrix_structured [field_name]

    return reconstructed_array





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

        # in case inferred array has zero size we will mark it as Python object
            # '0'

        shape = class_member_val.shape
        if shape[0] == 0:
            numpy_type_char_abbreviation = 'O'


        # numpy_type_char_abbreviation = (class_member_val.dtype.descr[0][1], shape)
        print 'numpy_type_char_abbreviation=',numpy_type_char_abbreviation
        # sys.exit()
    else:

        # ('f2', '>f8', (2, 3))
        numpy_type_char_abbreviation = numpy_type_dict[class_member_type][0]
        if numpy_type_char_abbreviation == 'S':
            numpy_type_char_abbreviation = 'S'+str(default_string_length)
        elif numpy_type_char_abbreviation == 'U':
            numpy_type_char_abbreviation = 'U'+str(default_string_length)

    return numpy_type_char_abbreviation

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

        print 'class_member=',class_member
        print 'class_member_name=',class_member_name
        print 'class_member_value',class_member_value


        try:
            numpy_type_abbreviation = determine_numpy_type_abbreviation(class_member)
            print 'numpy_type_abbreviation=',numpy_type_abbreviation
        except:
            print 'COULD NOT DETERMINE FORMAT FOR:'
            print 'class_member_name=',class_member_value
            print 'class_member_values=',class_member_value
            print 'SKIPPING this'
            continue

        names_list.append(class_member_name)
        format_list.append(numpy_type_abbreviation)


    return {'names': names_list, 'formats': format_list}



# class MatlabIO(object):
#     __class_name = ''
#     def __init__(self):
#         pass
#
#
#     def fill_dict(self,a_dict):
#         for class_member in inspect.getmembers(self, lambda a : not(inspect.isroutine(a))):
#
#             class_member_name = class_member[0]
#             class_member_val = class_member[1]
#
#             if not(class_member_name.startswith('__') and class_member_name.endswith('__')):
#                 # print 'class_member_name=', class_member_name
#                 if isinstance(class_member_val, MatlabIO):
#                     a_dict[class_member_name] = {}
#                     class_member_val.fill_dict(a_dict[class_member_name])
#                     # print 'GOT MATLAB IO CLASS'
#                 else:
#                     # print 'LEAF CLASS'
#                     a_dict[class_member_name] = class_member_val
#
#     def serialize(self, name, format='matlab'):
#         a_dict={}
#         self.fill_dict(a_dict)
#
#         print a_dict
#         sio.savemat(name, a_dict)
#
#
#     def deserialize(self, name, format='matlab'):
#         res = sio.loadmat(name,squeeze_me=True, struct_as_record=False)
#         # res = sio.loadmat(name,squeeze_me=True, struct_as_record=True)
#
#         # print res
#         # print '\n\n\n'
#
#         for attr_name, attr_val in res.items():
#             if not(attr_name .startswith('__') and attr_name .endswith('__')):
#                 # print 'attr_name=',attr_name
#                     # , ' val=', val, 'type =', type(val)
#                 # print 'fetching ',attr_name
#                 setattr(self, attr_name , attr_val)




# class MatlabIO_OLD(object):
#     __class_name = ''
#     def __init__(self):
#         pass
#
#         # self._name = ''
#
#     # def items(self):
#     #     '''
#     #     Generator that returns followin pairs: class member name, class member value
#     #     It only returns non-special members i.e. those whose names do not start with '__' and end with '__'
#     #     :return:
#     #     '''
#     #     for class_member in inspect.getmembers(self, lambda a : not(inspect.isroutine(a))):
#     #
#     #         class_member_name = class_member[0]
#     #         class_member_val = class_member[1]
#     #
#     #         if not(class_member_name.startswith('__') and class_member_name.endswith('__')):
#     #             print 'class_member_name=', class_member_name
#     #             yield class_member_name, {class_member_name:class_member_val}
#
#     # def serialize(self, name, format='matlab'):
#     #     sio.savemat(name, self)
#
#
#
#
#     def fill_dict(self,a_dict):
#         for class_member in inspect.getmembers(self, lambda a : not(inspect.isroutine(a))):
#
#             class_member_name = class_member[0]
#             class_member_val = class_member[1]
#
#             if not(class_member_name.startswith('__') and class_member_name.endswith('__')):
#                 # print 'class_member_name=', class_member_name
#                 if isinstance(class_member_val, MatlabIO_OLD):
#                     a_dict[class_member_name] = {}
#                     class_member_val.fill_dict(a_dict[class_member_name])
#                     # print 'GOT MATLAB IO CLASS'
#                 else:
#                     # print 'LEAF CLASS'
#                     a_dict[class_member_name] = class_member_val
#
#
#     def serialize(self, name, format='matlab'):
#         a_dict={}
#         top_level_name = type(self).__name__
#
#         print '\n\n\n top_level_name=',top_level_name, ' self__name = ', self.__class_name
#         if top_level_name == 'MatlabIO':
#             if self.__class_name != '':
#                 top_level_name = self.__class_name
#
#         # a_dict[type(self).__name__] = {}
#         a_dict[top_level_name] = {}
#         self.fill_dict( a_dict[top_level_name])
#         # self.fill_dict( a_dict[type(self).__name__])
#
#         # print 'a_dict=', a_dict
#         sio.savemat(name, a_dict)
#
#
#
#
#     def deserialize(self, name, format='matlab'):
#         res = sio.loadmat(name,squeeze_me=True, struct_as_record=False)
#         print res
#         print '\n\n\n'
#
#         #
#         # #count stored items at the top level dict
#         # count = 0
#         # for name,val in res.items():
#         #     if not(name.startswith('__') and name.endswith('__')):
#         #         count += 1
#         #
#         # print 'top_level_items_stored = ',count
#
#         # name and val are names and values of the attributes read from .mat file
#
#
#
#         # skip first level
#         # for first_level_name, first_level_val in res.items():
#         #     print 'first_level_name, first_level_val=',(first_level_name, first_level_val)
#         #     if not(first_level_name.startswith('__') and first_level_name.endswith('__')):
#         #         print 'first_level_name, first_level_val=',(first_level_name, first_level_val)
#         #         print 'dir(first_level_val)=',dir(first_level_val)
#         #
#         #         print res['durationMS']
#         #
#         #         for name,val in first_level_val.items():
#         #             if not(name.startswith('__') and name.endswith('__')):
#         #                 print 'name=',name, ' val=', val, 'type =', type(val)
#         #                 setattr(self, name, val)
#
#         # #
#         # for first_level_name, first_level_val in res.items():
#         #     print 'first_level_name, first_level_val=',(first_level_name, first_level_val)
#         #     if not(first_level_name.startswith('__') and first_level_name.endswith('__')):
#         #         print 'setting '
#         #         self.__class_name = first_level_name
#         #
#         #         print 'first_level_name, first_level_val=',(first_level_name, first_level_val)
#         #         print 'dir(first_level_val)=',dir(first_level_val)
#         #         # print first_level_val.HilbertBands
#         #         # print res['durationMS']
#         #
#         #         for attr_name in dir(first_level_val):
#         #             if not(attr_name .startswith('__') and attr_name .endswith('__')):
#         #                 print 'attr_name=',attr_name
#         #                     # , ' val=', val, 'type =', type(val)
#         #                 setattr(self, attr_name , getattr(first_level_val,attr_name ))
#
#
#         #
#
#
#
#
#         for first_level_name, first_level_val in res.items():
#             print 'first_level_name, first_level_val=',(first_level_name, first_level_val)
#             if not(first_level_name.startswith('__') and first_level_name.endswith('__')):
#                 print 'setting '
#                 self.__class_name = first_level_name
#
#                 print 'first_level_name, first_level_val=',(first_level_name, first_level_val)
#                 print 'dir(first_level_val)=',dir(first_level_val)
#                 print 'first_level_val[0]=',first_level_val[0]
#                 # print first_level_val.HilbertBands
#                 # print res['durationMS']
#
#
#                 print 'first_level_val=', first_level_val
#
#                 for attr_name in dir(first_level_val):
#                     # if not(attr_name .startswith('__') and attr_name .endswith('__')):
#                     if not attr_name in ['__class__', '__delattr__', '__dict__', '__doc__', '__format__',\
#                                          '__getattribute__', '__hash__', '__init__', '__module__', \
#                                          '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__',\
#                                          '__sizeof__', '__slotnames__', '__str__', '__subclasshook__', '__weakref__']:
#                         # print 'attr_name=',attr_name
#                             # , ' val=', val, 'type =', type(val)
#                         print 'fetching ',attr_name
#                         setattr(self, attr_name , getattr(first_level_val,attr_name ))
#
#         print 'after setting dir(self) ', dir(self)
#         print self.__getitem__(0)
#
#
#         #
#         # for name,val in res.items():
#         #     if not(name.startswith('__') and name.endswith('__')):
#         #         print 'name=',name, ' val=', val, 'type =', type(val)
#         #         setattr(self,name,val)
#         print 'after deserializetion_name =  ', self.__class_name
#         pass
#
#
# class MatlabIOREADER(object):
#     __class_name = ''
#     def __init__(self):
#         pass
#
#         # self._name = ''
#
#     # def items(self):
#     #     '''
#     #     Generator that returns followin pairs: class member name, class member value
#     #     It only returns non-special members i.e. those whose names do not start with '__' and end with '__'
#     #     :return:
#     #     '''
#     #     for class_member in inspect.getmembers(self, lambda a : not(inspect.isroutine(a))):
#     #
#     #         class_member_name = class_member[0]
#     #         class_member_val = class_member[1]
#     #
#     #         if not(class_member_name.startswith('__') and class_member_name.endswith('__')):
#     #             print 'class_member_name=', class_member_name
#     #             yield class_member_name, {class_member_name:class_member_val}
#
#     # def serialize(self, name, format='matlab'):
#     #     sio.savemat(name, self)
#
#
#
#
#     def fill_dict(self,a_dict):
#         for class_member in inspect.getmembers(self, lambda a : not(inspect.isroutine(a))):
#
#             class_member_name = class_member[0]
#             class_member_val = class_member[1]
#
#             if not(class_member_name.startswith('__') and class_member_name.endswith('__')):
#                 # print 'class_member_name=', class_member_name
#                 if isinstance(class_member_val, MatlabIO_OLD):
#                     a_dict[class_member_name] = {}
#                     class_member_val.fill_dict(a_dict[class_member_name])
#                     # print 'GOT MATLAB IO CLASS'
#                 else:
#                     # print 'LEAF CLASS'
#                     a_dict[class_member_name] = class_member_val
#
#
#     def serialize(self, name, format='matlab'):
#         a_dict={}
#         top_level_name = type(self).__name__
#
#         print '\n\n\n top_level_name=',top_level_name, ' self__name = ', self.__class_name
#         if top_level_name == 'MatlabIO':
#             if self.__class_name != '':
#                 top_level_name = self.__class_name
#
#         # a_dict[type(self).__name__] = {}
#         a_dict[top_level_name] = {}
#         self.fill_dict( a_dict[top_level_name])
#         # self.fill_dict( a_dict[type(self).__name__])
#
#         # print 'a_dict=', a_dict
#         sio.savemat(name, a_dict)
#
#
#
#
#     def deserialize(self, name, format='matlab'):
#         res = sio.loadmat(name,squeeze_me=True, struct_as_record=False)
#         print res
#         print '\n\n\n'
#
#
#         # for first_level_name, first_level_val in res.items():
#         #     if not(first_level_name.startswith('__') and first_level_name.endswith('__')):
#         #
#         #         setattr(self, attr_name , getattr(first_level_val,attr_name ))
#         #         # self.__class_name = first_level_name
#         #
#         #
#         #
#         #         print 'first_level_val=', first_level_val
#
#         for attr_name, attr_val in res.items():
#             if not(attr_name .startswith('__') and attr_name .endswith('__')):
#                 # print 'attr_name=',attr_name
#                     # , ' val=', val, 'type =', type(val)
#                 print 'fetching ',attr_name
#                 setattr(self, attr_name , attr_val)
#
#
#
#
#         #
#         # for name,val in res.items():
#         #     if not(name.startswith('__') and name.endswith('__')):
#         #         print 'name=',name, ' val=', val, 'type =', type(val)
#         #         setattr(self,name,val)
#         # print 'after deserializetion_name =  ', self.__class_name
#         pass
#

