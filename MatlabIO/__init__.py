__author__ = 'm'



from MatlabIO import *




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

