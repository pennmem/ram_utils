__author__ = 'm'

import inspect
import scipy.io as sio


class MatlabIO(object):
    __class_name = ''
    def __init__(self):
        pass

        # self._name = ''

    # def items(self):
    #     '''
    #     Generator that returns followin pairs: class member name, class member value
    #     It only returns non-special members i.e. those whose names do not start with '__' and end with '__'
    #     :return:
    #     '''
    #     for class_member in inspect.getmembers(self, lambda a : not(inspect.isroutine(a))):
    #
    #         class_member_name = class_member[0]
    #         class_member_val = class_member[1]
    #
    #         if not(class_member_name.startswith('__') and class_member_name.endswith('__')):
    #             print 'class_member_name=', class_member_name
    #             yield class_member_name, {class_member_name:class_member_val}

    # def serialize(self, name, format='matlab'):
    #     sio.savemat(name, self)




    def fill_dict(self,a_dict):
        for class_member in inspect.getmembers(self, lambda a : not(inspect.isroutine(a))):

            class_member_name = class_member[0]
            class_member_val = class_member[1]

            if not(class_member_name.startswith('__') and class_member_name.endswith('__')):
                # print 'class_member_name=', class_member_name
                if isinstance(class_member_val, MatlabIO):
                    a_dict[class_member_name] = {}
                    class_member_val.fill_dict(a_dict[class_member_name])
                    # print 'GOT MATLAB IO CLASS'
                else:
                    # print 'LEAF CLASS'
                    a_dict[class_member_name] = class_member_val


    def serialize(self, name, format='matlab'):
        a_dict={}
        top_level_name = type(self).__name__

        print '\n\n\n top_level_name=',top_level_name, ' self__name = ', self.__class_name
        if top_level_name == 'MatlabIO':
            if self.__class_name != '':
                top_level_name = self.__class_name

        # a_dict[type(self).__name__] = {}
        a_dict[top_level_name] = {}
        self.fill_dict( a_dict[top_level_name])
        # self.fill_dict( a_dict[type(self).__name__])

        # print 'a_dict=', a_dict
        sio.savemat(name, a_dict)




    def deserialize(self, name, format='matlab'):
        res = sio.loadmat(name,squeeze_me=True, struct_as_record=False)
        print res
        print '\n\n\n'

        #
        # #count stored items at the top level dict
        # count = 0
        # for name,val in res.items():
        #     if not(name.startswith('__') and name.endswith('__')):
        #         count += 1
        #
        # print 'top_level_items_stored = ',count

        # name and val are names and values of the attributes read from .mat file



        # skip first level
        # for first_level_name, first_level_val in res.items():
        #     print 'first_level_name, first_level_val=',(first_level_name, first_level_val)
        #     if not(first_level_name.startswith('__') and first_level_name.endswith('__')):
        #         print 'first_level_name, first_level_val=',(first_level_name, first_level_val)
        #         print 'dir(first_level_val)=',dir(first_level_val)
        #
        #         print res['durationMS']
        #
        #         for name,val in first_level_val.items():
        #             if not(name.startswith('__') and name.endswith('__')):
        #                 print 'name=',name, ' val=', val, 'type =', type(val)
        #                 setattr(self, name, val)


        for first_level_name, first_level_val in res.items():
            print 'first_level_name, first_level_val=',(first_level_name, first_level_val)
            if not(first_level_name.startswith('__') and first_level_name.endswith('__')):
                print 'setting '
                self.__class_name = first_level_name

                print 'first_level_name, first_level_val=',(first_level_name, first_level_val)
                print 'dir(first_level_val)=',dir(first_level_val)
                # print first_level_val.HilbertBands
                # print res['durationMS']

                for attr_name in dir(first_level_val):
                    if not(attr_name .startswith('__') and attr_name .endswith('__')):
                        print 'attr_name=',attr_name
                            # , ' val=', val, 'type =', type(val)
                        setattr(self, attr_name , getattr(first_level_val,attr_name ))



        #
        # for name,val in res.items():
        #     if not(name.startswith('__') and name.endswith('__')):
        #         print 'name=',name, ' val=', val, 'type =', type(val)
        #         setattr(self,name,val)
        print 'after deserializetion_name =  ', self.__class_name
        pass