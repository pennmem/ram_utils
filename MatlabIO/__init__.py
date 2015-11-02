__author__ = 'm'

import inspect
import scipy.io as sio


class MatlabIO(object):
    def __init__(self):pass

    def items(self):
        for class_member in inspect.getmembers(self, lambda a : not(inspect.isroutine(a))):

            class_member_name = class_member[0]
            class_member_val = class_member[1]

            if not(class_member_name.startswith('__') and class_member_name.endswith('__')):
                # print 'class_member_name=', class_member_name
                yield class_member_name, class_member_val

    def serialize(self, name, format='matlab'):
        sio.savemat(name, self)


    def deserialize(self, name, format='matlab'):
        res = sio.loadmat(name,squeeze_me=True, struct_as_record=False)
        # print res
        # print '\n\n\n'

        # name and val are names and values of the attributes read from .mat file
        for name,val in res.items():
            if not(name.startswith('__') and name.endswith('__')):
                # print 'name=',name, ' val=', val, 'type =', type(val)
                setattr(self,name,val)

        pass