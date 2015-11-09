__author__ = 'm'

import numpy as np
import types

# np.array.__ini

# class MyArray(np.array):
#     def __init__(self):
#         np.array.__init__(self)


class Data(object):
    def __init__(self, a, b):
        self.a = a
        self.b = b
        self.repr = 'Value_'+str(self.a)+'_'+str(self.b)

    def __getitem__(self,item):
        print 'original __getitem__'
        return getattr(self,item)

    def my_get_item(self,item):
        print 'item=',item
        return




d = Data(10,12)

d.c = 12

def my_get_item_new(self,item):
    print 'THIS IS my_get_item_new'
    return getattr(self,item)


# d.my_get_item = types.MethodType(my_get_item_new, d)
d.__getitem__ = types.MethodType(my_get_item_new, d)


# setattr(d, 'getitem_new', my_getitem)
# print d.c


print d['a']

# print d.my_get_item('a')

# a_array = np.array([Data(i, i*2) for i in xrange(10)])
#
#
#
# print a_array.__getitem__(0).repr
#
#
#
# print a_array.__getitem__
#
#
#
# b_array = np.arange(10)
#
#
#
#
# print np.where(b_array<5)
#
