import json
from collections import OrderedDict as OD
import collections
from itertools import izip

import sys

class JSONNode:
    def to_JSON(self):
        return json.dumps(self, default=lambda o: o.__dict__,
                          sort_keys=True, indent=4)

class JSONNodeList(list):
    def __init__(self, *args):
        list.__init__(self, *args)

    def output(self,node_name='', indent=0):
        indent_loc = indent
        # s_loc = ' '*indent_loc+'[\n'
        s_loc = ' ' * indent_loc +'"'+ node_name+ '"'+': [\n'
        num_items = len(self)

        for item_num,item in enumerate(self):
            if isinstance(item, JSONNode):
                s_loc += item.output(indent=indent_loc)
                if item_num == num_items-1:
                    s_loc += ' '*indent_loc+'\n'
                else:
                    s_loc += ',\n'
            else:
                raise TypeError('Trying to output element of type '+str(type(item))+'. Note that '+self.__class__.__name__+' members can be only of type JSONNode')

        s_loc += ' '*indent_loc+']'

        return s_loc

class JSONNode(collections.OrderedDict):
    def __init__(self, *args):
        collections.OrderedDict.__init__(self)

        self.default_indent = 4

        l = list(args)
        if len(l) / 2:
            l.append('')
        l_iter = iter(l)

        # Note: izip will call internally next() on every iterable passed to izip
        # this in effect produces "pairwise iteration" if we pass an iterator pointing to the same
        for key, val in izip(l_iter, l_iter):
            print (str(key), val)
            self[key] = val

    def get_header(self,indent):
        pass

    def output(self, node_name='',indent=0):
        indent_loc = indent
        if node_name:
            s_loc = ' ' * indent_loc +'"'+ node_name+ '"'+': {\n'
        else:
            s_loc = ' ' * indent_loc + '{\n'
        indent_loc += self.default_indent

        num_keys = len(self.keys())
        for i, (k, v) in enumerate(self.items()):

            if isinstance(v, JSONNode) or isinstance(v, JSONNodeList):
                s_loc += v.output(k,indent_loc)

                if i == num_keys-1:
                    s_loc += '\n'
                    # s_loc += ' ' * indent_loc + '}\n'
                else:
                    s_loc += ',\n'
                    # s_loc += ' ' * indent_loc+'},\n'
                    # s_loc += ' ' * indent_loc+',\n'
            else:
                s_loc += ' ' * indent_loc
                # s_loc += '"'+k + '"'+ ': ' + v + ',\n'
                s_loc += '"'+k + '"'+ ': ' + v

                if i == num_keys-1:
                    s_loc += '\n'
                    # s_loc += ' ' * indent_loc + '}\n'
                else:
                    # s_loc += ',\n'
                    s_loc += ',\n'

        # s_loc=s_loc[:-2]+'\n'
        indent_loc -= self.default_indent
        # s_loc += ' ' * indent_loc + '},\n'

        s_loc += ' ' * indent_loc + '}'

        return s_loc






t_jn = JSONNode('tal_bipolar', 'path')
subject_jn =  JSONNode('code','12','code1','13')
subject_jn1 =  JSONNode('code','112','code1','113')

j_list = JSONNodeList([subject_jn,subject_jn1])

print j_list.output(indent = 4)



jn = JSONNode('navigation', subject_jn,'mavigation', subject_jn1)

jn = JSONNode()
jn['navigation'] = subject_jn
jn['mavigation'] = subject_jn1

jn['list_mavigation'] = j_list

# j_list = JSONNodeList([JSONNode('a0',subject_jn),JSONNode('a1',subject_jn1)])





print jn.output()

jfile = open('j.json','w')
print >>jfile,jn.output()
jfile.close()

print json.dumps(jn,indent=4)


import json

with open("j.json",'r') as json_file:
    json_data = json.load(json_file)
    print json_data
    print json.dumps(json_data,indent=4)


subject_navigation_R1060M={
    'subject':{'code':'R1060M'},
    'electrode_info':{
        'tal_bipolar':{'path':'eeg/R1060M/tal/R1060M_talLocs_database_bipol.mat','sha1':'sjhdgskjhdgwueygfuysdgvkjhsadi'},
        'tal_monopolar':{'path':'eeg/R1060M/tal/R1111M_talLocs_database_monopol.mat', 'sha1':'sjhdgskjhdgwueygfuysdgvkjhsadi'},
    },
    'eeg_data':{
        'eeg_noreref_dir':{'path':'eeg/R1060M/eeg.noreref'},
        'eeg_reref_dir':{'path':'eeg/R1060M/eeg.reref'},
    },
    'experiments':{
        'FR1':{
            'events':{'path':'events/RAM_FR1/R1060M_events.mat','sha1':'sjhdgskjhdgwueygfuysdgvkjhsadi'},
            'math_events':{'path':'events/RAM_FR1/R1060M_math.mat','sha1':'sjhdgskjhdgwueygfuysdgvkjhsadi'},
            'experiment_info':{'path':'events/RAM_FR1/R1060M_expinfo.mat','sha1':'sjhdgskjhdgwueygfuysdgvkjhsadi'},
            'experiment_description':'Free Recall - record only'
        },
        'FR2':{
            'events':{'path':'events/RAM_FR2/R1060M_events.mat','sha1':'sjhdgskjhdgwueygfuysdgvkjhsadi'},
            'math_events':{'path':'events/RAM_FR2/R1060M_math.mat','sha1':'sjhdgskjhdgwueygfuysdgvkjhsadi'},
            'experiment_info':{'path':'events/RAM_FR2/R1060M_expinfo.mat','sha1':'sjhdgskjhdgwueygfuysdgvkjhsadi'},
            'experiment_description':'Free Recall - open-loop 5 second stimulation '
        },
        'PS':{
            'events':{'path':'events/RAM_PS/R1060M_events.mat','sha1':'sjhdgskjhdgwueygfuysdgvkjhsadi'},
            'experiment_info':{'path':'events/RAM_FR1/R1060M_expinfo.mat','sha1':'sjhdgskjhdgwueygfuysdgvkjhsadi'},
            'experiment_description':'Parameter Search - Stimulation Experiments'
        }

    }

}



# class JSONNode(collections.OrderedDict):
#     def __init__(self, name=None, *args):
#         collections.OrderedDict.__init__(self)
#
#         self.node_dict = None
#         self.default_indent = 4
#
#         if name is not None:
#             # self.node_dict = JSONNode()
#             self[name]=JSONNode()
#
#
#
#
#         l = list(args)
#         if len(l) / 2:
#             l.append('')
#         l_iter = iter(l)
#
#         # Note: izip will call internally next() on every iterable passed to izip
#         # this in effect produces "pairwise iteration" if we pass an iterator pointing to the same
#         # for key, val in izip(l_iter, l_iter):
#         #     print (str(key), val)
#         #     self[key] = val
#
#         for key, val in izip(l_iter, l_iter):
#             print (str(key), val)
#             self[name][key] = val
#
#
#     def output(self, indent=0):
#         indent_loc = indent
#         s_loc = ''
#         s_loc += ' ' * indent_loc + '{\n'
#         indent_loc += self.default_indent
#         for k, v in self.items():
#             if isinstance(v, JSONNode):
#                 # s_loc += ' '*indent_loc + k + ':'
#                 s_loc += ' '*indent_loc + k + ':\n'
#                 s_loc += v.output(indent_loc)
#             else:
#                 s_loc += ' ' * indent_loc
#                 s_loc += k + ':' + v + ',\n'
#         indent_loc -= self.default_indent
#         s_loc += ' ' * indent_loc + '}\n'
#         return s_loc
#
#
#     # def output(self, indent=0):
#     #     indent_loc = indent
#     #     s_loc = ''
#     #     s_loc += ' ' * indent_loc + '{\n'
#     #     indent_loc += self.default_indent
#     #     for k, v in self.items():
#     #         if isinstance(v, JSONNode):
#     #             # s_loc += ' '*indent_loc + k + ':'
#     #             s_loc += ' '*indent_loc + k + ':\n'
#     #             s_loc += v.output(indent_loc)
#     #         else:
#     #             s_loc += ' ' * indent_loc
#     #             s_loc += k + ':' + v + ',\n'
#     #     indent_loc -= self.default_indent
#     #     s_loc += ' ' * indent_loc + '}\n'
#     #     return s_loc
#
#     # def output(self, indent=0):
#     #     indent_loc = indent
#     #     s_loc = ''
#     #     s_loc += ' ' * indent_loc + '{\n'
#     #     indent_loc += self.default_indent
#     #     for k, v in self.items():
#     #         if isinstance(v, JSONNode):
#     #             s_loc += v.output(indent_loc)
#     #         else:
#     #             s_loc += ' ' * indent_loc
#     #             s_loc += k + ':' + v + ',\n'
#     #     indent_loc -= self.default_indent
#     #     s_loc += ' ' * indent_loc + '}\n'
#     #     return s_loc
#
#     def __str__(self):
#         return repr(self)
#
#
# tal_jn = JSONNode('tal_bipolar', 'path', 'eeg/R1060M/tal/R1060M_talLocs_database_bipol.mat')
#
# # jn = JSONNode('subject', 'code', 'R1111M', 'electrode_info',tal_jn)
#
# # jn = JSONNode('subject', 'electrode_info', tal_jn)
# # jn = JSONNode('subject', 'code', 'R1111M')
# # jn = JSONNode('subject', 'code', 'R1111M','code_new','R1111M_1')
#
# t_jn = JSONNode('tal_bipolar', 'path')
# jn = JSONNode('subject', 'field', t_jn)
#
# print jn.output()
#
# # subject_navigation_R1060M = OD(
# #     [
# #         ('subject',OD(['code','R1060M'])),
# #         ('electrode_info',
# #             OD([
# #
# #             ])
# #          )
# #
# #     ]
# # )
# #
# # subject_navigation_R1060M = OD(
# #     ('subject',OD(code='R1060M')),
# #     ('electrode_info',OD(
# #             ('tal_bipolar',OD(
# #                 ('path','eeg/R1060M/tal/R1060M_talLocs_database_bipol.mat')
# #                 )
# #              ),
# #             ('tal_monopolar',OD(
# #                 ('path','eeg/R1060M/tal/R1060M_talLocs_database_monopol.mat')
# #                 )
# #              ),
# #
# #
# #         )
# #      )
# # )
# #
# # # subject_navigation_R1060M={
# # #     'subject':{'code':'R1060M'},
# # #     'electrode_info':{
# # #         'tal_bipolar':{'path':'eeg/R1060M/tal/R1060M_talLocs_database_bipol.mat'},
# # #         'tal_monopolar':{'path':'eeg/R1060M/tal/R1111M_talLocs_database_monopol.mat'},
# # #     },
# # #
# # # }
# #
# # od = OD([('subject', 'R1060M')])
# # # od['subject']='R1060M'
# # od['electrode'] = 'ddsdsds'
# # #
# # # # print json.dumps(subject_navigation_R1060M,sort_keys=True, indent=4)
# # #
# # # # print json.dumps(subject_navigation_R1060M, indent=4, sort_keys=False)
# # #
# # # # print subject_navigation_R1060M
# # print od
# # #
# # #
