import json
import collections
import sys
from itertools import izip

class JSONNode(collections.OrderedDict):
    def __init__(self, *args, **kwds):
        collections.OrderedDict.__init__(self, *args, **kwds)

        self.default_indent = 4

    @staticmethod
    def read(filename):

        with open(filename, 'r') as json_file:
            json_node = json.load(json_file, object_pairs_hook=JSONNode)

        return json_node

    @staticmethod
    def initialize_form_list(*args):
        jn = JSONNode()
        l = list(args)
        if len(l) / 2:
            l.append('')
        l_iter = iter(l)

        # Note: izip will call internally next() on every iterable passed to izip
        # this in effect produces "pairwise iteration" if we pass an iterator pointing to the same
        for key, val in izip(l_iter, l_iter):
            # print (str(key), val)
            jn[key] = val

        return jn


    def write(self, filename):

        with open(filename, 'w') as json_file:
            json_file.write(self.output())

    def add_child_node(self, *args, **kwds):
        try:
            self[args[0]]=JSONNode()
            return self[args[0]]
        except IndexError:
            return None


    def output_list(self, lst, node_name='', indent=0):
        indent_loc = indent
        s_loc = ' ' * indent_loc + '"' + node_name + '"' + ': [\n'
        num_items = len(lst)

        for item_num, item in enumerate(lst):
            if isinstance(item, JSONNode):
                s_loc += item.output(indent=indent_loc)
                if item_num == num_items - 1:
                    s_loc += ' ' * indent_loc + '\n'
                else:
                    s_loc += ',\n'
            else:
                raise TypeError('Trying to output element of type ' + str(type(
                    item)) + '. Note that list elements in ' + self.__class__.__name__ + '  can be only of type JSONNode')

        s_loc += ' ' * indent_loc + ']'

        return s_loc

    def output(self, node_name='', indent=0):
        indent_loc = indent
        if node_name:
            s_loc = ' ' * indent_loc + '"' + node_name + '"' + ': {\n'
        else:
            s_loc = ' ' * indent_loc + '{\n'
        indent_loc += self.default_indent

        num_keys = len(self.keys())
        for i, (k, v) in enumerate(self.items()):

            if isinstance(v, JSONNode):
                s_loc += v.output(k, indent_loc)

                if i == num_keys - 1:
                    s_loc += '\n'

                else:
                    s_loc += ',\n'
            elif isinstance(v, list):

                s_loc += self.output_list(lst=v, node_name=k, indent=indent_loc)

                if i == num_keys - 1:
                    s_loc += '\n'

                else:
                    s_loc += ',\n'

            else:
                s_loc += ' ' * indent_loc
                s_loc += '"' + k + '"' + ': ' + '"' + str(v)+ '"'

                if i == num_keys - 1:
                    s_loc += '\n'
                else:
                    s_loc += ',\n'

        indent_loc -= self.default_indent

        s_loc += ' ' * indent_loc + '}'

        return s_loc


if __name__ =='__main__':


    j_read = JSONNode().read(filename="j.json")


    print j_read.output()

    j_read.write('json_new.json')
    sys.exit()
#
#     t_jn = JSONNode(tal_bipolar='path')
#     subject_jn =  JSONNode(code='12',code1='13')
#     subject_jn1 =  JSONNode(code='112',code1='113')
#
#     j_list  = [subject_jn,subject_jn1]
#
#
#
#
#
#     jn = JSONNode(navigation='subject_jn',mavigation=subject_jn1)
#
#     jn = JSONNode()
#     jn['navigation'] = subject_jn
#     jn['mavigation'] = subject_jn1
#
#     jn['list_mavigation'] = j_list
#
#     # j_list = JSONNodeList([JSONNode('a0',subject_jn),JSONNode('a1',subject_jn1)])
#
#
#
#
#
#     print jn.output()
#
#     jfile = open('j.json','w')
#     print >>jfile,jn.output()
#     jfile.close()
#
#     print json.dumps(jn,indent=4)
#
#
#     import json
#
#     with open("j.json",'r') as json_file:
#         json_data = json.load(json_file)
#         print json_data
#         print json.dumps(json_data,indent=4)
#
#
#     with open("j.json",'r') as json_file:
#         # json_data = json.load(json_file, object_pairs_hook=collections.OrderedDict)
#         # json_data = json.load(json_file, object_pairs_hook=OrdDictSub)
#         # json_data = json.load(json_file, object_pairs_hook=MyOrderedDict)
#         json_data = json.load(json_file, object_pairs_hook=JSONNode)
#         print json_data
#         print json_data.output()
#
#
#     # subject_navigation_R1060M={
#     #     'subject':{'code':'R1060M'},
#     #     'electrode_info':{
#     #         'tal_bipolar':{'path':'eeg/R1060M/tal/R1060M_talLocs_database_bipol.mat','sha1':'sjhdgskjhdgwueygfuysdgvkjhsadi'},
#     #         'tal_monopolar':{'path':'eeg/R1060M/tal/R1111M_talLocs_database_monopol.mat', 'sha1':'sjhdgskjhdgwueygfuysdgvkjhsadi'},
#     #     },
#     #     'eeg_data':{
#     #         'eeg_noreref_dir':{'path':'eeg/R1060M/eeg.noreref'},
#     #         'eeg_reref_dir':{'path':'eeg/R1060M/eeg.reref'},
#     #     },
#     #     'experiments':{
#     #         'FR1':{
#     #             'events':{'path':'events/RAM_FR1/R1060M_events.mat','sha1':'sjhdgskjhdgwueygfuysdgvkjhsadi'},
#     #             'math_events':{'path':'events/RAM_FR1/R1060M_math.mat','sha1':'sjhdgskjhdgwueygfuysdgvkjhsadi'},
#     #             'experiment_info':{'path':'events/RAM_FR1/R1060M_expinfo.mat','sha1':'sjhdgskjhdgwueygfuysdgvkjhsadi'},
#     #             'experiment_description':'Free Recall - record only'
#     #         },
#     #         'FR2':{
#     #             'events':{'path':'events/RAM_FR2/R1060M_events.mat','sha1':'sjhdgskjhdgwueygfuysdgvkjhsadi'},
#     #             'math_events':{'path':'events/RAM_FR2/R1060M_math.mat','sha1':'sjhdgskjhdgwueygfuysdgvkjhsadi'},
#     #             'experiment_info':{'path':'events/RAM_FR2/R1060M_expinfo.mat','sha1':'sjhdgskjhdgwueygfuysdgvkjhsadi'},
#     #             'experiment_description':'Free Recall - open-loop 5 second stimulation '
#     #         },
#     #         'PS':{
#     #             'events':{'path':'events/RAM_PS/R1060M_events.mat','sha1':'sjhdgskjhdgwueygfuysdgvkjhsadi'},
#     #             'experiment_info':{'path':'events/RAM_FR1/R1060M_expinfo.mat','sha1':'sjhdgskjhdgwueygfuysdgvkjhsadi'},
#     #             'experiment_description':'Parameter Search - Stimulation Experiments'
#     #         }
#     #
#     #     }
#     #
#     # }
#     #
#     #
#     #
