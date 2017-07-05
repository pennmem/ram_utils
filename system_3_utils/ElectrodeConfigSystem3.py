# from rampy.config import ConfigBase
from collections import OrderedDict
import numpy as np
import os
import errno
from os.path import *
from ptsa.data.readers.IndexReader import JsonIndexReader


class UnparseableConfigException(Exception):
    pass


class Contact():
    SURFACE_AREAS = dict(
        S=.5,  # TODO: These are all probably wrong!
        G=.5,
        D=.25,
    )

    CSV_FORMAT = '{name},{port_num},{jack_num},{surface_area:.4f},{description}'

    def __init__(self, name, port_num, jack_num, surface_area, description):
        self.name = name
        self.port_num = int(port_num)
        self.jack_num = int(jack_num)
        self.surface_area = float(surface_area)
        self.description = description

    def as_dict(self):
        d = dict(
            name=self.name,
            port_num=self.port_num,
            jack_num=self.jack_num,
            surface_area=self.surface_area,
            description=self.description
        )
        return d

    def as_csv(self):
        return self.CSV_FORMAT.format(**self.as_dict())


class SenseChannel():
    # TODO: What is 'x'?
    CSV_FORMAT = '{contact_name},{name},{contact_num},{ref},x,{description}'

    def __init__(self, contact, name, mux, ref, x, description):
        self.contact = contact
        self.name = name
        self.mux = mux
        self.ref = ref
        self.x = x
        self.description = description

    def as_dict(self):
        d = dict(
            contact=self.contact.as_dict(),
            name=self.name,
            mux=self.mux,
            ref=self.ref,
            description=self.description
        )
        return d

    def as_csv(self):
        return self.CSV_FORMAT.format(contact_name=self.contact.name, contact_num=self.contact.port_num,
                                      **self.as_dict())


class StimChannel():
    # TODO: What is x?
    CSV_FORMAT = 'StimChannel:,{name},x,#{comments}#\nAnodes:,{anode_csv},#\nCathodes:,{cathode_csv},#'

    def __init__(self, name, anodes, cathodes, comments):
        self.name = name
        self.anodes = [int(x) for x in anodes]
        self.cathodes = [int(x) for x in cathodes]
        self.comments = comments

    def as_dict(self):
        d = dict(
            name=self.name,
            anodes=self.anodes,
            cathodes=self.cathodes,
            comments=self.comments
        )
        return d

    def as_csv(self):
        anode_csv = ','.join([str(a) for a in self.anodes])
        cathode_csv = ','.join([str(c) for c in self.cathodes])
        return self.CSV_FORMAT.format(anode_csv=anode_csv, cathode_csv=cathode_csv, **self.as_dict())


class ElectrodeConfig(object):
    # TODO: SenseChannelSubclasses, StimulationChannelSubclasses??
    CSV_FORMAT = \
        'ODINConfigurationVersion:,#{config_version}#\n' \
        'ConfigurationName:,{config_name}\n' \
        'SubjectID:,{subject_id}\n' \
        'Contacts:\n' \
        '{contacts_csv}\n' \
        'SenseChannelSubclasses:\n' \
        'SenseChannels:\n' \
        '{sense_channels_csv}\n' \
        'StimulationChannelSubclasses:\n' \
        'StimulationChannels:' \
        '{stim_channels_csv}\n' \
        '{ref}\n' \
        'EOF\n'

    @property
    def contacts_csv(self):
        return '\n'.join([contact.as_csv() for contact in
                          sorted(self.contacts.values(), key=lambda c: c.jack_num)])

    @property
    def sense_channels_csv(self):
        return '\n'.join([sense_channel.as_csv() for sense_channel in
                          sorted(self.sense_channels.values(), key=lambda s: s.contact.jack_num)])

    @property
    def stim_channels_csv(self):
        stim_channels_as_csv = '\n'.join([stim_channel.as_csv() for stim_channel in
                          sorted(self.stim_channels.values(), key=lambda s: s.anodes[0])])
        if stim_channels_as_csv:
            stim_channels_as_csv = '\n'+stim_channels_as_csv
        return stim_channels_as_csv

    def as_csv(self):
        if not self.initialized:
            raise UnparseableConfigException("Config not initialized!")
        return self.CSV_FORMAT.format(
            contacts_csv=self.contacts_csv,
            sense_channels_csv=self.sense_channels_csv,
            stim_channels_csv=self.stim_channels_csv,
            **self.as_dict()
        )

    def as_pairs_csv(self):
        if not self.initialized:
            raise UnparseableConfigException("Config not initialized!")
        return self.CSV_FORMAT.format(
            contacts_csv=self.contacts_csv,
            sense_channels_csv=self.sense_channels_csv,
            stim_channels_csv=self.stim_channels_csv,
            **self.as_dict()
        )


    def __init__(self, filename=None):

        self.electrode_array_dtype = np.dtype(
            [('jack_box_num', '<i8'), ('contact_name', '|S256'), ('port_electrode', '|S256'), ('surface_area', '<f8'),
             ('description', '|S256')])
        self.config_version = None
        self.config_name = None
        self.subject_id = None
        self.contacts = OrderedDict()
        self.sense_channels = OrderedDict()
        self.stim_channels = OrderedDict()
        self.ref = None

        self.parse_fields = dict(
            ODINConfigurationVersion=self.parse_version,
            ConfigurationName=self.parse_name,
            SubjectID=self.parse_id,
            Contacts=self.parse_contacts,
            SenseChannelSubclasses=self.parse_sense_subclasses,
            SenseChannels=self.parse_sense_channels,
            StimulationChannelSubclasses=self.parse_stim_subclasses,
            StimulationChannels=self.parse_stim_channels,
            StimChannel=self.parse_stim_channel,
            REF=self.parse_ref,
            EOF=self.parse_eof
        )

        self.initialized = False

        if filename is not None:
            self.initialize(filename)

    def as_dict(self):
        contacts = OrderedDict()
        sense_channels = OrderedDict()
        stim_channels = OrderedDict()
        for k, v in self.contacts.items(): contacts[k] = v.as_dict()
        for k, v in self.sense_channels.items(): contacts[k] = v.as_dict()
        for k, v in self.stim_channels.items(): contacts[k] = v.as_dict()
        d = OrderedDict(
            config_version=self.config_version,
            config_name=self.config_name,
            subject_id=self.subject_id,
            contacts=contacts,
            sense_channels=sense_channels,
            stim_channels=stim_channels,
            ref=self.ref
        )
        return d

        # contacts={k:v.as_dict() for k,v in self.contacts.items()},
        # sense_channels={k:v.as_dict() for k,v in self.sense_channels.items()},
        # stim_channels={k:v.as_dict() for k,v in self.stim_channels.items()},

    def sense_channels_as_recarray(self):

        e_array = np.recarray((len(self.sense_channels),), dtype=self.electrode_array_dtype)

        for counter, (chan_name, chan_data) in enumerate(self.sense_channels.items()):
            contact_data = chan_data.contact
            e_array[counter]['jack_box_num'] = int(contact_data.jack_num)
            e_array[counter]['contact_name'] = str(contact_data.name)
            e_array[counter]['port_electrode'] = str(contact_data.port_num)
            e_array[counter]['surface_area'] = float(contact_data.surface_area)
            e_array[counter]['description'] = str(contact_data.description)

        return e_array

    def initialize(self, config_filename):
        with open(config_filename, 'r') as config_file:
            line = next(config_file)
            while (line != False):
                line = line.strip()
                label = line.split(':')[0]
                if label not in self.parse_fields:
                    raise UnparseableConfigException("Could not parse field {}".format(label))
                parser = self.parse_fields[label]
                line = parser(line, config_file)

            self.initialized = True

    def intitialize_from_dict(self, contacts_dict, config_name):
        self.config_version = '1.2'
        self.config_name = config_name
        content = contacts_dict.values()[0]
        self.subject_id = content['code']
        self.ref = 'REF:,0,common'
        for contact_entry in content['contacts'].values():
            code = contact_entry['code']
            channel = contact_entry['channel']
            area = Contact.SURFACE_AREAS[contact_entry['type']]
            description = contact_entry['description']

            # getting rid of commas from description - this fools csv parser
            if description is None:
                description = ''
            description = description.replace(',','-')

            self.contacts[code] = Contact(code, channel, channel, area, '#{}#'.format(description))
            self.sense_channels[code] = SenseChannel(self.contacts[code], code, channel / 32 + 1, '0', 'x',
                                                     '#{}#'.format(description))
        self.initialized = True

    def intitialize_from_pairs_dict(self, pairs_dict, config_name):
        self.config_version = '1.2'
        self.config_name = config_name
        content = pairs_dict.values()[0]
        self.subject_id = content['code']
        self.ref = 'REF:,0,common'
        for pair_entry in content['pairs'].values():
            code = pair_entry['code']
            ch1_code, ch2_code = code.split('-')
            ch1_num, ch2_num = pair_entry['channel_1'], pair_entry['channel_2']
            ch1_type, ch2_type = pair_entry['type_1'], pair_entry['type_2']
            ch1_area, ch2_area = Contact.SURFACE_AREAS[ch1_type], Contact.SURFACE_AREAS[ch2_type]
            description = None

            # getting rid of commas from description - this fools csv parser
            if description is None:
                description = ''
            description = description.replace(',','-')

            self.contacts[ch1_code] = Contact(ch1_code, ch1_num, ch1_num, ch1_area, '#{}#'.format(description))
            self.contacts[ch2_code] = Contact(ch2_code, ch2_num, ch2_num, ch2_area, '#{}#'.format(description))

            self.sense_channels[code] = SenseChannel(self.contacts[ch1_code], ch2_code, ch1_num, ch2_num, 'x',
                                                     '#{}#'.format(description))
        self.initialized = True


    def parse_version(self, line, file):
        self.config_version = line.split(',')[1].strip('#')
        return next(file)

    def parse_name(self, line, file):
        self.config_name = line.split(',')[1].strip()
        return next(file)

    def parse_id(self, line, file):
        self.subject_id = line.split(',')[1].strip()
        return next(file)

    def parse_contacts(self, line, file):
        line = next(file).strip()
        split_line = line.split(',')
        while (len(split_line) == 5):
            self.contacts[split_line[0]] = Contact(*split_line)
            line = next(file).strip()
            split_line = line.split(',')
        return line

    def parse_sense_subclasses(self, line, file):
        # What is this???
        return next(file)

    def parse_sense_channels(self, line, file):
        line = next(file).strip()
        split_line = line.split(',')
        while (len(split_line) == 6):
            self.sense_channels[split_line[1]] = \
                SenseChannel(self.contacts[split_line[0]], *split_line[1:])
            line = next(file).strip()
            split_line = line.split(',')
        return line

    def parse_stim_subclasses(self, line, file):
        # What is this??
        return next(file)

    def parse_stim_channels(self, line, file):
        return next(file)

    def parse_stim_channel(self, line, file):
        split_line = line.split(',')
        name = split_line[1]
        comment = split_line[3]

        # Get anodes
        line = next(file)
        split_line = line.split(':')
        if split_line[0] != 'Anodes':
            raise UnparseableConfigException("Expected \"Anodes\", found {}".format(split_line[0]))
        split_line = line.split(',')
        anodes = split_line[1:-1]
        if len(anodes) == 0:
            raise UnparseableConfigException("Found no anodes for stim channel {}".format(name))

        # Get cathodes
        line = next(file)
        split_line = line.split(':')
        if split_line[0] != "Cathodes":
            raise UnparseableConfigException("Expected \"Cathodes\", found {}".format(split_line[0]))
        split_line = line.split(',')
        cathodes = split_line[1:-1]
        if len(cathodes) == 0:
            raise UnparseableConfigException("Found no cathodes for stim channel {}".format(name))

        if len(cathodes) != len(anodes):
            raise UnparseableConfigException("Number of anodes ({}) "
                                             "did not match number of cathodes ({})".format(len(anodes), len(cathodes)))

        self.stim_channels[name] = StimChannel(name, anodes, cathodes, comment)

        return next(file)

    def parse_ref(self, line, file):
        split_line = line.split(',')
        self.ref = line
        return next(file)

    def parse_eof(self, line, file):
        return False


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def contacts_json_2_configuration_csv(contacts_json_path, output_dir, configuration_label='_ODIN',anodes=(),cathodes=()):
    import json
    ec = ElectrodeConfig()
    contacts_dict = json.load(open(contacts_json_path))
    ec.intitialize_from_dict(contacts_dict, "FromJson")
    for anode,cathode in zip(anodes,cathodes):
        name = '_'.join([anode,cathode])
        ec.stim_channels[name]=StimChannel(name=name,anodes=[ec.contacts[anode].jack_num],
                                               cathodes=[ec.contacts[cathode].jack_num],comments='')
    csv_out = ec.as_csv()
    try:
        mkdir_p(output_dir)
    except AttributeError:
        print '\n\nERROR IN CREATING DIRECTORY:'
        print 'Could not create %s directory' % output_dir
        return False

    out_file_name = join(output_dir,'contacts'+configuration_label+'.csv')

    open(out_file_name, 'w').write(csv_out)
    return True

def pairs_json_2_configuration_csv(pairs_json_path, output_dir, configuration_label='_ODIN',anodes=(),cathodes=()):
    import json
    ec = ElectrodeConfig()
    pairs_dict = json.load(open(pairs_json_path))
    ec.intitialize_from_pairs_dict(pairs_dict, "FromJson")
    for anode,cathode in zip(anodes,cathodes):
        name = '_'.join([anode,cathode])
        ec.stim_channels[name]=StimChannel(name=name,anodes=[ec.contacts[anode].jack_num],
                                               cathodes=[ec.contacts[cathode].jack_num],comments='')
    csv_out = ec.as_csv()
    try:
        mkdir_p(output_dir)
    except AttributeError:
        print '\n\nERROR IN CREATING DIRECTORY:'
        print 'Could not create %s directory' % output_dir
        return False

    out_file_name = join(output_dir,'contacts'+configuration_label+'.csv')

    open(out_file_name, 'w').write(csv_out)
    return True



def test_as_csv():
    import difflib
    ec = ElectrodeConfig()
    # csv_file = r"C:\Users\OdinUser\Desktop\configurations\ThisIsSubjectId_ThisIsConfigName.csv"
    csv_file = r"c:\OdinWiFiServer\ns2\R1170J_ALLCHANNELS107.csv"
    csv_contents = open(csv_file).read()
    ec.initialize(csv_file)
    if not csv_contents == ec.as_csv():
        print(''.join(difflib.ndiff(csv_contents.splitlines(True), ec.as_csv().splitlines(True))))
        #   assert False, "CSV not replicated!"
    else:
        print "CSV successfully replicated"
    return ec


def test_from_dict():
    import json
    ec = ElectrodeConfig()

    contacts_dict = json.load(open(r"C:\OdinWiFiServer\ns2\montage\contacts.json"))

    # contacts_dict = json.load(open(
    #     r"d:\protocols\r1\subjects\R1247P\localizations\1\montages\1\neuroradiology\current_processed\contacts.json"))
    ec.intitialize_from_dict(contacts_dict, "FromJson")
    csv_out = ec.as_csv()
    open(r"C:\OdinWiFiServer\ns2\montage\contacts.csv", 'w').write(csv_out)


# if __name__ == '__main__':
#     from pprint import pprint
#     subject = 'R1232N'
#     localization = 0
#     montage = 0
#     jr = JsonIndexReader('/protocols/r1.json')
#     output_dir = 'D:/experiment_configs1'
#     pairs_json = jr.get_value('pairs',subject=subject,montage=montage)
#
#     stim_channels = ['LAT1-LAT2','LAT3-LAT4']
#
#     (anodes,cathodes) = zip(*[pair.split('-') for pair in stim_channels]) if stim_channels else ([],[])
#
#     success_flag = pairs_json_2_configuration_csv(
#         pairs_json_path=pairs_json,
#         output_dir=output_dir, configuration_label=subject, anodes=anodes, cathodes=cathodes
#     )
#
#     # subject = 'R1232N'
#     # localization = 0
#     # montage = 0
#     # jr = JsonIndexReader('/protocols/r1.json')
#     # output_dir = 'D:/experiment_configs1'
#     # contacts_json = jr.get_value('contacts', subject=subject, montage=montage)
#     #
#     # success_flag = contacts_json_2_configuration_csv(
#     #     contacts_json_path=contacts_json,
#     #     output_dir=output_dir, configuration_label=subject, anodes=[], cathodes=[]
#     # )

if __name__ == '__main__':
    from pprint import pprint

    # ec = test_as_csv()
    test_from_dict()
    # pprint(ec.as_dict())
    # print(ec.as_csv())

