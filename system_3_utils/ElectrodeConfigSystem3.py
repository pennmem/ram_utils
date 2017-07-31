# from rampy.config import ConfigBase
from collections import OrderedDict, defaultdict
import numpy as np
import os
import errno
from os.path import *
from ptsa.data.readers.IndexReader import JsonIndexReader
from ReportTasks.hdf5_utils import save_arrays_as_hdf5
from JSONUtils import JSONNode
import json


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
            stim_channels_as_csv = '\n' + stim_channels_as_csv
        return stim_channels_as_csv

    @property
    def monopolar_trans_matrix(self):
        """
        Function that computes transformation matrix that takes mixed mode (a.k. Medtronic bipolar) recording and
        transforms it into monopolar recordings
        The formula that we implement to do bipolar referencing is the following (example):
        E5E1 = E5EREF - E1EREF => E5EREF= E5E1 + E!EREF
        So we measure E1EREF and E5E1 and the task is to recover E5EREF

        :return: {ndarray} transformation matrix - mixed-mode -> monopolar
        """

        num_channels = len(self.sense_channels.keys())

        tr_mat = np.zeros((256, 256), dtype=np.int)
        for i, sense_channel in enumerate(sorted(self.sense_channels.values(), key=lambda s: s.contact.jack_num)):
            port_num = sense_channel.contact.port_num
            tr_mat[port_num - 1, port_num - 1] = 1

            ref_num = int(sense_channel.ref)
            if ref_num != 0:
                # tr_mat[port_num - 1, ref_num - 1] = -1 # this was based on the original document from Medtronic - possibly buggy version

                tr_mat[
                    port_num - 1, ref_num - 1] = 1  # correct implementation that follows pattern described in the docstring

        # removing zero rows and columns
        non_zero_mask = np.any(tr_mat != 0, axis=0)
        tr_mat = tr_mat[non_zero_mask, :]
        tr_mat = tr_mat[:, non_zero_mask]
        return tr_mat

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
        self.num_comm_ref_channels = 2  # number of channels in each bank connected to common reference
        self.bank_capacity = 16 # number of channels in each bank (not each mux)

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

    def initialize_mixed_mode(self,config_filename):
        self.initialize(config_filename=config_filename)
        self.set_mixed_mode_references()

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
            description = description.replace(',', '-')

            self.contacts[code] = Contact(code, channel, channel, area, '#{}#'.format(description))
            self.sense_channels[code] = SenseChannel(self.contacts[code], code, channel / 32 + 1, '0', 'x',
                                                     '#{}#'.format(description))
        self.initialized = True

    @staticmethod
    def bank_id(channel,bank_capacity):
        return channel/bank_capacity  if channel % bank_capacity else (channel/bank_capacity)-1

    def intitialize_from_dict_bipol_medtronic(self, contacts_dict, config_name, references=()):
        """
        This iterates over all contacts stored in contacts.json and produces a list of sense channels
        that implement mixed-mode referencing scheme based on banks of 16 electrodes whenre first 2
        electrodes are connected to C/R and the remaining ones (in a given bank) are referenced to the first electrode
        (in a given bank). The function does not return anything but instead it alters the state of self
        (i.e. ElectrodeConfig class)

        :param contacts_dict: dict representing content of contacts.json
        :param config_name: {str} name of the configuration
        :param references: cufrrently unused
        :return: None
        """
        step_fcn = lambda x: 1 * (x > 0)

        self.config_version = '1.2'
        self.config_name = config_name
        content = contacts_dict.values()[0]
        self.subject_id = content['code']
        self.ref = 'REF:,0,common'
        sorted_contact_values = sorted(content['contacts'].values(), key=lambda x: x['channel'])
        bank_16_capacity = 16

        # reference_contact = -1 # determines reference contact for the current bank

        # first_channel = int(sorted_contact_values[0]['channel'])
        # bank_16_num = first_channel/bank_16_capacity # specifies to which bank of 16 a given contact belongs to

        bank_list_dict = defaultdict(list)

        # bank_id = lambda x, bank_capacity: x / bank_capacity if x % bank_capacity else (x / bank_capacity) - 1

        for contact_entry in sorted_contact_values:
            code = contact_entry['code']
            channel = contact_entry['channel']
            area = Contact.SURFACE_AREAS[contact_entry['type']]
            description = contact_entry['description']

            # getting rid of commas from description - this fools csv parser
            if description is None:
                description = ''
            description = description.replace(',', '-')

            self.contacts[code] = Contact(code, channel, channel, area, '#{}#'.format(description))

            sense_channel_obj = SenseChannel(contact=self.contacts[code],
                                             name=code,
                                             mux=channel / 32 + 1,
                                             ref='0',
                                             x='x',
                                             description='#{}#'.format(description))

            # bank_list_dict[(int(channel)-1)/bank_16_capacity].append(sense_channel_obj)
            bank_list_dict[self.bank_id(int(channel), bank_16_capacity)].append(sense_channel_obj)

        for bank_num in sorted(bank_list_dict.keys()):
            sense_channel_list = bank_list_dict[bank_num]
            ref_sense_channel = sense_channel_list[0]
            for i, sense_channel in enumerate(sense_channel_list):
                if i >= self.num_comm_ref_channels:
                    sense_channel.ref = str(ref_sense_channel.contact.port_num)
                self.sense_channels[sense_channel.contact.name] = sense_channel

        tr_mat = self.monopolar_trans_matrix
        self.initialized = True


    def set_mixed_mode_references(self):
        bank_list_dict = defaultdict(list)
        for sense_channel in self.sense_channels.values():
            channel = sense_channel.contact.port_num
            bank_list_dict[self.bank_id(int(channel), self.bank_capacity)].append(sense_channel)
        for bank_num in sorted(bank_list_dict.keys()):
            sense_channel_list = bank_list_dict[bank_num]
            ref_sense_channel = sense_channel_list[0]
            for i, sense_channel in enumerate(sense_channel_list):
                if i >= self.num_comm_ref_channels:
                    sense_channel.ref = str(ref_sense_channel.contact.port_num)
                self.sense_channels[sense_channel.contact.name] = sense_channel






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
            description = description.replace(',', '-')

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
        comment = split_line[3][1:-1]

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


def contacts_json_2_configuration_csv(contacts_json_path, output_dir, configuration_label='_ODIN', anodes=(),
                                      cathodes=()):
    import json
    ec = ElectrodeConfig()
    contacts_dict = json.load(open(contacts_json_path))
    ec.intitialize_from_dict(contacts_dict, "FromJson")
    for anode, cathode in zip(anodes, cathodes):
        name = '_'.join([anode, cathode])
        ec.stim_channels[name] = StimChannel(name=name, anodes=[ec.contacts[anode].jack_num],
                                             cathodes=[ec.contacts[cathode].jack_num], comments='')
    csv_out = ec.as_csv()
    try:
        mkdir_p(output_dir)
    except AttributeError:
        print '\n\nERROR IN CREATING DIRECTORY:'
        print 'Could not create %s directory' % output_dir
        return False

    out_file_name = join(output_dir, 'contacts' + configuration_label + '.csv')

    open(out_file_name, 'w').write(csv_out)
    return True


def monopolar_to_mixed_mode_config(config_file,output_dir):
    """
    Loads an electrode config file, and saves a mixed-mode config file with the same name and contacts to output_dir.
    :param config_file:
    :param output_dir:
    :return:
    """
    ec = ElectrodeConfig()
    ec.initialize_mixed_mode(config_filename=config_file)
    config_base = os.path.basename(config_file)
    if 'mixed_mode' not in config_base:
        config_base += '_mixed_mode'
    mkdir_p(os.path.abspath(output_dir))
    with open(os.path.join(output_dir,config_base),'w') as out:
        print('Saving %s'%config_base)
        out.write(ec.as_csv())
    return True


def jacksheet_leads_2_contacts_json(jacksheet_path, leads_path, subject):
    """
    Generates "emulated" contact.json that does not include coordinates biut contains channel name, jacksheet number type, description netc
    It is used to interface with the functions that expect contact.json

    :param jacksheet_path: path to jacksheet file
    :param leads_path: path to leads.txt file
    :return: {json-dict} emulated contact.json
    """

    jacksheet_lines = []
    jacksheet_array = None
    jacksheet_dtype = [('jacksheet_num','i4'),('label','|S256')]

    with open(jacksheet_path,'r') as jf:
        for line in jf.readlines():
            line = line.strip()
            jacksheet_lines.append(line.split())

        jacksheet_array = np.empty(len(jacksheet_lines),dtype=jacksheet_dtype)
        for i, line in enumerate(jacksheet_lines):
            jacksheet_array['jacksheet_num'][i] = int(line[0])
            jacksheet_array['label'][i] = line[1]

    leads_list = []
    with open(leads_path,'r') as lf:
        for line in lf.readlines():
            line = line.strip()
            leads_list.append(int(line))

    leads_array = np.array(leads_list,dtype=np.int)

    filter_mask = np.in1d(jacksheet_array['jacksheet_num'],leads_array)

    jacksheet_array = jacksheet_array[filter_mask]

    contacts_jn = JSONNode()
    contacts_jn[subject] = JSONNode(code=subject,contacts='contacts')
    contacts_jn[subject]['contacts']=JSONNode()
    contacts_entry_jn = contacts_jn[subject]['contacts']
    for jacksheet_entry in jacksheet_array:
        label = jacksheet_entry['label']
        jack_num = int(jacksheet_entry['jacksheet_num'])

        contacts_entry_jn[label] = JSONNode(channel=jack_num,code=label,description=None,type='S')

    return contacts_jn


def contacts_json_2_bipol_medtronic_configuration_csv(contacts_json_path, output_dir, configuration_label='_ODIN',
                                                      anodes=(), cathodes=()):
    """
    Converts contacts.json file into Odin Tool .csv file that implements bipolar referencing based on banks of 16 electrodes
    where first 4 electrodes int he bank are connected to C/R. Ut also saves transformation matrix (in the hdf5 format)
    that allows recovery of monopolar recordings

    :param contacts_json_path: path to contacts_json_path
    :param output_dir: output directory for the .csv and .h5
    :param configuration_label: label that gets inserted into .csv file name
    :param anodes: list of stim anodes
    :param cathodes: list of stim cathodes
    :return: {boolean} flag that tells if the execution of function finished or not
    """
    import json
    ec = ElectrodeConfig()
    contacts_dict = json.load(open(contacts_json_path))
    ec.intitialize_from_dict_bipol_medtronic(contacts_dict, "FromJsonBpolAuto")
    for anode, cathode in zip(anodes, cathodes):
        name = '_'.join([anode, cathode])
        ec.stim_channels[name] = StimChannel(name=name, anodes=[ec.contacts[anode].jack_num],
                                             cathodes=[ec.contacts[cathode].jack_num], comments='')
    csv_out = ec.as_csv()
    monopolar_trans_matrix = ec.monopolar_trans_matrix
    try:
        mkdir_p(output_dir)
    except AttributeError:
        print '\n\nERROR IN CREATING DIRECTORY:'
        print 'Could not create %s directory' % output_dir
        return False

    out_file_name = join(output_dir, 'contacts' + configuration_label + '.csv')
    open(out_file_name, 'w').write(csv_out)

    # we decided not to save monopolar_trans_matrix

    # monopolar_trans_matrix_fname = join(output_dir, 'monopolar_trans_matrix' + configuration_label + '.h5')
    #
    # save_arrays_as_hdf5(fname=monopolar_trans_matrix_fname,
    #                     array_dict={'monopolar_trans_matrix': monopolar_trans_matrix})
    #
    return True


def pairs_json_2_configuration_csv(pairs_json_path, output_dir, configuration_label='_ODIN', anodes=(), cathodes=()):
    import json
    ec = ElectrodeConfig()
    pairs_dict = json.load(open(pairs_json_path))
    ec.intitialize_from_pairs_dict(pairs_dict, "FromJson")
    for anode, cathode in zip(anodes, cathodes):
        name = '_'.join([anode, cathode])
        ec.stim_channels[name] = StimChannel(name=name, anodes=[ec.contacts[anode].jack_num],
                                             cathodes=[ec.contacts[cathode].jack_num], comments='')
    csv_out = ec.as_csv()
    try:
        mkdir_p(output_dir)
    except AttributeError:
        print '\n\nERROR IN CREATING DIRECTORY:'
        print 'Could not create %s directory' % output_dir
        return False

    out_file_name = join(output_dir, 'contacts' + configuration_label + '.csv')

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


if __name__ == '__main__':
    from pprint import pprint

    # subject = 'R1232N'
    subject = 'R1111M'
    localization = 0
    montage = 0
    jr = JsonIndexReader('/protocols/r1.json')
    output_dir = 'D:/experiment_configs1'
    contacts_json_path = jr.get_value('contacts', subject=subject, montage=montage)

    # # from JSONUtils import JSONNode
    # # j_read = JSONNode().read(filename="d:\experiment_configs1\contacts_R1111M_demo.json")
    # # contacts_json = j_read['R1111M']['contacts']
    # contacts_json_path = 'd:\experiment_configs1\contacts_R1111M_demo.json'

    jacksheet_path = 'D:/experiment_configs1/jacksheet_R1111M.txt'
    leads_path = 'D:/experiment_configs1/leads_R1111M.txt'
    contacts_json_content = jacksheet_leads_2_contacts_json(jacksheet_path=jacksheet_path, leads_path=leads_path, subject='R1111M')

    # stim_channels = ['LAT1-LAT2', 'LAT3-LAT4']
    stim_channels = ['LPOG14-LPOG15', 'LPOG15-LPOG16']
    (anodes, cathodes) = zip(*[pair.split('-') for pair in stim_channels]) if stim_channels else ([], [])

    success_flag = contacts_json_2_bipol_medtronic_configuration_csv(
        contacts_json_path=contacts_json_path,
        output_dir=output_dir, configuration_label=subject, anodes=anodes, cathodes=cathodes
    )




