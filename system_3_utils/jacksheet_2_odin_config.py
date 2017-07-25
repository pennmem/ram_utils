"""
This script takes jacksheet.txt and leads.txtx as an input and generates ODIN TOOL .csv electrode configuration tool.
It can generate.csv file for monopolar (legacy) and mixed-mode (aka bipolar) electrode referencing

USAGE:
python odin_config_tool_generator.py --subject SUBJECT  --output-dir OUTPUT_DIR [--stim-channels ANODE1-CATHODE1 [ANODE2-CATHODE2 ...]] [--bipolar]
--jacksheet=PATH_TO_JACKSHEET --leads=PATH_TO_LEADS
"""

from OdinConfigToolGeneratorParser import OdinConfigToolGeneratorParser
from system_3_utils.ElectrodeConfigSystem3 import contacts_json_2_configuration_csv, \
    contacts_json_2_bipol_medtronic_configuration_csv, jacksheet_leads_2_contacts_json
from ptsa.data.readers.IndexReader import JsonIndexReader
import argparse
import sys
from os.path import *


class CMLParser(object):
    def __init__(self, arg_count_threshold=1):
        self.parser = argparse.ArgumentParser(description='Report Generator')
        self.parser.add_argument('--subject', required=True, action='store')
        self.parser.add_argument('--contacts-json', required=False, action='store', default='')
        self.parser.add_argument('--stim-channels', nargs='+', action='store')
        self.parser.add_argument('--output-dir', required=True, action='store', default='')
        self.parser.add_argument('--jacksheet', required=True, action='store', default='')
        self.parser.add_argument('--leads', required=True, action='store', default='')
        self.parser.add_argument('--bipolar', action="store_true", default=False,
                                 help="Enable bipolar (aka mixed-mode) referencing inthe ENS")


        self.arg_list = []
        self.arg_count_threshold = arg_count_threshold

    def arg(self, name, *vals):
        self.arg_list.append(name)
        for val in vals:
            self.arg_list.append(val)

    def configure_python_paths(self, paths):
        for path in paths:
            sys.path.append(path)

    def parse(self):
        print sys.argv
        if len(sys.argv) <= self.arg_count_threshold and len(self.arg_list):
            args = self.parser.parse_args(self.arg_list)
        else:
            args = self.parser.parse_args()

        return args

cml_parser = CMLParser(arg_count_threshold=1)
cml_parser.arg('--stim-channels', 'LAT1-LAT2', 'LAT3-LAT4', )  # R1232N

args = cml_parser.parse()

print args.subject

subject_code = args.subject.split('_')[0]
montage = args.subject.split('_')[1] if len(args.subject.split('_')) > 1 else 0
print subject_code, montage

# if not args.contacts_json:
#     print 'finding contacts:'
#     contacts_json_path = jr.get_value('contacts', subject=subject_code, montage=montage)
# else:
#     contacts_json_path = args.contacts_json

contacts_json_content = jacksheet_leads_2_contacts_json(jacksheet_path=args.jacksheet, leads_path=args.leads, subject=args.subject)
contacts_json_path = join(args.output_dir,'emulated_contacts_%s.json'%args.subject)

contacts_json_content.write(contacts_json_path)


print contacts_json_path

(anodes, cathodes) = zip(*[pair.split('-') for pair in args.stim_channels]) if args.stim_channels else ([], [])
# generating .csv file for Odin Config Tool based on contacts.json
if args.contacts_json is not None:

    if args.bipolar:
        success_flag = contacts_json_2_bipol_medtronic_configuration_csv(
            contacts_json_path=contacts_json_path,
            output_dir=args.output_dir, configuration_label=args.subject, anodes=anodes, cathodes=cathodes
        )
    else:
        success_flag = contacts_json_2_configuration_csv(
            contacts_json_path=contacts_json_path,
            output_dir=args.coutput_dir, configuration_label=args.subject, anodes=anodes, cathodes=cathodes
        )
    if success_flag:
        print 'GENERATED CSV FILE in %s FOR Odin Config Tool' % args.output_dir
    else:
        print 'ERRORS WERE ENCOUNTERED. NO FILE WAS GENERATED'
