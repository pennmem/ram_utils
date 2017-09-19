"""
This script takes contact.json as an input and generates ODIN TOOL .csv electrode configuration tool. It can generate
.csv file for monopolar (legacy) and mixed-mode (aka bipolar) electrode referencing

USAGE:
python odin_config_tool_generator.py --subject SUBJECT --contacts-json CONTACTS --contacts-json-output-dir NEW_CONTACTS [--stim-channels ANODE1-CATHODE1 [ANODE2-CATHODE2 ...]] [--bipolar]

"""

from OdinConfigToolGeneratorParser import OdinConfigToolGeneratorParser
from system_3_utils.ElectrodeConfigSystem3 import contacts_json_2_configuration_csv, \
    contacts_json_2_bipol_medtronic_configuration_csv
from ptsa.data.readers.IndexReader import JsonIndexReader

# from ElectrodeConfigSystem3 import contacts_json_2_configuration_csv

cml_parser = OdinConfigToolGeneratorParser(arg_count_threshold=1)
subject = 'R1232N'

localization = 0
montage = 0

cml_parser.arg('--subject', subject)
# cml_parser.arg('--contacts-json',r"d:\protocols\r1\subjects\%s\localizations\%s\montages\%s\neuroradiology\current_processed\contacts.json"%(subject, localization, localization))
jr = JsonIndexReader('/protocols/r1.json')
cml_parser.arg('--contacts-json', jr.get_value('contacts', subject=subject, montage=montage))

cml_parser.arg('--contacts-json-output-dir', 'D:/experiment_configs')
cml_parser.arg('--stim-channels', 'LAT1-LAT2', 'LAT3-LAT4', )  # R1232N

args = cml_parser.parse()

print args.subject

subject_code = args.subject.split('_')[0]
montage = args.subject.split('_')[1] if len(args.subject.split('_')) > 1 else 0
print subject_code, montage

if not args.contacts_json:
    print 'finding contacts:'
    contacts_json_path = jr.get_value('contacts', subject=subject_code, montage=montage)
else:
    contacts_json_path = args.contacts_json

print contacts_json_path

(anodes, cathodes) = zip(*[pair.split('-') for pair in args.stim_channels]) if args.stim_channels else ([], [])
# generating .csv file for Odin Config Tool based on contacts.json
if args.contacts_json is not None:

    if args.bipolar:
        success_flag = contacts_json_2_bipol_medtronic_configuration_csv(
            contacts_json_path=contacts_json_path,
            output_dir=args.contacts_json_output_dir, configuration_label=args.subject, anodes=anodes, cathodes=cathodes
        )
    else:
        success_flag = contacts_json_2_configuration_csv(
            contacts_json_path=contacts_json_path,
            output_dir=args.contacts_json_output_dir, configuration_label=args.subject, anodes=anodes, cathodes=cathodes
        )
    if success_flag:
        print 'GENERATED CSV FILE in %s FOR Odin Config Tool' % args.contacts_json_output_dir
    else:
        print 'ERRORS WERE ENCOUNTERED. NO FILE WAS GENERATED'
