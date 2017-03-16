"""
USAGE:
python odin_config_tool_generator.py --subject SUBJECT --contacts-json CONTACTS --contacts-json-output-dir NEW_CONTACTS [--stim-channels ANODE1-CATHODE1 [ANODE2-CATHODE2 ...]]

"""


from OdinConfigToolGeneratorParser import OdinConfigToolGeneratorParser
from system_3_utils.ElectrodeConfigSystem3 import contacts_json_2_configuration_csv
from ptsa.data.readers.IndexReader import JsonIndexReader
# from ElectrodeConfigSystem3 import contacts_json_2_configuration_csv

cml_parser = OdinConfigToolGeneratorParser(arg_count_threshold=1)
subject = 'R1250N'
localization=0
montage=0

cml_parser.arg('--subject',subject)
# cml_parser.arg('--contacts-json',r"d:\protocols\r1\subjects\%s\localizations\%s\montages\%s\neuroradiology\current_processed\contacts.json"%(subject, localization, localization))
jr=JsonIndexReader('/protocols/r1.json')
cml_parser.arg('--contacts-json',jr.get_value('contacts',subject=subject,montage=montage))

# cml_parser.arg('--subject','R1247P_1')
# cml_parser.arg('--contacts-json',r"d:\protocols\r1\subjects\R1247P\localizations\1\montages\1\neuroradiology\current_processed\contacts.json")
cml_parser.arg('--contacts-json-output-dir','/home1/leond/fr3_config')
cml_parser.arg('--stim-channels','PG10-PG11')

args = cml_parser.parse()
(anodes,cathodes) = zip(*[pair.split('-') for pair in args.stim_channels])
# generating .csv file for Odin Config Tool based on contacts.json
if args.contacts_json is not None:

    success_flag = contacts_json_2_configuration_csv(
        contacts_json_path=args.contacts_json,
        output_dir=args.contacts_json_output_dir,configuration_label=args.subject,anodes=anodes,cathodes=cathodes
    )
    if success_flag:
        print 'GENERATED CSV FILE in %s FOR Odin Config Tool'%args.contacts_json_output_dir
    else:
        print 'ERRORS WERE ENCOUNTERED. NO FILE WAS GENERATED'


