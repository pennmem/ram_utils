"""
USAGE:
python odin_config_tool_generator.py --subject SUBJECT --contacts-json CONTACTS --contacts-json-output-dir NEW_CONTACTS [--stim-channels ANODE1-CATHODE1 [ANODE2-CATHODE2 ...]]

"""


from OdinConfigToolGeneratorParser import OdinConfigToolGeneratorParser
from system_3_utils.ElectrodeConfigSystem3 import contacts_json_2_configuration_csv
from ptsa.data.readers.IndexReader import JsonIndexReader
# from ElectrodeConfigSystem3 import contacts_json_2_configuration_csv

cml_parser = OdinConfigToolGeneratorParser(arg_count_threshold=1)
# subject = 'R1002P'
# subject = 'R1065J'
# subject = 'R1095N'
# subject = 'R1118N'
# subject = 'R1121M'
# subject = 'R1162N'
# subject = 'R1284N'
# subject = 'R1250N'
# subject = 'R1247P'
# subject = 'R1175N'
# subject = 'R1212P'
subject = 'R1232N'

localization=0
montage=0

# cml_parser.arg('--subject',subject)
# # cml_parser.arg('--contacts-json',r"d:\protocols\r1\subjects\%s\localizations\%s\montages\%s\neuroradiology\current_processed\contacts.json"%(subject, localization, localization))
# jr=JsonIndexReader('/protocols/r1.json')
# cml_parser.arg('--contacts-json',jr.get_value('contacts',subject=subject,montage=montage))
#
# # cml_parser.arg('--subject','R1247P_1')
# # cml_parser.arg('--contacts-json',r"d:\protocols\r1\subjects\R1247P\localizations\1\montages\1\neuroradiology\current_processed\contacts.json")
# cml_parser.arg('--contacts-json-output-dir','/home1/leond/fr3_config')
# cml_parser.arg('--stim-channels','PG10-PG11')

cml_parser.arg('--subject',subject)
# cml_parser.arg('--contacts-json',r"d:\protocols\r1\subjects\%s\localizations\%s\montages\%s\neuroradiology\current_processed\contacts.json"%(subject, localization, localization))
jr=JsonIndexReader('D:/protocols/r1.json')
cml_parser.arg('--contacts-json',jr.get_value('contacts',subject=subject,montage=montage))

# cml_parser.arg('--subject','R1247P_1')
# cml_parser.arg('--contacts-json',r"d:\protocols\r1\subjects\R1247P\localizations\1\montages\1\neuroradiology\current_processed\contacts.json")
cml_parser.arg('--contacts-json-output-dir','D:/experiment_configs')
# cml_parser.arg('--stim-channels','PG10-PG11','PG11-PG12', ) # R1250N
# cml_parser.arg('--stim-channels','LMD1-LMD2','LMD3-LMD4', ) # R1284N
# cml_parser.arg('--stim-channels','RTT1-RTT2','RTT3-RTT4', ) # R1095N
# cml_parser.arg('--stim-channels','LPF1-LPF2','LPF3-LPF4', ) # R1002P
# cml_parser.arg('--stim-channels','LS1-LS2','LS3-LS4', ) # R1065J
# cml_parser.arg('--stim-channels','G11-G12','G13-G14', ) # R1118N
# cml_parser.arg('--stim-channels','RFG1-RFG2','RFG3-RFG4', ) # R1121M
# cml_parser.arg('--stim-channels','G11-G12','G13-G14', ) # R1162N
# cml_parser.arg('--stim-channels','LAT1-LAT2','LAT3-LAT4', ) # R1175N
# cml_parser.arg('--stim-channels','LXB1-LXB2','LXB3-LXB4', ) # R1212P
cml_parser.arg('--stim-channels','LAT1-LAT2','LAT3-LAT4', ) # R1232N

args = cml_parser.parse()

print args.subject

subject_code=args.subject.split('_')[0]
montage = args.subject.split('_')[1] if len(args.subject.split('_'))>1 else 0
print subject_code,montage

# montage = 1

if not args.contacts_json :
    print 'finding contacts:'
    contacts_json = jr.get_value('contacts',subject=subject_code,montage=montage)
else:
    contacts_json = args.contacts_json

print contacts_json

(anodes,cathodes) = zip(*[pair.split('-') for pair in args.stim_channels]) if args.stim_channels else ([],[])
# generating .csv file for Odin Config Tool based on contacts.json
if args.contacts_json is not None:

    success_flag = contacts_json_2_configuration_csv(
        contacts_json_path=contacts_json,
        output_dir=args.contacts_json_output_dir,configuration_label=args.subject,anodes=anodes,cathodes=cathodes
    )
    if success_flag:
        print 'GENERATED CSV FILE in %s FOR Odin Config Tool'%args.contacts_json_output_dir
    else:
        print 'ERRORS WERE ENCOUNTERED. NO FILE WAS GENERATED'


