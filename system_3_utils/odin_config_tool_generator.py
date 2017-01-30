# command line example:
# python fr3_util_system_3.py --workspace-dir=/scratch/busygin/FR3_biomarkers --subject=R1145J_1 --n-channels=128 --anode=RD2 --anode-num=34 --cathode=RD3 --cathode-num=35 --pulse-frequency=200 --pulse-duration=500 --target-amplitude=1000


from OdinConfigToolGeneratorParser import OdinConfigToolGeneratorParser
from system_3_utils.ElectrodeConfigSystem3 import contacts_json_2_configuration_csv


cml_parser = OdinConfigToolGeneratorParser(arg_count_threshold=1)
cml_parser.arg('--subject','R1247P_1')

cml_parser.arg('--contacts-json',r"d:\protocols\r1\subjects\R1247P\localizations\1\montages\1\neuroradiology\current_processed\contacts.json")
cml_parser.arg('--contacts-json-output-dir',r"d:\experiment_configs")


args = cml_parser.parse()


# generating .csv file for Odin Config Tool based on contacts.json
if args.contacts_json is not None:

    success_flag = contacts_json_2_configuration_csv(
        contacts_json_path=args.contacts_json,
        output_dir=args.contacts_json_output_dir,configuration_label=args.subject
    )
    if success_flag:
        print 'GENERATED CSV FILE in %s FOR Odin Config Tool'%args.contacts_json_output_dir
    else:
        print 'ERRORS WERE ENCOUNTERED. NO FILE WAS GENERATED'


