# command line example:
# python thr3_util_system_3.py --workspace-dir=/scratch/busygin/FR3_biomarkers --subject=R1145J_1 --n-channels=128 --anode=RD2 --anode-num=34 --cathode=RD3 --cathode-num=35 --pulse-frequency=200 --pulse-duration=500 --target-amplitude=1000

print "ATTN: Wavelet params and interval length are hardcoded!! To change them, recompile"
print "Windows binaries from https://github.com/busygin/morlet_for_sys2_biomarker"
print "See https://github.com/busygin/morlet_for_sys2_biomarker/blob/master/README for detail."

from os.path import *

from system_3_utils.ram_tasks.CMLParserClosedLoop3 import CMLParserCloseLoop3

cml_parser = CMLParserCloseLoop3(arg_count_threshold=1)
cml_parser.arg('--workspace-dir','/scratch/leond/FR3_biomarkers/R1247P_1')
cml_parser.arg('--experiment','catFR3')
cml_parser.arg('--mount-point','/')
cml_parser.arg('--subject','R1247P_1')
cml_parser.arg('--electrode-config-file',r'/home1/leond/fr3_config/contactsR1247P.csv')
cml_parser.arg('--pulse-frequency','100')
cml_parser.arg('--target-amplitude','1000')
cml_parser.arg('--anode-num','95')
cml_parser.arg('--anode','Rd7')
cml_parser.arg('--cathode-num','97')
cml_parser.arg('--cathode','RE1')



# cml_parser.arg('--workspace-dir','/scratch/busygin/FR3_biomarkers')
# cml_parser.arg('--subject','R1145J_1')
# cml_parser.arg('--n-channels','128')
# cml_parser.arg('--anode-num','3')
# cml_parser.arg('--cathode-num','4')
# cml_parser.arg('--pulse-frequency','200')
# cml_parser.arg('--pulse-count','100')
# cml_parser.arg('--target-amplitude','1000')


args = cml_parser.parse()

# ------------------------------- end of processing command line

from ReportUtils import ReportPipeline

from tests.thr3_biomarker.THREventPreparation import THREventPreparation

from tests.thr3_biomarker.ComputeTHRPowers import ComputeTHRPowers

from tests.thr3_biomarker.MontagePreparation import MontagePreparation

from system_3_utils.ram_tasks.CheckElectrodeConfigurationClosedLoop3 import CheckElectrodeConfigurationClosedLoop3

from tests.thr3_biomarker.ComputeClassifier import ComputeClassifier

from tests.thr3_biomarker.system3.ExperimentConfigGeneratorClosedLoop3 import ExperimentConfigGeneratorClosedLoop3

from tests.thr3_biomarker.thr3_biomarker import Params

import numpy as np





params = Params()



report_pipeline = ReportPipeline(subject=args.subject,
                                       workspace_dir=join(args.workspace_dir,args.subject), mount_point=args.mount_point, args=args)

report_pipeline.add_task(THREventPreparation(mark_as_completed=False))

report_pipeline.add_task(MontagePreparation(mark_as_completed=False))

report_pipeline.add_task(CheckElectrodeConfigurationClosedLoop3(params=params, mark_as_completed=False))

report_pipeline.add_task(ComputeTHRPowers(params=params, mark_as_completed=True))

report_pipeline.add_task(ComputeClassifier(params=params, mark_as_completed=True))

report_pipeline.add_task(ExperimentConfigGeneratorClosedLoop3(params=params, mark_as_completed=False))


#
# # report_pipeline.add_task(SaveMatlabFile(params=params, mark_as_completed=False))

# starts processing pipeline
report_pipeline.execute_pipeline()
