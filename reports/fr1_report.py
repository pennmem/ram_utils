from ReportTasks import EventPreparation,MontagePreparation,ComputePowers,ComputeTTest,ComputeClassifier,ComposeSessionSummary
from ReportTasks.GenerateReportTasks import *
from ReportUtils import ReportPipeline,CMLParser
import os
import numpy as np



parser = CMLParser()





args = parser.parse()

# **********************************************

class Params(object):
    def __init__(self):
        self.penalty_type = 'l2'
        self.C = 7.2e-4

        self.n_perm = 200


        self.log_powers = True
        self.width = 5
        self.filt_order = 4

        self.start_time = 0.0
        self.end_time = 1.366
        self.buf = 1.365

class HFParams(Params):
    def __init__(self):
        super(HFParams,self).__init__()
        self.hfs = np.logspace(np.log10(2), np.log10(200), 50)
        self.hfs = self.hfs[self.hfs>=70.0]


report_pipeline = ReportPipeline(subject=args.subject,task=args.task, sessions = args.sessions,
                                 workspace_dir = os.path.join(args.workspace_dir,args.subject),
                                 exit_on_no_change=args.exit_on_no_change,
                                 recompute_on_no_status=args.recompute_on_no_status)


report_pipeline.add_task(EventPreparation.FREventPreparation(sessions=args.sessions,task=args.task))

report_pipeline.add_task(MontagePreparation.MontagePreparation(params=args))

report_pipeline.add_task(ComputePowers.ComputePowers(params=lf_params))

report_pipeline.add_task(ComputePowers.ComputePowers(params=hf_params))

report_pipeline.add_task(ComputeTTest.ComputeTTest(params=hf_params))

report_pipeline.add_task(ComputeClassifier.ComputeClassifier(params=lf_params,mark_as_completed=True))

report_pipeline.add_task(ComposeSessionSummary.ComposeFR1Summary())

report_pipeline.add_task(GeneratePlots(mark_as_completed=False))

report_pipeline.add_task(GenerateTex(params=tex_params))

report_pipeline.add_task(GenerateReportPDF())


report_pipeline.execute_pipeline()




