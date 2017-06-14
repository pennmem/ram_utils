from ReportPipelineBase import ReportPipelineBase



# class ReportPipeline(ReportPipelineBase):
#     def __init__(self, subject=None, experiment=None, task=None, workspace_dir=None , mount_point=None, exit_on_no_change=False,recompute_on_no_status=False):
#         super(ReportPipeline,self).__init__( subject=subject, experiment=experiment, task=task, workspace_dir=workspace_dir, mount_point=mount_point, exit_on_no_change=exit_on_no_change,recompute_on_no_status=recompute_on_no_status)

class ReportPipeline(ReportPipelineBase):
    def __init__(self, **options):
        super(ReportPipeline,self).__init__(**options)

