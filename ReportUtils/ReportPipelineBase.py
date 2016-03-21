from RamPipeline import RamPipeline
from ReportUtils.DependencyChangeTrackerLegacy import DependencyChangeTrackerLegacy
from ReportUtils import ReportSummary

class ReportPipelineBase(RamPipeline):
    def __init__(self, subject, workspace_dir, mount_point=None, exit_on_no_change=False):
        RamPipeline.__init__(self)
        self.exit_on_no_change = exit_on_no_change
        self.subject = subject

        self.mount_point = mount_point
        self.set_workspace_dir(workspace_dir)

        dependency_tracker = DependencyChangeTrackerLegacy(subject=subject, workspace_dir=workspace_dir, mount_point=mount_point)
        self.set_dependency_tracker(dependency_tracker=dependency_tracker)

        self.report_summary = ReportSummary()

    def add_report_error(self,error):
        self.report_summary.add_report_error(error)

    def add_report_status_obj(self,status_obj):
        self.report_summary.add_report_status_obj(status_obj)


    def get_report_summary(self):
        return self.report_summary

    def execute_pipeline(self):
        super(ReportPipelineBase,self).execute_pipeline()
        self.report_summary.add_changed_resources(changed_resources=self.dependency_change_tracker.get_changed_resources())