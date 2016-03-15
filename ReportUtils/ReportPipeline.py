from RamPipeline import RamPipeline

from ReportUtils.DependencyChangeTrackerLegacy import DependencyChangeTrackerLegacy

class ReportPipeline(RamPipeline):
    def __init__(self, subject, experiment, workspace_dir, mount_point=None):
        RamPipeline.__init__(self)
        self.subject = subject
        #self.task = 'RAM_FR1'
        self.experiment = experiment
        self.mount_point = mount_point
        self.set_workspace_dir(workspace_dir)
        dependency_tracker = DependencyChangeTrackerLegacy(subject=subject, workspace_dir=workspace_dir, mount_point=mount_point)

        self.set_dependency_tracker(dependency_tracker=dependency_tracker)
