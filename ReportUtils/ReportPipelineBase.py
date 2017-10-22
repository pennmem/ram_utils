import sys
from RamPipeline import RamPipeline
from ReportUtils.DependencyChangeTrackerLegacy import DependencyChangeTrackerLegacy
from ReportUtils.ReportSummary import ReportSummary
from ReportUtils.ReportExceptions import MissingExperimentError, MissingDataError, NumericalError
from ReportUtils import ReportDeployer


class ReportPipelineBase(RamPipeline):
    def __init__(self, **options):
        RamPipeline.__init__(self)
        # experiment_label is used to label experiment in the JSON status output file
        self.__option_list = [
            'args', 'subject', 'experiment', 'experiment_label', 'task',
            'workspace_dir', 'mount_point', 'exit_on_no_change',
            'recompute_on_no_status', 'sessions'
        ]

        # sanity check
        for option_name, option_val in options.iteritems():
            if option_name not in self.__option_list:
                raise AttributeError('Unknown option: '+option_name)

        try:
            args = options['args']
        except KeyError:
            args = None

        for option_name  in self.__option_list:
            try:
                # first check in kwds
                option_val = options[option_name]
            except KeyError:
                try:
                    # then check in args object
                    option_val = getattr(args,option_name)
                except AttributeError:
                    # if both fail, set value to None
                    option_val=None

            setattr(self, option_name, option_val)

        if self.experiment and not self.task:
            self.task = self.experiment

        if not self.experiment and  self.task:
            self.experiment = self.task

        self.set_workspace_dir(self.workspace_dir)

        dependency_tracker = DependencyChangeTrackerLegacy(subject=self.subject,
                                                           workspace_dir=self.workspace_dir,
                                                           mount_point=self.mount_point)
        self.set_dependency_tracker(dependency_tracker=dependency_tracker)

        self.report_summary = ReportSummary()
        self.report_site_URL = 'https://memory.psych.upenn.edu/public/'

    def set_experiment_label(self, label):
        self.experiemnt_label = label

    def add_report_error(self, error, stacktrace=None):
        self.report_summary.add_report_error(error, stacktrace=stacktrace)

    def add_report_status_obj(self, status_obj):
        self.report_summary.add_report_status_obj(status_obj)

    def get_report_summary(self):
        return self.report_summary

    def deploy_report(self, report_path='', classifier_experiment=None, suffix=None):
        rd = ReportDeployer.ReportDeployer(pipeline=self)
        rd.deploy_report(report_path=report_path, classifier_experiment=classifier_experiment, suffix=suffix)

    def execute_pipeline(self):
        if hasattr(self,'experiment_label') and self.experiment_label is not None:
            self.report_summary.set_experiment_name(exp_name=self.experiment_label)
        elif hasattr(self, 'experiment'):
            self.report_summary.set_experiment_name(exp_name=self.experiment)
        elif hasattr(self, 'task'):
            self.report_summary.set_experiment_name(exp_name=self.task)
        else:
            self.report_summary.set_experiment_name(exp_name='Unknown_Experiment')

        try:
            super(ReportPipelineBase, self).execute_pipeline()

        except KeyboardInterrupt:
            print 'GOT KEYBOARD INTERUPT. EXITING'
            sys.exit()
        except MissingExperimentError as mee:
            pass
        except MissingDataError:
            pass
        except NumericalError:
            pass
        except Exception as e:
            import traceback
            print(traceback.format_exc())

            self.add_report_error(error=e, stacktrace=traceback.format_exc())

        self.report_summary.add_changed_resources(
            changed_resources=self.dependency_change_tracker.get_changed_resources())
