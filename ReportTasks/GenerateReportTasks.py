from ReportUtils import ReportRamTask
from PlotUtils import PlotDataCollection,PlotData,PanelPlot,BarPlotData
from tornado import template



class GeneratePlots(ReportRamTask):
    def get_session_summary(self):
        return self.get_passed_object('session_summary')

    def run(self):
        raise NotImplementedError


    @staticmethod
    def roc_plot(fpr,tpr):
        return PlotData(x=fpr, y=tpr, xlim=[0.0, 1.0], ylim=[0.0, 1.0],
                       xlabel='False Alarm Rate\n(a)', ylabel='Hit Rate', levelline=((0.001, 0.999), (0.001, 0.999)),
                       color='k', markersize=1.0, xlabel_fontsize=18, ylabel_fontsize=18)


class GenerateTex(ReportRamTask):

    def __init__(self,params):
        super(ReportRamTask,self).__init__(mark_as_completed=False)
        if isinstance(params,dict):
            self.params=params
        else:
            self.params=vars(params)


    @property
    def template(self):
        raise NotImplementedError

    @property
    def plots(self):
        return self.get_passed_object('plot_files')

    def run(self):
        tpl = template.Template(self.template)
        self.params.update(**self.plots)
        self.params.update(**self.get_passed_object('session_summary'))
        tex_string = tpl.generate(**self.params)

        self.pass_object('tex_string',tex_string)


class DeployReportPDF(ReportRamTask):
    def __init__(self, mark_as_completed=True):
        super(DeployReportPDF, self).__init__(mark_as_completed)

    def run(self):
        report_file = self.get_passed_object('report_file')
        self.pipeline.deploy_report(report_path=report_file)
