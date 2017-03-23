from ReportUtils import ReportRamTask
import numpy as np
from PlotUtils import PanelPlot,PlotData,PlotDataCollection

class GeneratePlots(ReportRamTask):
    def __init__(self):
        super(GeneratePlots,self).__init__(mark_as_completed=False)



    def run(self):
        ps_events = self.get_passed_object('ps_events')
        ps_sessions = np.unique(ps_events.session)
        ps4_session_summaries = self.get_passed_object('ps4_session_summaries')
        if ps4_session_summaries:
            for session in ps_sessions:

                session_summary = ps4_session_summaries[session]


                panel_plot  = PanelPlot(i_max = 2, j_max = 1)
                panel_plot.add_plot_data(0,1,x=session_summary.amplitudes,y=session_summary.delta_classifiers,
                                         linestyle='',marker='x',color='black')
                panel_plot.add_plot_data(1,1,x=session_summary)



