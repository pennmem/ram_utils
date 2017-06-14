__author__ = 'm'
import numpy as np
from ...RamPipeline import *

from ram_utils.PlotUtils import PlotData, PanelPlot

#
# class PlotData(object):
#     # def __init__(self, x, y, xerr=None, yerr=None, x_tick_labels=None, y_tick_labels=None, title=''):
#     def __init__(self, x, y, **options):
#         self.ylabel_fontsize = 12
#         self.xlabel_fontsize = 12
#
#         for option_name in ['xerr', 'yerr', 'x_tick_labels', 'y_tick_labels','title', 'ylabel_fontsize','ylabel_fontsize', 'xlim','ylim']:
#             try:
#                 setattr(self, option_name, options[option_name])
#                 print 'option_name=',option_name,' val=',options[option_name], ' value_check = ', getattr(self, option_name)
#             except LookupError:
#                 setattr(self, option_name, None)
#
#         self.x = x
#         self.y = y
#
#         # self.xerr = xerr
#         # self.yerr = yerr
#         # self.x_tick_labels = x_tick_labels
#         # self.y_tick_labels = y_tick_labels
#         #
        # self.title = title





class XValResults(object):
    def __init__(self):
        self.auc_results = None


class XValPlots(RamTask):
    def __init__(self, params, mark_as_completed=True):
        RamTask.__init__(self, mark_as_completed)
        self.params = params

    def run(self):
        xval_results = self.get_passed_object('xval_results')
        auc_results = xval_results.auc_results

        threshold_values = np.unique(auc_results.t_thresh)

        x = threshold_values
        y = np.empty_like(x)
        yerr = np.empty_like(x)
        y_features = np.empty_like(x)
        y_features_err = np.empty_like(x)

        for i,t_thresh in enumerate(threshold_values):
            auc_results_t_thresh = auc_results[auc_results.t_thresh == t_thresh]

            auc_mean = np.mean(auc_results_t_thresh.auc)
            auc_median = np.median(auc_results_t_thresh.auc)
            auc_min = np.min(auc_results_t_thresh.auc)
            auc_max = np.max(auc_results_t_thresh.auc)

            y[i] = auc_median
            yerr[i] = auc_max-auc_median


            num_features_mean = np.mean(auc_results_t_thresh.num_features)
            num_features_median = np.median(auc_results_t_thresh.num_features)
            num_features_min = np.min(auc_results_t_thresh.num_features)
            num_features_max = np.max(auc_results_t_thresh.num_features)
            
            y_features[i] = num_features_median
            y_features_err[i] = num_features_max - num_features_median


        pd = PlotData(x=x, y=y, yerr=yerr, xhline_pos=0.5, ylim=(0.0, 1.0),ylabel='AUC', xlabel='t-threshold')
        pd_features = PlotData(x=x, y=y_features,yerr=y_features_err,ylabel='Number of features', xlabel='t-threshold')

        pd1 = PlotData(x=x, y=y, yerr=yerr, xhline_pos=0.5, ylim=(0.0, 1.0),ylabel='AUC', xlabel='t-threshold_1')
        pd_features1 = PlotData(x=x, y=y_features,yerr=y_features_err,ylabel='Number of features', xlabel='t-threshold_1')



        # pd_dict={(0,0):pd, (0,1):pd_features,(1,0):pd1, (1,1):pd_features1}

        pd_dict={(0,0):pd, (0,1):pd_features}

        panel_plot = PanelPlot(i_max=2, j_max=2, title='', ytitle='AUC',wspace=0.3,hspace=0.3)

        for plot_panel_index, pd in pd_dict.iteritems():
            print 'plot_panel_index=', plot_panel_index
            print 'pd.x=', pd.x
            print 'pd.y=', pd.y
            # plot_letter = chr(ord('a') + 2 * plot_panel_index[0] + plot_panel_index[1])
            panel_plot.add_plot_data(plot_panel_index[0], plot_panel_index[1], plot_data=pd)
                # ,
                #                      x_tick_labels=pd.x_tick_labels, title='(' + plot_letter + ')', ylim=pd.ylim)

        plot = panel_plot.generate_plot()
        # plot.subplots_adjust(wspace=0.3, hspace=0.3)
        # # plt.savefig(join(plotsDir, quantity_name+'.png'), dpi=300,bboxinches='tight')
        #
        # plot_out_fname = self.get_path_to_resource_in_workspace('reports/' + self.pipeline.experiment + '-' + self.pipeline.subject + '-report_plot_' + session_summary.name + '.pdf')
        #
        # plot.savefig(plot_out_fname, dpi=300, bboxinches='tight')
        #
        # cumulative_plot_data_dict = self.get_passed_object('cumulative_plot_data_dict')
        #
        # panel_plot = PanelPlot(i_max=3, j_max=2, title='', y_axis_title='$\Delta$ Post-Pre Classifier Output')
        #
        # for plot_panel_index, pd in cumulative_plot_data_dict.iteritems():
        #     # print 'plot_panel_index=', plot_panel_index
        #     # print 'pd.x=', pd.x
        #     # print 'pd.y=', pd.y
        #     plot_letter = chr(ord('a') + 2 * plot_panel_index[0] + plot_panel_index[1])
        #     panel_plot.add_plot_data(plot_panel_index[0], plot_panel_index[1], x=pd.x, y=pd.y, yerr=pd.yerr,
        #                              x_tick_labels=pd.x_tick_labels, title='(' + plot_letter + ')', ylim=pd.ylim)
        #
        # plot = panel_plot.generate_plot()
        # plot.subplots_adjust(wspace=0.3, hspace=0.3)
        # # plt.savefig(join(plotsDir, quantity_name+'.png'), dpi=300,bboxinches='tight')
        #
        # plot_out_fname = self.get_path_to_resource_in_workspace('reports/' + self.pipeline.experiment + '-' + self.pipeline.subject + '-report_plot_Cumulative.pdf')

        plot_out_fname = join(self.pipeline.output_dir, 'plot_'+self.pipeline.subject+'.pdf')
        plot.savefig(plot_out_fname, dpi=300, bboxinches='tight')



