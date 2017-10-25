__author__ = 'm'

from RamPipeline import *
from PlotUtils import *
import numpy as np


class PlotTask(RamTask):
    def __init__(self, mark_as_completed=True):
        RamTask.__init__(self, mark_as_completed)

    def run(self):
        # prepare data sample
        x1 = np.arange(10)
        y1 = 2 * x1 ** 2 + 1
        yerr1 = np.random.rand(10)

        # create PlotData - a convenient way to gather all information about data series
        plot_data_1 = PlotData(
        x=x1, y=y1, yerr=yerr1, xlabel='attempt number', ylabel='score',
        linestyle='dashed',
        color='r', marker='s'
        )

        # create plot data for Bar plotting
        plot_data_2 = BarPlotData(
        x=np.arange(10), y=np.random.rand(10),
        title='data01', yerr=np.random.rand(10) * 0.1,
        x_tick_labels=['a0','a1','a2','a3','a4','a5','a6','a7','a8','a9'],
        barcolors=['r','g','b','r','g','b','r','g','b','r']
        )

        data_frame = np.random.rand(6, 5)
        annotation_dictionary = {(0, 0): 10, (1, 2): 20}
        from itertools import product

        annotation_dictionary = {(i, j): i * j for i, j in product(range(6), range(6))}

        x_tick_labels = ['x0', 'x1', 'x2', 'x3', 'x4']
        y_tick_labels = ['y0', 'y1', 'y2', 'y3', 'y4', 'y5']

        # create plot data for BrickHeatmap plotting
        plot_data_3 = BrickHeatmapPlotData(
        df=data_frame, annot_dict=annotation_dictionary,
        title='random_data_brick_plot',
        x_tick_labels=x_tick_labels, y_tick_labels=y_tick_labels,
        xlabel='XLABEL', ylabel='YLABEL',
        val_lim=[-1.5, 1.5]
        )


        # create Panel Plot (panel contains only 1 plot)
        panel_plot = PanelPlot(i_max=2, j_max=2, title='Scores')

        # add plot data to the (0,0) location in the panel plot
        panel_plot.add_plot_data(0, 0, plot_data=plot_data_1)

        # add bar plot data to the (0,1) location in the panel plot
        panel_plot.add_plot_data(0, 1, plot_data=plot_data_2)

        # add brick heratmap plot data to the (0,1) location in the panel plot
        panel_plot.add_plot_data(1, 1, plot_data=plot_data_3)

        # generate plot - plot is a matplotlib.pyplot module
        plot = panel_plot.generate_plot()

        # save plot as pdf
        plot.savefig('scores.pdf', dpi=300, bboxinches='tight')
