__author__ = 'm'

import numpy as np
import matplotlib.pyplot as plt


class PlotData(object):
    def __init__(self, x, y, y_err=None, title=''):
        self.x = x
        self.y = y
        self.y_err = y_err
        self.title = title


class PanelPlot(object):
    def __init__(self, i_max, j_max, title='', x_axis_title='', y_axis_title=''):
        self.i_max = self.i_max
        self.j_max = self.j_max
        self.title = title
        self.x_axis_title = x_axis_title
        self.y_axis_title = y_axis_title

        self.plot_data_matrix = [[None for x in range(j_max)] for x in range(i_max)]

    def add_plot_data(self,i_panel, j_panel, x, y, y_err=None, title=''):

        self.plot_data_matrix[i_panel][j_panel] = PlotData(x, y, y_err, title)

    def generate_plot(self):
        fig  = plt.figure(figsize=(15,15))
        fig.suptitle(self.title, fontsize=16, fontweight='bold')
        # fig.text(x=0.5, y=0.95, s='Minimum 2 cells per cluster' ,fontsize=14, horizontalalignment='center')
        # fig.text(x=0.5, y=0.02, s='Time [Months]' ,fontsize=16, fontweight='bold',horizontalalignment='center')
        import itertools
        for i, j in itertools.product(xrange(self.i_max), xrange(self.j_max)):

            pd = self.plot_data_matrix[i][j]

            ax = plt.subplot2grid((self.i_max,self.j_max),(i, j))

            ax.plot(pd.x,df[function_name+'_'+quantity_name],draw_styles[function_name], label=function_name)


def generate_panel_plot():
    pass