__author__ = 'm'

import numpy as np

# this makes matplotlib independend of the X server - comes handy on clusters
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt


class PlotData(object):
    # def __init__(self, x, y, xerr=None, yerr=None, x_tick_labels=None, y_tick_labels=None, title=''):
    def __init__(self, x, y, **options):
        self.ylabel_fontsize = 12
        self.xlabel_fontsize = 12

        for option_name in ['xerr', 'yerr', 'x_tick_labels', 'y_tick_labels','title', 'ylabel_fontsize','ylabel_fontsize', 'xlim','ylim']:
            try:
                setattr(self, option_name, options[option_name])
                print 'option_name=',option_name,' val=',options[option_name], ' value_check = ', getattr(self, option_name)
            except LookupError:
                setattr(self, option_name, None)

        self.x = x
        self.y = y

        # self.xerr = xerr
        # self.yerr = yerr
        # self.x_tick_labels = x_tick_labels
        # self.y_tick_labels = y_tick_labels
        #
        # self.title = title


class PanelPlot(object):

    def __init__(self, i_max, j_max, title='', x_axis_title='', y_axis_title=''):
        self.i_max = i_max
        self.j_max = j_max
        self.title = title
        self.x_axis_title = x_axis_title
        self.y_axis_title = y_axis_title

        self.plot_data_matrix = [[None for x in range(j_max)] for x in range(i_max)]

    def add_plot_data(self,i_panel, j_panel, x, y, **options):


        print 'i',i_panel,' j ',j_panel, ' x ',x, ' y ',y
        print 'options=',options
        self.plot_data_matrix[i_panel][j_panel] = PlotData(x, y, **options)


    # def add_plot_data(self,i_panel, j_panel, x, y,xerr=None, yerr=None, title=''):
    #
    #     self.plot_data_matrix[i_panel][j_panel] = PlotData(x, y, xerr=xerr, yerr=yerr, title=title)

    def generate_plot(self):
        fig  = plt.figure(figsize=(15,15))
        fig.suptitle(self.title, fontsize=16, fontweight='bold')
        # fig.text(x=0.5, y=0.95, s='Minimum 2 cells per cluster' ,fontsize=14, horizontalalignment='center')

        fig.text(x=0.5, y=0.02, s=self.x_axis_title ,fontsize=16, fontweight='bold',horizontalalignment='center')
        import itertools
        for i, j in itertools.product(xrange(self.i_max), xrange(self.j_max)):

            pd = self.plot_data_matrix[i][j]
            if pd is None:
                print 'Could not find plot data for panel coordinates (i,j)= ',(i,j)
                continue

            ax = plt.subplot2grid((self.i_max,self.j_max),(i, j))


            if j == 0 :

                ax.set_ylabel(self.y_axis_title,fontsize=pd.ylabel_fontsize)

            print 'pd=',pd

            if pd.xerr is not None or pd.yerr is not None:
                # xerr=[xerr, 2*xerr],
                ax.errorbar(pd.x, pd.y, yerr=pd.yerr, fmt='--o')
                ax.set_xlim([np.min(pd.x)-0.5, np.max(pd.x)+0.5])
                # if pd.xlim:
                    # ax.set_xlim(pd.xlim)


                if pd.x_tick_labels is not None:
                    ax.set_xticks(pd.x)
                    ax.set_xticklabels(pd.x_tick_labels)



            else:

                ax.plot(pd.x,pd.y,'bs', label=pd.title)
                ax.set_xlim([np.min(pd.x)-0.5, np.max(pd.x)+0.5])

            if pd.ylim:
                ax.set_ylim(pd.ylim)



            # if pd.xlim:
            #     # ax.set_xlim(pd.xlim)
            #     ax.set_xlim([np.min(pd.x)-0.5, np.max(pd.x)+0.5])
            #
            # if pd.x_tick_labels is not None:
            #     ax.set_xticks(pd.x)
            #     ax.set_xticklabels(pd.x_tick_labels)

            ax.set_xlabel(pd.title, fontsize=pd.xlabel_fontsize)

        return plt

def generate_panel_plot():
    pass


print ''
if __name__== '__main__':

    panel_plot = PanelPlot(i_max=2, j_max=2, title='Random Data 1', x_axis_title='x_axis_label', y_axis_title='y_axis_random')

    panel_plot.add_plot_data(0,0,x=np.arange(10),y=np.random.rand(10), title='data00')
    panel_plot.add_plot_data(0,1,x=np.arange(10),y=np.random.rand(10), title='data01')
    panel_plot.add_plot_data(1,0,x=np.arange(10),y=np.random.rand(10), title='data10')
    panel_plot.add_plot_data(1,1,x=np.arange(10),y=np.random.rand(10), yerr=np.random.rand(10), title='data11')
    plot = panel_plot.generate_plot()
    plot.subplots_adjust(wspace=0.3, hspace=0.3)
    # plt.savefig(join(plotsDir, quantity_name+'.png'), dpi=300,bboxinches='tight')
    plot.savefig('demo_1.pdf', dpi=300, bboxinches='tight')
    # plot.savefig('demo.png', dpi=300, bboxinches='tight')
    # plot.savefig('demo.png')
    # plot.show()

    print 'GOT HERE'
