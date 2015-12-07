__author__ = 'm'

import numpy as np

# this makes matplotlib independend of the X server - comes handy on clusters
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt


class PlotData(object):
    # def __init__(self, x, y, xerr=None, yerr=None, x_tick_labels=None, y_tick_labels=None, title=''):
    def __init__(self, **options):
        '''
        Initializes PlotData
        :param options: options are  'x', 'y', 'xerr', 'yerr', 'x_tick_labels', 'y_tick_labels','title',
        'ylabel_fontsize','ylabel_fontsize', 'xlim','ylim','xhline_pos','xlabel','ylabel','linestyle','color','marker'
        :return:
        '''
        self.ylabel_fontsize = 12
        self.xlabel_fontsize = 12

        for option_name in ['x', 'y', 'xerr', 'yerr', 'x_tick_labels', 'y_tick_labels','title',
                            'ylabel_fontsize','ylabel_fontsize', 'xlim','ylim','xhline_pos', 'xlabel','ylabel','linestyle','color','marker']:
            try:
                setattr(self, option_name, options[option_name])
                print 'option_name=',option_name,' val=',options[option_name], ' value_check = ', getattr(self, option_name)
            except LookupError:
                setattr(self, option_name, None)

        # setting reasonable defaults
        if self.linestyle is None:
            self.linestyle='-'
        if self.color is None:
            self.color='black'
        if self.marker is None:
            self.marker=''



        if self.x is None or self.y is None:
            raise AttributeError('PlotData requires that x and y attributes are initialized. Use PlotData(x=x_array,y=y_array) syntax')

class BarPlotData(object):
    # def __init__(self, x, y, xerr=None, yerr=None, x_tick_labels=None, y_tick_labels=None, title=''):
    def __init__(self, **options):
        '''
        Initializes PlotData
        :param options: options are  'x', 'y', 'xerr', 'yerr', 'x_tick_labels', 'y_tick_labels','title',
        'ylabel_fontsize','ylabel_fontsize', 'xlim','ylim','xhline_pos','xlabel','ylabel','linestyle','color','marker',
        'barcolors'
        :return:
        '''
        self.ylabel_fontsize = 12
        self.xlabel_fontsize = 12

        for option_name in ['x', 'y', 'xerr', 'yerr', 'x_tick_labels', 'y_tick_labels','title',
                            'ylabel_fontsize','ylabel_fontsize', 'xlim','ylim','xhline_pos', 'xlabel','ylabel','linestyle','color','marker','barcolors']:
            try:
                setattr(self, option_name, options[option_name])
                print 'option_name=',option_name,' val=',options[option_name], ' value_check = ', getattr(self, option_name)
            except LookupError:
                setattr(self, option_name, None)

        # setting reasonable defaults
        if self.linestyle is None:
            self.linestyle='-'
        if self.color is None:
            self.color='black'
        if self.marker is None:
            self.marker=''



        if self.x is None or self.y is None:
            raise AttributeError('PlotData requires that x and y attributes are initialized. Use PlotData(x=x_array,y=y_array) syntax')


class PanelPlot(object):

    def __init__(self, **options):
        '''
        Initializes PanelPlot
        :param options: options are: 'i_max', 'j_max', 'title', 'xtitle', 'ytitle', 'wspace', 'hspace','xfigsize','yfigsize'
        :return: None
        '''
        for option_name in ['i_max', 'j_max', 'title', 'xtitle', 'ytitle', 'wspace', 'hspace','xfigsize','yfigsize']:
            try:
                setattr(self, option_name, options[option_name])
                print 'option_name=',option_name,' val=',options[option_name], ' value_check = ', getattr(self, option_name)
            except LookupError:
                setattr(self, option_name, None)


        self.plot_data_matrix = [[None for x in range(self.j_max)] for x in range(self.i_max)]

    def add_plot_data(self,i_panel, j_panel, **options):
        '''
        Adds PlotData to the proper location in the panel plot
        :param i_panel: x position of the plot in the panel grid
        :param j_panel: y position of the plot in the panel grid
        :param options: same options you would pass to PlotData. if one of the options is plot_data than
        the rest of the options gets ignored
        :return:None
        '''

        print 'i',i_panel,' j ',j_panel
        print 'options=',options
        try:
            pd = options['plot_data']
        except LookupError:
            pd = PlotData(**options)

        self.plot_data_matrix[i_panel][j_panel] = pd
        # self.plot_data_matrix[i_panel][j_panel] = PlotData(x, y, **options)


    # def add_plot_data(self,i_panel, j_panel, x, y,xerr=None, yerr=None, title=''):
    #
    #     self.plot_data_matrix[i_panel][j_panel] = PlotData(x, y, xerr=xerr, yerr=yerr, title=title)

    def generate_plot(self):
        '''
        grid layout numbering is as follows:
        ---------------------------------------------
        |
        |   (0,0)       (0,1)       (0,2)
        |
        |
        |
        |
        |   (1,0)       (1,1)       (1,2)
        |
        |
        |
        |
        |   (2,0)       (2,1)       (2,2)
        |
        ---------------------------------------------
        :return:
        '''

        fig = None
        if self.xfigsize is None or self.yfigsize is None:
            fig  = plt.figure(figsize=(15,15))
        else:
            fig  = plt.figure(figsize=(self.xfigsize,self.yfigsize))

        if self.title is None:
            self.title = ''
        if self.xtitle is None:
            self.xtitle = ''


        fig.suptitle(self.title, fontsize=16, fontweight='bold')
        # fig.text(x=0.5, y=0.95, s='Minimum 2 cells per cluster' ,fontsize=14, horizontalalignment='center')

        fig.text(x=0.5, y=0.02, s=self.xtitle ,fontsize=16, fontweight='bold',horizontalalignment='center')
        import itertools
        for i, j in itertools.product(xrange(self.i_max), xrange(self.j_max)):

            pd = self.plot_data_matrix[i][j]
            if pd is None:
                print 'Could not find plot data for panel coordinates (i,j)= ',(i,j)
                continue

            ax = plt.subplot2grid((self.i_max,self.j_max),(i, j))

            # ax.set_aspect('equal', adjustable='box')


            # y axis labels
            if pd.ylabel is None:
                if j == 0 :
                    ax.set_ylabel(self.ytitle,fontsize=pd.ylabel_fontsize)
            else:
                ax.set_ylabel(pd.ylabel,fontsize=pd.ylabel_fontsize)

            # x axis labels
            if pd.xlabel is None:
                pass
            else:
                ax.set_xlabel(pd.xlabel,fontsize=pd.xlabel_fontsize)



            print 'pd=',pd

            if isinstance(pd,PlotData):

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

                    # lines = ax.plot(pd.x,pd.y,'bs', label=pd.title)
                    # flierprops = dict(marker='o', markerfacecolor='green', markersize=12,
                    #   linestyle='none')
    # linestyles[axisNum], color=color, markersize=10
                    lines = ax.plot(pd.x,pd.y, pd.marker, ls=pd.linestyle, color=pd.color, label=pd.title)

                    ax.set_xlim([np.min(pd.x)-0.5, np.max(pd.x)+0.5])

            # BAR_PLOT_DATA - bar plots
            elif isinstance(pd,BarPlotData):
                inds = np.arange(len(pd.x))
                width = 0.33;
                rects = ax.bar(inds, pd.y, width, color='r',yerr=pd.yerr)
                if pd.x_tick_labels is not None:
                    ax.set_xticks(pd.x)
                    ax.set_xticklabels(pd.x_tick_labels)

                if pd.barcolors is not None:
                    for i, rect  in enumerate(rects):
                        rect.set_color(pd.barcolors[i])



            if pd.ylim:
                ax.set_ylim(pd.ylim)

            if pd.xhline_pos is not None:
                ax.axhline(y=pd.xhline_pos, color='black', ls='dashed')
            # ax.axhline(y=0.5, color='k', ls='dashed')


            # if pd.xlim:
            #     # ax.set_xlim(pd.xlim)
            #     ax.set_xlim([np.min(pd.x)-0.5, np.max(pd.x)+0.5])
            #
            # if pd.x_tick_labels is not None:
            #     ax.set_xticks(pd.x)
            #     ax.set_xticklabels(pd.x_tick_labels)

            # ax.set_xlabel(pd.title, fontsize=pd.xlabel_fontsize)
        if self.wspace is None or self.hspace is None:
            pass
        else:
            fig.subplots_adjust(wspace=self.wspace, hspace=self.hspace)

        return plt

def generate_panel_plot():
    pass


print ''
if __name__== '__main__':

    panel_plot = PanelPlot(xfigsize=15,yfigsize=7.5,  i_max=1, j_max=2, title='Random Data 1', xtitle='x_axis_label', ytitle='y_axis_random')

    panel_plot.add_plot_data(0,0,x=np.arange(10),y=np.random.rand(10), title='data00',linestyle='dashed',color='green',marker='s')
    bpd = BarPlotData(x=np.arange(10),y=np.random.rand(10), title='data01',yerr=np.random.rand(10)*0.1,
                      x_tick_labels=['a0','a1','a2','a3','a4','a5','a6','a7','a8','a9'],
                      barcolors=['r','g','b','r','g','b','r','g','b','r'])
    panel_plot.add_plot_data(0,1,plot_data=bpd)
    # panel_plot.add_plot_data(0,1,x=np.arange(10),y=np.random.rand(10), title='data01')
    # panel_plot.add_plot_data(1,0,x=np.arange(10),y=np.random.rand(10), title='data10')
    # panel_plot.add_plot_data(1,1,x=np.arange(10),y=np.random.rand(10), yerr=np.random.rand(10), title='data11')
    plot = panel_plot.generate_plot()
    plot.subplots_adjust(wspace=0.3, hspace=0.3)
    # plt.savefig(join(plotsDir, quantity_name+'.png'), dpi=300,bboxinches='tight')
    plot.savefig('demo_1.pdf', dpi=300, bboxinches='tight')
    # plot.savefig('demo.png', dpi=300, bboxinches='tight')
    # plot.savefig('demo.png')
    # plot.show()

    print 'GOT HERE'
