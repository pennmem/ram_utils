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
        'ylabel_fontsize','ylabel_fontsize', 'xlim','ylim','xhline_pos','xlabel','ylabel','linestyle','color','marker',
        'levelline'
        :return:
        '''
        self.ylabel_fontsize = 12
        self.xlabel_fontsize = 12

        for option_name in ['x', 'y', 'xerr', 'yerr', 'x_tick_labels', 'y_tick_labels', 'title',
                            'ylabel_fontsize', 'ylabel_fontsize', 'xlim', 'ylim', 'xhline_pos', 'xlabel', 'ylabel',
                            'linestyle', 'color', 'marker', 'levelline','label']:
            try:
                setattr(self, option_name, options[option_name])
                print 'option_name=', option_name, ' val=', options[option_name], ' value_check = ', getattr(self,
                                                                                                             option_name)
            except LookupError:
                setattr(self, option_name, None)

        # setting reasonable defaults
        if self.linestyle is None:
            self.linestyle = '-'
        if self.color is None:
            self.color = 'black'
        if self.marker is None:
            self.marker = ''

        if self.label is None:
            self.label = ''

        if self.x is None or self.y is None:
            raise AttributeError(
                'PlotData requires that x and y attributes are initialized. Use PlotData(x=x_array,y=y_array) syntax')


class BarPlotData(object):
    # def __init__(self, x, y, xerr=None, yerr=None, x_tick_labels=None, y_tick_labels=None, title=''):
    def __init__(self, **options):
        '''
        Initializes PlotData
        :param options: options are  'x', 'y', 'xerr', 'yerr', 'x_tick_labels', 'y_tick_labels','title',
        'ylabel_fontsize','ylabel_fontsize', 'xlim','ylim','xhline_pos','xlabel','ylabel','linestyle','color','marker',
        'levelline', 'barcolors','barwidth'
        :return:
        '''
        self.ylabel_fontsize = 12
        self.xlabel_fontsize = 12

        for option_name in ['x', 'y', 'xerr', 'yerr', 'x_tick_labels', 'y_tick_labels', 'title',
                            'ylabel_fontsize', 'ylabel_fontsize', 'xlim', 'ylim', 'xhline_pos', 'xlabel', 'ylabel',
                            'linestyle', 'color', 'marker', 'levelline', 'barcolors', 'barwidth', 'alpha','label']:
            try:
                setattr(self, option_name, options[option_name])
                print 'option_name=', option_name, ' val=', options[option_name], ' value_check = ', getattr(self,
                                                                                                             option_name)
            except LookupError:
                setattr(self, option_name, None)

        # setting reasonable defaults
        if self.linestyle is None:
            self.linestyle = '-'
        if self.color is None:
            self.color = 'black'
        if self.marker is None:
            self.marker = ''
        if self.barwidth is None:
            self.barwidth = 0.5

        if self.label is None:
            self.label = ''


        if self.x is None or self.y is None:
            raise AttributeError(
                'PlotData requires that x and y attributes are initialized. Use PlotData(x=x_array,y=y_array) syntax')


class BrickHeatmapPlotData(object):
    # def __init__(self, x, y, xerr=None, yerr=None, x_tick_labels=None, y_tick_labels=None, title=''):
    def __init__(self, **options):
        '''
        Initializes PlotData
        :param options: options are 'df', 'annot_dict','val_lim','x', 'y', 'xerr', 'yerr', 'x_tick_labels', 'y_tick_labels','title',
        'ylabel_fontsize','ylabel_fontsize', 'xlim','ylim','xhline_pos','xlabel','ylabel','linestyle','color','marker',
        'levelline', 'barcolors','colorbar_title','colorbar_title_location'
        :return:
        '''
        self.ylabel_fontsize = 12
        self.xlabel_fontsize = 12

        for option_name in ['df', 'annot_dict', 'val_lim', 'x', 'y', 'xerr', 'yerr', 'x_tick_labels', 'y_tick_labels',
                            'title',
                            'ylabel_fontsize', 'ylabel_fontsize', 'xlim', 'ylim', 'xhline_pos', 'xlabel', 'ylabel',
                            'linestyle', 'color', 'marker', 'levelline', 'barcolors', 'colorbar_title',
                            'colorbar_title_location','label']:
            try:
                setattr(self, option_name, options[option_name])
                print 'option_name=', option_name, ' val=', options[option_name], ' value_check = ', getattr(self,
                                                                                                             option_name)
            except LookupError:
                setattr(self, option_name, None)

        # setting reasonable defaults
        if self.linestyle is None:
            self.linestyle = '-'
        if self.color is None:
            self.color = 'black'
        if self.marker is None:
            self.marker = ''

        if self.label is None:
            self.label = ''

        if self.df is None:
            raise AttributeError(
                'BrickHeatmapPlotData requires that df attribute is initialized - it can be pandas DataFrame object of simply 2D numpy array. Use PlotData(df=df) syntax')


class PlotDataCollection(object):
    def __init__(self, **options):

        for option_name in ['df', 'annot_dict', 'val_lim', 'x', 'y', 'xerr', 'yerr', 'x_tick_labels', 'y_tick_labels',
                            'title',
                            'ylabel_fontsize', 'ylabel_fontsize', 'xlim', 'ylim', 'xhline_pos', 'xlabel', 'ylabel',
                            'linestyle', 'color', 'marker', 'levelline', 'barcolors', 'colorbar_title',
                            'colorbar_title_location','legend_pos','legend_on']:
            try:
                setattr(self, option_name, options[option_name])
                print 'option_name=', option_name, ' val=', options[option_name], ' value_check = ', getattr(self,
                                                                                                             option_name)
            except LookupError:
                setattr(self, option_name, None)

        self.plot_data_list = []
        if self.legend_on is None:
            self.legend_on=False

    def add_plot_data(self, pd):
        self.plot_data_list.append(pd)


class PanelPlot(object):
    def __init__(self, **options):
        '''
        Initializes PanelPlot
        :param options: options are: 'i_max', 'j_max', 'title', 'xtitle', 'ytitle', 'wspace', 'hspace','xfigsize','yfigsize'
        :return: None
        '''
        for option_name in ['i_max', 'j_max', 'title', 'xtitle','xtitle_fontsize', 'ytitle', 'ytitle_fontsize', 'wspace', 'hspace', 'xfigsize', 'yfigsize']:
            try:
                setattr(self, option_name, options[option_name])
                print 'option_name=', option_name, ' val=', options[option_name], ' value_check = ', getattr(self,
                                                                                                             option_name)
            except LookupError:
                setattr(self, option_name, None)

        self.plot_data_matrix = [[None for x in range(self.j_max)] for x in range(self.i_max)]

    def add_plot_data(self, i_panel, j_panel, **options):
        '''
        Adds PlotData to the proper location in the panel plot
        :param i_panel: x position of the plot in the panel grid
        :param j_panel: y position of the plot in the panel grid
        :param options: same options you would pass to PlotData. if one of the options is plot_data than
        the rest of the options gets ignored
        :return:None
        '''

        print 'i', i_panel, ' j ', j_panel
        print 'options=', options
        try:
            pd = options['plot_data']
        except LookupError:
            pd = PlotData(**options)

        self.plot_data_matrix[i_panel][j_panel] = pd
        # self.plot_data_matrix[i_panel][j_panel] = PlotData(x, y, **options)

    def add_plot_data_collection(self, i_panel, j_panel, **options):
        '''
        Adds PlotData to the proper location in the panel plot
        :param i_panel: x position of the plot in the panel grid
        :param j_panel: y position of the plot in the panel grid
        :param options: same options you would pass to PlotData. if one of the options is plot_data than
        the rest of the options gets ignored
        :return:None
        '''

        print 'i', i_panel, ' j ', j_panel
        print 'options=', options
        try:
            pd = options['plot_data_collection']
        except LookupError:
            pd = PlotDataCollection(**options)

        self.plot_data_matrix[i_panel][j_panel] = pd

    # def add_plot_data(self,i_panel, j_panel, x, y,xerr=None, yerr=None, title=''):
    #
    #     self.plot_data_matrix[i_panel][j_panel] = PlotData(x, y, xerr=xerr, yerr=yerr, title=title)


    def draw_brick_heatmap(self, plot_data, ax):
        pd = plot_data
        import seaborn as sns
        import matplotlib.pyplot as plt

        import pandas
        from pandas import DataFrame

        # data = np.arange(30).reshape(6,5)

        if isinstance(pd.df, pandas.DataFrame):
            df = pd.df
        else:

            df = pandas.DataFrame(pd.df, columns=pd.x_tick_labels, index=np.array(pd.y_tick_labels))


        # colormap = sns.palplot(sns.color_palette("coolwarm", 7))
        # sns.set_palette(colormap)

        if pd.val_lim:
            ax = sns.heatmap(df, cmap='bwr', fmt="d", vmin=pd.val_lim[0], vmax=pd.val_lim[1])
        else:
            ax = sns.heatmap(df, cmap='bwr', fmt="d")

        xpos, ypos = np.meshgrid(ax.get_xticks(), ax.get_yticks())

        for x, y in zip(xpos.flat, ypos.flat):
            print 'x,y=', (x, y)
            # ax.text(x, y, 20.0, color='k', ha="center", va="center",)


            # for (i,x), (j,y) in zip(enumerate(xpos.flat), enumerate(ypos.flat)):
            #     print 'i,j=',(i,j)
            #     print 'x,y=',(x,y)

            # ax.text(x, y, 20.0, color='k', ha="center", va="center")
            # ax.text(x, y, annotate_dict[(i,j)], color='k', ha="center", va="center")

        from itertools import product
        xticks = ax.get_xticks()
        yticks = ax.get_yticks()

        xticks_numbered = zip(np.arange(len(xticks)), xticks)
        # yticks_numbered = zip(np.arange(len(yticks)), yticks)
        yticks_numbered = zip(np.arange(len(yticks))[::-1],
                              yticks)  # had to invert y axis to achieve numpy matrix ordering

        if pd.annot_dict is not None:

            # implementing numpy matrix ordering - (0,0) is upper left corner
            for (j, y), (i, x), in product(yticks_numbered, xticks_numbered):
                # print 'x_tuple=', (i, x), ' y_tuple=', (j, y)
                # print pd.annot_dict[(j,i)]

                try:
                    ax.text(x, y, pd.annot_dict[(j, i)], color='k', ha="center", va="center")
                except LookupError:
                    print 'COULD NOT GET i,j = ', (j, i)
                    pass

        # from itertools import product
        # xticks = ax.get_xticks()
        # yticks = ax.get_yticks()
        #
        # xticks_numbered = zip(np.arange(len(xticks)), xticks)
        # yticks_numbered = zip(np.arange(len(yticks)), yticks)
        #
        # annotate_dict = {(i, j): i * j for i, j in product(range(6), range(6))}
        #
        # if pd.annot_dict is not None:
        #
        #     for (i, x), (j, y) in product(xticks_numbered, yticks_numbered):
        #         print 'x_tuple=', (i, x), ' y_tuple=', (i, y)
        #         try:
        #             ax.text(x, y, pd.annot_dict[(i, j)], color='k', ha="center", va="center")
        #         except LookupError:
        #             pass

        if pd.xlabel:
            ax.set_xlabel(pd.xlabel, fontsize=pd.xlabel_fontsize)
        if pd.ylabel:
            ax.set_ylabel(pd.ylabel, fontsize=pd.ylabel_fontsize)
        if pd.title:
            ax.set_title(pd.title)

    def process_PlotDataCollection(self,pd, ax):

        min_x_list=[]
        max_x_list=[]

        for pd_instance in pd.plot_data_list:
            min_x_list.append(np.min(pd_instance.x))
            max_x_list.append(np.max(pd_instance.x))
            if isinstance(pd_instance, PlotData):
                self.process_PlotData(pd_instance, ax)
            elif isinstance(pd_instance, BarPlotData):
                self.process_BarPlotData(pd_instance, ax)
            elif isinstance(pd_instance, BrickHeatmapPlotData):
                self.process_BrickHeatmapPlotData(pd_instance, ax)

        if pd.xlim:
            ax.set_xlim(pd.xlim)
        else:
            ax.set_xlim([np.min(min_x_list) - 0.5, np.max(max_x_list) + 0.5])


        if pd.legend_on:
            if pd.legend_pos is not None:
                ax.legend(bbox_to_anchor=pd.legend_pos)
            else:
                ax.legend()

    def process_PlotData(self, pd, ax):

        if pd.xerr is not None or pd.yerr is not None:
            # xerr=[xerr, 2*xerr],
            lines = ax.errorbar(pd.x, pd.y, yerr=pd.yerr, fmt='--o', label=pd.label)
            # ax.set_xlim([np.min(pd.x)-0.5, np.max(pd.x)+0.5])
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
            lines = ax.plot(pd.x, pd.y, pd.marker, ls=pd.linestyle, color=pd.color, label=pd.label)

        if pd.xlim:
            ax.set_xlim(pd.xlim)

        else:
            ax.set_xlim([np.min(pd.x) - 0.5, np.max(pd.x) + 0.5])

        if pd.xlim:
            ax.set_xlim(pd.xlim)

        if pd.ylim:
            ax.set_ylim(pd.ylim)

        self.process_extra_lines(pd,ax)


    def process_BarPlotData(self, pd, ax):
        inds = np.arange(len(pd.x))
        alpha = 1.0
        if pd.alpha is not None:
            alpha = pd.alpha
        rects = ax.bar(inds - 0.5 * pd.barwidth, pd.y, pd.barwidth, color='r', yerr=pd.yerr, alpha=alpha, label=pd.label)
        if pd.x_tick_labels is not None:
            ax.set_xticks(pd.x)
            ax.set_xticklabels(pd.x_tick_labels)

        if pd.barcolors is not None:
            for i, rect in enumerate(rects):
                rect.set_color(pd.barcolors[i])

        if pd.xlim:
            ax.set_xlim(pd.xlim)

        if pd.ylim:
            ax.set_ylim(pd.ylim)


        self.process_extra_lines(pd,ax)


    def process_BrickHeatmapPlotData(self, pd, ax):

        self.draw_brick_heatmap(pd, ax)
        # ax.text(0.9, 0.95, "Bin:", ha ='left', fontsize = 15)
        if pd.colorbar_title:
            # colorbar locqation coordinates are measured in number of data samples (rows, columns)
            if pd.colorbar_title_location is not None:

                ax.text(pd.colorbar_title_location[0], pd.colorbar_title_location[1], pd.colorbar_title,
                        fontsize=12, rotation=270)
            else:

                ax.text(6.0, 5, pd.colorbar_title,
                        fontsize=12, rotation=270)

        self.process_extra_lines(pd,ax)

    def process_extra_lines(self,pd,ax):
        # LEVEL_LINE
        if pd.levelline is not None:
            levelline = ax.plot(pd.levelline[0], pd.levelline[1], ls='--', color='black')

        # HORIZONTAL LINE
        if pd.xhline_pos is not None:
            ax.axhline(y=pd.xhline_pos, color='black', ls='dashed')


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
            fig = plt.figure(figsize=(15, 15))
        else:
            fig = plt.figure(figsize=(self.xfigsize, self.yfigsize))

        if self.title is None:
            self.title = ''
        if self.xtitle is None:
            self.xtitle = ''

        fig.suptitle(self.title, fontsize=16, fontweight='bold')
        # fig.text(x=0.5, y=0.95, s='Minimum 2 cells per cluster' ,fontsize=14, horizontalalignment='center')

        # fig.text(x=0.5, y=0.02, s=self.xtitle, fontsize=16, fontweight='bold', horizontalalignment='center')
        #
        xtitle_fontsize = 16
        if self.xtitle_fontsize is not None:
            xtitle_fontsize = self.xtitle_fontsize

        ytitle_fontsize = 16
        if self.ytitle_fontsize is not None:
            ytitle_fontsize = self.ytitle_fontsize


        fig.text(x=0.5, y=0.02, s=self.xtitle, fontsize=xtitle_fontsize,  horizontalalignment='center')

        import itertools
        for i, j in itertools.product(xrange(self.i_max), xrange(self.j_max)):

            pd = self.plot_data_matrix[i][j]
            if pd is None:
                print 'Could not find plot data for panel coordinates (i,j)= ', (i, j)
                continue

            ax = plt.subplot2grid((self.i_max, self.j_max), (i, j))

            # ax.set_aspect('equal', adjustable='box')


            # # y axis labels
            # if pd.ylabel is None:
            #     if j == 0:
            #         ax.set_ylabel(self.ytitle, fontsize=pd.ylabel_fontsize)
            #
            # else:
            #     ax.set_ylabel(pd.ylabel, fontsize=pd.ylabel_fontsize)

            # y axis labels
            if pd.ylabel is None:
                if j == 0:
                    ax.set_ylabel(self.ytitle, fontsize=ytitle_fontsize)

            else:
                ax.set_ylabel(pd.ylabel, fontsize=ytitle_fontsize)


            # x axis labels
            if pd.xlabel is None:
                pass
            else:
                ax.set_xlabel(pd.xlabel, fontsize=pd.xlabel_fontsize)


            # print 'pd=',pd


            if isinstance(pd, PlotDataCollection):
                plot_data_list = pd.plot_data_list
            else:
                plot_data_list = [pd]

            # for pd_instance in plot_data_list:
            #     if isinstance(pd_instance, PlotData):
            #         self.process_PlotData(pd_instance, ax)
            #     elif isinstance(pd_instance, BarPlotData):
            #         self.process_BarPlotData(pd_instance, ax)
            #     elif isinstance(pd_instance, BrickHeatmapPlotData):
            #         self.process_BrickHeatmapPlotData(pd_instance, ax)
            #
            if isinstance(pd, PlotDataCollection):
                self.process_PlotDataCollection(pd, ax)
            elif isinstance(pd, PlotData):
                self.process_PlotData(pd, ax)
            elif isinstance(pd, BarPlotData):
                self.process_BarPlotData(pd, ax)
            elif isinstance(pd, BrickHeatmapPlotData):
                self.process_BrickHeatmapPlotData(pd, ax)


            # if isinstance(pd, PlotDataCollection):
            #     plot_data_list = pd.plot_data_list
            # else:
            #     plot_data_list = [pd]
            # 
            # for pd_instance in plot_data_list:
            #     if isinstance(pd_instance, PlotData):
            #         self.process_PlotData(pd_instance, ax)
            #     elif isinstance(pd_instance, BarPlotData):
            #         self.process_BarPlotData(pd_instance, ax)
            #     elif isinstance(pd_instance, BrickHeatmapPlotData):
            #         self.process_BrickHeatmapPlotData(pd_instance, ax)

                    # ax.set_xlabel(pd.title, fontsize=pd.xlabel_fontsize)
        if self.wspace is None or self.hspace is None:
            pass
        else:
            fig.subplots_adjust(wspace=self.wspace, hspace=self.hspace)

        return plt


def generate_panel_plot():
    pass


def draw_brick_heatmap(plot_data):
    import seaborn as sns
    import matplotlib.pyplot as plt
    import pandas

    fig, ax = plt.subplots()

    pd = plot_data

    if isinstance(pd.df, pandas.DataFrame):
        df = pd.df
    else:

        df = pandas.DataFrame(pd.df, columns=pd.x_tick_labels, index=np.array(pd.y_tick_labels))


    # colormap = sns.palplot(sns.color_palette("coolwarm", 7))
    # sns.set_palette(colormap)

    if pd.val_lim:
        ax = sns.heatmap(df, cmap='bwr', fmt="d", vmin=pd.val_lim[0], vmax=pd.val_lim[1])
    else:
        ax = sns.heatmap(df, cmap='bwr', fmt="d")

    xpos, ypos = np.meshgrid(ax.get_xticks(), ax.get_yticks())

    for x, y in zip(xpos.flat, ypos.flat):
        print 'x,y=', (x, y)
        # ax.text(x, y, 20.0, color='k', ha="center", va="center",)


        # for (i,x), (j,y) in zip(enumerate(xpos.flat), enumerate(ypos.flat)):
        #     print 'i,j=',(i,j)
        #     print 'x,y=',(x,y)

        # ax.text(x, y, 20.0, color='k', ha="center", va="center")
        # ax.text(x, y, annotate_dict[(i,j)], color='k', ha="center", va="center")

    from itertools import product
    xticks = ax.get_xticks()
    yticks = ax.get_yticks()

    xticks_numbered = zip(np.arange(len(xticks)), xticks)
    # yticks_numbered = zip(np.arange(len(yticks)), yticks)
    yticks_numbered = zip(np.arange(len(yticks))[::-1], yticks)  # had to invert y axis to achieve numpy matrix ordering

    if pd.annot_dict is not None:

        # implementing numpy matrix ordering - (0,0) is upper left corner
        for (j, y), (i, x), in product(yticks_numbered, xticks_numbered):
            # print 'x_tuple=', (i, x), ' y_tuple=', (j, y)
            # print pd.annot_dict[(j,i)]

            try:
                ax.text(x, y, pd.annot_dict[(j, i)], color='k', ha="center", va="center")
            except LookupError:
                print 'COULD NOT GET i,j = ', (j, i)
                pass


    # ax.text(6.0, 3.5,'Right the plot', fontsize=10, rotation=270)

    if pd.colorbar_title:
        if pd.colorbar_title_location is not None:

            ax.text(pd.colorbar_title_location[0], pd.colorbar_title_location[1], pd.colorbar_title,
                    fontsize=12, rotation=270)
        else:
            ax.text(6.0, 5, pd.colorbar_title,
                    fontsize=12, rotation=270)

    if pd.xlabel:
        ax.set_xlabel(pd.xlabel, fontsize=pd.xlabel_fontsize)
    if pd.ylabel:
        ax.set_ylabel(pd.ylabel, fontsize=pd.ylabel_fontsize)
    if pd.title:
        ax.set_title(pd.title)

    return fig, ax


if __name__ == '__main__':
    panel_plot_0 = PanelPlot(xfigsize=15, yfigsize=7.5, i_max=1, j_max=1, title='SHIFTED DATA 1', xtitle='x_axis_label',xtitle_fontsize=36,
                           ytitle='y_axis_random', ytitle_fontsize=36, xlabel='skdjhskdhksjhksdhk')

    pdc = PlotDataCollection(legend_on=True)
# yerr=np.random.rand(10),
# yerr=np.random.rand(10),

    pd_1 = PlotData(x=np.arange(10,dtype=np.float), y=np.random.rand(10), yerr=np.random.rand(10),  title='', linestyle='',
                    color='green', marker='s', levelline=[[0, 10], [0, 1]],label='green_series')
    pd_2 = PlotData(x=np.arange(5,dtype=np.float)-0.1, y=np.random.rand(5), yerr=np.random.rand(5), title='', linestyle='',
                    color='blue', marker='*',label='blue_series')



    pdc.add_plot_data(pd_1)
    pdc.add_plot_data(pd_2)


    panel_plot_0.add_plot_data_collection(0, 0, plot_data_collection=pdc)



    plot = panel_plot_0.generate_plot()
    plot.subplots_adjust(wspace=0.3, hspace=0.3)

    plot.savefig('demo_shift.pdf', dpi=300, bboxinches='tight')



    panel_plot = PanelPlot(xfigsize=15, yfigsize=7.5, i_max=2, j_max=2, title='Random Data 1', xtitle='x_axis_label',
                           ytitle='y_axis_random')


    panel_plot.add_plot_data(0, 0, x=np.arange(10), y=np.random.rand(10), title='data00', linestyle='dashed',
                             color='green', marker='s', levelline=[[0, 10], [0, 1]])




    bpd = BarPlotData(x=np.arange(10), y=np.random.rand(10), title='data01', yerr=np.random.rand(10) * 0.1,
                      x_tick_labels=['a0', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7', 'a8', 'a9'],
                      barcolors=['r', 'g', 'b', 'r', 'g', 'b', 'r', 'g', 'b', 'r'])
    panel_plot.add_plot_data(0, 1, plot_data=bpd)
    # panel_plot.add_plot_data(0,1,x=np.arange(10),y=np.random.rand(10), title='data01')
    # panel_plot.add_plot_data(1,0,x=np.arange(10),y=np.random.rand(10), title='data10')
    # panel_plot.add_plot_data(1,1,x=np.arange(10),y=np.random.rand(10), yerr=np.random.rand(10), title='data11')
    # plt.savefig(join(plotsDir, quantity_name+'.png'), dpi=300,bboxinches='tight')

    # plot.savefig('demo.png', dpi=300, bboxinches='tight')
    # plot.savefig('demo.png')
    # plot.show()



    data_frame = np.random.rand(6, 5)
    annotation_dictionary = {(0, 0): 10, (1, 2): 20}
    from itertools import product

    annotation_dictionary = {(i, j): i * j for i, j in product(range(6), range(6))}

    x_tick_labels = ['x0', 'x1', 'x2', 'x3', 'x4']
    y_tick_labels = ['y0', 'y1', 'y2', 'y3', 'y4', 'y5']

    hpd = BrickHeatmapPlotData(df=data_frame, annot_dict=annotation_dictionary, title='random_data_brick_plot',
                               x_tick_labels=x_tick_labels, y_tick_labels=y_tick_labels, xlabel='XLABEL',
                               ylabel='YLABEL', val_lim=[-1.5, 1.5],
                               colorbar_title='t-stat for random data',
                               colorbar_title_location=[6.0, 4.5],
                               )

    panel_plot.add_plot_data(1, 1, plot_data=hpd)

    ###################################### PLOT DATA COLLECTION
    pdc = PlotDataCollection()

    bpd_1 = BarPlotData(x=np.arange(10), y=np.random.rand(10), title='data01', yerr=np.random.rand(10) * 0.1,
                        x_tick_labels=['a0', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7', 'a8', 'a9'],
                        barcolors=['r', 'g', 'b', 'r', 'g', 'b', 'r', 'g', 'b', 'r'], alpha=0.2)

    pd_1 = PlotData(x=np.arange(1,11,2), y=np.random.rand(5), title='', linestyle='',
                    color='green', marker='s', levelline=[[0, 10], [0, 1]])
    pd_2 = PlotData(x=np.arange(0,10,2), y=np.random.rand(5), title='', linestyle='',
                    color='blue', marker='*' )

    pdc.add_plot_data(pd_1)
    pdc.add_plot_data(pd_2)
    pdc.add_plot_data(bpd_1)

    panel_plot.add_plot_data_collection(1, 0, plot_data_collection=pdc)


    ###################################### END OF PLOT DATA COLLECTION

    plot = panel_plot.generate_plot()
    plot.subplots_adjust(wspace=0.3, hspace=0.3)

    plot.savefig('demo_1.pdf', dpi=300, bboxinches='tight')


    # standalone brick heatmap plot
    data_frame = np.random.rand(7, 5)
    annotation_dictionary = {(0, 0): 10, (1, 2): 20}
    from itertools import product

    annotation_dictionary = {(i, j): i * j for i, j in product(range(7), range(5))}

    x_tick_labels = ['x0', 'x1', 'x2', 'x3', 'x4']
    y_tick_labels = ['y0', 'y1', 'y2', 'y3', 'y4', 'y5', 'y6']

    hpd = BrickHeatmapPlotData(df=data_frame, annot_dict=annotation_dictionary, title='random_data_brick_plot',
                               x_tick_labels=x_tick_labels, y_tick_labels=y_tick_labels, xlabel='XLABEL',
                               ylabel='YLABEL', val_lim=[-1.5, 1.5],
                               colorbar_title='t-stat for random data',
                               colorbar_title_location=[6.0, 4.5]
                               )

    fig, ax = draw_brick_heatmap(hpd)
    fig.savefig('heatmap_example.png')
