__author__ = 'm'

import numpy as np

# this makes matplotlib independend of the X server - comes handy on clusters
import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
from collections import namedtuple

PlotDataOption = namedtuple('PlotDataOption', ['name', 'default_value'])


class OptionsObject(object):
    def __init__(self):
        pass

    def PDO(self, name, default_value=None):
        return PlotDataOption(name=name, default_value=default_value)

    def init_options(self, option_list, options={}):
        for option in option_list:
            if hasattr(self, option.name): continue
            try:
                setattr(self, option.name, options[option.name])
                print 'option_name=', option.name, ' val=', options[option.name], ' value_check = ', getattr(self,
                                                                                                             option.name)
            except LookupError:
                setattr(self, option.name, option.default_value)


class PlotDataBase(OptionsObject):
    def __init__(self, **options):

        PDO = self.PDO

        option_list = [
            PDO(name='x'),
            PDO(name='y'),
            PDO(name='xerr'),
            PDO(name='yerr'),
            PDO(name='x_tick_labels'),
            PDO(name='y_tick_labels'),
            PDO(name='title'),
            PDO(name='xlabel_fontsize', default_value=12),
            PDO(name='ylabel_fontsize', default_value=12),
            PDO(name='xlim'),
            PDO(name='ylim'),
            PDO(name='xhline_pos'),
            PDO(name='xlabel'),
            PDO(name='ylabel'),
            PDO(name='linestyle', default_value='-'),
            PDO(name='color', default_value='black'),
            PDO(name='marker', default_value='o'),
            PDO(name='markersize', default_value=5.0),
            PDO(name='levelline'),
            PDO(name='label', default_value=''),
            PDO(name='elinewidth', default_value=1), # width of the error bar


        ]

        self.init_options(option_list, options)

    def sanity_check(self):
        if self.x is None or self.y is None:
            raise AttributeError(

                self.__class__.__name__ + ' requires that x and y attributes are initialized. Use PlotData(x=x_array,y=y_array) syntax')

    def get_yrange(self):
        if self.yerr is not None:
            try:
                return [np.min(self.y - self.yerr), np.max(self.y + self.yerr)]
            except:
                return [None, None]
        else:
            try:
                return [np.min(self.y), np.max(self.y)]
            except:
                return [None, None]

    def get_xrange(self):
        if self.xerr is not None:
            try:
                return [np.min(self.x - self.xerr), np.max(self.x + self.xerr)]
            except:
                return [None, None]
        else:
            try:
                return [np.min(self.x), np.max(self.x)]
            except:
                return [None, None]


class PlotData(PlotDataBase):
    # def __init__(self, x, y, xerr=None, yerr=None, x_tick_labels=None, y_tick_labels=None, title=''):
    def __init__(self, **options):
        '''
        Initializes PlotData
        :param options: options are  'x', 'y', 'xerr', 'yerr', 'x_tick_labels', 'y_tick_labels','title',
        'xlabel_fontsize','ylabel_fontsize', 'xlim','ylim','xhline_pos','xlabel','ylabel','linestyle','color','marker',
        'levelline'
        :return:
        '''
        PlotDataBase.__init__(self, **options)
        self.sanity_check()


class BarPlotData(PlotDataBase):
    # def __init__(self, x, y, xerr=None, yerr=None, x_tick_labels=None, y_tick_labels=None, title=''):
    def __init__(self, **options):
        '''
        Initializes PlotData
        :param options: options are  'x', 'y', 'xerr', 'yerr', 'x_tick_labels', 'y_tick_labels','title',
        'xlabel_fontsize','ylabel_fontsize', 'xlim','ylim','xhline_pos','xlabel','ylabel','linestyle','color','marker',
        'levelline', 'barcolors','barwidth'
        :return:
        '''
        PlotDataBase.__init__(self, **options)

        PDO = self.PDO

        option_list = [

            PDO(name='barcolors', default_value=''),
            PDO(name='barwidth', default_value=0.5),
            PDO(name='alpha', default_value=0.5),
        ]
        self.init_options(option_list, options)
        self.sanity_check()


class BrickHeatmapPlotData(PlotDataBase):
    # def __init__(self, x, y, xerr=None, yerr=None, x_tick_labels=None, y_tick_labels=None, title=''):
    def __init__(self, **options):
        '''
        Initializes PlotData
        :return:
        '''
        PlotDataBase.__init__(self, **options)

        PDO = self.PDO

        option_list = [

            PDO(name='df'),
            PDO(name='annot_dict'),
            PDO(name='val_lim'),
            PDO(name='colorbar_title'),
            PDO(name='colorbar_title_location'),
            PDO(name='cmap'),
            PDO(name='annotation_font_color'),
        ]
        self.init_options(option_list, options)
        self.sanity_check()

    def sanity_check(self):
        if self.df is None:
            raise AttributeError(
                self.__class__.__name__ + ' requires that df attribute is initialized - it can be pandas DataFrame object of simply 2D numpy array. Use PlotData(df=df) syntax')


class PlotDataCollection(PlotDataBase):
    def __init__(self, **options):
        PlotDataBase.__init__(self, **options)

        self.plot_data_list = []
        PDO = self.PDO

        option_list = [
            PDO(name='legend_pos'),
            PDO(name='legend_on', default_value=False),
        ]
        self.init_options(option_list, options)

    def add_plot_data(self, pd):
        self.plot_data_list.append(pd)

    def get_yrange(self):
        try:
            min_list = []
            max_list = []

            for pd_instance in self.plot_data_list:
                if pd_instance.yerr is not None:
                    yerr = pd_instance.yerr
                else:
                    yerr = np.zeros_like(pd_instance.y)

                min_list.append(np.min(pd_instance.y - yerr))
                max_list.append(np.max(pd_instance.y + yerr))

            return [np.min(min_list), np.max(max_list)]
        except:
            return [None, None]

    def get_xrange(self):
        try:
            min_list = []
            max_list = []

            for pd_instance in self.plot_data_list:
                if pd_instance.xerr is not None:
                    xerr = pd_instance.xerr
                else:
                    xerr = np.zeros_like(pd_instance.x)

                min_list.append(np.min(pd_instance.x - xerr))
                max_list.append(np.max(pd_instance.x + xerr))

            return [np.min(min_list), np.max(max_list)]
        except:
            return [None, None]


class PanelPlot(OptionsObject):
    def __init__(self, **options):
        '''
        Initializes PanelPlot
        :param options: options are: 'i_max', 'j_max', 'title', 'xtitle', 'ytitle', 'wspace', 'hspace','xfigsize','yfigsize'
        :return: None
        '''
        OptionsObject.__init__(self)

        PDO = self.PDO
        option_list = [
            PDO(name='i_max', default_value=1),
            PDO(name='j_max', default_value=1),
            PDO(name='title'),
            PDO(name='xtitle'),
            PDO(name='xtitle_fontsize', default_value=None),
            PDO(name='ytitle'),
            PDO(name='ytitle_fontsize', default_value=None),
            PDO(name='wspace', default_value=0.3),
            PDO(name='hspace', default_value=0.3),
            PDO(name='xfigsize', default_value=15.0),
            PDO(name='yfigsize', default_value=15.0),
            PDO(name='labelsize', default_value=None),  # determines font size for tick labels

        ]

        self.init_options(option_list, options)

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

    def add_plot_data_collection(self, i_panel, j_panel, **options):
        '''
        Adds PlotData to the proper location in the panel plot
        :param i_panel: x position of the plot in the panel grid
        :param j_panel: y position of the plot in the panel grid
        :param options: same options you would pass to PlotData. if one of the options is plot_data than
        the rest of the options gets ignored
        :return:None
        '''

        try:
            pd = options['plot_data_collection']
        except LookupError:
            pd = PlotDataCollection(**options)

        self.plot_data_matrix[i_panel][j_panel] = pd

    def draw_brick_heatmap(self, plot_data, ax):
        pd = plot_data
        import seaborn as sns
        import matplotlib.pyplot as plt

        import pandas

        if isinstance(pd.df, pandas.DataFrame):
            df = pd.df
        else:

            df = pandas.DataFrame(pd.df, columns=pd.x_tick_labels, index=np.array(pd.y_tick_labels))

        # colormap = sns.palplot(sns.color_palette("coolwarm", 7))
        # sns.set_palette(colormap)

        cmap = 'bwr'
        if pd.cmap is not None:
            cmap = pd.cmap

        if pd.val_lim:
            ax = sns.heatmap(df, cmap=cmap, fmt="d", vmin=pd.val_lim[0], vmax=pd.val_lim[1])
        else:
            ax = sns.heatmap(df, cmap=cmap, fmt="d")

        xpos, ypos = np.meshgrid(ax.get_xticks(), ax.get_yticks())

        from itertools import product
        xticks = ax.get_xticks()
        yticks = ax.get_yticks()

        xticks_numbered = zip(np.arange(len(xticks)), xticks)

        yticks_numbered = zip(np.arange(len(yticks))[::-1],
                              yticks)  # had to invert y axis to achieve numpy matrix ordering

        annotation_font_color = 'k'
        if pd.annotation_font_color is not None:
            annotation_font_color = pd.annotation_font_color

        if pd.annot_dict is not None:

            # implementing numpy matrix ordering - (0,0) is upper left corner
            for (j, y), (i, x), in product(yticks_numbered, xticks_numbered):
                # print 'x_tuple=', (i, x), ' y_tuple=', (j, y)
                # print pd.annot_dict[(j,i)]

                try:
                    ax.text(x, y, pd.annot_dict[(j, i)], color=annotation_font_color, ha="center", va="center")
                except LookupError:
                    print 'COULD NOT GET i,j = ', (j, i)
                    pass

        if pd.xlabel:
            ax.set_xlabel(pd.xlabel, fontsize=pd.xlabel_fontsize)
        if pd.ylabel:
            ax.set_ylabel(pd.ylabel, fontsize=pd.ylabel_fontsize)
        if pd.title:
            ax.set_title(pd.title)

    def process_PlotDataCollection(self, pd, ax):

        min_x_list = []
        max_x_list = []

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

        if pd.ylim:
            ax.set_ylim(pd.ylim)

        if pd.legend_on:
            if pd.legend_pos is not None:
                ax.legend(bbox_to_anchor=pd.legend_pos)
            else:
                ax.legend()

    def process_PlotData(self, pd, ax):

        if pd.xerr is not None or pd.yerr is not None:
            lines = ax.errorbar(pd.x, pd.y, yerr=pd.yerr, elinewidth=pd.elinewidth, fmt='--o',marker=pd.marker, markersize=pd.markersize,color=pd.color, label=pd.label)

            if pd.x_tick_labels is not None:
                ax.set_xticks(pd.x)
                ax.set_xticklabels(pd.x_tick_labels)

        else:

            if not pd.markersize:
                pd.markersize = 5.0
            lines = ax.plot(pd.x, pd.y, pd.marker, markersize=pd.markersize, ls=pd.linestyle, color=pd.color,
                            label=pd.label)

        if pd.xlim:
            ax.set_xlim(pd.xlim)

        else:
            ax.set_xlim([np.min(pd.x) - 0.5, np.max(pd.x) + 0.5])

        if pd.xlim:
            ax.set_xlim(pd.xlim)

        if pd.ylim:
            ax.set_ylim(pd.ylim)

        self.process_extra_lines(pd, ax)

    def process_BarPlotData(self, pd, ax):
        inds = np.arange(len(pd.x))
        alpha = 1.0
        if pd.alpha is not None:
            alpha = pd.alpha
        rects = ax.bar(inds - 0.5 * pd.barwidth, pd.y, pd.barwidth, color='r', yerr=pd.yerr, alpha=alpha,
                       label=pd.label)
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

        self.process_extra_lines(pd, ax)

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

        self.process_extra_lines(pd, ax)

    def process_extra_lines(self, pd, ax):
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

        fig = plt.figure(figsize=(self.xfigsize, self.yfigsize))

        if self.title is None:
            self.title = ''
        if self.xtitle is None:
            self.xtitle = ''

        fig.suptitle(self.title, fontsize=16, fontweight='bold')
        # fig.text(x=0.5, y=0.95, s='Minimum 2 cells per cluster' ,fontsize=14, horizontalalignment='center')

        # fig.text(x=0.5, y=0.02, s=self.xtitle, fontsize=16, fontweight='bold', horizontalalignment='center')
        #
        xtitle_fontsize = self.xtitle_fontsize
        ytitle_fontsize = self.ytitle_fontsize

        fig.text(x=0.5, y=0.02, s=self.xtitle, fontsize=xtitle_fontsize, horizontalalignment='center')

        import itertools
        for i, j in itertools.product(xrange(self.i_max), xrange(self.j_max)):

            pd = self.plot_data_matrix[i][j]
            if pd is None:
                print 'Could not find plot data for panel coordinates (i,j)= ', (i, j)
                continue

            ax = plt.subplot2grid((self.i_max, self.j_max), (i, j))

            # ax.set_aspect('equal', adjustable='box')

            # y axis labels
            if pd.ylabel is None:
                if j == 0:
                    ax.set_ylabel(self.ytitle, fontsize=ytitle_fontsize)

            else:
                ax.set_ylabel(pd.ylabel, fontsize=pd.ylabel_fontsize)

            # x axis labels
            if self.xtitle is None:
                pass
            else:
                ax.set_xlabel(pd.xlabel, fontsize=pd.xlabel_fontsize)

            if isinstance(pd, PlotDataCollection):
                plot_data_list = pd.plot_data_list
            else:
                plot_data_list = [pd]

            if isinstance(pd, PlotDataCollection):
                self.process_PlotDataCollection(pd, ax)
            elif isinstance(pd, PlotData):
                self.process_PlotData(pd, ax)
            elif isinstance(pd, BarPlotData):
                self.process_BarPlotData(pd, ax)
            elif isinstance(pd, BrickHeatmapPlotData):
                self.process_BrickHeatmapPlotData(pd, ax)

            if self.labelsize:
                ax.tick_params(axis='both', which='major', labelsize=self.labelsize)

                # [tick.label.set_fontsize(self.labelsize) for tick in ax.xaxis.get_major_ticks()]
                # [tick.label.set_fontsize(self.labelsize) for tick in ax.yaxis.get_major_ticks()]

        if self.wspace is None or self.hspace is None:
            pass
        else:
            fig.subplots_adjust(wspace=self.wspace, hspace=self.hspace)

        fig.tight_layout()

        # plt.tight_layout(pad=3.0, w_pad=0.0, h_pad=0.0)

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

    cmap = 'bwr'
    if pd.cmap is not None:
        cmap = pd.cmap

    if pd.val_lim:
        ax = sns.heatmap(df, cmap=cmap, fmt="d", vmin=pd.val_lim[0], vmax=pd.val_lim[1])
    else:
        ax = sns.heatmap(df, cmap=cmap, fmt="d")

    xpos, ypos = np.meshgrid(ax.get_xticks(), ax.get_yticks())

    from itertools import product
    xticks = ax.get_xticks()
    yticks = ax.get_yticks()

    xticks_numbered = zip(np.arange(len(xticks)), xticks)
    # yticks_numbered = zip(np.arange(len(yticks)), yticks)
    yticks_numbered = zip(np.arange(len(yticks))[::-1], yticks)  # had to invert y axis to achieve numpy matrix ordering

    annotation_font_color = 'k'
    if pd.annotation_font_color is not None:
        annotation_font_color = pd.annotation_font_color

    if pd.annot_dict is not None:

        # implementing numpy matrix ordering - (0,0) is upper left corner
        for (j, y), (i, x), in product(yticks_numbered, xticks_numbered):
            # print 'x_tuple=', (i, x), ' y_tuple=', (j, y)
            # print pd.annot_dict[(j,i)]

            try:
                ax.text(x, y, pd.annot_dict[(j, i)], color=annotation_font_color, ha="center", va="center")
            except LookupError:
                print 'COULD NOT GET i,j = ', (j, i)
                pass

    # ax.text(6.0, 3.5,'Right the plot', fontsize=10, rotation=270)

    if pd.colorbar_title:
        if pd.colorbar_title_location is not None:

            ax.text(len(xticks) * pd.colorbar_title_location[0], len(yticks) * pd.colorbar_title_location[1],
                    pd.colorbar_title,
                    fontsize=12, rotation=270)

        else:

            ax.text(len(xticks) * 1.2, len(yticks) * 0.5, pd.colorbar_title,
                    fontsize=12, rotation=270)

    if pd.xlabel:
        ax.set_xlabel(pd.xlabel, fontsize=pd.xlabel_fontsize)
    if pd.ylabel:
        ax.set_ylabel(pd.ylabel, fontsize=pd.ylabel_fontsize)
    if pd.title:
        ax.set_title(pd.title)

    fig.tight_layout()

    return fig, ax

