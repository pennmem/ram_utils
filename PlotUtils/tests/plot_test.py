from PlotUtils import  *
import sys

pd_1 = PlotData(x=np.arange(10, dtype=np.float), y=np.random.rand(10), yerr=np.random.rand(10),
                ylabel='series_1',xlabel='x_axis_1',
                color='green',  levelline=[[0, 10], [0, 1]], label='green_series')
pd_2 = PlotData(x=np.arange(5, dtype=np.float) - 0.1, y=np.random.rand(5), yerr=np.random.rand(5),
                xlabel='x_axis_2' , ylabel='series_2',
                color='blue', marker='*', label='blue_series')

panel_plot = PanelPlot(xfigsize=15, yfigsize=7.5, i_max=1, j_max=2, title='PANEL TITLE')

panel_plot.add_plot_data(0,0,plot_data=pd_1)
panel_plot.add_plot_data(0,1,plot_data=pd_2)

plot = panel_plot.generate_plot()
plot.subplots_adjust(wspace=0.3, hspace=0.3)

plot.savefig('panel_separate_y_titles.png')


pd_1 = PlotData(x=np.arange(10, dtype=np.float), y=np.random.rand(10), yerr=np.random.rand(10),
                color='green', marker='s', levelline=[[0, 10], [0, 1]], label='green_series')
pd_2 = PlotData(x=np.arange(5, dtype=np.float) - 0.1, y=np.random.rand(5), yerr=np.random.rand(5),
                color='blue', marker='*', label='blue_series')

pdc = PlotDataCollection(legend_on=True)
pdc.xlabel = 'x_axis_pdc'
pdc.ylabel = 'y_pdc'
pdc.xlabel_fontsize = 20
pdc.ylabel_fontsize = 20

pdc.add_plot_data(pd_1)
pdc.add_plot_data(pd_2)


pd_3 = PlotData(x=np.arange(5, dtype=np.float) - 0.1, y=np.random.rand(5), yerr=np.random.rand(5),
                xlabel='x_axis_3_new' , ylabel='series_3',
                xlabel_fontsize = 20,ylabel_fontsize = 20,
                color='black', marker='*', markersize=10.0, elinewidth=3.0, label='black_series')


panel_plot = PanelPlot(xfigsize=15, yfigsize=7.5, i_max=1, j_max=2, title='PANEL TITLE PDC')


panel_plot.add_plot_data(0,0,plot_data=pdc)
panel_plot.add_plot_data(0,1,plot_data=pd_3)

plot = panel_plot.generate_plot()
plot.subplots_adjust(wspace=0.3, hspace=0.3)

plot.savefig('panel_separate_y_titles_pdc.png')





##############################################################


panel_plot_0 = PanelPlot(xfigsize=15, yfigsize=7.5, i_max=1, j_max=1, title='SHIFTED DATA 1', xtitle='x_axis_label',
                         xtitle_fontsize=36,
                         ytitle='y_axis_random', ytitle_fontsize=36, xlabel='skdjhskdhksjhksdhk')

pdc = PlotDataCollection(legend_on=True)
# yerr=np.random.rand(10),
# yerr=np.random.rand(10),

pd_1 = PlotData(x=np.arange(10, dtype=np.float), y=np.random.rand(10), yerr=np.random.rand(10), title='',
                linestyle='',
                color='green', marker='s', levelline=[[0, 10], [0, 1]], label='green_series')
pd_2 = PlotData(x=np.arange(5, dtype=np.float) - 0.1, y=np.random.rand(5), yerr=np.random.rand(5), title='',
                linestyle='',
                color='blue', marker='*', label='blue_series')

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

pd_1 = PlotData(x=np.arange(1, 11, 2), y=np.random.rand(5), title='', linestyle='',
                color='green', marker='s', levelline=[[0, 10], [0, 1]])
pd_2 = PlotData(x=np.arange(0, 10, 2), y=np.random.rand(5), title='', linestyle='',
                color='blue', marker='*')

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
#
cmap = matplotlib.cm.get_cmap('Blues')
import seaborn as sns

hpd = BrickHeatmapPlotData(df=data_frame, annot_dict=annotation_dictionary, title='random_data_brick_plot',
                           x_tick_labels=x_tick_labels, y_tick_labels=y_tick_labels, xlabel='XLABEL',
                           ylabel='YLABEL', val_lim=[-1.5, 1.5],
                           colorbar_title='t-stat for random data',
                           colorbar_title_location=[1.2, 0.5],

                           cmap=cmap
                           )

fig, ax = draw_brick_heatmap(hpd)
fig.savefig('heatmap_example.png')

###################################### BAR PLOT DATA COLLECTION
panel_plot_1 = PanelPlot(xfigsize=5, yfigsize=5, i_max=1, j_max=1, title='BAR_PLOT_DEMO', xtitle='',
                         )

bpd = BarPlotData(x=np.arange(20), y=np.random.rand(20), xlabel='x_axis_label', ylabel='y_axis_label', title='data01',
                  yerr=np.random.rand(20) * 0.1,
                  elinewidth=2.5,
                  x_tick_labels=['a0', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7', 'a8', 'a9'] * 2,
                  barcolors=['r', 'g', 'b', 'r', 'g', 'b', 'r', 'g', 'b', 'r'] * 2
                  )

panel_plot_1.add_plot_data(0, 0, plot_data=bpd)

plot = panel_plot_1.generate_plot()
# plot.subplots_adjust(wspace=0.3, hspace=0.3)


# plot.savefig('bar_plot_demo_1.png', dpi=300, bboxinches='tight')
# plot.savefig('bar_plot_demo_1.png', bboxinches='tight')
plot.savefig('bar_plot_demo_1.png')
plot.savefig('bar_plot_demo_1.pdf', dpi=300, bboxinches='tight')
# plot.savefig('bar_plot_demo_1.pdf', dpi=100)

###################################### END BAR PLOT DATA COLLECTION
