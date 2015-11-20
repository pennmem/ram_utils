__author__ = 'm'

from collections import OrderedDict

class SessionSummary(object):
    def __init__(self):
        self.plot_data_dict = OrderedDict() # {panel_plot_coordinate (0,0) : PlotData}
        self.constant_name = self.constant_value = self.constant_unit = None
        self.stimtag = None
        self.parameter1 = self.parameter2 = None
        self.name = None
        self.date = None
        self.length = None
        self.isi_mid = None
        self.isi_half_range = None
