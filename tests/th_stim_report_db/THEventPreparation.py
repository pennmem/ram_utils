__author__ = 'm'

import os
import os.path
import numpy as np
from ptsa.data.readers import BaseEventReader

from RamPipeline import *
from ReportUtils import ReportRamTask
from numpy.lib.recfunctions import append_fields

class THEventPreparation(ReportRamTask):
    def __init__(self, params, mark_as_completed=True):
        super(THEventPreparation, self).__init__(mark_as_completed)
        self.params = params

    def run(self):
        #try:

        events = None
        th1_e_path = ''

        if self.params.include_th1:
            #try:
            th1_e_path = os.path.join(self.pipeline.mount_point, 'data/events/RAM_TH1',
                                      self.pipeline.subject + '_events.mat')
            e_reader = BaseEventReader(filename=th1_e_path, eliminate_events_with_no_eeg=True)
            events = e_reader.read()

            # change the item field name to item_name to not cause issues with item()
            events.dtype.names = ['item_name' if i=='item' else i for i in events.dtype.names]
            ev_order = np.argsort(events, order=('session','trial','mstime'))
            events = events[ev_order]
            print 'sessions: ',np.unique(events['session'])

            # add in error if object locations are transposed
            xCenter = 384.8549
            yCenter = 358.834
            x = events.chosenLocationX-xCenter;
            y = events.chosenLocationY-yCenter;
            xChest = events.locationX-xCenter;
            yChest = events.locationY-yCenter;
            distErr_transpose = np.sqrt((-xChest - x)**2 + (-yChest - y)**2);
            events = append_fields(events,'distErr_transpose',distErr_transpose,dtypes=float,usemask=False,asrecarray=True)

            # add field for correct if transposed
            recalled_ifFlipped = np.zeros(np.shape(distErr_transpose),dtype=bool)
            recalled_ifFlipped[events.isRecFromStartSide==0] = events.distErr_transpose[events.isRecFromStartSide==0]<=13
            events = append_fields(events,'recalled_ifFlipped',recalled_ifFlipped,dtypes=float,usemask=False,asrecarray=True)

            # add field for error percentile (performance factor)
            error_percentiles = self.calc_norm_dist_error(events.locationX,events.locationY,events.distErr)
            events = append_fields(events,'norm_err',error_percentiles,dtypes=float,usemask=False,asrecarray=True)
            # self.pass_object(self.pipeline.task+'_all_events', events)

            chest_events = events[events.type == 'CHEST']

            rec_events = events[events.type == 'REC']

            events = events[(events.type == 'CHEST') & (events.confidence >= 0)]
            print len(events), 'TH1 item presentation events'

            self.pass_object('th_events', events)


    def calc_norm_dist_error(self,x_pos,y_pos,act_errs):
        rand_x = np.random.uniform(359.9,409.9,100000)
        rand_y = np.random.uniform(318.0,399.3,100000)

        error_percentiles = np.zeros(np.shape(act_errs),dtype=float)
        for i,this_item in enumerate(zip(x_pos,y_pos,act_errs)):
            if np.isnan(this_item[2]):
                error_percentiles[i] = np.nan
            else:
                possible_errors = np.sqrt((rand_x - this_item[0])**2 + (rand_y - this_item[1])**2)
                error_percentiles[i] = np.mean(possible_errors < this_item[2])
        return error_percentiles


        #except Exception:
        #    self.raise_and_log_report_exception(
        #        exception_type='MissingDataError',
        #        exception_message='Missing TH1 events data (%s)' % (th1_e_path)
        #    )
