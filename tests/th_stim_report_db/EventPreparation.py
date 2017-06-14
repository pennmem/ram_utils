import os
import os.path

import scipy.io as spio
import numpy as np
from numpy.lib.recfunctions import append_fields
from ptsa.data.readers import BaseEventReader
from ram_utils.RamPipeline import *
from ReportUtils import ReportRamTask

class EventPreparation(ReportRamTask):
    def __init__(self, mark_as_completed=True):
        super(EventPreparation, self).__init__(mark_as_completed)

    def run(self):
        task = self.pipeline.task

        e_path = os.path.join(self.pipeline.mount_point, 'data/events', task, self.pipeline.subject + '_events.mat')
        e_reader = BaseEventReader(filename=e_path, eliminate_events_with_no_eeg=False)
        events = e_reader.read()

        # change the item field name to item_name to not cause issues with item()
        events.dtype.names = ['item_name' if i=='item' else i for i in events.dtype.names]
        ev_order = np.argsort(events, order=('session','trial','mstime'))
        events = events[ev_order]

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
        self.pass_object(self.pipeline.task+'_all_events', events)

        chest_events = events[events.type == 'CHEST']

        rec_events = events[events.type == 'REC']

        events = events[(events.type == 'CHEST') & (events.confidence >= 0)]
        print len(events), task, 'item presentation events'

        self.pass_object(task+'_events', events)
        self.pass_object(self.pipeline.task+'_chest_events', chest_events)
        self.pass_object(self.pipeline.task+'_rec_events', rec_events)


        score_path = os.path.join(self.pipeline.mount_point , 'data', 'events', task, self.pipeline.subject + '_score.mat')
        # score_events = self.loadmat(score_path)
        score_events = spio.loadmat(score_path,squeeze_me=True,struct_as_record=True)
        self.pass_object(task+'_score_events', score_events)


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
