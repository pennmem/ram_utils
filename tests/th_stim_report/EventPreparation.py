from os.path import *
import sys
sys.path.append(join(dirname(__file__),'..','..'))

import scipy.io as spio
import numpy as np
from numpy.lib.recfunctions import append_fields
from ptsa.data.readers import BaseEventReader
from ptsa.data.readers.IndexReader import JsonIndexReader
from RamPipeline import *
from ReportUtils import ReportRamTask

class EventPreparation(ReportRamTask):
    def __init__(self, mark_as_completed=True):
        super(EventPreparation, self).__init__(mark_as_completed)

    def run(self):
        task = self.pipeline.task
        subject = self.pipeline.subject

        tmp = subject.split('_')
        subj_code = tmp[0]
        # montage = 0 if len(tmp) == 1 else int(tmp[1])
        montage = 1


        json_reader = JsonIndexReader(os.path.join(self.pipeline.mount_point, 'protocols/r1.json'))
        event_files = sorted(list(json_reader.aggregate_values('task_events', subject=subj_code, montage=montage,
                                                               experiment=task)))

        evs_field_list = ['mstime', 'type','item_name','stim_list', 'trial', 'block', 'chestNum', 'locationX', 'locationY',
                          'chosenLocationX',
                          'chosenLocationY', 'navStartLocationX', 'navStartLocationY', 'recStartLocationX',
                          'recStartLocationY',
                          'isRecFromNearSide', 'isRecFromStartSide', 'reactionTime', 'confidence', 'session',
                          'radius_size',
                          'listLength', 'distErr', 'recalled', 'eegoffset', 'eegfile','is_stim',
                          ]
        events = None
        for sess_file in event_files:
            e_path = os.path.join(self.pipeline.mount_point, str(sess_file))
            e_reader = BaseEventReader(filename=e_path, eliminate_events_with_no_eeg=True)
            sess_events = e_reader.read()
            events = sess_events if events is None else np.hstack((events, sess_events))

        events = events.view(np.recarray)
        # events = events[evs_field_list]
        ev_order = np.argsort(events, order=('session', 'trial', 'mstime'))
        events = events[ev_order] # For some reason, leaving stim_params in crashes Python whenever I run it
                                  # This is possibly because stim_params is an empty list; revisit when stim_params loaded correctly
        # add in error if object locations are transposed
        xCenter = 384.8549
        yCenter = 358.834

        x = events.chosenLocationX - xCenter;
        y = events.chosenLocationY - yCenter;
        xChest = events['locationX'] - xCenter;
        yChest = events['locationY'] - yCenter;
        distErr_transpose = np.sqrt((-xChest - x) ** 2 + (-yChest - y) ** 2);
        events = append_fields(events, 'distErr_transpose', distErr_transpose, dtypes=float, usemask=False,
                               asrecarray=True)

        # add field for correct if transposed
        recalled_ifFlipped = np.zeros(np.shape(distErr_transpose), dtype=bool)
        recalled_ifFlipped[events.isRecFromStartSide == 0] = events.distErr_transpose[
                                                                 events.isRecFromStartSide == 0] <= 13
        events = append_fields(events, 'recalled_ifFlipped', recalled_ifFlipped, dtypes=float, usemask=False,
                               asrecarray=True)

        # add field for error percentile (performance factor)
        error_percentiles = self.calc_norm_dist_error(events.locationX, events.locationY, events.distErr)
        events = append_fields(events, 'norm_err', error_percentiles, dtypes=float, usemask=False, asrecarray=True)
        self.pass_object(self.pipeline.task + '_all_events', events)

        chest_events = events[events.type == 'CHEST']

        rec_events = events[events.type == 'REC']

        events = events[(events.type == 'CHEST') & (events.confidence >= 0)]
        print len(events), task, 'item presentation events'

        self.pass_object(task+'_events', events)
        self.pass_object(self.pipeline.task + '_chest_events', chest_events)
        self.pass_object(self.pipeline.task + '_rec_events', rec_events)

        #TODO: replace with JSON code once this is properly integrated
        score_path = os.path.join(self.pipeline.mount_point , 'data', 'events', 'RAM_'+task, self.pipeline.subject + '_score.mat')
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

