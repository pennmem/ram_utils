__author__ = 'm'

import os
import os.path
import numpy as np
import scipy.io as spio
from ptsa.data.readers import BaseEventReader
from numpy.lib.recfunctions import append_fields

from RamPipeline import *


class THEventPreparation(RamTask):
    def __init__(self, mark_as_completed=True):
        RamTask.__init__(self, mark_as_completed)

    def run(self):
        e_path = os.path.join(self.pipeline.mount_point , 'data/events/RAM_TH1', self.pipeline.subject + '_events.mat')
        e_reader = BaseEventReader(filename=e_path, eliminate_events_with_no_eeg=True)
        events = e_reader.read()

        # removing stim fileds that shouldn't be in non-stim experiments
        evs_field_list = ['mstime','type','item','trial','block','chestNum','locationX','locationY','chosenLocationX',
                          'chosenLocationY','navStartLocationX','navStartLocationY','recStartLocationX','recStartLocationY',
                          'isRecFromNearSide','isRecFromStartSide','reactionTime','confidence','session','radius_size',
                          'listLength','distErr','recalled','eegoffset','eegfile'
                          ]
        events = events[evs_field_list]

        # change the item field name to item_name to not cause issues with item()
        events.dtype.names = ['item_name' if i=='item' else i for i in events.dtype.names]
        ev_order = np.argsort(events, order=('session','trial','mstime'))
        events = events[ev_order]
        
        # add in error if object locations are transposed
        xCenter = 384.8549
        yCenter = 358.834
        x = events.chosenLocationX-xCenter
        y = events.chosenLocationY-yCenter
        xChest = events.locationX-xCenter
        yChest = events.locationY-yCenter
        distErr_transpose = np.sqrt((-xChest - x)**2 + (-yChest - y)**2)
        events = append_fields(events,'distErr_transpose',distErr_transpose,dtypes=float,usemask=False,asrecarray=True)

        # add field for correct if transposed
        recalled_ifFlipped = np.zeros(np.shape(distErr_transpose),dtype=bool)
        recalled_ifFlipped[events.isRecFromStartSide==0] = events.distErr_transpose[events.isRecFromStartSide==0]<=13
        events = append_fields(events,'recalled_ifFlipped',recalled_ifFlipped,dtypes=float,usemask=False,asrecarray=True)        
        
        # add field for error percentile (performance factor)
        error_percentiles = self.calc_norm_dist_error(events.locationX,events.locationY,events.distErr)        
        events = append_fields(events,'norm_err',error_percentiles,dtypes=float,usemask=False,asrecarray=True)        
        self.pass_object('TH_all_events', events)

        chest_events = events[events.type == 'CHEST']

        rec_events = events[events.type == 'REC']

        events = events[(events.type == 'CHEST') & (events.confidence >= 0)]
        print len(events), 'item presentation events'
        
        self.pass_object('TH_events', events)
        self.pass_object('TH_chest_events', chest_events)
        self.pass_object('TH_rec_events', rec_events)


        score_path = os.path.join(self.pipeline.mount_point , 'data/events/RAM_TH1', self.pipeline.subject + '_score.mat')
        # score_events = self.loadmat(score_path)
        score_events = spio.loadmat(score_path,squeeze_me=True,struct_as_record=True)
        self.pass_object('TH_score_events', score_events)
        
        timing_path = os.path.join(self.pipeline.mount_point , 'data/events/RAM_TH1', self.pipeline.subject + '_timing.mat')
        timing_events = self.loadmat(timing_path)
        self.pass_object('TH_time_events', timing_events)
        
        # timing_info = dict()
        # if isinstance(timing_events['events'], spio.matlab.mio5_params.mat_struct):
        #     # timing_info = np.recarray((1,),dtype=[('session', int), ('trial_times', float)])
        #     timing_info['session'] = np.array([timing_events['events'].session])
        #     timing_info['trial_times'] = timing_events['events'].trialInfo
        # else:
        #     timing_info['session'] = np.zeros(len(timing_events['events']),dtype=np.int)
        #     timing_info['trial_times'] = np.array([list() for _ in xrange(len(timing_events['events']))],dtype=np.int)
        #     for i, sess_events in enumerate(timing_events['events']):
        #         timing_info['session'][i] = sess_events.session
        #         timing_info['trial_times'][i] = sess_events.trialInfo
            
        # test = self.loadmat('/data/events/RAM_TH1/testevents.mat')
        # print test['events'][0].sessionScore
        
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
    
    # Better loadmat, from http://stackoverflow.com/questions/7008608/scipy-io-loadmat-nested-structures-i-e-dictionaries
    def loadmat(self,filename):
        '''
        this function should be called instead of direct spio.loadmat
        as it cures the problem of not properly recovering python dictionaries
        from mat files. It calls the function check keys to cure all entries
        which are still mat-objects
        '''
        data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)
        return self._check_keys(data)

    def _check_keys(self,dict):
        '''
        checks if entries in dictionary are mat-objects. If yes
        todict is called to change them to nested dictionaries
        '''
        for key in dict:
            if isinstance(dict[key], spio.matlab.mio5_params.mat_struct):
                dict[key] = self._todict(dict[key])
            return dict        

    def _todict(self,matobj):
        '''
        A recursive function which constructs from matobjects nested dictionaries
        '''
        dict = {}
        for strg in matobj._fieldnames:
            elem = matobj.__dict__[strg]
            if isinstance(elem, spio.matlab.mio5_params.mat_struct):
                dict[strg] = self._todict(elem)
            elif isinstance(elem,np.ndarray):
                dict[strg] = self._tolist(elem)
            else:
                dict[strg] = elem
        return dict

    def _tolist(self,ndarray):
        '''
        A recursive function which constructs lists from cellarrays 
        (which are loaded as numpy ndarrays), recursing into the elements
        if they contain matobjects.
        '''
        elem_list = []            
        for sub_elem in ndarray:
            if isinstance(sub_elem, spio.matlab.mio5_params.mat_struct):
                elem_list.append(self._todict(sub_elem))
            elif isinstance(sub_elem,np.ndarray):
                elem_list.append(self._tolist(sub_elem))
            else:
                elem_list.append(sub_elem)
        return elem_list