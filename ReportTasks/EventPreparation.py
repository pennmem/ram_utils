from ReportUtils import ReportRamTask
from ptsa.data.readers.IndexReader import JsonIndexReader
from ptsa.data.readers import BaseEventReader
import os
import numpy as np
import pandas as pd

class EventPreparation(ReportRamTask):
    ''' Base class for preparing events. '''


    @staticmethod
    def event_field(task):
        ''' Field in JSON index that points to the events file.
            For PAL and FR experiments, "all_events" combines the task and the math distractor;
            Other experiments don't have a distractor, and so the events are stored as "task_events"
        '''
        return 'all_events' if ('PAL'in task or 'FR' in task) else 'task_events'

    def __init__(self,task,sessions):
        super(ReportRamTask,self).__init__(mark_as_completed=False)
        self.task = task
        self.sessions = sessions

    def load_events(self):
        ''' Load the events for this object's task and sessions. '''
        task = self.task
        jr = JsonIndexReader(os.path.join(self.pipeline.mount_point,'protocols/r1.json'))
        subject_parts = self.pipeline.subject.split('_')
        sessions = self.pipeline.sessions
        montage = 0 if len(subject_parts)==1 else subject_parts[1]
        subject=subject_parts[0]
        if sessions is None:
            event_paths = jr.aggregate_values(self.event_field(task),subject=subject,montage=montage,experiment=task)
        else:
            event_paths = [jr.get_value(self.event_field,subject=subject,montage=montage,experiment=task,session=s)
                           for s in sessions]

        return np.concatenate([BaseEventReader(filename=event_path).read() for event_path in event_paths]).view(np.recarray)

    def process_events(self,events):
        return {'events' : events}

    def run(self):
        events = self.load_events()
        events_to_pass = self.process_events(events)
        for name in events_to_pass:
            self.pass_object(name,events_to_pass[name])



class FREventPreparation(EventPreparation):
    def process_events(self,events):
        ''' (cat)FR events get split into:
        * "events" (i.e. WORD events)
        * math_events
        * rec_events (recall events)
        * intr_events (intrusions) '''

        math_events = events[events.type == 'PROB']

        rec_events = events[events.type == 'REC_WORD']

        intr_events = rec_events[(rec_events.intrusion!=-999) & (rec_events.intrusion!=0)]

        events = events[events.type == 'WORD']
        return {
            '%s_events'%(self.task): events,
            'math_events': math_events,
            'rec_events': rec_events,
            'intr_events': intr_events
        }

class JointFR1EventPreparation(FREventPreparation):
    ''' JointFR1EventPreparation treats FR1 events and catFR1 events as being from the same task.
        This class should only be used for subjects with both FR1 and catFR1 sessions
    '''
    def __init__(self,sessions):
        super(JointFR1EventPreparation,self).__init__(None,sessions)

    def load_events(self):
        sessions = self.sessions
        self.task='FR1'
        self.sessions =[s for s in sessions if s<100]
        fr1_events = super(JointFR1EventPreparation,self).load_events()
        fr1_fields = fr1_events.dtype.names
        self.task = 'catFR1'
        self.sessions = [s-100 for s in sessions if s>=100]
        catfr1_events=super(JointFR1EventPreparation,self).load_events()[fr1_fields]
        catfr1_events.session += 100
        return np.concatenate((fr1_events,catfr1_events)).view(np.recarray)

class PALEventPreparation(EventPreparation):

    def process_events(self,events):
        ''' PAL1 events get split into:
            * events (i.e. STUDY_PAIR events)
            * math_events
            * recall events
            * intrusion events
            * test probe events
        '''
        math_events = events[events.type == 'PROB']
        rec_events = events[(events.type == 'REC_EVENT') & (events.vocalization!=1)]
        intr_events = rec_events[(rec_events.intrusion!=-999) & (rec_events.correct==0)]
        test_probe_events = events[events.type == 'TEST_PROBE']
        task_events = events[(events.type == 'STUDY_PAIR') & (events.correct!=-999)]
        return {
            'math_events':math_events,
            'rec_events':rec_events,
            'intr_events':intr_events,
            'test_probe_events':test_probe_events,
            '%s_events'%(self.task):task_events,
            'all_events': events
        }

class PSEventPreparation(EventPreparation):

    def process_events(self,events):
        task = self.task
        stim_params = pd.DataFrame.from_records(events.stim_params)
        events = pd.DataFrame.from_records(events)
        del events['stim_params']
        events = pd.concat([events, stim_params], axis=1)

        self.propagate_stim_params_to_all_events(events)

        control_events = events[events.type == 'SHAM']
        control_events = control_events.to_records(index=False)

        events = self.compute_isi(events)

        is_stim_event_type_vec = np.vectorize(self.is_stim_event_type)
        stim_mask = is_stim_event_type_vec(events.type)
        if task == 'PS3':
            # stim_inds = np.where(stim_mask)[0]
            # stim_events = pd.DataFrame(events[stim_mask])
            # last_burst_inds = stim_inds + stim_events['nBursts'].values
            # last_bursts = events.ix[last_burst_inds]
            # events = stim_events
            # events['train_duration'] = last_bursts['mstime'].values - events['mstime'].values + last_bursts['pulse_duration'].values
            Exception('PS3 not supported')
        else:
            events = events[stim_mask]

        events = events.to_records(index=False)

        print len(events), 'stim', task, 'events'
        return {
            'control_events': control_events,
            '%s_events'%(self.task): events
        }

    @staticmethod
    def is_stim_event_type(event_type):
        return event_type == 'STIM_ON'

    @staticmethod
    def compute_isi(events):
        print 'Computing ISI'

        events['isi'] = events['mstime'] - events['mstime'].shift(1)
        events.loc[events['type'].shift(1) != 'STIM_OFF', 'isi'] = np.nan
        events.loc[events['isi'] > 7000.0, 'isi'] = np.nan

        return events

    @staticmethod
    def propagate_stim_params_to_all_events(events):
        events_by_session = events.groupby(['session'])
        for sess, session_events in events_by_session:
            last_stim_event = session_events[session_events.type == 'STIM_ON'].iloc[-1]
            session_mask = (events.session == sess)
            events.loc[session_mask, 'anode_label'] = last_stim_event.anode_label
            events.loc[session_mask, 'cathode_label'] = last_stim_event.cathode_label
            events.loc[session_mask, 'anode_number'] = last_stim_event.anode_number
            events.loc[session_mask, 'cathode_number'] = last_stim_event.cathode_number


class THEventPreparation(EventPreparation):
    def process_events(self,events):
        '''
        For TH events, we add in fields to account for transposition, as well as percentile error
        They get split into chest events (encoding) and recall events (retrieval).
        '''
        task = self.task
        ev_order = np.argsort(events, order=('session','trial','mstime'))
        events = events[ev_order]
        # add in error if object locations are transposed
        xCenter = 384.8549
        yCenter = 358.834
        x = events['chosenLocationX']-xCenter
        y = events['chosenLocationY']-yCenter
        xChest = events['locationX']-xCenter
        yChest = events['locationY']-yCenter
        distErr_transpose = np.sqrt((-xChest - x)**2 + (-yChest - y)**2);
        events = np.recarray.append_fields(events,'distErr_transpose',distErr_transpose,dtypes=float,usemask=False,asrecarray=True)

        # add field for correct if transposed
        recalled_ifFlipped = np.zeros(np.shape(distErr_transpose),dtype=bool)
        recalled_ifFlipped[events.isRecFromStartSide==0] = events.distErr_transpose[events.isRecFromStartSide==0]<=13
        events = np.recarray.append_fields(events,'recalled_ifFlipped',recalled_ifFlipped,dtypes=float,usemask=False,asrecarray=True)

        # add field for error percentile (performance factor)
        error_percentiles = self.calc_norm_dist_error(events.locationX,events.locationY,events.distErr)
        events = np.recarray.append_fields(events,'norm_err',error_percentiles,dtypes=float,usemask=False,asrecarray=True)
        self.pass_object(self.pipeline.task+'_all_events', events)

        chest_events = events[events.type == 'CHEST']

        rec_events = events[events.type == 'REC']

        events = events[(events.type == 'CHEST') & (events.confidence >= 0)]
        print len(events), task, 'item presentation events'

        return {
            'chest_events': chest_events,
            'rec_events': rec_events,
            '%s_events'%(self.task): events
        }

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








