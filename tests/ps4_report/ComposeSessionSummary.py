from ReportUtils import ReportRamTask
import ChooseLocation
import pandas as pd
import numpy as np
import time

class PS4SessionSummary(object):
    def __init__(self):
        self.info_by_location = {} # location
        self.sham_dc  = np.nan
        self.sham_sem = np.nan
        self.best_location = ''
        self.best_amplitude= np.nan
        self.pval = ''
        self.tstat = -999.
        self.tie = True
        self.pval_vs_sham = np.nan
        self.tstat_vs_sham = np.nan
        self.decision_info = {}



class PS4LocationSummary(object):
    def __init__(self, loc_tag):
        self.loc_tag=loc_tag
        self.delta_classifiers = {}
        self.post_stim_biomarkers = {}
        self.amplitudes = {}
        self.post_stim_amplitudes=  {}
        self.sem  = -999.
        self.snr = -999.
        self.best_amplitude = -999.
        self.best_delta_classifier = -999.


class ComposeSessionSummary(ReportRamTask):
    def __init__(self,mark_as_completed):
        super(ComposeSessionSummary, self).__init__(mark_as_completed)
    def run(self):
        events = self.get_passed_object('ps_events')
        event_frame = pd.DataFrame.from_records([e for e in events],columns=events.dtype.names)
        session_summaries = {}
        session_data = []
        for session, sess_events in event_frame.groupby('session'):

            first_time_stamp = sess_events.iloc[0].mstime
            timestamps = sess_events.mstime
            last_time_stamp = np.max(timestamps)
            session_length = '%.2f' % ((last_time_stamp - first_time_stamp) / 60000.0)
            session_date = time.strftime('%d-%b-%Y', time.localtime(last_time_stamp / 1000))
            n_lists = len(sess_events.list.unique())

            session_data.append([session, session_date, session_length, n_lists])
            decision  =events[(events.session==session) & (events.type=='OPTIMIZATION_DECISION')]
            session_summary =PS4SessionSummary()

            if len(decision):
                session_summary.sham_dc = decision.sham.delta_classifier[0]
                session_summary.sham_sem = decision.sham.sem[0]
                session_summary.best_location = decision.decision.best_location[0]
                session_summary.best_amplitude = (decision.loc1 if decision.loc1.loc_name==session_summary.best_location else decision.loc2).amplitude[0]
                session_summary.pval = decision.decision.p_val[0]
                session_summary.tstat = decision.decision.t_stat[0]
                session_summary.tie = decision.decision.tie[0]
                session_summary.tstat_vs_sham = decision.sham.t_stat[0]
                session_summary.pval_vs_sham = decision.sham.p_val[0]
            else:

                opt_events = sess_events.loc[sess_events.type=='OPTIMIZATION']
                (locations, loc_datasets) = zip(*[('_'.join(name),table.loc[:,['amplitude','delta_classifier']].values) for (name,table) in opt_events.groupby(('anode_label','cathode_label'))])

                # TODO: include sham delta classifiers when we need to reconstruct results
                decision,loc_info = ChooseLocation.choose_location(loc_datasets[0],locations[0],loc_datasets[1],locations[1],
                                                                np.array([(ld.min(),ld.max()) for ld in loc_datasets]),
                                                                None)
                for i,k in enumerate(loc_info):
                    loc_info[k]['amplitude'] = loc_info[k]['amplitude']/1000
                    decision['loc%s'%(i+1)] = loc_info[k]

                session_summary.tie= decision['Tie']
                session_summary.best_location = decision['best_location_name']
                session_summary.best_amplitude = loc_info[session_summary.best_location]['amplitude']
                session_summary.pval = decision['p_val']
                session_summary.tstat = decision['t_stat']


            for location,loc_events in sess_events.groupby(('anode_label','cathode_label')):
                if location[0] and location[1]:
                    loc_tag= '%s_%s'%(location[0],location[1])
                    location_summary = PS4LocationSummary(loc_tag=loc_tag)
                    opt_events = loc_events.loc[loc_events.type=='OPTIMIZATION'].groupby('list_phase')

                    for i,(phase,phase_opt_events) in enumerate(opt_events):
                        post_stim_phase_events = loc_events.loc[(event_frame.list_phase==phase)
                                                                 & (event_frame.type=='BIOMARKER')
                                                                 & (event_frame.position=='POST')]
                        loc_decision_info = decision['loc1'] if decision['loc1']['loc_name']==loc_tag else decision['loc2']
                        location_summary.amplitudes[phase]=phase_opt_events.amplitude.values/1000.
                        location_summary.delta_classifiers[phase] = phase_opt_events.delta_classifier.values
                        location_summary.post_stim_biomarkers[phase]=post_stim_phase_events.biomarker_value
                        location_summary.post_stim_amplitudes[phase] = post_stim_phase_events.amplitude.values/1000.
                        if len(loc_decision_info):
                            location_summary.best_amplitude = loc_decision_info['amplitude']
                            location_summary.best_delta_classifier = loc_decision_info['delta_classifier']
                            location_summary.sem = loc_decision_info['sem']
                            location_summary.snr = loc_decision_info['snr']

                    session_summary.info_by_location[loc_tag] = location_summary



            session_summaries[session] = session_summary

        self.pass_object('session_data',session_data)
        self.pass_object('session_summaries',session_summaries)



