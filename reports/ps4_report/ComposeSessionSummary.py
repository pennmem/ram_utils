from ReportUtils import ReportRamTask
import pandas as pd
import numpy as np
import time

class PS4SessionSummary(object):
    def __init__(self):
        self.info_by_location = {} # location
        self.sham_dc  = -999
        self.sham_sem = -999
        self.best_location = ''
        self.best_amplitude=-999
        self.pval = -999
        self.tstat = -999
        self.tie = True
        self.pval_vs_sham = -999
        self.tstat_vs_sham = -999
        self.decision_info = {}



class PS4LocationSummary(object):
    def __init__(self, loc_tag):
        self.loc_tag=loc_tag
        self.delta_classifiers = {}
        self.amplitudes = {}
        self.sem  = -999
        self.snr = -999
        self.best_amplitude = -999
        self.best_delta_classifier = -999


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

            for location,loc_events in sess_events.groupby(('anode_label','cathode_label')):
                if location[0] and location[1]:
                    loc_tag= '%s_%s'%(location[0],location[1])
                    location_summary = PS4LocationSummary(loc_tag=loc_tag)
                    opt_events = loc_events.loc[loc_events.type=='OPTIMIZATION'].groupby('list_phase')

                    for i,(phase,phase_opt_events) in enumerate(opt_events):
                        loc_decision_info = decision.loc1 if decision.loc1.loc_name==loc_tag else decision.loc2
                        location_summary.amplitudes[phase]=phase_opt_events.amplitude.values/1000.
                        location_summary.delta_classifiers[phase] = phase_opt_events.delta_classifier.values
                        location_summary.best_amplitude = loc_decision_info.amplitude[0]
                        location_summary.best_delta_classifier = loc_decision_info.delta_classifier[0]
                        location_summary.sem = loc_decision_info.sem[0]
                        location_summary.snr = loc_decision_info.snr[0]

                    session_summary.info_by_location[loc_tag] = location_summary
            session_summary.sham_dc = decision.sham.delta_classifier[0]
            session_summary.sham_sem = decision.sham.sem[0]
            session_summary.best_location = decision.decision.best_location[0]
            session_summary.best_amplitude = (decision.loc1 if decision.loc1.loc_name==session_summary.best_location else decision.loc2).amplitude[0]
            session_summary.pval = decision.decision.p_val[0]
            session_summary.tstat = decision.decision.t_stat[0]
            session_summary.tie = decision.decision.tie[0]
            session_summary.tstat_vs_sham = decision.sham.t_stat[0]
            session_summary.pval_vs_sham = decision.sham.p_val[0]

            session_summaries[session] = session_summary

        self.pass_object('session_data',session_data)
        self.pass_object('session_summaries',session_summaries)



