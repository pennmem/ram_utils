from ReportUtils import *
import numpy as np
import pandas as pd
from scipy.stats import norm,chisquare

class SessionSummary(object):
    pass


class ComposeSessionSummary(ReportRamTask):

    THRESH = 0.444 # need the actual threshold value

    @staticmethod
    def spc(events):
        return np.array(
            [np.mean((events[events.serialpos == p].recalled==True).astype(np.int))
             for p in sorted(np.unique(events.serialpos))]
        )

    @staticmethod
    def pfr(events, stim=True):
        first_recalls = np.zeros(np.unique(events.serialpos[events.serialpos>=0]).shape,dtype=np.float)
        words = events[events.type=='WORD']
        rec_words = events[events.type=='REC_WORD']
        sessions = np.unique(events.session)
        for sess in sessions:
            lists = np.unique(events[events.session==sess].list)
            for lst in lists:
                sess_list_recalls=np.sort(rec_words[(rec_words.session==sess) & (rec_words.list==lst)],order='mstime')
                if len(sess_list_recalls) == 0:
                    continue
                else:
                    name=sess_list_recalls[0].item_name
                    recalled_position = words[(words.session==sess) & (words.list==lst) & (words.item_name==name)].serialpos
                    first_recalls[recalled_position]+=1.0
        return first_recalls/first_recalls.sum()



    @staticmethod
    def make_recog_table(targets, lures):
        def n_hits(targets, stim):
            return targets[targets.stim_list == stim].recognized.sum()

        def n_fa(lures, stim):
            return (lures[lures.stim_list == stim].rejected == False).sum()

        def d_prime(targets, lures, stim):
            return norm.ppf(n_hits(targets, stim) / len(targets[targets.stim_list == stim])) - norm.ppf(
                n_fa(lures, stim) / len(lures[lures.stim_list == stim]))

        statistics = np.array([n_hits(targets, stim=True), n_hits(targets, stim=False),
                               n_fa(lures, stim=True), n_fa(lures, stim=False),
                               d_prime(targets, lures, stim=True), d_prime(targets, lures, stim=False)]).reshape(2, 3)
        return pd.DataFrame(data=statistics, index=['hits', 'false alarms', 'd_prime'],
                            columns=['Stim List', 'Non-Stim List'])


    @staticmethod
    def make_recall_table(events, encoding_biomarkers, retrieval_biomarkers, thresh):
        def n_recalls(events):
            return events.recalled.astype(np.float).sum()
        e_stim = events.is_stim
        r_stim = events.rec_stim
        e_low = encoding_biomarkers<thresh
        r_low = retrieval_biomarkers < thresh
        stim_conditions,nostim_conditions = [ [(a & b),(a & ~b), ((~a) & (b)), ~(a | b)]
                                              for (a,b) in [(e_stim,r_stim),(e_low,r_low)]]
        recall_array = np.array([[n_recalls(events[mask]) for mask in stim_conditions],
                                 [n_recalls(events[mask]) for mask in nostim_conditions]])
        chi,p = chisquare(recall_array,axis=0)
        table = np.concatenate((recall_array,chi,p),axis=0)
        return pd.DataFrame(data=table,index=['Stim items','low items',r'\CHI^2','p'],columns=['Both','Encoding only','Retrieval only','Neither'])

    @staticmethod
    def is_stim_item(events):
        """
        Index words that were stimulated at encoding
        :param events:
        :return:
        """
        return np.array([(event.type=='WORD') & (events[i+1].type=='STIM_ON') or (events[i+1].type=='WORD_OFF' and (events[i+2].type=='STIM_ON' or (events[i+2].type=='DISTRACT_START' and events[i+3].type=='STIM_ON')))
                for (i,event) in enumerate(events[:-3])])

    def is_post_stim_item(self,events):
        stim_words = list(events[self.is_stim_item(events)])
        all_words = events[events.type=='WORD']
        post_stim_words= [all_words[i-1] in stim_words for i,_ in enumerate(all_words)]
        return np.array([event in post_stim_words for event in events])

    @staticmethod
    def is_rec_stim_item(events):
            return np.array([ (event.type=='REC_WORD') & ((events[i-1].type=='STIM_ON') | events[i-2].type=='STIM_ON') for i,event in enumerate(events)])

    def is_post_rec_stim_item(self,events):
        stim_words = list(events(self.is_rec_stim_item(events)))
        rec_words = events[events.type=='REC_WORD']
        post_stim_words = [rec_words[i-1] in stim_words for i,_ in enumerate(rec_words)]
        return np.array([event in post_stim_words for event in events])



    def make_performance_table(self,events):
        stim_list_events = events[events.stim_list==True]

        encoding_stim_events = stim_list_events[self.is_stim_item(stim_list_events)]
        post_encoding_stim_events = stim_list_events[self.is_post_stim_item(stim_list_events)]

        retrieval_stim_events = stim_list_events[self.is_rec_stim_item(stim_list_events)]
        post_retrieval_stim_events = stim_list_events[self.is_post_rec_stim_item(stim_list_events)]

        nostim_list_events = events[events.stim_list==False]

        baseline = np.nanmean(nostim_list_events.recalled)
        recall_change = [(x.recalled.nanmean()-baseline)/baseline for x in encoding_stim_events,post_encoding_stim_events,
                         retrieval_stim_events,post_retrieval_stim_events]

        return pd.DataFrame(data=np.array(recall_change),index=['% recall change'],
                            columns=['Encoding stim','Encoding post-stim','Retrieval stim','Retrieval post-stim'])

    @staticmethod
    def make_stim_recall_table(events):
        n_recalls_by_list = [events[events.list==lst].recalled.sum() for lst in np.unique(events.list)]
        n_stims_by_list = [(events[events.list==lst].type=='STIM_ON').sum() for lst in np.unique(events.list)]
        is_stim_list = [events[events.list==lst][0].stim_list for lst in np.unique(events.list)]
        table=np.concatenate([n_recalls_by_list,n_stims_by_list,is_stim_list])
        return pd.DataFrame(data=table,index=['n_recalls','n_stims','is_stim_list'],columns=np.unique(events.list))







    def run(self):
        all_events =self.get_passed_object('all_events')
        sessions = np.unique(all_events.session)
        encoding_events=self.get_passed_object('encoding_events')
        retrieval_events= self.get_passed_object('retrieval_events')
        session_summaries = {}
        encoding_probs  = self.get_passed_object('encoding_probs')
        retrieval_probs = self.get_passed_object('retrieval_probs')

        recog_events = self.get_passed_object('recog_events')

        for session in sessions:
            session_summary = SessionSummary()
            events = all_events[all_events.session==session]
            words = events[events.type=='WORD']
            stim_words = words[words.stim_list==True]
            nostim_words = words[words.stim_list==False]
            sess_encode_probs = encoding_probs[words.session==session]
            sess_retr_probs = retrieval_probs[retrieval_events.session==session]
            lures= events[events.type=='RECOG_LURE']


            session_summary.stim_spc ,session_summary.nostim_spc = [self.spc(wrds) for wrds in (stim_words,nostim_words)]

            session_summary.stim_pfc, session_summary.nostim_pfc = [self.pfr(events, stim=True), self.pfr(events, stim=False)]


            session_summary.recall_table = self.make_recall_table(events,sess_encode_probs,sess_retr_probs,self.THRESH)

            session_summary.recog_table = self.make_recog_table(recog_events[recog_events.session==session],lures)

            session_summary.stim_rate_tables = {}

            for (name,group) in [('Encoding',encoding_events), ('Retrieval',retrieval_events),('All',events)]:
                session_summary.stim_rate_tables[name] = self.make_stim_recall_table(group[group.session==session])

            session_summaries[session]=session_summary



        self.pass_object('session_summaries',session_summaries)













