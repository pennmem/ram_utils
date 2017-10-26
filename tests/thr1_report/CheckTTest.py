import numpy as np
from ramutils.pipeline import RamTask


class CheckTTest(RamTask):
    def __init__(self, params, mark_as_completed=True):
        RamTask.__init__(self, mark_as_completed)
        self.params = params


    def run(self):
        subject = self.pipeline.subject
        task = self.pipeline.task

        events = self.get_passed_object(task + '_events')
        sessions = np.unique(events.session)

        channels = self.get_passed_object('channels')
        tal_info = self.get_passed_object('tal_info')

        ttest_output = self.get_passed_object('ttest')

        print 'Subject', subject

        for sess in sessions:
            print 'Session', sess

            sess_events = events[events.session==sess]
            sess_ttest_output = ttest_output[sess]

            #print sess_ttest_output[0].shape
            #print sess_ttest_output[1].shape

            t = sess_ttest_output[0]
            p = sess_ttest_output[1]
            inds = np.argsort(p)
            print "%15s %10s %10s" % ('Elec pair', 't-stat', 'p-value')
            for i in xrange(10):
                idx = inds[i]
                print "%15s %10.5f %10.5f" % (tal_info[idx].tagName, t[idx], p[idx])
