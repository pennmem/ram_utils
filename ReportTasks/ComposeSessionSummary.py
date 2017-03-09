import SessionSummary
from ReportUtils import  ReportRamTask

SESSION_SUMMARY={
    'catFR1':SessionSummary.catFR1SessionSummary,
    'FR1' :  SessionSummary.FR1SessionSummary,
    'PAL1':  SessionSummary.PAL1SessionSummary,
    'TH1' :  SessionSummary.TH1SessionSummary
}

class ComposeSessionSummary(ReportRamTask):
    @property
    def events(self):
        return self.get_passed_object('events')

