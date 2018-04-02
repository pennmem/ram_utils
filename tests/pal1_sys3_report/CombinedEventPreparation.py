import os
import os.path
import numpy as np

from ptsa.data.readers import BaseEventReader
from ptsa.data.readers  import JsonIndexReader

from ReportUtils import RamTask

import hashlib
from sklearn.externals import joblib
from ReportTasks.RamTaskMethods import create_baseline_events
from numpy.lib.recfunctions import append_fields
from ramutils.pipeline import RamTask


class CombinedEventPreparation(RamTask):
    def __init__(self, mark_as_completed=True):
        super(CombinedEventPreparation, self).__init__(mark_as_completed)

    def run(self):
        subject = self.pipeline.subject
        tmp = subject.split('_')
        subj_code = tmp[0]

        fr1_evs = self.get_passed_object('FR_events_with_recall')
        catfr1_evs = self.get_passed_object('CatFR_events_with_recall')
        pal1_evs = self.get_passed_object('PAL1_events_with_recall')

        columns_to_keep = ['session', 'type', 'correct', 'eegoffset', 'msoffset', 'eegfile', 'exp_name']

        columns_to_keep_fr1 = ['session', 'type', 'recalled', 'eegoffset', 'msoffset', 'eegfile']
        columns_to_keep_pal1 = ['session', 'type', 'correct', 'eegoffset', 'msoffset', 'eegfile']

        # filtering fr1_evs
        fr1_evs = self.process_events(evs=fr1_evs, session_increment=100, columns_to_keep=columns_to_keep,
                                      columns_to_keep_fr=columns_to_keep_fr1, task_name='FR1')

        # filtering catfr1_evs
        # NOTICE: we will relabel catFR1 events as FR1 events, hence task_name='FR1' below.
        catfr1_evs = self.process_events(evs=catfr1_evs, session_increment=200, columns_to_keep=columns_to_keep,
                                         columns_to_keep_fr=columns_to_keep_fr1, task_name='FR1')

        # filtering pal1_evs
        pal1_evs = pal1_evs[columns_to_keep_pal1]
        pal1_evs = pal1_evs[(pal1_evs.type == 'STUDY_PAIR') | (pal1_evs.type == 'REC_EVENT')]

        pal1_evs.type[(pal1_evs.type == 'STUDY_PAIR')] = 'WORD'

        pal1_evs = append_fields(pal1_evs, 'exp_name', pal1_evs.type,
                                 dtypes=pal1_evs.type.dtype, usemask=False,
                                 asrecarray=True)

        pal1_evs.exp_name = 'PAL1'

        pal1_evs = pal1_evs[columns_to_keep]

        recarrays_to_concatenate = []
        if fr1_evs is not None:
            recarrays_to_concatenate.append(fr1_evs)

        if catfr1_evs is not None:
            recarrays_to_concatenate.append(catfr1_evs)

        recarrays_to_concatenate.append(pal1_evs)

        combined_evs = np.concatenate(recarrays_to_concatenate)

        combined_evs = combined_evs.view(np.recarray)

        self.pass_object('combined_evs', combined_evs)

    def process_events(self, evs, session_increment, columns_to_keep, columns_to_keep_fr, task_name='FR1'):
        """

        :param evs: recarray fo events
        :param session_increment: this is a fixed number we add to a session column to allow concatenation
        of events from different tasks into a single recarray (we add 100 for FR1 and 200 for CatFR1)
        :param columns_to_keep: list of column names to retain in the final recarray
        :param columns_to_keep_fr: list of columns to process in the FR or CatFR1 session
        :param task_name: name of the task
        :return: recarray with
        """
        if evs is None:
            return evs

        evs = evs[columns_to_keep_fr]
        evs = evs[(evs.type == 'WORD') | (evs.type == 'REC_WORD') | (evs.type == 'REC_BASE')]

        evs = append_fields(evs, 'correct', evs.recalled,
                            dtypes=evs.recalled.dtype, usemask=False,
                            asrecarray=True)

        # in case we forgot to do this in earlier tasks
        evs.recalled[(evs.type == 'REC_WORD')] = 1
        evs.recalled[(evs.type == 'REC_BASE')] = 0

        evs.correct = np.copy(evs.recalled)

        evs.type[(evs.type == 'REC_WORD') | (evs.type == 'REC_BASE')] = 'REC_EVENT'

        evs = append_fields(evs, 'exp_name', evs.type,
                            dtypes=evs.type.dtype, usemask=False,
                            asrecarray=True)

        evs.exp_name = task_name
        evs.session += session_increment

        # finalizing list of columns

        evs = evs[columns_to_keep]

        return evs
