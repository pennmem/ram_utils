__author__ = 'm'

import os
import os.path
import numpy as np

from ptsa.data.readers import BaseEventReader
from ptsa.data.readers.IndexReader import JsonIndexReader

from RamPipeline import *
from ReportUtils import RamTask

import hashlib
from sklearn.externals import joblib
from ReportTasks.RamTaskMethods import create_baseline_events
from numpy.lib.recfunctions import append_fields


class CombinedEventPreparation(RamTask):
    def __init__(self, mark_as_completed=True):
        super(CombinedEventPreparation, self).__init__(mark_as_completed)

    def run(self):
        subject = self.pipeline.subject
        tmp = subject.split('_')
        subj_code = tmp[0]

        fr1_evs = self.get_passed_object('FR_events_with_recall')
        pal1_evs = self.get_passed_object('PAL1_events_with_recall')

        colums_to_keep = ['session', 'type', 'correct', 'eegoffset', 'msoffset', 'eegfile', 'exp_name']

        colums_to_keep_fr1 = ['session', 'type', 'recalled', 'eegoffset', 'msoffset', 'eegfile']
        colums_to_keep_pal1 = ['session', 'type', 'correct', 'eegoffset', 'msoffset', 'eegfile']

        if fr1_evs is not None:
            # filtering fr1_evs
            fr1_evs = fr1_evs[colums_to_keep_fr1]
            fr1_evs = fr1_evs[(fr1_evs.type == 'WORD') | (fr1_evs.type == 'REC_WORD') | (fr1_evs.type == 'REC_BASE')]

            fr1_evs = append_fields(fr1_evs, 'correct', fr1_evs.recalled,
                                    dtypes=fr1_evs.recalled.dtype, usemask=False,
                                    asrecarray=True)

            # in case we forgot to do this in earlier tasks
            fr1_evs.recalled[(fr1_evs.type == 'REC_WORD')] = 1
            fr1_evs.recalled[(fr1_evs.type == 'REC_BASE')] = 0

            fr1_evs.correct = np.copy(fr1_evs.recalled)

            fr1_evs.type[(fr1_evs.type == 'REC_WORD') | (fr1_evs.type == 'REC_BASE')] = 'REC_EVENT'

            fr1_evs = append_fields(fr1_evs, 'exp_name', fr1_evs.type,
                                    dtypes=fr1_evs.type.dtype, usemask=False,
                                    asrecarray=True)

            fr1_evs.exp_name = 'FR1'
            fr1_evs.session += 100

            # finalizing list of columns

            fr1_evs = fr1_evs[colums_to_keep]

        # filtering pal1_evs
        pal1_evs = pal1_evs[colums_to_keep_pal1]
        pal1_evs = pal1_evs[(pal1_evs.type == 'STUDY_PAIR') | (pal1_evs.type == 'REC_EVENT')]

        pal1_evs.type[(pal1_evs.type == 'STUDY_PAIR')] = 'WORD'

        pal1_evs = append_fields(pal1_evs, 'exp_name', pal1_evs.type,
                                 dtypes=pal1_evs.type.dtype, usemask=False,
                                 asrecarray=True)

        pal1_evs.exp_name = 'PAL1'

        pal1_evs = pal1_evs[colums_to_keep]


        if fr1_evs is not None:
            combined_evs = np.concatenate((fr1_evs, pal1_evs))
        else:
            combined_evs = pal1_evs

        combined_evs = combined_evs.view(np.recarray)

        self.pass_object('combined_evs', combined_evs)
