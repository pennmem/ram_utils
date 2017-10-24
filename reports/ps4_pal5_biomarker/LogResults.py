import os.path

import json
from copy import deepcopy
import numpy as np
import pandas as pd
import io

from sklearn.externals import joblib

from ReportUtils import RamTask

import hashlib



class LogResults(RamTask):
    def __init__(self, params,mark_as_completed=False, log_filename=''):
        super(LogResults,self).__init__(mark_as_completed)
        self.log_filename = log_filename

        self.columns = ['subject','AUC_enc', 'AUC_ret', 'AUC_both', 'median_classifier','retrieval_thresh']


    def run(self):

        auc_enc = self.get_passed_object('auc_encoding')
        auc_ret = self.get_passed_object('auc_retrieval')
        auc_both = self.get_passed_object('auc_both')
        xval_output = self.get_passed_object('xval_output')


        try:
            retrieval_thresh = self.get_passed_object('retrieval_biomarker_threshold')
        except:
            retrieval_thresh = -999

        try:
            median_classifier = [xval_output[-1].jstat_thresh]
        except:
            median_classifier = -999

        auc_enc_sess = deepcopy(auc_enc)
        auc_ret_sess = deepcopy(auc_ret)
        auc_both_sess = deepcopy(auc_both)

        # del auc_enc_sess[-1]
        # del auc_ret_sess[-1]
        # del auc_both_sess[-1]

         # 'AUC_enc':[0.75]

        try:
            df_from_disk = pd.read_csv(self.log_filename)
        except:
            df_from_disk = pd.DataFrame(columns=self.columns)


        d = pd.DataFrame(
            {'subject':[self.pipeline.subject],
             'AUC_enc':[np.mean(auc_enc_sess)],
             'AUC_ret':[np.mean(auc_ret_sess)],
             'AUC_both':[np.mean(auc_both_sess)],
             'retrieval_thresh':[retrieval_thresh],
             'median_classifier':[median_classifier],
             },
            columns=self.columns
        )

        df_from_disk = df_from_disk.append(d)

        df_from_disk = df_from_disk.fillna(-999)

        df_from_disk.to_csv(self.log_filename, sep=',', index=False)


        # formatting
        for col in df_from_disk.columns:
            df_from_disk[col] = df_from_disk[col].apply(lambda x: '|' +  col + ' ' + str(x))

        with io.open(self.log_filename+'.formatted', 'wb') as file:
            df_from_disk.to_csv(file, sep=" ", header=False, index=False, quotechar=' ')





if __name__ == '__main__':

    columns = ['subject','AUC_enc']

    try:
        df1 = pd.read_csv('dupa.csv')
        print df1
    except:
        df1 = pd.DataFrame(columns=columns)
        pass


    df = pd.DataFrame(columns=columns)

    d = pd.DataFrame({'subject':['R1234M'],
         # 'AUC_enc':[0.75]

    }, columns=columns)
    # print d

    df1 = df1.append(d)
    df1 = df1.fillna(-1)
    #
    print df1

    for col in df1.columns:
        df1[col] = df1[col].apply(lambda x: '|' +  col + ' ' + str(x))
    with io.open('dupa.csv', 'wb') as file: df1.to_csv(file, sep=" ", header=False, index=False, quotechar=' ')

    # df1.to_csv('dupa.csv',index=False)


