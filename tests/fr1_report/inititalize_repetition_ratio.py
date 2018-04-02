from ptsa.data.readers  import JsonIndexReader
from ptsa.data.readers import BaseEventReader
import RepetitionRatio
import numpy as np
from sklearn.externals import joblib
import os

if __name__ =='__main__':
    task = 'catFR1'
    j_reader = JsonIndexReader('/protocols/r1.json')
    subjects = j_reader.subjects(experiment=task)
    all_repetition_rates = {}

    for subject in subjects:
        try:
            print 'Repetition ratios for subject: ',subject

            evs_field_list = ['item_num', 'serialpos', 'session', 'subject', 'rectime', 'experiment', 'mstime', 'type',
                              'eegoffset', 'iscorrect', 'answer', 'recalled', 'item_name', 'intrusion', 'montage', 'list',
                              'eegfile', 'msoffset']
            if task == 'catFR1':
                evs_field_list += ['category', 'category_num']

            tmp = subject.split('_')
            subj_code = tmp[0]
            montage = 0 if len(tmp) == 1 else int(tmp[1])

            json_reader = JsonIndexReader(os.path.join('/','protocols','r1.json'))

            event_files = sorted(
                list(json_reader.aggregate_values('all_events', subject=subj_code, montage=montage, experiment=task)))
            events = None
            for sess_file in event_files:
                e_path = os.path.join(str(sess_file))
                e_reader = BaseEventReader(filename=e_path, eliminate_events_with_no_eeg=True)

                sess_events = e_reader.read()[evs_field_list]

                if events is None:
                    events = sess_events
                else:
                    events = np.hstack((events, sess_events))

            events = events.view(np.recarray)
            print len(events),' events found'
            recalls = events[events.recalled==1]
            sessions = np.unique(recalls.session)
            lists=np.unique(recalls.list)
            repetition_rates = np.empty([len(sessions),len(lists)])

            for i,r in enumerate(repetition_rates.flat):
                repetition_rates.flat[i] = np.nan
            for session in sessions:
                sess_recalls = recalls[recalls.session == session]
                lists = np.unique(sess_recalls.list)
                repetition_rates[session][:len(lists)] = [RepetitionRatio.repetition_ratio(sess_recalls[sess_recalls.list == l])
                                             for l in lists]
            all_repetition_rates[subject,'catFR1'] = repetition_rates.copy()
        except Exception as e:
            print 'Subject ',subject,'failed:'
            print e
    print 'saving repetition ratio dictionary'
    joblib.dump(all_repetition_rates,'/scratch/leond/catFR1_all_repetition_ratios.pkl')