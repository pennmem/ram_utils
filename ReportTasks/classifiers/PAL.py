from . import RAMClassifier
from .joint_classifiers import JointEncodingRetrievalClassifier
import  numpy as np
from ..RamTaskMethods import compute_powers

class PAL3Classifier(RAMClassifier):
    start_time = 0.3
    end_time = 2.0
    buffer_time = 1.0

    freqs = np.logspace(np.log10(3), np.log10(180), 8)

    def recalls(self):
        return self.events.correct




class PAL5Classsifier(JointEncodingRetrievalClassifier):
    start_time = 0.3
    end_time = 2.00
    buf = 1.2

    retrieval_start_time = -0.625
    retrieval_end_time = -0.1
    retrieval_buf = 0.524

    @property
    def encoding(self):
        return (self.types =='STUDY_PAIR') | (self.types=='PRACTICE_PAIR')


    def compute_features(self):
        encoding_events = self.events[self.encoding]
        retrieval_events = self.events[~self.encoding]
        encoding_powers, encoding_events = compute_powers(encoding_events, self.channels, self.start_time,
                                                          self.end_time, self.buffer_time, self.freqs, True, self.pairs)
        retrieval_powers,retrieval_events = compute_powers(retrieval_events, self.channels, self.retrieval_start_time,
                                                           self.retrieval_end_time, self.retrieval_buf, self.freqs,
                                                           True, self.pairs)

        self.events = np.concatenate([encoding_events,retrieval_events]).astype(np.recarray).sort(order=['session','list','mstime'])
        self.pow_mat = np.empty((len(self.events),encoding_powers.shape[-1]))
        self.pow_mat[self.encoding] = encoding_powers
        self.pow_mat[~self.encoding] = retrieval_powers

        return (self.pow_mat,self.events)

