from . import RAMClassifier
import numpy as np
from ..RamTaskMethods import compute_powers,ModelOutput

class JointEncodingRetrievalClassifier(RAMClassifier):

    encoding_samples_weight = 1

    retrieval_start_time = -1.0
    retrieval_end_time  = 0.0
    retrival_buf = 0.666

    @property
    def encoding(self):
        return np.array([True]*len(self.events))

    def compute_features(self):
        encoding_events = self.events[self.encoding]
        retrieval_events = self.events[~self.encoding]
        encoding_powers, encoding_events = compute_powers(encoding_events, self.channels, self.pairs,
                                                            self.start_time, self.end_time,
                                                            self.buffer_time,
                                                            self.freqs, True
                                                            )
        retrieval_powers,retrieval_events = compute_powers(retrieval_events,self.channels,self.pairs,
                                                           self.retrieval_start_time,self.retrieval_end_time,self.retrival_buf,
                                                           self.freqs,True
                                                           )

        self.events = np.concatenate([encoding_events,retrieval_events]).astype(np.recarray).sort(order=['session','list','mstime'])
        self.pow_mat = np.empty((len(self.events),encoding_powers.shape[-1]))
        self.pow_mat[self.encoding] = encoding_powers
        self.pow_mat[~self.encoding] = retrieval_powers
        return (self.pow_mat,self.events)

    def get_sample_weights(self,events):
        insample_enc_mask = self.encoding
        insample_retrieval_mask = ~self.encoding

        n_enc_0 = events[insample_enc_mask & ~self.recalls].shape[0]
        n_enc_1 = events[insample_enc_mask &  self.recalls].shape[0]

        n_ret_0 = events[insample_retrieval_mask & ~self.recalls].shape[0]
        n_ret_1 = events[insample_retrieval_mask & self.recalls].shape[0]

        n_vec = np.array([1.0 / n_enc_0, 1.0 / n_enc_1, 1.0 / n_ret_0, 1.0 / n_ret_1], dtype=np.float)
        n_vec /= np.mean(n_vec)

        n_vec[:2] *= self.encoding_samples_weight

        n_vec /= np.mean(n_vec)


        sample_weights = np.ones(events.shape[0], dtype=np.float)

        sample_weights[insample_enc_mask & ~self.recalls] = n_vec[0]
        sample_weights[insample_enc_mask &  self.recalls] = n_vec[1]
        sample_weights[insample_retrieval_mask & ~self.recalls] = n_vec[2]
        sample_weights[insample_retrieval_mask &  self.recalls] = n_vec[3]

        return sample_weights

def cross_validate_encoding_only(fr5_classifier, pow_mat):
    events = fr5_classifier.events
    is_encoding = fr5_classifier.encoding
    sessions = np.unique(events.session)
    probs = np.array([np.nan] * len(events))
    for session in sessions:
        insample = events.session != session
        outsample = (~insample) & is_encoding
        fr5_classifier.fit(pow_mat[insample], events[insample])
        probs[outsample] = fr5_classifier.predict_proba(pow_mat[outsample])

    return ModelOutput(fr5_classifier.recalls[is_encoding], probs[is_encoding])
