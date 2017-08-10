from . import RAMClassifier
import numpy as np

class TH3Classifier(RAMClassifier):

    start_time = -1.2
    end_time = 0.5
    buffer_time = 1.7

    freqs = np.logspace(np.log10(3),np.log10(180),8)

    @property
    def recalls(self):
        return ((self.events.distErr <= np.max([self.events[0].radius_size, np.median(self.events.distErr)]))
                | self.events.confidence==0)


class THRClassifier(RAMClassifier):
    start_time = 0.0
    end_time = 1.3
    buffer_time = 1.299

    @property
    def recalls(self):
        return self.events.recalled