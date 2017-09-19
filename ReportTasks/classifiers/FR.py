from . import RAMClassifier
from .joint_classifiers import JointEncodingRetrievalClassifier
from ..RamTaskMethods import compute_powers
import numpy as np


class FR3Classifier(RAMClassifier):
    start_time = 0.0
    end_time = 1.366
    buffer_time = 1.365


class FR5Classifier(JointEncodingRetrievalClassifier):
    """
    Joint encoding/retrieval classifier for free recall
    """
    start_time = 0.0
    end_time = 1.366
    buffer_time = 1.365

    retrieval_start_time = -0.525
    retrieval_end_time =  0.0
    retrieval_buf = 0.524

    solver = 'newton-cg'
    class_weight = None

    encoding_samples_weight = 2.5


    @property
    def encoding(self):
        return self.types=='WORD'








