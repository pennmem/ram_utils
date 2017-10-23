"""Schema definitions for files to be ingested by web reporting."""

from __future__ import division

import numpy as np
import h5py
from traits.api import HasTraits, Int, Float, String, Array


class _Schema(HasTraits):
    """Base class for defining serializable schema."""
    def __init__(self, **kwargs):
        super(_Schema, self).__init__(**kwargs)

        traits = self.class_visible_traits()
        for key, value in kwargs.items():
            if key not in traits:
                raise RuntimeError("trait {} is not in {}".format(
                    key, self.__class__.__name__
                ))
            setattr(self, key, value)

    def _get_type_handler(self, trait):
        return {
            Int: self._scalar_handler,
            Float: self._scalar_handler,
            String: self._string_handler,
            Array: self._array_handler,
        }[trait.trait_type]

    def to_hdf(self, filename, mode='w'):
        """Serialize schema to HDF5.

        :param str filename:
        :param str mode: Default: ``'w'``

        """
        with h5py.File(filename, mode) as hfile:
            for name in self.class_visible_traits():
                trait = self.trait(name)
                # tt = trait.trait_type

                chunks = True if trait.array else False
                dset = hfile.create_dataset('/{}'.format(name),
                                            data=getattr(self, name),
                                            chunks=chunks)
                dset.attrs['desc'] = trait.desc


class FRSessionSummary(_Schema):
    number = Int(desc='session number')  # FIXME: not seemingly used
    name = String(desc='experiment name')
    start = Float(desc='start timestamp')
    end = Float(desc='end timestamp')

    n_words = Int(desc='number of words')
    n_correct_words = Int(desc='number of correctly recalled words')
    pc_correct_words = Float(desc='percentage of correctly recalled words')

    n_pli = Int(desc='number of prior-list intrusions')
    pc_pli = Float(desc='percentage of prior-list intrusions')
    n_eli = Int(desc='number of extra-list intrusions')
    pc_eli = Float(desc='percentage of extra-list intrusions')

    prob_recall = Array(dtype=np.float64, desc='probability of recall by encoding position')
    prob_first_recall = Array(desc='probability of encoding position being recalled first')

    n_math = Int(desc='number of math problems')
    n_correct_math = Int(desc='number of correctly answered math problems')
    pc_correct_math = Float(desc='percentage of correctly answered math problems')
    math_per_list = Float(desc='mean number of math problems per list')

    auc = Float(desc='classifier AUC')
    fpr = Array(dtype=np.float64, desc='false positive rate')
    tpr = Array(dtype=np.float64, desc='true positive rate')

    pc_diff_from_mean = Array(dtype=np.float64, desc='percentage difference from mean')
    perm_AUCs = Array(dtype=np.float64, desc='permutation test AUCs')
    perm_test_pvalue = Float(desc='permutation test p-value')
    jstat_thresh = Float(desc='J statistic')
    jstat_percentile = Float(desc='J statistic percentile')


class CatFRSessionSummary(FRSessionSummary):
    irt_within_cat = Float(desc='average inter-response time within categories')
    irt_between_cat = Float(desc='average inter-response time between categories')


if __name__ == "__main__":
    import time

    data = dict(
        number=0,
        name="FR1",
        start=time.time() - 100,
        end=time.time(),
        n_words=100,
        n_correct_words=30,
        pc_correct_words=100/30.,
        n_pli=10,
        pc_pli=10/100.,
        n_eli=0,
        pc_eli=0.,
        prob_recall=np.random.random((12,)),
        prob_first_recall=np.random.random((12,)),
        n_math=100,
        n_correct_math=30,
        pc_correct_math=30/100.,
        math_per_list=10.,
        auc=0.5,
        fpr=np.random.random((32,)),  # FIXME: length
        tpr=np.random.random((32,)),  # FIXME: length
        pc_diff_from_mean=np.random.random((3,)),  # FIXME: length
        perm_AUCs=np.random.random((32,)),  # FIXME: length
        perm_test_pvalue=0.001,
        jstat_thresh=0.5,
        jstat_percentile=0.5,
    )

    summary = FRSessionSummary(**data)
    summary.to_hdf('/tmp/summary.h5')

    cat_data = data.copy()
    cat_data['irt_within_cat'] = 0.5
    cat_data['irt_between_cat'] = 0.5
    cat_summary = CatFRSessionSummary(**cat_data)
    cat_summary.to_hdf('/tmp/cat_summary.h5')
