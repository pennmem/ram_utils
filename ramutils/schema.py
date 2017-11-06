from __future__ import division

import numpy as np
import h5py
from traits.api import HasTraits, Int, Float, String, Array


class Schema(HasTraits):
    """Base class for defining serializable schema/data classes."""
    def __init__(self, **kwargs):
        super(Schema, self).__init__(**kwargs)

        traits = self.class_visible_traits()
        for key, value in kwargs.items():
            if key not in traits:
                raise RuntimeError("trait {} is not in {}".format(
                    key, self.__class__.__name__
                ))
            setattr(self, key, value)

    def __str__(self):
        attr_strs = ["{}={}".format(attr, getattr(self, attr))
                     for attr in self.visible_traits()]
        return "<{}({})>".format(self.__class__.__name__, '\n    '.join(attr_strs))

    def __repr__(self):
        return self.__str__()

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
                if trait.desc is not None:
                    dset.attrs['desc'] = trait.desc

    @classmethod
    def from_hdf(cls, filename):
        """Deserialize from HDF5.

        :param str filename:
        :returns: Deserialized instance

        """
        self = cls()
        with h5py.File(filename, 'r') as hfile:
            for name in self.visible_traits():
                setattr(self, name, hfile['/{}'.format(name)][:])
        return self


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
    #
    # summary = FRSessionSummary(**data)
    # summary.to_hdf('/tmp/summary.h5')
    #
    # cat_data = data.copy()
    # cat_data['irt_within_cat'] = 0.5
    # cat_data['irt_between_cat'] = 0.5
    # cat_summary = CatFRSessionSummary(**cat_data)
    # cat_summary.to_hdf('/tmp/cat_summary.h5')
