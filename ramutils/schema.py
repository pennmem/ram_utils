from __future__ import division

import json

import h5py
import numpy as np
from traits.api import HasTraits


class _NumpyJsonEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        else:
            return json.JSONEncoder.default(self, o)


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

    def to_json(self):
        """Serialize as a JSON string.

        Notes
        -----
        This uses a custom JSON encoder to handle numpy arrays but could
        conceivably lose precision. If this is important, please consider
        serializing in HDF5 format instead.

        """
        d = {name: getattr(self, name) for name in self.visible_traits()}
        return json.dumps(d, cls=_NumpyJsonEncoder)

    @classmethod
    def from_json(cls, data):
        """Deserialize from a JSON string or file.

        :param str or file data:

        """
        if not isinstance(data, str):
            loaded = json.load(data)
        else:
            loaded = json.loads(data)

        return cls(**{key: value for key, value in loaded.items()})
