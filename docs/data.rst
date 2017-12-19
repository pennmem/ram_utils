Serializable data structures
============================

Defining data classes
---------------------

Data can be defined in a serializable manner using the
:class:`traitschema.Schema` base class which adds serialization methods to
data classes that are defined using the :mod:`traits` package. To ensure
serializability, use the ``Array`` type whenever possible.


Experimental parameters
-----------------------

Experimental parameters (e.g., timing windows) are defined as
:class:`traitschema.Schema` subclasses so that the parameters used when
training a classifier can be easily saved.

.. automodule:: ramutils.parameters
    :members:
