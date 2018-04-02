Pipelines
=========

Experiment configuration file generation and post-experiment reporting is done
via a series of tasks that are built up into a pipeline using dask_.

.. _dask: http://dask.pydata.org/en/latest/index.html

Existing Pipelines
^^^^^^^^^^^^^^^^
.. autofunction:: ramutils.pipelines.ramulator_config.make_ramulator_config
.. autofunction:: ramutils.pipelines.report.make_report
.. autofunction:: ramutils.pipelines.aggregated_report.make_aggregated_report

Defining tasks
--------------

Tasks are created by using the :func:`ramutils.tasks.task` decorator or wrapping
a function with :func:`ramutils.tasks.make_task`. These simply apply the
:func:`dask.delayed` and (optionally) ``joblib`` caching decorators. The former
is important for adding the ability to parallelize a pipeline (for tasks that
can run independently) while the latter allows for resuming a pipeline when
something goes wrong or if only changing one parameter which does not affect all
tasks.

.. autofunction:: ramutils.tasks.task

.. autofunction:: ramutils.tasks.make_task

Reference
---------

Common tasks come predefined in the :mod:`ramutils.tasks` package and are
documented below.

Classifier tasks
^^^^^^^^^^^^^^^^

.. automodule:: ramutils.tasks.classifier
    :members:

Events tasks
^^^^^^^^^^^^

.. automodule:: ramutils.tasks.events
    :members:

Miscellaneous tasks
^^^^^^^^^^^^^^^^^^^

.. automodule:: ramutils.tasks.misc
    :members:

Montage tasks
^^^^^^^^^^^^^

.. automodule:: ramutils.tasks.montage
    :members:

Odin/Ramulator tasks
^^^^^^^^^^^^^^^^^^^^

.. automodule:: ramutils.tasks.odin
    :members:

Power computation tasks
^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: ramutils.tasks.powers
    :members:

Report summary tasks
^^^^^^^^^^^^^^^^^^^^

.. automodule:: ramutils.tasks.summary
    :members:

