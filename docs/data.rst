Serializable data structures
============================

Defining data classes
---------------------

Data can be defined in a serializable manner using the
:class:`traitschema.Schema` base class which adds serialization methods to
data classes that are defined using the :mod:`traits` package. To ensure
serializability, use the ``Array`` type whenever possible.


Experiment parameters
-----------------------

Experiment parameters (e.g., timing windows) are defined as
:class:`traitschema.Schema` subclasses so that the parameters used when
training a classifier can be easily saved.

.. automodule:: ramutils.parameters
    :members: FilePaths, ExperimentParameters, FRParameters, PALParameters, PS5Parameters, StimParameters

Underlying Data
---------------
All data necessary to rebuild a report is saved in a binary format as part of generating the report. All data is dumped
into a single directory with differentiation between subjects/sessions/data done by following a strict naming convention:
{subject}_{experiment}_{session}_{data_type}.{file_type}. Most saved objects are unique to a particular subject/experiment/session.
In cases where this is not true, {session} wil be an underscore-separated list of the sessions used to generate the data.
To see how this is done, see  :func:`ramutils.tasks.misc.save_all_output` and :func:`ramutils.tasks.misc.load_existing_results`.
For example, if a target selection table was generated using sessions 1, 2, and 3 for subject R1XXX and experiment
XYZ, then the file would be saved as R1XXX_XYZ_1_2_3_target_selection_table.csv. Listed below are the types of data stored. Their
corresponding objects are also noted. The properties and methods defined for each of these objects can be found in the
documentation below.

* target_selection_table -- A csv file containing metadata for each electrode
* classifier_summary -- Metadata related to classifier performance :class:`ClassifierSummary`
* math_summary -- Math events and useful helper methods for assessing performance on the distractor task :class:`MathSummary`
* session_summary -- Events and helper methods for conducting behavioral analyses and generating plots. In many cases,
  there are summary objects specific to the type of session, i.e. stim vs. nonstim, FR vs. CatFR vs. PS, etc.


.. automodule:: ramutils.reports.summary
    :members: ClassifierSummary, MathSummary, Summary, SessionSummary, FRSessionSummary, CatFRSessionSummary, StimSessionSummary, FRStimSessionSummary, PSSessionSummary
