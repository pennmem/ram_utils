RAM reporting and other utilities
=================================

**Note**: This repo is undergoing a massive cleanup. Things will be
gradually either moved into separate repositories (where common code can
be shared in with other projects) or into a single, top-level
``ramutils`` package. The information below applies to the pre-cleanup
code and will need to be updated once cleanup is complete.

The Pipeline
------------

RAM reports are produced by instantiating a ``ReportPipeline`` and
adding tasks to it in sequence. The ``ReportPipeline`` provides two ways
for the tasks to share data:

-  Pipeline attributes

Each RAM Task in the pipeline holds a reference to the pipeline object
itself in self.pipeline. The subject, task, and mount point are stored
as attributes of the pipeline, and are accessed in this way

-  Object Passing

The pipeline maintains a dictionary of objects that can be referenced
from each task. To add an object to this dictionary from a Task, call
``Task.pass_object(key,value)``. To retrieve an object, call
``Task.get_passed_object(key)``.

Once all the tasks have been added to the pipeline, call
``ReportPipeline.execute_pipeline()`` to perform all the tasks.

For each task in the queue, ``execute_pipeline`` does the following:

-  Check the workspace directory for the file ``Task_Name.completed``
-  If this file exists, call ``Task.input_hashsum()`` and check for
   changes to the tasks’ dependencies
-  If there are no changes, call ``Task.restore()``
-  If there are changes, or if there is no ``.completed`` file, call
   ``Task.run()``

The Report Tasks
----------------

The various RAM reports typically follow a common structure:

-  Load the events for the subject and experiment
-  Load the montage for the subject and experiment
-  Load the EEG for the subject and experiment, and compute a spectral
   decomposition
-  Train a classifier on the power matrix, and evaluate its performance
-  Assemble any other statistics that should be included in the report
-  Generate the relevant graphs
-  Fill in the Latex template
-  Compile the report into a PDF

Biomarker preparation also follows this pattern, but rather than
preparing a PDF report it saves the classifier and the biomarker
parameters as a ``.mat`` file.

Some Directories
----------------

-  ``MatlabIO`` contains wrappers for spicy.io to convert between Python
   objects and Matlab structs.
-  ``PlotUtils`` contains a number of convenience classes for producing
   elegant plots via matplotlib. The RAM report pipelines use this
   module extensively

-  ``RamPipeline`` contains the base pipeline and task classes
-  ``ReportUtils`` contains subclasses from RamPipeline specific to the
   RAM reports, along with some modules used in automated report
   generation
-  ``TexUtils`` contains code to format matrices nicely in Latex
-  ``clinical_affairs`` contains shell scripts to call the various
   reports
-  ``tests`` contains all the RAM reporting pipelines, as well as shell
   scripts for purposes of automation.

For more information, please consult ``docs``.
