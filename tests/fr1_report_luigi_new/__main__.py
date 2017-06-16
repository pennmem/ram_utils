import luigi
from Params import Params
from Pipeline import Pipeline
from Report import Report

"""
to run:

see the comment in the Reports.py
or when running from pycharm you may run this file directly

"""

params = Params()
pipeline = Pipeline(params)

luigi.build([
                Report(pipeline=pipeline, mark_as_completed=False, workspace_dir='d:\sc_lui', subject='R1065J')
            ],
            local_scheduler=True,
            )

