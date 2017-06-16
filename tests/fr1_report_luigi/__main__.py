import luigi
from Params import Params
from Pipeline import Pipeline
from Setup import Setup
from FR1EventPreparation_1 import FR1EventPreparation_1
from Sink import Sink


"""
to run
python -m  tests.fr1_report_luigi

"""

params = Params()
pipeline = Pipeline(params)


# luigi.build([
#                 # Setup(pipeline=pipeline, mark_as_completed=False,workspace_dir='d:\sc_lui', subject='R1065J'),
#                 FR1EventPreparation_1(pipeline=pipeline, mark_as_completed=True)
#             ],
#             local_scheduler=True,
#             )

# luigi.build([
#                 # Setup(pipeline=pipeline, mark_as_completed=False,workspace_dir='d:\sc_lui', subject='R1065J'),
#                 FR1EventPreparation_1(pipeline=pipeline, mark_as_completed=True,workspace_dir='d:\sc_lui', subject='R1065J')
#             ],
#             local_scheduler=True,
#             )


luigi.build([
                # Setup(pipeline=pipeline, mark_as_completed=False,workspace_dir='d:\sc_lui', subject='R1065J'),
                Sink(pipeline=pipeline, mark_as_completed=False,workspace_dir='d:\sc_lui', subject='R1065J')
            ],
            local_scheduler=True,
            )

