class Pipeline(object):
    def __init__(self, params):

        self.workspace_dir = 'd:/scratch/luigi_demo_fr1'
        self.params = params
        self.mount_point = 'd:/'
        self.subject = 'R1065J'
        # subject = 'R1065J'
        self.task = 'FR1'
        self.experiment = 'FR1'
        self.sessions = None
        self.exit_on_no_change = True
        self.recompute_on_no_status = True

        self.passed_objects_dict = {}
