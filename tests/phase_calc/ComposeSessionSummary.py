from ReportUtils import ReportRamTask


def make_connectivity_strength_table(bp_tal_structs, connectivity_strength):
    ttest_data = [list(a) for a in zip(bp_tal_structs.etype.values, bp_tal_structs.index.values, bp_tal_structs.bp_atlas_loc, connectivity_strength)]
    return ttest_data

def format_connectivity_strength_table(table_data):
    for i,line in enumerate(table_data):
        if abs(line[-1]) < 1.5:
            table_data[:] = table_data[:i]
            return table_data
        color = 'red' if line[-1]>=2.0 else 'blue' if line[-1]<=-2.0 else None
        line[-1] = '%.3f' % line[-1]
        if color is not None:
            if color == 'red':
                line[:] = ['\\textbf{\\textcolor{BrickRed}{%s}}' % s for s in line]
            elif color == 'blue':
                line[:] = ['\\textbf{\\textcolor{blue}{%s}}' % s for s in line]


class ComposeSessionSummary(ReportRamTask):
    def __init__(self, mark_as_completed=True):
        super(ComposeSessionSummary, self).__init__(mark_as_completed)

    def run(self):
        bp_tal_structs = self.get_passed_object('bp_tal_structs')
        connectivity_strength = self.get_passed_object('connectivity_strength')

        connectivity_strength_data = make_connectivity_strength_table(bp_tal_structs, connectivity_strength)
        connectivity_strength_data.sort(key=lambda row: abs(row[-1]), reverse=True)
        connectivity_strength_table = format_connectivity_strength_table(connectivity_strength_data)

        self.pass_object('connectivity_strength_table', connectivity_strength_table)
