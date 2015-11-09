__author__ = 'm'

from Table import *
from sigfig import *

def generate_tex_table(caption='My Table', header=[], columns=[], label = 'tab:mytable',sigfigs=2):
    import StringIO
    out_str_io = StringIO.StringIO()

    t = Table(len(header), caption=caption, label=label)
    t.add_header_row(header)
    t.add_data(columns, sigfigs=sigfigs)
    t.print_table(out_str_io)

    return out_str_io.getvalue()


