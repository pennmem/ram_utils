import re
import numpy as np

def parse_biomarker_output(filename):
    probs = []
    thresh = None
    pattern = re.compile(r"prob=(?P<prob>\d.\d+), threshold=(?P<thresh>\d.\d+)")
    f = open(filename, 'r')
    # after_3rd_list = False
    for line in f:
        # if line[0:7]=='KS test':
        #     after_3rd_list = True
        # else:
        #     if after_3rd_list:
        m = pattern.match(line)
        if m:
            probs.append(float(m.group('prob')))
            thresh = float(m.group('thresh'))
    f.close()
    return np.array(probs, dtype=float), thresh
