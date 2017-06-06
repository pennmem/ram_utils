import re

def latex_table(matrix, hlines=True):
    result = '\hline\n' if hlines else ''
    for row in matrix:
        for j,elem in enumerate(row):
            s = str(elem)
            if re.match('-?\d+\.?\d+?$', s):
                s = '$'+s+'$'
            result += s
            result += ' & ' if j<len(row)-1 else ' \\\\\n'
        if hlines:
            result += '\hline\n'
    return result