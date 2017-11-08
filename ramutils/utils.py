""" Some utility functions for calculations that are done during the course of the reporting pipeline """


def safe_divide(a, b):
    """ Attempts to perform a/b and catches zero division errors to prevent crashing 

    Parameters:
    -----------
    a: float
        Numerator
    b: float
        Denominator

    Returns
    -------
    result: float
        0 if denominator is 0, else a/b

    """
    try:
        result = a / b
    except ZeroDivisionError:
        result = 0
    
    return result

def combine_tag_names(tag_name_list):
    """ Generate sensible output from a list of tuples containing anode and cathode contact names """
    targets = [join_tag_tuple(target) for target in tag_name_list]
    return targets

def join_tag_tuple(tag_tuple):
    # Check if there is single-target stimulation
    if tag_tuple[0].find(",") == -1:
        return "-".join(tag_tuple)

    # First element of the tag tuple will be anodes, the second cathodes
    anodes = [el for el in tag_tuple[0].split(",")]
    cathodes = [el for el in tag_tuple[1].split(",")]

    pairs = ["-".join((anodes[i], cathodes[i])) for i in range(len(anodes))]
    joined = ":".join(pairs)

    return joined

def sanitize_comma_sep_list(input_list):
    """ Clean up a string with comma-separated values to remove 0 elements"""
    tokens = input_list.split(",")
    tokens = [token for token in tokens if token != "0"]
    output = ",".join(tokens)
    return output