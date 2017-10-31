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