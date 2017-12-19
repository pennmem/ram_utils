import pytest
from ramutils.utils import *


@pytest.mark.parametrize("experiment, expected", [
    ('FR1', '1'),
    ('FR2', '2'),
    ('FR3', '3'),
    ('FR5', '5'),
    ('FR6', '6'),
    ('catFR1', '1'),
    ('catFR2', '2'),
    ('catFR3', '3'),
    ('catFR5', '5'),
    ('catFR6', '6'),
    ('PS1', '1'),
    ('PS2', '2'),
    ('PS2.1', '2.1'),
    ('PS3', '3')
])
def test_extract_experiment_series(experiment, expected):
    actual = extract_experiment_series(experiment)
    assert actual == expected


@pytest.mark.parametrize("tag_list, expected_output", [
    ([("LAD8,LAHCD7", "LAD9,LAHCD8"), ("LAD8", "LAD9"), ("LAHCD7", "LAHCD8")],
     ["LAD8-LAD9:LAHCD7-LAHCD8", "LAD8-LAD9", "LAHCD7-LAHCD8"]),
])
def test_combine_tag_names(tag_list, expected_output):
    result = combine_tag_names(tag_list)
    assert result == expected_output


@pytest.mark.parametrize("tag_tuple, expected_output", [
    (("LAD8,LAHCD7", "LAD9,LAHCD8"), "LAD8-LAD9:LAHCD7-LAHCD8"),
    (("LAD8", "LAD9"), "LAD8-LAD9"),
])
def test_join_tag_tuple(tag_tuple, expected_output):
    result = join_tag_tuple(tag_tuple)
    assert result == expected_output


def test_mkdir_p(tmpdir):
    dirs = [
        "something/without/preceding/slash",
        "/something/with/preceding/slash",
        "something/with/trailing/slash",
        "/",
    ]
    for path in dirs:
        mkdir_p(path)


def test_bytes_to_str():
    string = u'string'
    bstring = b'bytestring'

    assert bytes_to_str(string) == string
    assert bytes_to_str(bstring) == u'bytestring'
