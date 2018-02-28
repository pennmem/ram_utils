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


def test_safe_divide():
    assert safe_divide(1, 0) == 0.
    assert safe_divide(1.0, 0) == 0.


def test_safe_divide_decorator():
    @safe_divide
    def division():
        return 1/0

    assert division() == 0.


@pytest.mark.parametrize("file_path, subject, experiment, montage, sessions, file_name, file_type", [
    ('/R1345D_catFR5_0_math_summary.h5', 'R1345D', 'catFR5', 0, [0], 'math_summary', 'h5'),
    ('/R1345D_FR1_0_1_2_target_selection_table.csv', 'R1345D', 'FR1', 0, [0, 1, 2], 'target_selection_table', 'csv'),
    ('/R1345D_FR1_0_classifier_session_0.h5', 'R1345D', 'FR1', 0, [0], 'classifier_session_0', 'h5'),
    ('/R1345D_1_FR1_0_classifier_session_0.h5', 'R1345D', 'FR1', 1, [0], 'classifier_session_0', 'h5'),
])
def test_extract_report_info_from_path(file_path, subject, experiment, montage, sessions, file_name, file_type):
    results = extract_report_info_from_path(file_path)
    assert results['subject'] == subject
    assert results['experiment'] == experiment
    assert results['montage'] == montage
    assert results['sessions'] == sessions
    assert results['file_name'] == file_name
    assert results['file_type'] == file_type
