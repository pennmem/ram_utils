import pytest

from ramutils import utils


@pytest.mark.parametrize("tag_list, expected_output", [
    ([("LAD8,LAHCD7", "LAD9,LAHCD8"), ("LAD8", "LAD9"), ("LAHCD7", "LAHCD8")], ["LAD8-LAD9:LAHCD7-LAHCD8", "LAD8-LAD9", "LAHCD7-LAHCD8"]),
])
def test_combine_tag_names(tag_list, expected_output):
    result = utils.combine_tag_names(tag_list)
    assert result == expected_output
    return


@pytest.mark.parametrize("tag_tuple, expected_output", [
    (("LAD8,LAHCD7", "LAD9,LAHCD8"), "LAD8-LAD9:LAHCD7-LAHCD8"),
    (("LAD8", "LAD9"), "LAD8-LAD9"),
])
def test_join_tag_tuple(tag_tuple, expected_output):
    result = utils.join_tag_tuple(tag_tuple)
    assert result == expected_output
    return