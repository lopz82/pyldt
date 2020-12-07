import itertools
import pathlib

import numpy as np
import pytest
from pandas.core.frame import DataFrame

from pyldt._ldt import (
    get_data_from_string,
    clean_tail_empty_strings,
    numerical_conversion,
    LDT,
)

EXPECTED_SHAPE = (181, 361)


@pytest.mark.basics
def test_shape(lum):
    assert lum.dist.shape == EXPECTED_SHAPE


def assert_rotational_symmetry(lum):
    for col in range(lum._original_dist.shape[1]):
        assert (lum._original_dist[:, 0] == lum._original_dist[:, col]).all()


def assert_C0_C180_symmetry(lum):
    assert_has_reflectional_symmetry_C180(lum)


def assert_C90_C270_symmetry(lum):
    c_planes = lum._original_dist.shape[1]
    half = int(c_planes / 2)
    # We create two matrixes pointing two the result of each symmetry.
    C90sym = lum._original_dist[:, : half + 1]
    C270sym = lum._original_dist[:, half + 1 :]
    for n in np.arange(C90sym.shape[1]):
        assert (C90sym[:, n] == C90sym[:, -1 - n]).all()
    for n in np.arange(C270sym.shape[1]):
        assert (C270sym[:, n] == C270sym[:, -1 - n]).all()


def assert_C0_C180_C90_C270_symmetry(lum):
    assert_has_reflectional_symmetry_C180(lum)


def assert_has_reflectional_symmetry_C180(lum):
    for n in range(0, lum._original_dist.shape[1]):
        assert (lum._original_dist[:, n] == lum._original_dist[:, -n]).all()


def test_classmethod_from_raw_data(string1, string2):
    str1 = LDT.from_raw_data(string1)
    str2 = LDT.from_raw_data(string2)


def test_path_and_raw_data_exception():
    with pytest.raises(TypeError):
        _ = LDT("path", "raw_data")


def test_get_ldt_file_IOError():
    with pytest.raises(IOError):
        error = LDT()
        error.get_ldt_file("./fake-path")


def test__eq__(lum1, lum2):
    assert lum1 == lum1
    assert lum2 == lum2
    assert lum1 != lum2
    assert lum2 != lum1


def test__hash__(lum1, lum2):
    assert len(str(hash(lum1))) == 10
    assert len(str(hash(lum2))) == 10
    assert hash(lum1) != hash(lum2)


# @pytest.mark.skip(
#     "Fails without any explanation. It works alone, but "
#     "if the whole test suit is run, it fails."
# )
def test_compare_to():
    """assert (0 <= lum2.compare_to(lum1) <= 100) == True is not working in pytest.
    To avoid performing the comparison twice (to check <= 0 and <= 100), we store the value."""
    DIRECTORY = pathlib.Path(__file__).parent.absolute()

    def l1():
        return LDT.from_file(DIRECTORY / "data/ldt/1/21013614.ldt")

    def l2():
        return LDT.from_file(DIRECTORY / "data/ldt/3/22169664.ldt")

    a = l1()
    b = l2()
    res12 = a.compare_to(b)
    assert 0 <= res12 <= 100
    res21 = b.compare_to(a)
    assert 0 <= res21 <= 100
    assert res12 == res21
    assert a.compare_to(a) == 100
    assert b.compare_to(b) == 100


def test__str__(lum1, lum2):
    assert (
        str(lum1)
        == "[[1768.  1768.  1768.  ... 1768.  1768.  1768. ]\n [1757.6 1757.6 1757.6 ... 1757.6 1757.6 "
        "1757.6]\n [1747.2 1747.2 1747.2 ... 1747.2 1747.2 1747.2]\n ...\n [   0.     0.     0.  ...  "
        "  0.     0.     0. ]\n [   0.     0.     0.  ...    0.     0.     0. ]\n [   0.     0.     "
        "0.  ...    0.     0.     0. ]]"
    )
    assert (
        str(lum2)
        == "[[2396.   2396.   2396.   ... 2396.   2396.   2396.  ]\n [2392.   2391.84 2391.68 ... 2389.6 "
        " 2389.6  2389.6 ]\n [2388.   2387.68 2387.36 ... 2383.2  2383.2  2383.2 ]\n ...\n [   0.     "
        " 0.      0.   ...    0.      0.      0.  ]\n [   0.      0.      0.   ...    0.      0.      "
        "0.  ]\n [   0.      0.      0.   ...    0.      0.      0.  ]]"
    )


def test_max_property(lum1, lum2):
    assert lum1.max == 1768.0
    assert lum1.min == 0.0
    assert lum2.max == 2396.0
    assert lum2.min == 0


def test_has_nan(lum1, lum2):
    assert lum1.has_nan == False
    assert lum2.has_nan == False


def test_as_array(lum1, lum2):
    assert len(lum1.as_array.shape) == 1
    assert len(lum2.as_array.shape) == 1


def test_as_dataframe(lum1, lum2):
    assert type(lum1.as_dataframe) == DataFrame
    assert type(lum2.as_dataframe) == DataFrame


@pytest.mark.symmetry
def test_symmetry(lum):
    if lum.symmetry == 1:
        assert_rotational_symmetry(lum)
    elif lum.symmetry == 2:
        assert_C0_C180_symmetry(lum)
    elif lum.symmetry == 3:
        assert_C90_C270_symmetry(lum)
    elif lum.symmetry == 4:
        assert_C0_C180_C90_C270_symmetry(lum)


@pytest.mark.numeric
@pytest.mark.parametrize(
    "args, expected",
    [
        ("3,5", 3.5),
        ("3.5", 3.5),
        ("0", 0),
        ("0,0001", 0.0001),
        ("0.0001", 0.0001),
        ("STRING", "STRING"),
        ("string", "string"),
    ],
)
def test_numerical_conversion(args, expected):
    assert numerical_conversion(args) == expected


BASE = ["A", "B", "C", "D", "E", "F"]
TO_DELETE = ["\r", ""]
COMBINATIONS = []
for perm in itertools.permutations(TO_DELETE):
    COMBINATIONS.append(BASE + list(perm))


@pytest.mark.strings
@pytest.mark.parametrize("args", COMBINATIONS)
def test_clean_tail_empty_strings(args):
    assert clean_tail_empty_strings(args) == BASE


@pytest.mark.strings
def test_get_data_from_string(string):
    result = get_data_from_string(string)
    assert isinstance(result, list)
    assert "\n" not in result
    assert result[-1] not in TO_DELETE
