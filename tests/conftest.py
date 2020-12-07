import os
import pathlib

import pytest

from pyldt import LDT

ALL_C_PLANES_INCLUDED = 0
ROTATIONAL_SYMMETRY = 1
C0_C180_SYMMETRY = 2
C90_C270_SYMMETRY = 3
C0_C180_C90_C270_SYMMETRY = 4

NUM = 1

DIRECTORY = pathlib.Path(__file__).parent.absolute()


def find_files_with_symmetry(symmetry, num=50):
    files = []
    with os.scandir(DIRECTORY / f"data/ldt/{symmetry}") as dir:
        if num:
            for i, entry in enumerate(dir):
                if i <= num and entry.name.endswith(".ldt") and entry.is_file():
                    files.append(entry.path)
                else:
                    break
        else:
            for i, entry in enumerate(dir):
                if entry.name.endswith(".ldt") and entry.is_file():
                    files.append(entry.path)
    return files


@pytest.fixture(
    scope="session",
    params=find_files_with_symmetry(ROTATIONAL_SYMMETRY, NUM)
    + find_files_with_symmetry(C0_C180_SYMMETRY, NUM)
    + find_files_with_symmetry(C90_C270_SYMMETRY, NUM)
    + find_files_with_symmetry(C0_C180_C90_C270_SYMMETRY, NUM),
)
def lum(request):
    return LDT.from_file(request.param)


@pytest.fixture(
    scope="session",
    params=find_files_with_symmetry(ROTATIONAL_SYMMETRY, NUM)
    + find_files_with_symmetry(C0_C180_SYMMETRY, NUM)
    + find_files_with_symmetry(C90_C270_SYMMETRY, NUM)
    + find_files_with_symmetry(C0_C180_C90_C270_SYMMETRY, NUM),
)
def string(request):
    with open(request.param) as fp:
        return fp.read()


@pytest.fixture(scope="session")
def lum1():
    return LDT.from_file(DIRECTORY / "data/ldt/1/21013614.ldt")


@pytest.fixture(scope="session")
def lum2():
    return LDT.from_file(DIRECTORY / "data/ldt/3/22169664.ldt")


@pytest.fixture(scope="session")
def string1():
    with open(DIRECTORY / "data/ldt/1/21013614.ldt") as f:
        return f.read()


@pytest.fixture(scope="session")
def string2():
    with open(DIRECTORY / "data/ldt/3/22169664.ldt") as f:
        return f.read()
