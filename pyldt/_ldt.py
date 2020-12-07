import hashlib
import math
from pathlib import Path
from typing import List, Optional, Tuple, Union, TypeVar

import numpy as np
from scipy.interpolate import interp2d

TypeLDT = TypeVar("TypeLDT", bound="LDT")


def clean_tail_empty_strings(data: List[str]) -> List[str]:
    """Removes empty strings from the provided data"""
    while data[-1] in ["", "\r"]:
        data.pop()
    return data


def get_data_from_string(raw_data: str) -> List[str]:
    """Reads, splits and clean raw LDT data"""
    split = raw_data.split("\n")
    return clean_tail_empty_strings(split)


def numerical_conversion(n: str) -> Union[int, float]:
    """Helper method to convert strings into ints and floats"""
    res = n
    n = n.replace(",", ".", 1)
    try:
        res = int(n)
    except ValueError:
        try:
            res = float(n)
        except ValueError:
            pass
    return res


# Default values for interpolation. Using a 1 unit step is slower but guarantee compatibility.
# The most usual step is 5 units, but there are some files from other manufacturers using 2.5 or 1.
DEFAULT_C_PLANES_STEP_INTERPOLATION = 1
DEFAULT_INTENSITIES_STEP_INTERPOLATION = 1
DEFAULT_NUM_C_PLANES = 24

# The precision provided by a float16 is enough and reduces the memory usage and improves the speed
DEFAULT_PRECISION = np.float16


class LDT:
    # Mapping of all existing symmetries types according to the LDT file standard
    ALL_C_PLANES_INCLUDED = 0
    ROTATIONAL_SYMMETRY = 1
    C0_C180_SYMMETRY = 2
    C90_C270_SYMMETRY = 3
    C0_C180_C90_C270_SYMMETRY = 4

    # Starting line of the photometric data in the LDT file
    START_LINE_PHOTO_DATA = 42

    # Mapping of attributes and line number in the LDT file
    ATTRIBUTE_LINE_MAP_LDT_FILE = {
        "manufacturer": 0,
        "symmetry": 2,
        "num_c_planes": 3,
        "dist_c_planes": 4,
        "num_lum_int": 5,
        "dist_lum_int": 6,
        "model": 8,
    }

    def __init__(
        self,
        *,
        path: Optional[str] = None,
        raw_data: Optional[str] = None,
        c_planes_step_interpolation=DEFAULT_C_PLANES_STEP_INTERPOLATION,
        intensities_step_interpolation=DEFAULT_INTENSITIES_STEP_INTERPOLATION,
        num_c_planes=DEFAULT_NUM_C_PLANES,
        precision=DEFAULT_PRECISION,
    ) -> None:
        if path and raw_data:
            raise TypeError("Use a path or data argument")
        self.raw_data = ""
        self.path = None
        self.manufacturer = None
        self.filename = None
        self.model = None
        self.symmetry = None
        self.num_c_planes = None
        self.dist_c_planes = None
        self.num_lum_int = None
        self.dist_lum_int = None
        self.c_planes = None
        self.intensities = None
        self._original_dist = None
        self.dist = None
        self._c_planes_step_interpolation = c_planes_step_interpolation
        self._intensities_step_interpolation = intensities_step_interpolation
        self._num_c_planes = num_c_planes
        self._precision = precision
        if path:
            self.path = Path(path)
            self.filename = self.path.name
            self.clean_data = self.get_ldt_file(self.path)
            self.fit()
        if raw_data:
            self.clean_data = get_data_from_string(raw_data)
            self.fit()

    @classmethod
    def from_file(cls, path: str) -> TypeLDT:
        """Simple shortcut for creating a LDT instance from a file"""
        return cls(path=path)

    @classmethod
    def from_raw_data(cls, raw_data: str) -> TypeLDT:
        """Simple shortcut for creating a LDT instance from raw data"""
        return cls(raw_data=raw_data)

    def get_ldt_file(self, path: str) -> List[str]:
        """Opens and reads the LDT file. Cleans trailing whitespaces"""
        try:
            data = []
            with open(path) as fp:
                for i, line in enumerate(fp):
                    self.raw_data += line
                    if line == "" and i > len(fp) / 2:
                        continue
                    data.append(line.strip())
            return clean_tail_empty_strings(data)
        except IOError:
            raise IOError(f"Error opening {path}")

    def __hash__(self):
        return (
            int(hashlib.sha256(self.raw_data.encode("utf-8")).hexdigest(), 16)
            % 10 ** 10
        )

    def __eq__(self, other: TypeLDT) -> bool:
        return True if self.compare_to(other) == 100 else False

    def __ne__(self, other: TypeLDT) -> bool:
        return False if self.compare_to(other) == 100 else True

    def __repr__(self) -> str:
        if self.manufacturer or self.filename:
            return f"{[self.manufacturer]} - {self.filename}"
        elif self.path:
            return f"{self.path}"
        elif self.raw_data:
            return f"Raw data"

    def __str__(self) -> str:
        return str(self.dist)

    def compare_to(self, other: TypeLDT) -> float:
        """Performs a comparison with another LDT file. Returns a likeness percentage"""
        if self._precision != other._precision:
            raise ValueError(
                f"Cannot compare two photometries with different precisions {self._precision} != {other._precision}"
            )
        if (self.dist == other.dist).all():
            return 100
        dividend = np.abs(self.dist - other.dist)
        divisor = np.maximum(self.dist, other.dist)
        condition = divisor != 0
        return (1 - (np.divide(dividend, divisor, where=condition)).mean()) * 100

    def fit(self) -> None:
        """Performs all the necessary operations to calculate a 360° photometric data from the provided LDT file"""
        self.converted_data = self.convert(self.clean_data)
        self.extract_photometric_data(self.converted_data)
        self.calculate_photometry()
        self._original_dist = np.copy(self.dist)
        self.interpolate(
            self._c_planes_step_interpolation, self._intensities_step_interpolation
        )

    def convert(self, raw_data: List[str]) -> List[Union[int, float]]:
        """Helper method to convert numerical strings in their appropriate type"""
        return [numerical_conversion(value.strip()) for value in raw_data]

    def extract_data_from_rows(self, converted_data: List[str]) -> None:
        """Extracts data from the provided array and creates attributes according to ATTRIBUTE_LINE_MAP_LDT_FILE map.

        The keys are the attribute names and the values the row number from they must be extracted.
        """
        for attr, line in LDT.ATTRIBUTE_LINE_MAP_LDT_FILE.items():
            self.__dict__[attr] = converted_data[line]

    def extract_photometric_data(self, converted_data: List[str]) -> None:
        """Extracts all photometric data from the provided data"""
        self.extract_data_from_rows(converted_data)
        self.extract_c_planes(converted_data)
        self.extract_intensities(converted_data)
        self.extract_distribution(converted_data)

    def extract_intensities(self, converted_data: List[str]) -> None:
        self.intensities = np.array(
            converted_data[
                LDT.START_LINE_PHOTO_DATA
                + self.num_c_planes : LDT.START_LINE_PHOTO_DATA
                + self.num_c_planes
                + self.num_lum_int
            ]
        ).astype(self._precision)

    def extract_c_planes(self, converted_data: List[str]) -> None:
        self.c_planes = np.array(
            converted_data[
                LDT.START_LINE_PHOTO_DATA : LDT.START_LINE_PHOTO_DATA
                + self.num_c_planes
            ],
        ).astype(self._precision)

    def extract_distribution(self, converted_data: List[str]) -> None:
        self.dist = np.array(
            converted_data[
                LDT.START_LINE_PHOTO_DATA + self.num_c_planes + self.num_lum_int :
            ]
        ).astype(self._precision)
        # Splitting according lum intensity measures number
        self.dist = np.array_split(self.dist, len(self.dist) / self.num_lum_int)
        self.dist = np.array(self.dist).astype(self._precision).T

    def calculate_photometry(self) -> None:
        # If all c planes are included there is nothing to do!
        if self.symmetry == LDT.ALL_C_PLANES_INCLUDED:  # pragma: no cover
            pass  # pragma: no cover
        elif self.symmetry == LDT.ROTATIONAL_SYMMETRY:
            self.apply_rotational_symmetry()
        elif self.symmetry == LDT.C0_C180_SYMMETRY:
            self.apply_C0_C180_symmetry()
        elif self.symmetry == LDT.C90_C270_SYMMETRY:
            self.apply_C90_C270_symmetry()
        elif self.symmetry == LDT.C0_C180_C90_C270_SYMMETRY:
            self.apply_C0_180_C90_C270_symmetry()

    def apply_rotational_symmetry(self) -> None:
        # We tile c_planes times the same column. Some photometric files include more than one column.
        # So we tile just the first column. Otherwise the matrix could get bigger than expected.
        if self.num_c_planes == 1:
            self.num_c_planes = self._num_c_planes
            self.c_planes = np.arange(0, 361, 360 / self._num_c_planes)
        self.dist = np.tile(self.dist[:, :1], len(self.c_planes))

    def apply_C0_C180_symmetry(self) -> None:
        self.dist = np.hstack(
            [
                self.dist,
                self.dist[:, -2:0:-1],
            ]
        )

    def apply_C90_C270_symmetry(self) -> None:
        n_cols = self.dist.shape[1]
        self.dist = np.hstack(
            [
                self.dist[:, round(n_cols / 2) :],  # C0-90
                self.dist[:, -2 : round(n_cols / 2) - 1 : -1],  # C105-180
                self.dist[:, round(n_cols / 2) - 1 : 0 : -1],  # C195-255
                self.dist[:, : round(n_cols / 2)],  # C270-345
            ]
        )

    def apply_C0_180_C90_C270_symmetry(self) -> None:
        # Some files have the proper data and do not need to be reshaped.
        if not self.has_complete_photometry():
            # C0-C180 symmetry
            self.dist = np.hstack([self.dist, self.dist[:, -2::-1]])
            # C90-C270 symmetry
            self.dist = np.hstack(
                [
                    self.dist,
                    self.dist[:, -2:0:-1],
                ]
            )

    def has_complete_photometry(self) -> Tuple[int, int]:
        return self.dist.shape == (self.num_lum_int, self.num_c_planes)

    def interpolate(self, c_planes_step: float, intensities_step: float) -> None:
        interp_func = interp2d(self.c_planes, self.intensities, self.dist)
        self.c_planes = np.arange(0, 361, c_planes_step)
        self.intensities = np.arange(0, 181, intensities_step)
        self.dist = interp_func(self.c_planes, self.intensities)

    @property
    def max(self) -> float:
        return self.dist.max()

    @property
    def min(self) -> float:
        return self.dist.min()

    @property
    def has_nan(self) -> bool:
        return np.isnan(self.dist).all()

    @property
    def as_array(self) -> np.ndarray:
        return self.dist.flatten()

    @property
    def as_dataframe(self) -> "pd.DataFrame":
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("To use this feature you must install pandas")

        return pd.DataFrame(self.dist, columns=self.c_planes, index=self.intensities)

    def plot(self, angles: Optional[str] = "all") -> None:  # pragma: no cover
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("To use this feature you must install matplotlib")

        fig, ax = plt.subplots(subplot_kw=dict(polar=True))
        if angles == "all":
            ax.plot(
                [math.radians(x) for x in self.intensities],
                self.dist[:, 0],
                c="r",
                linewidth=2,
                label="C0-180",
            )
            ax.plot(
                [math.radians(-x) for x in self.intensities],
                self.dist[:, 180],
                c="r",
                linewidth=2,
            )
            ax.plot(
                [math.radians(-x) for x in self.intensities],
                self.dist[:, 90],
                "b--",
                linewidth=2,
                label="C90-270",
            )
            ax.plot(
                [math.radians(x) for x in self.intensities],
                self.dist[:, 270],
                "b--",
                linewidth=2,
            )
        elif angles == "0-180":
            ax.plot(
                [math.radians(x) for x in self.intensities],
                self.dist[:, 0],
                "r",
                linewidth=2,
                label="C0-180",
            )
            ax.plot(
                [math.radians(-x) for x in self.intensities],
                self.dist[:, 180],
                "r",
                linewidth=2,
            )
        elif angles == "90-270":
            ax.plot(
                [math.radians(-x) for x in self.intensities],
                self.dist[:, 90],
                "b--",
                linewidth=2,
                label="C90-270",
            )
            ax.plot(
                [math.radians(x) for x in self.intensities],
                self.dist[:, 270],
                "b--",
                linewidth=2,
            )
        ax.set_theta_zero_location("S")
        ax.set_theta_direction(1)
        ax.set_rmax(self.dist.max() * 1.1)
        # Setting Y scale for cdl
        rgrids = np.linspace(0, self.dist.max() * 1.1, 6)[:-1]
        ax.set_rgrids(np.round(rgrids, -2), angle=180)
        # Setting C planes tickers locations
        tlt = list(np.linspace(0, 150, 6)) + list(np.linspace(180, 30, 6))
        tlt = [int(coord) for coord in tlt]
        thetaticks = np.arange(0, 360, 30)
        ax.set_thetagrids(thetaticks, tlt)
        ax.legend(loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.2))
        fig.suptitle(
            f"{self.manufacturer}\n{self.model}\n{self.filename}", y=1.12, fontsize=13
        )

    def save_plot(self) -> None:  # pragma: no cover
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("To use this feature you must install matplotlib")

        self.plot()
        plt.savefig(self.filename)
