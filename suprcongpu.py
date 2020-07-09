#!/usr/bin/env python3

from abc import ABC, abstractmethod
import argparse
import json
import os
from pathlib import Path
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QFrame,
    QScrollArea,
    QMessageBox,
    QFileDialog,
    QHBoxLayout,
    QVBoxLayout,
    QFormLayout,
    QCheckBox,
    QComboBox,
    QSpinBox,
    QWidget,
    QPushButton,
    QLineEdit,
    QLabel,
)
from PyQt5.QtCore import QObject, Qt, QRegExp, pyqtSlot
from PyQt5.QtGui import QIntValidator, QDoubleValidator, QRegExpValidator, QFont
import subprocess
import sys
from typing import Optional, Union


# Global settings.
CONFIG_FILE_NAME = ".tmp_config.json"
VALID_SYMMETRIES = ["s", "dx2-y2", "dxy", "g"]

# Color definitions
colorOk = "#ffffff"
colorErr = "#f6989d"
colorErrMsg = "#ff4500"


# -----------------------------------------------------------------------------
# COMMAND-LINE ARGUMENTS
# -----------------------------------------------------------------------------
def make_default_parser():
    parser = argparse.ArgumentParser(
        prog="SuprConGPU",
        description="SuprConGPU Main file for setting up self-consistent simulations in a confined geometry (a mesoscopic grain). Different superconducting order parameters and geometries can be specified, as well as physical parameters such as temperature and external fields. To set these, create a .json parameter-file as in examples and give the path to this file with the -f option. All parameters can be overridden using the command-line options specified below.",
    )

    # Display GUI.
    parser.add_argument("--gui", action="store_true", help="Use GUI to run simulation.")

    # Config file.
    parser.add_argument(
        "-f",
        "--file",
        type=str,
        default=None,
        help="Path to JSON file with parameters specifying the simulation settings.",
    )

    # C++ binary.
    parser.add_argument(
        "-b",
        "--cpp-binary",
        type=str,
        default="build/release/bin/main",
        help="Path to the C++ binary.",
    )

    # =============== PHYSICS =============== #

    # Symmetry.
    parser.add_argument(
        "-H",
        "--op-sym",
        type=str,
        default=None,
        choices=VALID_SYMMETRIES,
        nargs="+",
        help="Order parameter pairing symmetry. Multiple values can be passed, one for each order parameter component.",
    )

    # Relative Tc.
    parser.add_argument(
        "-C",
        "--op-tc",
        type=float,
        default=None,
        nargs="+",
        help="Order parameter critical temperature. Multiple values can be passed, one for each order parameter component. Note that all values will be scaled so that the maximum is 1.",
    )

    # Phase shift.
    parser.add_argument(
        "-J",
        "--op-init-phase-shift",
        type=float,
        default=None,
        nargs="+",
        help="The phase shift of the initial order parameter guess, in units of 2pi. Multiple values can be passed, one for each order parameter component.",
    )

    # Num phase windings.
    parser.add_argument(
        "-W",
        "--op-init-phase-winding",
        type=float,
        default=None,
        nargs="+",
        help="The phase winding of the initial order parameter guess, in units of 2pi. Multiple values can be passed, one for each order parameter component.",
    )

    # Random phase weight.
    parser.add_argument(
        "-R",
        "--op-init-rand-phase-weight",
        type=float,
        default=None,
        nargs="+",
        help="The weight of the random phase of the initial order parameter guess. Multiple values can be passed, one for each order parameter component.",
    )

    # Temperature.
    parser.add_argument(
        "-T",
        "--temperature",
        type=float,
        default=None,
        help="Temperature in units of T_c.",
    )

    # External flux quanta.
    parser.add_argument(
        "-B",
        "--external-flux-quanta",
        type=float,
        default=None,
        help="External magnetic field, perpendicular to sample, in units of flux quanta. For non-solenoid field, it is distributed uniformly over the sample.",
    )

    # Grain side length.
    parser.add_argument(
        "-s",
        "--grain-side-length",
        type=float,
        default=None,
        help="Side length of grain in units of coherence lengths (hbar*v_F)/(2*pi*k_B*T_c).",
    )

    # Crystal axes rotation.
    parser.add_argument(
        "-r",
        "--crystal-axes-rotation",
        type=float,
        default=None,
        help="Rotation between grain and crystal ab-axes, in units of 2pi.",
    )

    # Geometry.
    parser.add_argument(
        "-z",
        "--num-polygon-edges",
        type=int,
        default=None,
        help="Number edges of the grain polygon. Negative values yield a circle.",
    )

    # Solenoid.
    group_solenoid = parser.add_mutually_exclusive_group()
    group_solenoid.add_argument(
        "--solenoid",
        action="store_true",
        help="Set up a solenoid, and use solenoid gauge for the external field.",
    )
    group_solenoid.add_argument(
        "--no-solenoid", action="store_true", help="Do not set up a solenoid."
    )

    # =============== ACCURACY =============== #

    # Convergence criterion.
    parser.add_argument(
        "-c",
        "--convergence-criterion",
        type=float,
        default=None,
        help="Convergence criteron for self-consistency.",
    )

    # Lattice side length.
    parser.add_argument(
        "-N",
        "--lattice-side-length",
        type=int,
        default=None,
        help="Number of discrete lattice points in the grain. Set to odd value for best convergence. Recommended to be at least 10 times the grain size.",
    )

    # Num Fermi momenta.
    parser.add_argument(
        "-p",
        "--num-fermi-momenta",
        type=int,
        default=None,
        help="Number of discrete angles in Fermi-surface average. Set to "
        "odd value for best convergence. Recommended minumum: 101.",
    )

    # Num iterations max.
    parser.add_argument(
        "-M",
        "--num-iterations-max",
        type=int,
        default=None,
        help="Maximum number of self-consistent iterations to run.",
    )

    # Num iterations min.
    parser.add_argument(
        "-m",
        "--num-iterations-min",
        type=int,
        default=None,
        help="Minimum number of self-consistent iterations to run.",
    )

    # Num Ozaki energies.
    parser.add_argument(
        "-e",
        "--num-ozaki-energies",
        type=int,
        default=None,
        help="Number of Ozaki energies (similar to Matsubara energies, see the documentation).",
    )

    # =============== MISC =============== #

    # Data path.
    parser.add_argument(
        "-o",
        "--data-path",
        type=str,
        default=None,
        help="Base folder for output (data and parameter files).",
    )

    # Verbose.
    group_verbose = parser.add_mutually_exclusive_group()
    group_verbose.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print iteration status and other output to terminal.",
    )
    group_verbose.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Do not print iteration status and other output to terminal.",
    )

    # Save.
    group_save = parser.add_mutually_exclusive_group()
    group_save.add_argument(
        "--save-data", action="store_true", help="Save data files to disk."
    )
    group_save.add_argument(
        "--no-save-data", action="store_true", help="Do not save data files to disk."
    )

    # Save data interval.
    parser.add_argument(
        "--save-data-interval",
        type=int,
        default=None,
        help="How frequently to save data.",
    )

    # Visualize.
    group_visualize = parser.add_mutually_exclusive_group()
    group_visualize.add_argument(
        "--visualize", action="store_true", help="Run with real-time plotting."
    )
    group_visualize.add_argument(
        "--no-visualize", action="store_true", help="Run without real-time plotting."
    )

    return parser


def set_value(
    simparams: dict,
    key: str,
    value: Union[int, float, list],
    group: Optional[str] = None,
) -> None:
    # If the value is None then it is not set in the command line arguments, so it should not be used.
    if value is not None:
        if group is None:
            simparams[key] = value
        else:
            # This should always be true as we get a list from the parser.
            assert isinstance(value, list)

            if group in simparams:
                # The group, e.g. 'order_parameter' already exists.
                # Check that the lengths of the lists agree.
                assert isinstance(simparams[group], list)
                assert len(value) == len(simparams[group])
                for i, elem in enumerate(value):
                    # TODO(niclas): Add dummy value so one can only specify e.g. the second critical temperature.
                    simparams[group][i][key] = elem
            else:
                # The group, e.g. 'order_parameter' does not exist, so lets create a dict for every element of 'value'.
                tmp = []
                for elem in value:
                    tmp.append({key: elem})
                simparams[group] = tmp


def set_flag(simparams: dict, key: str, on: bool, off: bool) -> None:
    # The flags are mutually exclusive, so both cannot be true. If both are false then they have not been set in the command line arguments, and should therefore not be used.
    if on or off:
        # Off = True implies On = False and vice versa. So only the on value needs to be considered.
        simparams[key] = on


def check_value(key: str, value: Union[bool, float, int, str], checks: dict) -> bool:
    # Check that the value is of the expected type.
    expected_type = checks["type"]
    if not isinstance(value, expected_type):
        print(f"--The field '{key}' must be of type {expected_type}.")
        return False

    # Check that the value is larger than 'low'.
    if "low" in checks:
        low = checks["low"]
        if not value > low:
            print(f"--The field '{key}' must be larger than {low}.")
            return False

    # Check that the value is not in the excluded list.
    if "excluded" in checks:
        for exc in checks["excluded"]:
            if value == exc:
                print(f"--The field '{key}' cannot have value {exc}.")
                return False

    # Check that the value is in some pre-defined set.
    if "one_of" in checks:
        valid_values = checks["one_of"]
        if not value in valid_values:
            print(f"--The field '{key}' must be one of {valid_values}.")
            return False

    # No errors.
    return True


def check_dict(simparams: dict, checks: dict) -> bool:
    success = True

    # Loop over the keys in the checks so we can spot if something is missing.
    for key, value_checks in checks.items():
        if not key in simparams:
            print(f"--The field '{key}' must exist.")
            success = False
            continue

        value = simparams[key]

        # Check that the value is of the expected type.
        expected_type = value_checks["type"]
        if not isinstance(value, expected_type):
            print(f"--The field '{key}' must be of type {expected_type}.")
            success = False
            continue

        if isinstance(value, list):
            for elem in value:
                if isinstance(elem, dict):
                    success &= check_dict(simparams=elem, checks=value_checks["sub"])
                else:
                    print(f"--The field '{key}' must be a list of dicts.")
                    success = False
                    continue
        else:
            # If not a dict then check the values.
            success &= check_value(key=key, value=value, checks=value_checks)

    return success


def validate(simparams: dict) -> bool:
    # A dict with the checks needed for all parameters.
    checks = {
        "convergence_criterion": {"type": float, "low": 0.0},
        "crystal_axes_rotation": {"type": float},
        "data_path": {"type": str},
        "external_flux_quanta": {"type": float},
        "grain_side_length": {"type": float, "low": 0.0},
        "lattice_side_length": {"type": int, "low": 1},
        "num_fermi_momenta": {"type": int, "low": 1},
        "num_iterations_max": {"type": int, "low": 1},
        "num_iterations_min": {"type": int, "low": 1},
        "num_ozaki_energies": {"type": int, "low": 1},
        "num_polygon_edges": {"type": int, "excluded": [0, 1, 2]},
        "order_parameter": {
            "type": list,
            "sub": {
                "critical_temperature": {"type": float, "low": 0.0},
                "initial_phase_shift": {"type": float},
                "initial_phase_winding": {"type": float},
                "initial_random_phase_weight": {"type": float},
                "symmetry": {"type": str, "one_of": VALID_SYMMETRIES},
            },
        },
        "save_data": {"type": bool},
        "save_data_interval": {"type": int, "low": 0},
        "solenoid": {"type": bool},
        "temperature": {"type": float, "low": 0.0},
        "verbose": {"type": bool},
        "visualize": {"type": bool},
    }

    # General checks.
    success = check_dict(simparams, checks)

    # Special checks.
    if success:
        # Normalize critical temperature so that the maximum is one.
        Tc_max = 0
        for comp in simparams["order_parameter"]:
            Tc_max = max(Tc_max, comp["critical_temperature"])
        for i in range(len(simparams["order_parameter"])):
            simparams["order_parameter"][i]["critical_temperature"] /= Tc_max

        # Check that the minimum number of iterations is smaller than the maximum.
        if not (simparams["num_iterations_max"] > simparams["num_iterations_min"]):
            print("--'num_iterations_max' must be larger than 'num_iterations_min'.")
            success = False

    return success


# -----------------------------------------------------------------------------
# GUI
# -----------------------------------------------------------------------------
class QMeta(type(QObject), type(ABC)):
    pass


class Serializable(ABC):
    @property
    @abstractmethod
    def jsonKey(self):
        pass

    @abstractmethod
    def json(self):
        pass

    @abstractmethod
    def setValue(self, value):
        pass

    @abstractmethod
    def validate(self):
        pass


class Input(Serializable, QWidget, metaclass=QMeta):
    def __init__(self, validator, jsonKey, errMsg, parent):
        super().__init__(parent)

        self._jsonKey = jsonKey
        self._input = QLineEdit(parent)
        if validator is not None:
            self._input.setValidator(validator)
        self._input.textChanged.connect(self.handle_text_change)

        self._errInfo = QLabel(parent)
        self._errInfo.setText(errMsg)
        self._errInfo.setStyleSheet("QLabel {color: %s;}" % colorErrMsg)
        self._errInfo.setVisible(False)

        layout = QVBoxLayout(self)
        layout.addWidget(self._input)
        layout.addWidget(self._errInfo)
        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)

    @pyqtSlot(str)
    def handle_text_change(self, text):
        color = colorOk if self.valid() or len(text) == 0 else colorErr
        self._input.setStyleSheet("QLineEdit {background-color: %s;}" % color)
        self.validate()

    def valid(self):
        return self._input.hasAcceptableInput()

    def validate(self):
        if not self.valid():
            self._errInfo.setVisible(True)
        else:
            self._errInfo.setVisible(False)

    @property
    def jsonKey(self):
        return self._jsonKey

    def setValue(self, value):
        self._input.setText(str(value))


class InputFloat(Input):
    def __init__(
        self,
        *,
        jsonKey,
        onlyPositive=False,
        minValue=None,
        errMsg="Invalid input!",
        parent=None,
    ):
        if onlyPositive:
            regExFloat = "^\+?(\d+([.]\d*)?([eE][+-]?\d+)?|[.]\d+([eE][+-]?\d+)?)$"
        else:
            regExFloat = "^[+-]?(\d+([.]\d*)?([eE][+-]?\d+)?|[.]\d+([eE][+-]?\d+)?)$"
        validator = QRegExpValidator(QRegExp(regExFloat))
        super().__init__(validator, jsonKey, errMsg, parent)
        self.minValue = minValue

    def json(self):
        if self.valid():
            return {self.jsonKey: float(self._input.text())}
        else:
            return None

    def valid(self):
        return self._input.hasAcceptableInput() and self.validRange()

    def validRange(self):
        if self.minValue is not None:
            return float(self._input.text()) > self.minValue
        return True


class InputInt(Input):
    def __init__(self, *, jsonKey, minValue=0, errMsg="Invalid input!", parent=None):
        validator = QIntValidator()
        validator.setBottom(minValue)
        super().__init__(validator, jsonKey, errMsg, parent)

    def json(self):
        if self.valid():
            return {self.jsonKey: int(self._input.text())}
        else:
            return None


class InputStr(Input):
    def __init__(self, *, jsonKey, errMsg="Invalid input!", parent=None):
        rx = QRegExp("^(?!\s*$).+")
        validator = QRegExpValidator(rx)
        super().__init__(validator, jsonKey, errMsg, parent)

    def json(self):
        if self.valid():
            return {self.jsonKey: self._input.text()}
        else:
            return None


class CheckBox(Serializable, QCheckBox, metaclass=QMeta):
    def __init__(self, *, jsonKey, parent=None):
        super().__init__(parent)

        self._jsonKey = jsonKey
        self.setFocusPolicy(Qt.TabFocus)

    @property
    def jsonKey(self):
        return self._jsonKey

    def json(self):
        return {self.jsonKey: self.isChecked()}

    def setValue(self, value):
        if value:
            self.setChecked(True)
        else:
            self.setChecked(False)

    def validate(self):
        pass


class ChooseFile(QPushButton):
    def __init__(self, *, defaultPath, parent=None):
        super().__init__(parent)

        self._path = Path(defaultPath)
        self.setText(str(self._path.resolve()))
        self.clicked.connect(self.handle_click)

    @pyqtSlot()
    def handle_click(self):
        if self._path.exists():
            openDir = str(self._path.parent)
        else:
            openDir = ""
        filePath = QFileDialog.getOpenFileName(
            self, caption="SuprConGPU binary path", directory=openDir
        )[0]
        if filePath:
            self.setText(filePath)
            self._path = Path(filePath)

    def valid(self):
        return self._path.is_file() and os.access(self._path, os.X_OK)

    def getPath(self):
        if self.valid():
            return str(self._path)
        else:
            return None


class Geometry(Serializable, QWidget, metaclass=QMeta):
    def __init__(self, *, jsonKey, minValue=3, errMsg="Invalid input!", parent=None):
        super().__init__(parent)

        self._jsonKey = jsonKey
        self._minValue = minValue

        # Dropdown to choose shape
        self._shapeDropdown = QComboBox(parent)
        self._shapeDropdown.setPlaceholderText("Choose grain shape")
        self._shapeDropdown.addItem("Disc")
        self._shapeDropdown.addItem("Polygon")
        self._shapeDropdown.setFocusPolicy(Qt.TabFocus)
        self._shapeDropdown.currentIndexChanged.connect(self.handle_shape_choice)

        # Spin box to choose number of polygon edges
        # Note: when "Disc" is chosen, number of edges is assumed to be -1
        self._edgesSpinBox = QSpinBox(parent)
        self._edgesSpinBox.setFocusPolicy(Qt.NoFocus)

        if minValue is not None:
            self._edgesSpinBox.setMinimum(minValue)

        self._numEdgesWidget = QWidget(parent)
        numEdgesLayout = QHBoxLayout(self._numEdgesWidget)
        numEdgesLayout.addWidget(QLabel("Number of edges:"))
        numEdgesLayout.addWidget(self._edgesSpinBox)

        numEdgesLayout.setContentsMargins(5, 0, 0, 0)
        self._numEdgesWidget.setVisible(False)

        # Error message
        self._errInfo = QLabel(parent)
        self._errInfo.setText(errMsg)
        self._errInfo.setStyleSheet("QLabel {color: %s;}" % colorErrMsg)
        self._errInfo.setVisible(False)

        # Layout
        inputLayout = QHBoxLayout()
        inputLayout.addWidget(self._shapeDropdown)
        inputLayout.addWidget(self._numEdgesWidget)

        layout = QVBoxLayout(self)
        layout.addLayout(inputLayout)
        layout.addWidget(self._errInfo)
        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)

    @pyqtSlot(int)
    def handle_shape_choice(self, index):
        if index < 1:
            self._numEdgesWidget.setVisible(False)
            self._edgesSpinBox.setFocusPolicy(Qt.NoFocus)
        else:
            self._numEdgesWidget.setVisible(True)
            self._edgesSpinBox.setFocusPolicy(Qt.StrongFocus)
        self.validate_shape_choice(index)

    def validate_shape_choice(self, index):
        if index < 0:
            self._errInfo.setVisible(True)
        else:
            self._errInfo.setVisible(False)

    def validate(self):
        self.validate_shape_choice(self._shapeDropdown.currentIndex())

    @property
    def jsonKey(self):
        return self._jsonKey

    def json(self):
        if self._shapeDropdown.currentIndex() == 0:
            value = -1
        elif self._shapeDropdown.currentIndex() == 1:
            value = self._edgesSpinBox.value()
        else:
            value = None

        if value is not None:
            return {self.jsonKey: value}
        return None

    def setValue(self, value):
        valueInt = int(value)
        if valueInt < 0:
            self._shapeDropdown.setCurrentIndex(0)
        elif valueInt >= self._minValue:
            self._shapeDropdown.setCurrentIndex(1)
            self._edgesSpinBox.setValue(valueInt)
        else:
            self._shapeDropdown.setCurrentIndex(-1)


class InputDropdown(Serializable, QWidget, metaclass=QMeta):
    def __init__(
        self, *, jsonKey, choices, placeholderText, errMsg="Invalid input!", parent=None
    ):
        super().__init__(parent)
        self._jsonKey = jsonKey
        self._choices = choices

        # Dropdown
        self._dropdown = QComboBox(parent)
        self._dropdown.setPlaceholderText(placeholderText)
        for choice in choices:
            self._dropdown.addItem(choice)
        self._dropdown.currentIndexChanged.connect(self.validate_dropdown_choice)
        self._dropdown.setFocusPolicy(Qt.TabFocus)

        # Error message
        self._errInfo = QLabel(parent)
        self._errInfo.setText(errMsg)
        self._errInfo.setStyleSheet("QLabel {color: %s;}" % colorErrMsg)
        self._errInfo.setVisible(False)

        # Layout
        layout = QVBoxLayout(self)
        layout.addWidget(self._dropdown)
        layout.addWidget(self._errInfo)
        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)

    @pyqtSlot(int)
    def validate_dropdown_choice(self, index):
        if index < 0:
            self._errInfo.setVisible(True)
        else:
            self._errInfo.setVisible(False)

    def validate(self):
        self.validate_dropdown_choice(self._dropdown.currentIndex())

    @property
    def jsonKey(self):
        return self._jsonKey

    def json(self):
        value = None
        currentIndex = self._dropdown.currentIndex()
        if currentIndex >= 0:
            value = self._choices[currentIndex]

        if value is not None:
            return {self._jsonKey: value}
        return None

    def setValue(self, value):
        try:
            index = self._choices.index(value)
        except ValueError:
            index = -1
        self._dropdown.setCurrentIndex(index)


class OrderParameterComponent(Serializable, QWidget, metaclass=QMeta):
    def __init__(
        self, *, parent=None,
    ):
        super().__init__(parent)

        self._inputs = {}

        # Order parameter symmetry dropdown
        self._symmetryDropdown = InputDropdown(
            jsonKey="symmetry",
            choices=VALID_SYMMETRIES,
            placeholderText="Choose symmetry",
        )
        self._inputs[self._symmetryDropdown.jsonKey] = self._symmetryDropdown

        # Critical temperature
        self._tc = InputFloat(
            jsonKey="critical_temperature",
            onlyPositive=True,
            minValue=0.0,
            parent=parent,
        )
        self._inputs[self._tc.jsonKey] = self._tc

        # Initial phase shift
        self._init_phase_shift = InputFloat(
            jsonKey="initial_phase_shift", parent=parent
        )
        self._init_phase_shift.setValue(0.0)
        self._inputs[self._init_phase_shift.jsonKey] = self._init_phase_shift

        # Initial phase winding
        self._init_phase_winding = InputFloat(
            jsonKey="initial_phase_winding", parent=parent
        )
        self._init_phase_winding.setValue(0.0)
        self._inputs[self._init_phase_winding.jsonKey] = self._init_phase_winding

        # Initial random phase weight
        self._init_rand_phase_weight = InputFloat(
            jsonKey="initial_random_phase_weight", parent=parent
        )
        self._init_rand_phase_weight.setValue(0.0)
        self._inputs[
            self._init_rand_phase_weight.jsonKey
        ] = self._init_rand_phase_weight

        # Layout
        layout = QFormLayout(self)
        layout.setLabelAlignment(Qt.AlignLeft)
        layout.addRow(QLabel("Order parameter symmetry"), self._symmetryDropdown)
        layout.addRow(QLabel("Critical temperature"), self._tc)
        layout.addRow(QLabel("Initial phase shift"), self._init_phase_shift)
        layout.addRow(QLabel("Initial phase winding"), self._init_phase_winding)
        layout.addRow(
            QLabel("Initial random phase weight"), self._init_rand_phase_weight
        )
        layout.setContentsMargins(0, 1, 0, 0)

    def validate(self):
        for widget in self._inputs.values():
            widget.validate()

    @property
    def jsonKey(self):
        return None

    def json(self):
        result = {}
        for key, widget in self._inputs.items():
            widgetJson = widget.json()
            if widgetJson is None:
                return None
            else:
                result.update(widgetJson)
        return result

    def setValue(self, valueDict):
        if valueDict is not None:
            try:
                assert isinstance(valueDict, dict)
            except AssertionError:
                print(
                    "Input Error: Each order parameter component should be a dictionary of key-value pairs!"
                )
            for key, value in valueDict.items():
                widget = self._inputs.get(key)
                if widget is not None:
                    widget.setValue(value)


class OrderParameter(Serializable, QScrollArea, metaclass=QMeta):
    class HorizontalSpacer(QFrame):
        def __init__(self, parent=None):
            super().__init__(parent)
            self.setFrameStyle(QFrame.HLine | QFrame.Plain)
            self.setMinimumSize(1, 20)

    class ComponentLabel(QLabel):
        def __init__(self, labelText, parent=None):
            super().__init__(labelText, parent)
            font = QFont()
            font.setBold(True)
            self.setFont(font)

    def __init__(self, *, jsonKey, parent=None):
        super().__init__(parent)

        self._jsonKey = jsonKey
        self._opComponents = []
        self._separators = []

        # Controls
        self._addButton = QPushButton("Add component", parent)
        self._addButton.clicked.connect(self.handle_add)
        self._removeButton = QPushButton("Remove component", parent)
        self._removeButton.clicked.connect(self.handle_remove)
        self._removeButton.setEnabled(False)

        # Layout
        buttonLayout = QHBoxLayout()
        buttonLayout.addWidget(self._addButton)
        buttonLayout.addWidget(self._removeButton)

        self._opLayout = QFormLayout()

        masterWidget = QWidget(self)
        layout = QVBoxLayout(masterWidget)
        layout.addLayout(buttonLayout)
        layout.addLayout(self._opLayout)
        self.setWidget(masterWidget)
        self.setWidgetResizable(True)
        self.setMinimumHeight(255)

        # Add one empty order parameter component
        self.add_component()

    def add_component(self):
        opComponent = OrderParameterComponent()
        label = self.ComponentLabel("Component {}:".format(len(self._opComponents) + 1))
        self._opComponents.append(opComponent)
        self._opLayout.addRow(label, opComponent)
        return opComponent

    @pyqtSlot(result=OrderParameterComponent)
    def handle_add(self):
        separator = self.HorizontalSpacer()
        self._separators.append(separator)
        self._opLayout.addRow(separator)
        opComponent = self.add_component()
        self._removeButton.setEnabled(True)
        return opComponent

    def remove_last_component(self):
        if len(self._opComponents) > 0:
            opComponent = self._opComponents.pop()
            self._opLayout.removeRow(opComponent)

    def remove_last_separator(self):
        if len(self._separators) > 0:
            separator = self._separators.pop()
            self._opLayout.removeRow(separator)

    @pyqtSlot()
    def handle_remove(self):
        if len(self._opComponents) > 1:
            self.remove_last_component()
            self.remove_last_separator()

        if len(self._opComponents) < 2:
            self._removeButton.setEnabled(False)

    def clear(self):
        while len(self._opComponents) > 0:
            self.remove_last_component()
            self.remove_last_separator()
        self._removeButton.setEnabled(False)

    def validate(self):
        for opComponent in self._opComponents:
            opComponent.validate()

    @property
    def jsonKey(self):
        return self._jsonKey

    def json(self):
        result = []
        for opComponent in self._opComponents:
            opComponentJson = opComponent.json()
            if opComponentJson is None:
                return None
            else:
                result.append(opComponentJson)
        return {self._jsonKey: result}

    def setValue(self, opComponentsList):
        try:
            assert isinstance(opComponentsList, list)
        except AssertionError:
            print(
                "Input Error: Order parameter should be specified as a list of component dictionaries"
            )
        self.clear()
        for opComponentDict in opComponentsList:
            if len(self._opComponents) == 0:
                opComponent = self.add_component()
            else:
                opComponent = self.handle_add()
            opComponent.setValue(opComponentDict)


class MainWindow(QMainWindow):
    def __init__(self, *, cppBinaryPath):
        super(MainWindow, self).__init__()
        self.setWindowTitle("SuprConGPU Runner")
        font = QFont()
        font.setPixelSize(14)
        QApplication.setFont(font)

        # Widgets registry
        self._widgetRegistry = {}

        # Create main widget holding all the controls and inputs
        mainWidget = QFrame()

        # Create inputs form layout
        formLayout = QFormLayout()
        formLayout.setLabelAlignment(Qt.AlignLeft)

        # ------ #
        # Inputs #
        # ------ #

        # Convergence criterion
        self._convergence_criterion = InputFloat(
            jsonKey="convergence_criterion", minValue=0.0
        )
        self.register(self._convergence_criterion)
        formLayout.addRow(QLabel("Convergence criterion"), self._convergence_criterion)

        # Axes rotation
        self._crystal_axes_rotation = InputFloat(jsonKey="crystal_axes_rotation")
        self.register(self._crystal_axes_rotation)
        formLayout.addRow(
            QLabel("Crystal axes rotation"), self._crystal_axes_rotation,
        )

        # Set data path
        self._data_path = InputStr(jsonKey="data_path")
        self.register(self._data_path)
        formLayout.addRow(QLabel("Data path"), self._data_path)

        # External flux quanta
        self._external_flux_quanta = InputFloat(jsonKey="external_flux_quanta")
        self.register(self._external_flux_quanta)
        formLayout.addRow(QLabel("External flux quanta"), self._external_flux_quanta)

        # Grain side length
        self._grain_side_length = InputFloat(
            jsonKey="grain_side_length", onlyPositive=True
        )
        self.register(self._grain_side_length)
        formLayout.addRow(QLabel("Grain side length"), self._grain_side_length)

        # Lattice side length
        self._lattice_side_length = InputInt(jsonKey="lattice_side_length", minValue=1)
        self.register(self._lattice_side_length)
        formLayout.addRow(QLabel("Lattice side length"), self._lattice_side_length)

        # Number of Fermi momenta
        self._num_fermi_momenta = InputInt(jsonKey="num_fermi_momenta", minValue=1)
        self.register(self._num_fermi_momenta)
        formLayout.addRow(QLabel("Number of Fermi momenta"), self._num_fermi_momenta)

        # Max number of iterations
        self._num_iterations_max = InputInt(jsonKey="num_iterations_max", minValue=1)
        self.register(self._num_iterations_max)
        formLayout.addRow(
            QLabel("Max number of self-consistent iterations"), self._num_iterations_max
        )

        # Min number of iterations
        self._num_iterations_min = InputInt(jsonKey="num_iterations_min", minValue=1)
        self.register(self._num_iterations_min)
        formLayout.addRow(
            QLabel("Min number of self-consistent iterations"), self._num_iterations_min
        )

        # Number of Ozaki energies
        self._num_ozaki_energies = InputInt(jsonKey="num_ozaki_energies", minValue=1)
        self.register(self._num_ozaki_energies)
        formLayout.addRow(QLabel("Number of Ozaki energies"), self._num_ozaki_energies)

        # Number of polygon edges
        self._num_polygon_edges = Geometry(jsonKey="num_polygon_edges")
        self.register(self._num_polygon_edges)
        formLayout.addRow(QLabel("System geometry"), self._num_polygon_edges)

        # Order parameter
        self._order_parameter = OrderParameter(jsonKey="order_parameter")
        self.register(self._order_parameter)
        formLayout.addRow(QLabel("Order parameter"), self._order_parameter)

        # Save data
        self._save_data = CheckBox(jsonKey="save_data")
        self.register(self._save_data)
        formLayout.addRow(QLabel("Save data"), self._save_data)

        # Save data interval
        self._save_data_interval = InputInt(jsonKey="save_data_interval")
        self.register(self._save_data_interval)
        formLayout.addRow(
            QLabel("Interval to save data with"), self._save_data_interval
        )

        # Solenoid
        self._solenoid = CheckBox(jsonKey="solenoid")
        self.register(self._solenoid)
        formLayout.addRow(QLabel("Solenoid"), self._solenoid)

        # Temperature
        self._temperature = InputFloat(jsonKey="temperature", onlyPositive=True)
        self.register(self._temperature)
        formLayout.addRow(QLabel("Temperature"), self._temperature)

        # Verbose calculation
        self._verbose = CheckBox(jsonKey="verbose")
        self.register(self._verbose)
        formLayout.addRow(QLabel("Verbose calculation"), self._verbose)

        # Visualize
        self._visualize = CheckBox(jsonKey="visualize")
        self.register(self._visualize)
        formLayout.addRow(QLabel("Visualize"), self._visualize)

        # Compiled binary path
        self._binary_path = ChooseFile(defaultPath=cppBinaryPath)
        formLayout.addRow(QLabel("SuprConGPU binary path"), self._binary_path)

        # -------- #
        # Controls #
        # -------- #

        # Buttons
        self._load_config = QPushButton("Load config", mainWidget)
        self._load_config.setFocusPolicy(Qt.TabFocus)
        self._load_config.clicked.connect(self.handle_load_file)

        self._save_config = QPushButton("Save config", mainWidget)
        self._save_config.setFocusPolicy(Qt.TabFocus)
        self._save_config.clicked.connect(self.handle_save_file)

        self._run_simulation = QPushButton("Run simulation", mainWidget)
        self._run_simulation.setFocusPolicy(Qt.TabFocus)
        self._run_simulation.clicked.connect(self.handle_run_simulation)

        buttonsLayout = QHBoxLayout()
        buttonsLayout.addWidget(self._load_config)
        buttonsLayout.addWidget(self._save_config)
        buttonsLayout.addWidget(self._run_simulation)

        # ------------------ #
        # Application layout #
        # ------------------ #

        # Main widget layout
        layout = QVBoxLayout(mainWidget)
        layout.addLayout(formLayout)
        layout.addLayout(buttonsLayout)

        # Set the created form as the central part of the application window
        scrollArea = QScrollArea(self)
        scrollArea.setWidgetResizable(True)
        scrollArea.setWidget(mainWidget)
        self.setCentralWidget(scrollArea)
        self.setFocus()

        # Set startup size
        self.resize(820, 980)

        # Set application styles
        self.setAppStyle()

    def setAppStyle(self):
        self.setStyleSheet(
            "QMainWindow > QWidget { background-color: rgb(255, 255, 255);}"
        )

    def register(self, widget: Serializable):
        self._widgetRegistry[widget.jsonKey] = widget

    def serialize(self):
        config = {}
        success = True
        for inputWidget in self._widgetRegistry.values():
            inputWidget.validate()
            widgetJson = inputWidget.json()
            if widgetJson is not None:
                config.update(widgetJson)
            else:
                success = False
                config.update({inputWidget.jsonKey: None})
        return (config, success)

    def deserialize(self, config):
        for key, value in config.items():
            inputWidget = self._widgetRegistry.get(key)
            if inputWidget is not None:
                inputWidget.setValue(value)
                inputWidget.validate()

    def make_invalid_input_popup(self):
        msgBox = QMessageBox()
        msgBox.setIcon(QMessageBox.Warning)
        msgBox.setText(
            "Some of the inputs are invalid or empty.\nDo you want to proceed?"
        )
        msgBox.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        return msgBox

    @pyqtSlot()
    def handle_save_file(self):
        saveFileName = QFileDialog.getSaveFileName(
            self, caption="Save config file", filter="Config file (*.json)",
        )[0]

        if saveFileName:
            if not saveFileName.endswith(".json"):
                saveFileName += ".json"

            (config, shouldProceed) = self.serialize()
            if not shouldProceed:
                msgBox = self.make_invalid_input_popup()
                popupResponse = msgBox.exec_()
                if popupResponse == QMessageBox.Yes:
                    shouldProceed = True
            if shouldProceed:
                with open(saveFileName, "w") as file:
                    json.dump(config, file, indent="\t", sort_keys=True)
                    file.write("\n")

    @pyqtSlot()
    def handle_load_file(self):
        loadFileName = QFileDialog.getOpenFileName(
            self, caption="Open config file", filter="Config file (*.json)"
        )[0]

        if loadFileName:
            with open(loadFileName, "r") as file:
                config = json.load(file)
                self.deserialize(config)

    @pyqtSlot()
    def handle_run_simulation(self):
        binaryPath = self._binary_path.getPath()
        if binaryPath is not None:
            (config, shouldProceed) = self.serialize()
            if not shouldProceed:
                msgBox = self.make_invalid_input_popup()
                popupResponse = msgBox.exec_()
                if popupResponse == QMessageBox.Yes:
                    shouldProceed = True
            if shouldProceed:
                # Normalize critical temperatures to values <= 1.0
                tc_max = 0
                order_parameter = config["order_parameter"]
                for component in order_parameter:
                    tc_max = max(tc_max, component["critical_temperature"])
                for component in order_parameter:
                    component["critical_temperature"] /= tc_max

                with open(CONFIG_FILE_NAME, "w") as file:
                    json.dump(config, file, indent="\t", ensure_ascii=False)

                subprocess.call([str(binaryPath), CONFIG_FILE_NAME])
        else:
            msgBox = QMessageBox()
            msgBox.setIcon(QMessageBox.Critical)
            msgBox.setText(
                "Binary path is invalid!\n"
                "Please, make sure it points to the existing SuprConGPU compiled binary."
            )
            msgBox.exec_()


def showGui(simparams: dict, cpp_binary: str) -> int:
    # Every Qt application should contain exactly one instance of QApplication.
    # This handles interaction with the OS and screen, and runs the main event loop.
    app = QApplication(sys.argv)

    # Main part of the application
    mainWindow = MainWindow(cppBinaryPath=cpp_binary)

    if bool(simparams):
        mainWindow.deserialize(simparams)
    mainWindow.show()

    # Center on the screen.
    mainWindow.move(
        QApplication.desktop().screen().rect().center()
        - mainWindow.frameGeometry().center()
    )

    # Start the main event loop until finished.
    # The final exit code is passed over to python environment.
    return app.exec_()


# -----------------------------------------------------------------------------
# MAIN FUNCTION
# -----------------------------------------------------------------------------
def main() -> None:
    # Create the argument parser.
    parser = make_default_parser()

    # Parse all arguments.
    args = parser.parse_args()

    # Display help and exit if called without arguments.
    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit(1)

    # Simulation parameters dictionary.
    simparams = {}

    # If config file is set, load it first.
    if args.file is not None:
        # Read the JSON file.
        with open(args.file, "r") as f:
            data = f.read()

        # Parse the JSON file.
        simparams = json.loads(data)

    # Override values from the command line.
    set_value(simparams, "convergence_criterion", args.convergence_criterion)
    set_value(simparams, "crystal_axes_rotation", args.crystal_axes_rotation)
    set_value(simparams, "data_path", args.data_path)
    set_value(simparams, "external_flux_quanta", args.external_flux_quanta)
    set_value(simparams, "grain_side_length", args.grain_side_length)
    set_value(simparams, "lattice_side_length", args.lattice_side_length)
    set_value(simparams, "num_fermi_momenta", args.num_fermi_momenta)
    set_value(simparams, "num_iterations_max", args.num_iterations_max)
    set_value(simparams, "num_iterations_min", args.num_iterations_min)
    set_value(simparams, "num_ozaki_energies", args.num_ozaki_energies)
    set_value(simparams, "num_polygon_edges", args.num_polygon_edges)

    group = "order_parameter"
    set_value(simparams, "critical_temperature", args.op_tc, group)
    set_value(simparams, "initial_phase_shift", args.op_init_phase_shift, group)
    set_value(simparams, "initial_phase_winding", args.op_init_phase_winding, group)
    set_value(
        simparams, "initial_random_phase_weight", args.op_init_rand_phase_weight, group,
    )
    set_value(simparams, "symmetry", args.op_sym, group)

    set_flag(simparams, "save_data", args.save_data, args.no_save_data)
    set_value(simparams, "save_data_interval", args.save_data_interval)
    set_flag(simparams, "solenoid", args.solenoid, args.no_solenoid)
    set_value(simparams, "temperature", args.temperature)
    set_flag(simparams, "verbose", args.verbose, args.quiet)
    set_flag(simparams, "visualize", args.visualize, args.no_visualize)

    status_code = 0
    if args.gui:
        status_code = showGui(simparams, args.cpp_binary)
    else:
        # Validate parameters, exit if there are errors.
        success = validate(simparams)
        if not success:
            sys.exit(1)

        # Write temporary config file.
        with open(CONFIG_FILE_NAME, "w") as file:
            json.dump(simparams, file, indent="\t", ensure_ascii=False)

        # Run the C++ program.
        cpp_binary = Path(args.cpp_binary)
        if cpp_binary.is_file() and os.access(cpp_binary, os.X_OK):
            status_code = subprocess.call([str(cpp_binary), CONFIG_FILE_NAME])
        else:
            status_code = 1
            print(
                f"--Cannot find the C++ executable '{cpp_binary}'. See the README for how to compile it."
            )

    # Remove temporary config file.
    if Path(CONFIG_FILE_NAME).is_file():
        os.remove(CONFIG_FILE_NAME)
    sys.exit(status_code)


if __name__ == "__main__":
    main()
