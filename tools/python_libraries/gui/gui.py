#!/usr/bin/env python3
# Copyright (C) 2020 and after, SuprConGPU team.
# All rights reserved.
# This file is part of the SuprConGPU Project, under the GNU LGPL v3
# license or higher. See LICENSE.txt for license information.

"""Set up GUI for SuprConGPU frontend."""

# Built-in modules and third-party libraries.
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
    QGroupBox,
    QWidget,
    QPushButton,
    QLineEdit,
    QLabel,
)
from PyQt5.QtCore import QObject, Qt, QRegExp, pyqtSlot
from PyQt5.QtGui import QIntValidator, QDoubleValidator, QRegExpValidator, QFont
import subprocess
import sys

# Custom libraries.
sys.path.insert(0, "../../")
from python_libraries.parameter_setup import argument_validation as AV
from python_libraries.settings import configuration

# Global settings.
# Color definitions
colorOk = "#ffffff"
colorErr = "#f6989d"
colorErrMsg = "#ff4500"

# ------------------------------------------------------------------------------
# GUI
# ------------------------------------------------------------------------------
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
    def __init__(self, validator, jsonKey, jsonGroup, errMsg, parent):
        super().__init__(parent)

        self._jsonKey = jsonKey
        self._jsonGroup = jsonGroup
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

    @property
    def jsonGroup(self):
        return self._jsonGroup

    def setValue(self, value):
        self._input.setText(str(value))


class InputFloat(Input):
    def __init__(
        self,
        *,
        jsonKey,
        jsonGroup=None,
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
        super().__init__(validator, jsonKey, jsonGroup, errMsg, parent)
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
    def __init__(
        self,
        *,
        jsonKey,
        jsonGroup=None,
        minValue=0,
        errMsg="Invalid input!",
        parent=None,
    ):
        validator = QIntValidator()
        validator.setBottom(minValue)
        super().__init__(validator, jsonKey, jsonGroup, errMsg, parent)

    def json(self):
        if self.valid():
            return {self.jsonKey: int(self._input.text())}
        else:
            return None


class InputStr(Input):
    def __init__(
        self, *, jsonKey, jsonGroup=None, errMsg="Invalid input!", parent=None
    ):
        rx = QRegExp("^(?!\s*$).+")
        validator = QRegExpValidator(rx)
        super().__init__(validator, jsonKey, jsonGroup, errMsg, parent)

    def json(self):
        if self.valid():
            return {self.jsonKey: self._input.text()}
        else:
            return None


class CheckBox(Serializable, QCheckBox, metaclass=QMeta):
    def __init__(self, *, jsonKey, jsonGroup=None, parent=None):
        super().__init__(parent)

        self._jsonKey = jsonKey
        self._jsonGroup = jsonGroup
        self.setFocusPolicy(Qt.TabFocus)

    @property
    def jsonKey(self):
        return self._jsonKey

    @property
    def jsonGroup(self):
        return self._jsonGroup

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
    def __init__(self, *, default_path, parent=None):
        super().__init__(parent)

        self._path = Path(default_path)
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
    def __init__(
        self,
        *,
        jsonKey,
        jsonGroup=None,
        minValue=3,
        errMsg="Invalid input!",
        parent=None,
    ):
        super().__init__(parent)

        self._jsonKey = jsonKey
        self._jsonGroup = jsonGroup
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

    @property
    def jsonGroup(self):
        return self._jsonGroup

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
        self,
        *,
        jsonKey,
        choices,
        placeholderText,
        jsonGroup=None,
        errMsg="Invalid input!",
        parent=None,
    ):
        super().__init__(parent)
        self._jsonKey = jsonKey
        self._jsonGroup = jsonGroup
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

    @property
    def jsonGroup(self):
        return self._jsonGroup

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
        self,
        *,
        parent=None,
    ):
        super().__init__(parent)

        self._inputs = {}

        # Order parameter symmetry dropdown
        self._symmetryDropdown = InputDropdown(
            jsonKey="symmetry",
            choices=configuration.valid_symmetries,
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
        layout.addRow(QLabel("Critical temperature [Tc_max]"), self._tc)
        layout.addRow(QLabel("Initial phase shift [2\u03C0]"), self._init_phase_shift)
        layout.addRow(
            QLabel("Initial phase winding [2\u03C0]"), self._init_phase_winding
        )
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

    @property
    def jsonGroup(self):
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

    def __init__(self, *, jsonKey, jsonGroup=None, parent=None):
        super().__init__(parent)

        self._jsonKey = jsonKey
        self._jsonGroup = jsonGroup
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

    @property
    def jsonGroup(self):
        return self._jsonGroup

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
        if len(opComponentsList) == 0:
            print("Input Warning: Order parameter was not specified")
            opComponent = self.add_component()
        else:
            for opComponentDict in opComponentsList:
                if len(self._opComponents) == 0:
                    opComponent = self.add_component()
                else:
                    opComponent = self.handle_add()
                opComponent.setValue(opComponentDict)


class runner_main_window(QMainWindow):
    def __init__(self, *, cpp_binary_path):
        super(runner_main_window, self).__init__()
        self.setWindowTitle("SuprConGPU Runner")
        font = QFont()
        font.setPixelSize(13)
        QApplication.setFont(font)

        # Widgets registry
        self._widgetRegistry = {}

        # Create main widget holding all the controls and inputs
        mainWidget = QFrame()

        # -------------- #
        # Physics Inputs #
        # -------------- #
        physicsBox = QGroupBox("Physics Parameters")
        physicsBoxLayout = QFormLayout(physicsBox)
        physicsBoxLayout.setLabelAlignment(Qt.AlignLeft)

        # Temperature
        self._temperature = InputFloat(jsonKey="temperature", onlyPositive=True)
        self.register(self._temperature)
        physicsBoxLayout.addRow(QLabel("Temperature [Tc]"), self._temperature)

        # External flux quanta
        self._external_flux_quanta = InputFloat(jsonKey="external_flux_quanta")
        self.register(self._external_flux_quanta)
        physicsBoxLayout.addRow(
            QLabel("External flux quanta [\u03A6\u2080]"), self._external_flux_quanta
        )

        # Penetration depth
        self._prenetration_depth = InputFloat(jsonKey="penetration_depth")
        self.register(self._prenetration_depth)
        physicsBoxLayout.addRow(
            QLabel("Penetration depth [\u03BE\u2080]"), self._prenetration_depth
        )

        # Crystal axes rotation
        self._crystal_axes_rotation = InputFloat(jsonKey="crystal_axes_rotation")
        self.register(self._crystal_axes_rotation)
        physicsBoxLayout.addRow(
            QLabel("Crystal axes rotation [2\u03C0]"),
            self._crystal_axes_rotation,
        )

        # Grain side length
        self._grain_side_length = InputFloat(
            jsonKey="grain_side_length", onlyPositive=True
        )
        self.register(self._grain_side_length)
        physicsBoxLayout.addRow(
            QLabel("Grain side length [\u03BE\u2080]"), self._grain_side_length
        )

        # Number of polygon edges
        self._num_polygon_edges = Geometry(jsonKey="num_polygon_edges")
        self.register(self._num_polygon_edges)
        physicsBoxLayout.addRow(QLabel("System geometry"), self._num_polygon_edges)

        # Solenoid
        self._solenoid = CheckBox(jsonKey="solenoid")
        self.register(self._solenoid)
        physicsBoxLayout.addRow(QLabel("Solenoid"), self._solenoid)

        # Order parameter
        self._order_parameter = OrderParameter(jsonKey="order_parameter")
        self.register(self._order_parameter)
        physicsBoxLayout.addRow(QLabel("Order parameter"), self._order_parameter)

        # --------------------- #
        # Simulation Parameters #
        # --------------------- #
        simulationBox = QGroupBox("Simulation Parameters")
        simulationBoxLayout = QFormLayout(simulationBox)
        simulationBoxLayout.setLabelAlignment(Qt.AlignLeft)

        # Lattice grid size
        self._points_per_coherence_length = InputFloat(
            jsonKey="points_per_coherence_length", minValue=1.0
        )
        self.register(self._points_per_coherence_length)
        simulationBoxLayout.addRow(
            QLabel("Points per coherence length"),
            self._points_per_coherence_length,
        )

        # Number of Fermi momenta
        self._num_fermi_momenta = InputInt(jsonKey="num_fermi_momenta", minValue=1)
        self.register(self._num_fermi_momenta)
        simulationBoxLayout.addRow(
            QLabel("Number of Fermi momenta"), self._num_fermi_momenta
        )

        # Number of Ozaki energies
        self._num_ozaki_energies = InputInt(jsonKey="num_ozaki_energies", minValue=1)
        self.register(self._num_ozaki_energies)
        simulationBoxLayout.addRow(
            QLabel("Number of Ozaki energies"), self._num_ozaki_energies
        )

        # Max number of iterations
        self._num_iterations_max = InputInt(jsonKey="num_iterations_max", minValue=1)
        self.register(self._num_iterations_max)
        simulationBoxLayout.addRow(
            QLabel("Max number of self-consistent iterations"), self._num_iterations_max
        )

        # Min number of iterations
        self._num_iterations_min = InputInt(jsonKey="num_iterations_min", minValue=1)
        self.register(self._num_iterations_min)
        simulationBoxLayout.addRow(
            QLabel("Min number of self-consistent iterations"), self._num_iterations_min
        )

        # Convergence criterion
        self._convergence_criterion = InputFloat(
            jsonKey="convergence_criterion", minValue=0.0
        )
        self.register(self._convergence_criterion)
        simulationBoxLayout.addRow(
            QLabel("Convergence criterion"), self._convergence_criterion
        )

        # --------------------- #
        # Optimizer Parameters #
        # --------------------- #
        optimizerBox = QGroupBox("Optimizer Parameters")
        optimizerBoxLayout = QFormLayout(optimizerBox)
        optimizerBoxLayout.setLabelAlignment(Qt.AlignLeft)

        # Use Anderson all the time.
        self._anderson_always = CheckBox(
            jsonKey="anderson_always", jsonGroup="optimizer"
        )
        self.register(self._anderson_always)
        optimizerBoxLayout.addRow(QLabel("Anderson always"), self._anderson_always)

        # Inertia.
        self._inertia = InputFloat(
            jsonKey="inertia", jsonGroup="optimizer", minValue=0.0
        )
        self.register(self._inertia)
        optimizerBoxLayout.addRow(QLabel("Inertia"), self._inertia)

        # Max mixing factor.
        self._max_mixing = InputFloat(
            jsonKey="max_mixing", jsonGroup="optimizer", minValue=0.0
        )
        self.register(self._max_mixing)
        optimizerBoxLayout.addRow(QLabel("Max mixing factor"), self._max_mixing)

        # Min mixing factor.
        self._min_mixing = InputFloat(
            jsonKey="min_mixing", jsonGroup="optimizer", minValue=0.0
        )
        self.register(self._min_mixing)
        optimizerBoxLayout.addRow(QLabel("Max mixing factor"), self._min_mixing)

        # Rank.
        self._rank = InputInt(jsonKey="rank", jsonGroup="optimizer", minValue=0)
        self.register(self._rank)
        optimizerBoxLayout.addRow(QLabel("Rank"), self._rank)

        # Replace oldest.
        self._replace_oldest = CheckBox(jsonKey="replace_oldest", jsonGroup="optimizer")
        self.register(self._replace_oldest)
        optimizerBoxLayout.addRow(QLabel("Replace oldest"), self._replace_oldest)

        # Step size.
        self._step_size = InputFloat(
            jsonKey="step_size", jsonGroup="optimizer", minValue=0.0
        )
        self.register(self._step_size)
        optimizerBoxLayout.addRow(QLabel("Step size"), self._step_size)

        # ------------------------ #
        # Miscellaneous Parameters #
        # ------------------------ #
        miscBox = QGroupBox("Miscellaneous Parameters")
        miscBoxLayout = QFormLayout(miscBox)
        miscBoxLayout.setLabelAlignment(Qt.AlignLeft)

        # Set data path
        self._data_path = InputStr(jsonKey="data_path")
        self.register(self._data_path)
        miscBoxLayout.addRow(QLabel("Data path"), self._data_path)

        # Save data
        self._save_data = CheckBox(jsonKey="save_data")
        self.register(self._save_data)
        miscBoxLayout.addRow(QLabel("Save data"), self._save_data)

        # Save data interval
        self._save_data_interval = InputInt(jsonKey="save_data_interval")
        self.register(self._save_data_interval)
        miscBoxLayout.addRow(
            QLabel("Save every Nth iteration"), self._save_data_interval
        )

        # Verbose calculation
        self._verbose = CheckBox(jsonKey="verbose")
        self.register(self._verbose)
        miscBoxLayout.addRow(QLabel("Verbose calculation"), self._verbose)

        # Visualize
        self._visualize = CheckBox(jsonKey="visualize")
        self.register(self._visualize)
        miscBoxLayout.addRow(QLabel("Visualize"), self._visualize)

        # Compiled binary path
        self._binary_path = ChooseFile(default_path=cpp_binary_path)
        miscBoxLayout.addRow(QLabel("SuprConGPU binary path"), self._binary_path)

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
        layout.addWidget(physicsBox)
        layout.addWidget(simulationBox)
        layout.addWidget(optimizerBox)
        layout.addWidget(miscBox)
        layout.addLayout(buttonsLayout)

        # Set the created form as the central part of the application window
        scrollArea = QScrollArea(self)
        scrollArea.setWidgetResizable(True)
        scrollArea.setWidget(mainWidget)
        self.setCentralWidget(scrollArea)
        self.setFocus()

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
                if inputWidget.jsonGroup is None:
                    # If this item is not part of a group (i.e. sub-dict), update the config directly.
                    config.update(widgetJson)
                else:
                    # Check if the group exists, if not add it.
                    if not inputWidget.jsonGroup in config:
                        config[inputWidget.jsonGroup] = {}
                    # Update the group.
                    config[inputWidget.jsonGroup].update(widgetJson)
            else:
                success = False
                config.update({inputWidget.jsonKey: None})
        return (config, success)

    def deserialize(self, config):
        for key, value in config.items():
            if isinstance(value, dict):
                # If this item is a dict, run deserialize recursively.
                self.deserialize(value)
            else:
                # This item is not a dict, so we can set the values directly.
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
            self,
            caption="Save config file",
            filter="Config file (*.json)",
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
                # Validate parameters, do not start simulation if there are errors.
                success = AV.validate_arguments(arguments_dict=config)
                if success:
                    with open(configuration.temp_config_file_name, "w") as file:
                        json.dump(config, file, indent="\t", ensure_ascii=False)

                    subprocess.call(
                        [str(binaryPath), configuration.temp_config_file_name]
                    )
        else:
            msgBox = QMessageBox()
            msgBox.setIcon(QMessageBox.Critical)
            msgBox.setText(
                "Binary path is invalid!\n"
                "Please, make sure it points to the existing SuprConGPU compiled binary."
            )
            msgBox.exec_()


class plotter_main_window(QMainWindow):
    def __init__(self, plot_parameters):
        super(plotter_main_window, self).__init__()
        self.setWindowTitle("SuprConGPU Plotter")
        font = QFont()
        font.setPixelSize(13)
        QApplication.setFont(font)

        # Widgets registry
        self._widgetRegistry = {}

        # Create main widget holding all the controls and inputs
        main_widget = QFrame()

        # --------------- #
        # Current density #
        # --------------- #
        current_density_box = QGroupBox("Current density")
        current_density_box_layout = QFormLayout(current_density_box)
        current_density_box_layout.setLabelAlignment(Qt.AlignLeft)

        # Plot magnitude.
        self._plot_current_density = CheckBox(
            jsonKey="plot", jsonGroup="current_density"
        )
        self.register(self._plot_current_density)
        current_density_box_layout.addRow(QLabel("Plot"), self._plot_current_density)

        # --------------------- #
        # Magnetic flux density #
        # --------------------- #
        magnetic_flux_density_box = QGroupBox("Current density")
        magnetic_flux_density_box_layout = QFormLayout(magnetic_flux_density_box)
        magnetic_flux_density_box_layout.setLabelAlignment(Qt.AlignLeft)

        # Plot magnitude.
        self._plot_magnetic_flux_density = CheckBox(
            jsonKey="plot", jsonGroup="magnetic_flux_density"
        )
        self.register(self._plot_magnetic_flux_density)
        magnetic_flux_density_box_layout.addRow(
            QLabel("Plot"), self._plot_magnetic_flux_density
        )

        # --------------- #
        # Order Parameter #
        # --------------- #
        order_parameter_box = QGroupBox("Order parameter")
        order_parameter_box_layout = QFormLayout(order_parameter_box)
        order_parameter_box_layout.setLabelAlignment(Qt.AlignLeft)

        # Plot magnitude.
        self._plot_order_parameter = CheckBox(
            jsonKey="plot", jsonGroup="order_parameter"
        )
        self.register(self._plot_order_parameter)
        order_parameter_box_layout.addRow(QLabel("Plot"), self._plot_order_parameter)

        # ---------------- #
        # Vector potential #
        # ---------------- #
        vector_potential_box = QGroupBox("Vector potential")
        vector_potential_box_layout = QFormLayout(vector_potential_box)
        vector_potential_box_layout.setLabelAlignment(Qt.AlignLeft)

        # Plot magnitude.
        self._plot_vector_potential = CheckBox(
            jsonKey="plot", jsonGroup="vector_potential"
        )
        self.register(self._plot_vector_potential)
        vector_potential_box_layout.addRow(QLabel("Plot"), self._plot_vector_potential)

        # ------------------ #
        # General parameters #
        # ------------------ #
        general_box = QGroupBox("General plotting parameters")
        general_box_layout = QFormLayout(general_box)
        general_box_layout.setLabelAlignment(Qt.AlignLeft)

        # Compiled binary path
        self._data_path = ChooseFile(
            default_path=plot_parameters["general"]["data_path"]
        )
        general_box_layout.addRow(QLabel("Data path"), self._data_path)

        # -------- #
        # Controls #
        # -------- #

        # Buttons
        self._load_config = QPushButton("Load plot config", main_widget)
        self._load_config.setFocusPolicy(Qt.TabFocus)
        self._load_config.clicked.connect(self.handle_load_file)

        self._plot_observables = QPushButton("Plot", main_widget)
        self._plot_observables.setFocusPolicy(Qt.TabFocus)
        self._plot_observables.clicked.connect(self.handle_plot_observables)

        buttons_layout = QHBoxLayout()
        buttons_layout.addWidget(self._load_config)
        buttons_layout.addWidget(self._plot_observables)

        # ------------------ #
        # Application layout #
        # ------------------ #

        # Main widget layout
        layout = QVBoxLayout(main_widget)
        layout.addWidget(general_box)
        layout.addWidget(order_parameter_box)
        layout.addWidget(current_density_box)
        layout.addWidget(magnetic_flux_density_box)
        layout.addWidget(vector_potential_box)
        layout.addLayout(buttons_layout)

        # Set the created form as the central part of the application window
        scroll_area = QScrollArea(self)
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(main_widget)
        self.setCentralWidget(scroll_area)
        self.setFocus()

        # Set application styles
        self.setAppStyle()

    def setAppStyle(self):
        self.setStyleSheet(
            "QMainWindow > QWidget { background-color: rgb(255, 255, 255);}"
        )

    def register(self, widget: Serializable):
        self._widgetRegistry[widget.jsonKey] = widget

    def deserialize(self, config):
        for key, value in config.items():
            if isinstance(value, dict):
                # If this item is a dict, run deserialize recursively.
                self.deserialize(value)
            else:
                # This item is not a dict, so we can set the values directly.
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
    def handle_load_file(self):
        loadFileName = QFileDialog.getOpenFileName(
            self, caption="Open plot config file", filter="Plot config file (*.json)"
        )[0]

        if loadFileName:
            with open(loadFileName, "r") as file:
                config = json.load(file)
                self.deserialize(config)

    @pyqtSlot()
    def handle_plot_observables(self):
        binaryPath = self._binary_path.getPath()
        if binaryPath is not None:
            (config, shouldProceed) = self.serialize()
            if not shouldProceed:
                msgBox = self.make_invalid_input_popup()
                popupResponse = msgBox.exec_()
                if popupResponse == QMessageBox.Yes:
                    shouldProceed = True
            if shouldProceed:
                # Validate parameters, do not start simulation if there are errors.
                success = AV.validate_arguments(arguments_dict=config)
                if success:
                    with open(configuration.temp_config_file_name, "w") as file:
                        json.dump(config, file, indent="\t", ensure_ascii=False)

                    subprocess.call(
                        [str(binaryPath), configuration.temp_config_file_name]
                    )
        else:
            msgBox = QMessageBox()
            msgBox.setIcon(QMessageBox.Critical)
            msgBox.setText(
                "Binary path is invalid!\n"
                "Please, make sure it points to the existing SuprConGPU compiled binary."
            )
            msgBox.exec_()


# Create runner GUI.
def show_runner_gui(simulation_parameters: dict, cpp_binary_path: str) -> int:
    # Every Qt application should contain exactly one instance of QApplication.
    # This handles interaction with the OS and screen, and runs the main event loop.
    app = QApplication(sys.argv)

    # Main part of the application.
    main_window = runner_main_window(cpp_binary_path=cpp_binary_path)

    # Set startup size depending on the screen geometry.
    screen_rectangle = app.desktop().screenGeometry()
    main_window.resize(
        screen_rectangle.width() * 2 // 5, screen_rectangle.height() * 4 // 5
    )

    if bool(simulation_parameters):
        main_window.deserialize(simulation_parameters)
    main_window.show()

    # Center on the screen.
    main_window.move(
        QApplication.desktop().screen().rect().center()
        - main_window.frameGeometry().center()
    )

    # Start the main event loop until finished.
    # The final exit code is passed over to python environment.
    return app.exec_()


# Create plotter GUI.
def show_plotter_gui(plot_parameters: dict) -> int:
    # Every Qt application should contain exactly one instance of QApplication.
    # This handles interaction with the OS and screen, and runs the main event loop.
    app = QApplication(sys.argv)

    # Main part of the application.
    main_window = plotter_main_window(plot_parameters)

    # Set startup size depending on the screen geometry.
    screen_rectangle = app.desktop().screenGeometry()
    main_window.resize(
        screen_rectangle.width() * 1.5 // 5, screen_rectangle.height() * 1.5 // 5
    )

    if bool(plot_parameters):
        main_window.deserialize(plot_parameters)
    main_window.show()

    # Center on the screen.
    main_window.move(
        QApplication.desktop().screen().rect().center()
        - main_window.frameGeometry().center()
    )

    # Start the main event loop until finished.
    # The final exit code is passed over to python environment.
    return app.exec_()