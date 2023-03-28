# Copyright (C) 2021 Ikomia SAS
# Contact: https://www.ikomia.com
#
# This file is part of the IkomiaStudio software.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from ikomia import core, dataprocess
from ikomia.utils import pyqtutils, qtconversion
from infer_ofa.infer_ofa_process import InferOfaParam

# PyQt GUI framework
from PyQt5.QtWidgets import *


# --------------------
# - Class which implements widget associated with the process
# - Inherits PyCore.CWorkflowTaskWidget from Ikomia API
# --------------------
class InferOfaWidget(core.CWorkflowTaskWidget):

    def __init__(self, param, parent):
        core.CWorkflowTaskWidget.__init__(self, parent)

        if param is None:
            self.parameters = InferOfaParam()
        else:
            self.parameters = param

        model_sizes = ["tiny", "medium", "base", "large"]

        # Create layout : QGridLayout by default
        self.grid_layout = QGridLayout()
        self.combo_model = pyqtutils.append_combo(self.grid_layout, "Model size")
        for model_size in model_sizes:
            self.combo_model.addItem(model_size)
        self.combo_model.setCurrentText(self.parameters.size)

        self.edit_prompt = pyqtutils.append_edit(self.grid_layout, "Text prompt", self.parameters.prompt)
        # PyQt -> Qt wrapping
        layout_ptr = qtconversion.PyQtToQt(self.grid_layout)

        # Set widget layout
        self.set_layout(layout_ptr)

    def on_apply(self):
        # Apply button clicked slot

        # Get parameters from widget
        # Example : self.parameters.windowSize = self.spinWindowSize.value()
        model_size = self.combo_model.currentText()

        if self.parameters.size != model_size:
            self.parameters.update = True
            self.parameters.size = model_size

        self.parameters.prompt = self.edit_prompt.text()

        # Send signal to launch the process
        self.emit_apply(self.parameters)


# --------------------
# - Factory class to build process widget object
# - Inherits PyDataProcess.CWidgetFactory from Ikomia API
# --------------------
class InferOfaWidgetFactory(dataprocess.CWidgetFactory):

    def __init__(self):
        dataprocess.CWidgetFactory.__init__(self)
        # Set the name of the process -> it must be the same as the one declared in the process factory class
        self.name = "infer_ofa"

    def create(self, param):
        # Create widget object
        return InferOfaWidget(param, None)
