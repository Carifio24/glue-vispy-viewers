import os

import numpy as np

from qtpy import QtWidgets

from glue_qt.utils import load_ui
from echo.qt import autoconnect_callbacks_to_qt


__all__ = ['CuttingPlaneWidget']


class CuttingPlaneWidget(QtWidgets.QWidget):
    """Qt controls for the volume viewer's cutting plane.

    A toggle enables the cut; a Simple/Advanced selector switches between
    axis-aligned cuts (X/Y/Z radio buttons) and a free-orientation cut
    driven by tilt/rotation sliders. A depth slider and a flip button are
    always shown while the cut is enabled.
    """

    def __init__(self, viewer_state, parent=None):
        super().__init__(parent=parent)
        self.state = viewer_state

        self.ui = load_ui('cutting_plane_widget.ui', self,
                          directory=os.path.dirname(__file__))

        self._wire_radio_buttons()
        connect_kwargs = {
            'value_cut_tilt': dict(value_range=(0.0, np.pi)),
            'value_cut_rotation': dict(value_range=(0.0, 2.0 * np.pi)),
            'value_cut_depth': dict(value_range=(0.0, 1.0)),
            'value_cut_plane_image_opacity': dict(value_range=(0.0, 1.0)),
        }
        self._connections = autoconnect_callbacks_to_qt(
            self.state, self.ui, connect_kwargs)
        self.state.add_callback('cut_enabled', self._on_enabled_change)
        self.state.add_callback('cut_mode', self._on_mode_change)
        self._on_enabled_change()
        self._on_mode_change()

    def _wire_radio_buttons(self):
        # Axis radio buttons aren't covered by echo's autoconnect, so wire
        # them by hand: button toggle -> state.cut_axis, and the reverse.
        self._axis_group = QtWidgets.QButtonGroup(self)
        self._axis_buttons = {'X': self.ui.radio_axis_x,
                              'Y': self.ui.radio_axis_y,
                              'Z': self.ui.radio_axis_z}
        for i, btn in enumerate(self._axis_buttons.values()):
            self._axis_group.addButton(btn, i)

        def on_toggled(checked):
            for ax, btn in self._axis_buttons.items():
                if btn.isChecked():
                    if self.state.cut_axis != ax:
                        self.state.cut_axis = ax
                    return

        for btn in self._axis_buttons.values():
            btn.toggled.connect(on_toggled)
        self.state.add_callback('cut_axis', self._sync_axis_buttons)
        self._sync_axis_buttons()

    def _sync_axis_buttons(self, *args):
        btn = self._axis_buttons.get(self.state.cut_axis)
        if btn is not None and not btn.isChecked():
            btn.setChecked(True)

    def _on_enabled_change(self, *args):
        self.ui.controls.setEnabled(bool(self.state.cut_enabled))

    def _on_mode_change(self, *args):
        # Simple mode shows the axis radio buttons; Advanced mode shows the
        # tilt/rotation sliders. Toggle each row's label and field together.
        simple = (self.state.cut_mode == 'Simple')
        self.ui.label_axis.setVisible(simple)
        self.ui.simple_box.setVisible(simple)
        for w in (self.ui.label_tilt, self.ui.value_cut_tilt,
                  self.ui.label_rotation, self.ui.value_cut_rotation):
            w.setVisible(not simple)
