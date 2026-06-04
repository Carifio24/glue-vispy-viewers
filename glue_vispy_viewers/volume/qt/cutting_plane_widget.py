import numpy as np

from qtpy import QtCore, QtWidgets

from echo.qt import autoconnect_callbacks_to_qt


__all__ = ['CuttingPlaneWidget']


class CuttingPlaneWidget(QtWidgets.QWidget):
    """Qt controls for the volume viewer's cutting plane.

    A toggle enables the cut; a Simple/Advanced selector switches between
    axis-aligned cuts (X/Y/Z radio buttons) and a free-orientation cut
    driven by tilt/rotation sliders. A depth slider is always shown
    while the cut is enabled.
    """

    def __init__(self, viewer_state, parent=None):
        super().__init__(parent=parent)
        self.state = viewer_state
        self._build_ui()
        self._wire_radio_buttons()
        connect_kwargs = {
            'value_cut_tilt': dict(value_range=(0.0, np.pi)),
            'value_cut_rotation': dict(value_range=(0.0, 2.0 * np.pi)),
            'value_cut_depth': dict(value_range=(0.0, 1.0)),
        }
        self._connections = autoconnect_callbacks_to_qt(
            self.state, self, connect_kwargs)
        self.state.add_callback('cut_enabled', self._on_enabled_change)
        self.state.add_callback('cut_mode', self._on_mode_change)
        self._on_enabled_change()
        self._on_mode_change()

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)

        self.bool_cut_enabled = QtWidgets.QCheckBox('Enable cutting plane')
        layout.addWidget(self.bool_cut_enabled)

        mode_row = QtWidgets.QHBoxLayout()
        mode_row.addWidget(QtWidgets.QLabel('Mode:'))
        self.combosel_cut_mode = QtWidgets.QComboBox()
        mode_row.addWidget(self.combosel_cut_mode, 1)
        self._mode_row_widget = QtWidgets.QWidget()
        self._mode_row_widget.setLayout(mode_row)
        layout.addWidget(self._mode_row_widget)

        # Simple-mode axis radio buttons
        self._simple_box = QtWidgets.QGroupBox('Axis')
        simple_layout = QtWidgets.QHBoxLayout(self._simple_box)
        self.radio_axis_x = QtWidgets.QRadioButton('X')
        self.radio_axis_y = QtWidgets.QRadioButton('Y')
        self.radio_axis_z = QtWidgets.QRadioButton('Z')
        for w in (self.radio_axis_x, self.radio_axis_y, self.radio_axis_z):
            simple_layout.addWidget(w)
        layout.addWidget(self._simple_box)

        # Advanced-mode angle sliders -- subheader-above-slider layout so
        # each slider gets the full widget width.
        self._advanced_box = QtWidgets.QGroupBox('Orientation')
        adv_layout = QtWidgets.QVBoxLayout(self._advanced_box)
        adv_layout.addWidget(QtWidgets.QLabel('Tilt'))
        self.value_cut_tilt = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.value_cut_tilt.setRange(0, 180)
        adv_layout.addWidget(self.value_cut_tilt)
        adv_layout.addWidget(QtWidgets.QLabel('Rotation'))
        self.value_cut_rotation = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.value_cut_rotation.setRange(0, 360)
        adv_layout.addWidget(self.value_cut_rotation)
        layout.addWidget(self._advanced_box)

        # Depth slider
        self._depth_box = QtWidgets.QGroupBox('Depth')
        depth_layout = QtWidgets.QVBoxLayout(self._depth_box)
        self.value_cut_depth = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.value_cut_depth.setRange(0, 1000)
        depth_layout.addWidget(self.value_cut_depth)
        layout.addWidget(self._depth_box)

        # Flip button: swaps the shown and clipped sides of the plane. Named
        # ``button_flip_cut`` so echo's autoconnect wires its click straight to
        # ``state.flip_cut`` (Simple-mode cuts switch to Advanced when flipped).
        self.button_flip_cut = QtWidgets.QPushButton('Flip shown/clipped side')
        layout.addWidget(self.button_flip_cut)

        layout.addStretch(1)

    def _wire_radio_buttons(self):
        # Axis radio buttons aren't covered by echo's autoconnect, so wire
        # them by hand: button toggle -> state.cut_axis, and the reverse.
        self._axis_group = QtWidgets.QButtonGroup(self)
        for i, w in enumerate((self.radio_axis_x, self.radio_axis_y, self.radio_axis_z)):
            self._axis_group.addButton(w, i)
        self._axis_buttons = {'X': self.radio_axis_x,
                              'Y': self.radio_axis_y,
                              'Z': self.radio_axis_z}

        def on_toggled(checked, axis=None):
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
        on = bool(self.state.cut_enabled)
        for w in (self._mode_row_widget, self._simple_box,
                  self._advanced_box, self._depth_box, self.button_flip_cut):
            w.setEnabled(on)

    def _on_mode_change(self, *args):
        simple = (self.state.cut_mode == 'Simple')
        self._simple_box.setVisible(simple)
        self._advanced_box.setVisible(not simple)
