from qtpy import QtCore, QtWidgets

from ...common.qt.viewer_options import VispyOptionsWidget
from .cutting_plane_widget import CuttingPlaneWidget


__all__ = ['VolumeOptionsWidget']


class VolumeOptionsWidget(QtWidgets.QWidget):
    """Tabbed viewer-options widget for the volume viewer.

    "General" embeds the shared ``VispyOptionsWidget`` (axes, stretch,
    slices, resolution, reference data). "Limits" reparents the min/max
    inputs and flip buttons out of the shared widget into a tab modelled
    on glue-qt's 2D image viewer. "Cutting plane" holds the cutting plane
    controls.
    """

    def __init__(self, viewer_state=None, session=None, parent=None):
        super().__init__(parent=parent)

        self._tabs = QtWidgets.QTabWidget(self)
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(self._tabs)

        self.general = VispyOptionsWidget(
            viewer_state=viewer_state, session=session, parent=self)
        self._tabs.addTab(self.general, 'General')

        self.limits = self._build_limits_tab()
        self._tabs.addTab(self.limits, 'Limits')

        self.cutting_plane = CuttingPlaneWidget(viewer_state=viewer_state, parent=self)
        self._tabs.addTab(self.cutting_plane, 'Cutting plane')

        # Expose the helpers the rest of the viewer pokes at on the
        # shared widget (notably the slice helper) at the top level for
        # API compatibility with code that expects the flat widget.
        self.ui = self.general.ui
        if hasattr(self.general, 'slice_helper'):
            self.slice_helper = self.general.slice_helper

    def _build_limits_tab(self):
        ui = self.general.ui
        tab = QtWidgets.QWidget()
        grid = QtWidgets.QGridLayout(tab)
        grid.setContentsMargins(10, 10, 10, 10)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(6)

        header_min = QtWidgets.QLabel('min')
        header_max = QtWidgets.QLabel('max')
        for header in (header_min, header_max):
            f = header.font()
            f.setBold(True)
            header.setFont(f)
            header.setAlignment(QtCore.Qt.AlignCenter)
        grid.addWidget(header_min, 0, 1)
        grid.addWidget(header_max, 0, 3)

        axes = (('x axis', ui.valuetext_x_min, ui.valuetext_x_max, ui.button_flip_x),
                ('y axis', ui.valuetext_y_min, ui.valuetext_y_max, ui.button_flip_y),
                ('z axis', ui.valuetext_z_min, ui.valuetext_z_max, ui.button_flip_z))
        for row, (axis_label, lo, hi, flip) in enumerate(axes, start=1):
            label = QtWidgets.QLabel(axis_label)
            f = label.font()
            f.setBold(True)
            label.setFont(f)
            grid.addWidget(label, row, 0)
            grid.addWidget(lo, row, 1)
            grid.addWidget(flip, row, 2)
            grid.addWidget(hi, row, 3)

        grid.setRowStretch(len(axes) + 1, 1)
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(3, 1)

        # Hide the now-orphaned "min/max:" labels left behind in the
        # General tab. They share the same object name (``label_2``)
        # so ``self.ui.label_2`` only exposes the last one -- walk the
        # children to find them all.
        for child in self.general.findChildren(QtWidgets.QLabel):
            if child.text() == 'min/max:':
                child.hide()

        return tab
