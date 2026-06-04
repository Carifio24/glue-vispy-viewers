import os

from qtpy import QtWidgets

from echo.qt import autoconnect_callbacks_to_qt

from glue_qt.utils import load_ui
from glue_qt.viewers.common.slice_widget import MultiSliceWidgetHelper

from glue.viewers.volume3d.viewer_state import VolumeViewerState3D as Vispy3DVolumeViewerState
from glue.viewers.scatter3d.viewer_state import ScatterViewerState3D as Vispy3DScatterViewerState

__all__ = ["VispyOptionsWidget"]


class VispyOptionsWidget(QtWidgets.QWidget):
    """Tabbed viewer-options widget shared by the scatter and volume viewers.

    "General" holds the reference data, axis attributes, resolution, slices
    and the display toggles. "Limits" holds the per-axis min/max, flip and
    stretch controls. The volume viewer adds a "Cutting plane" tab on top
    (see ``VolumeOptionsWidget``).
    """

    def __init__(self, viewer_state=None, session=None, parent=None):

        super(VispyOptionsWidget, self).__init__(parent=parent)

        self._data_collection = session.data_collection

        self._tabs = QtWidgets.QTabWidget(self)
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(self._tabs)

        directory = os.path.dirname(__file__)
        self.general = QtWidgets.QWidget()
        self.ui = load_ui('viewer_options.ui', self.general, directory=directory)
        self._tabs.addTab(self.general, 'General')

        self.limits = QtWidgets.QWidget()
        self.limits_ui = load_ui('limits.ui', self.limits, directory=directory)
        self._tabs.addTab(self.limits, 'Limits')

        if not isinstance(viewer_state, Vispy3DScatterViewerState):
            self.ui.bool_clip_data.hide()

        if not hasattr(viewer_state, 'downsample'):
            self.ui.bool_downsample.hide()

        if not hasattr(viewer_state, 'resolution'):
            self.ui.label_resolution.hide()
            self.ui.combosel_resolution.hide()

        if not hasattr(viewer_state, 'reference_data'):
            self.ui.label_reference_data.hide()
            self.ui.combosel_reference_data.hide()

        if isinstance(viewer_state, Vispy3DVolumeViewerState):
            self.slice_helper = MultiSliceWidgetHelper(viewer_state=viewer_state,
                                                       layout=self.ui.layout_slices)

        self.ui.label_line_width.hide()
        self.ui.value_line_width.hide()

        stretch_kwargs = {'value_x_stretch': dict(value_range=(0.1, 10), log=True),
                          'value_y_stretch': dict(value_range=(0.1, 10), log=True),
                          'value_z_stretch': dict(value_range=(0.1, 10), log=True),
                          'valuetext_x_stretch': dict(fmt='{:6.2f}'),
                          'valuetext_y_stretch': dict(fmt='{:6.2f}'),
                          'valuetext_z_stretch': dict(fmt='{:6.2f}')}

        self._connections = autoconnect_callbacks_to_qt(viewer_state, self.ui)
        self._connections_limits = autoconnect_callbacks_to_qt(
            viewer_state, self.limits_ui, stretch_kwargs)
