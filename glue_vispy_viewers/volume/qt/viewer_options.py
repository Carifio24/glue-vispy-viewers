from qtpy import QtWidgets

from ...common.qt.viewer_options import VispyOptionsWidget
from .cutting_plane_widget import CuttingPlaneWidget


__all__ = ['VolumeOptionsWidget']


class VolumeOptionsWidget(QtWidgets.QWidget):
    """Tabbed viewer-options widget for the volume viewer.

    The first tab embeds the shared ``VispyOptionsWidget`` (axes, limits,
    stretch, slices, resolution). The second tab holds the cutting plane
    controls -- a feature only the volume viewer has, which is why this
    wrapper lives here rather than next to the shared widget.
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

        self.cutting_plane = CuttingPlaneWidget(viewer_state=viewer_state, parent=self)
        self._tabs.addTab(self.cutting_plane, 'Cutting plane')

        # Expose the helpers the rest of the viewer pokes at on the
        # shared widget (notably the slice helper) at the top level for
        # API compatibility with code that expects the flat widget.
        self.ui = self.general.ui
        if hasattr(self.general, 'slice_helper'):
            self.slice_helper = self.general.slice_helper
