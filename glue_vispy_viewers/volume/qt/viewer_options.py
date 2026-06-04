from ...common.qt.viewer_options import VispyOptionsWidget
from .cutting_plane_widget import CuttingPlaneWidget


__all__ = ['VolumeOptionsWidget']


class VolumeOptionsWidget(VispyOptionsWidget):
    """Volume viewer options: the shared General/Limits tabs plus a cutting
    plane tab.

    Extends the shared :class:`VispyOptionsWidget` (which provides the
    "General" and "Limits" tabs) by appending a "Cutting plane" tab with the
    volume-specific cutting plane controls.
    """

    def __init__(self, viewer_state=None, session=None, parent=None):
        super().__init__(viewer_state=viewer_state, session=session, parent=parent)

        self.cutting_plane = CuttingPlaneWidget(viewer_state=viewer_state, parent=self)
        self._tabs.addTab(self.cutting_plane, 'Cutting plane')
