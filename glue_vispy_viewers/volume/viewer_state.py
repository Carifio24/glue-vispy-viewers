from echo import CallbackProperty

from glue.viewers.volume3d.viewer_state import VolumeViewerState3D

__all__ = ['Vispy3DVolumeViewerState']


class Vispy3DVolumeViewerState(VolumeViewerState3D):
    """Volume viewer state with vispy-only extensions.

    Subclasses ``glue.viewers.volume3d.viewer_state.VolumeViewerState3D``
    and adds attributes that only make sense for the vispy-based volume
    renderer, so they aren't visible from non-vispy frontends (notably
    ipyvolume) which would otherwise ignore them. If/when those become
    cross-frontend concepts they can be lifted into glue-core.
    """

    # Cutting plane defined as ``a*x + b*y + c*z + d = 0``; the tuple
    # holds ``(a, b, c, d)``. ``None`` disables the cut.
    cutting_plane = CallbackProperty(None)
