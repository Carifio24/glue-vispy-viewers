import numpy as np
from echo import CallbackProperty, SelectionCallbackProperty

from glue.viewers.volume3d.viewer_state import VolumeViewerState3D

__all__ = ['Vispy3DVolumeViewerState']


# In the volume fragment shader, v_position spans 0..u_shape on each axis,
# with u_shape = (256, 256, 256) regardless of the underlying data shape.
# Cutting plane parameters are computed relative to this cube.
_CUBE_EXTENT = 256.0
_CUBE_CENTER = _CUBE_EXTENT / 2.0
_CUBE_HALF_DIAGONAL = (3.0 ** 0.5) * _CUBE_CENTER


def _axis_angles(axis):
    """Return ``(tilt, rotation)`` in radians for an axis-aligned cut.

    The normal is +x, +y or +z; the cube center sits on the kept side.
    """
    if axis == 'X':
        return np.pi / 2, 0.0
    if axis == 'Y':
        return np.pi / 2, np.pi / 2
    return 0.0, 0.0   # 'Z'


def cutting_plane_from_state(state):
    """Return the shader (a, b, c, d) tuple, or ``None`` if disabled.

    The cube extends 0..256 in each shader coordinate. The plane's signed
    distance from the cube centre is moved linearly between
    ``+_CUBE_HALF_DIAGONAL`` (depth=0, plane outside on the +normal side,
    nothing cut) and ``-_CUBE_HALF_DIAGONAL`` (depth=1, plane outside on
    the -normal side, everything cut).
    """
    if not state.cut_enabled:
        return None
    if state.cut_mode == 'Simple':
        tilt, rotation = _axis_angles(state.cut_axis)
    else:
        tilt, rotation = state.cut_tilt, state.cut_rotation
    a = np.sin(tilt) * np.cos(rotation)
    b = np.sin(tilt) * np.sin(rotation)
    c = np.cos(tilt)
    offset = _CUBE_HALF_DIAGONAL * (1.0 - 2.0 * state.cut_depth)
    d = -(a + b + c) * _CUBE_CENTER - offset
    return float(a), float(b), float(c), float(d)


class Vispy3DVolumeViewerState(VolumeViewerState3D):
    """Volume viewer state with vispy-only extensions.

    Subclasses ``glue.viewers.volume3d.viewer_state.VolumeViewerState3D``
    and adds attributes that only make sense for the vispy-based volume
    renderer, so they aren't visible from non-vispy frontends (notably
    ipyvolume) which would otherwise ignore them. If/when those become
    cross-frontend concepts they can be lifted into glue-core.
    """

    cut_enabled = CallbackProperty(False, docstring='Whether the cutting plane is active.')
    cut_mode = SelectionCallbackProperty(
        0, choices=['Simple', 'Advanced'],
        docstring='Simple (axis-aligned) or Advanced (free orientation).')
    cut_axis = SelectionCallbackProperty(
        2, choices=['X', 'Y', 'Z'],
        docstring='Axis perpendicular to the cutting plane in Simple mode.')
    cut_tilt = CallbackProperty(0.0, docstring='Polar angle of the plane normal, radians (0..pi).')
    cut_rotation = CallbackProperty(0.0, docstring='Azimuthal angle of the plane normal, radians (0..2*pi).')
    cut_depth = CallbackProperty(0.5, docstring='Depth of the cut, 0 (no cut) to 1 (full cut).')

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # SelectionCallbackProperty stores choices per-instance via a WeakKeyDictionary
        # populated lazily on first ``get_choices`` lookup. Force the dictionary to be
        # populated now so the choices are bound to the instance (rather than only to
        # the class) and survive a round-trip through ``__setgluestate__``.
        Vispy3DVolumeViewerState.cut_mode.set_choices(self, ['Simple', 'Advanced'])
        Vispy3DVolumeViewerState.cut_axis.set_choices(self, ['X', 'Y', 'Z'])
