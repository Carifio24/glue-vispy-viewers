import numpy as np

from ..viewer_state import (Vispy3DVolumeViewerState, cutting_plane_from_state,
                            _CUBE_CENTER, _CUBE_HALF_DIAGONAL)


def test_default_disabled():
    state = Vispy3DVolumeViewerState()
    assert state.cut_enabled is False
    assert cutting_plane_from_state(state) is None


def test_simple_axis_z():
    state = Vispy3DVolumeViewerState()
    state.cut_enabled = True
    state.cut_mode = 'Simple'
    state.cut_axis = 'Z'
    state.cut_depth = 0.5  # plane through cube centre
    a, b, c, d = cutting_plane_from_state(state)
    assert (a, b) == (0.0, 0.0)
    assert c == 1.0
    assert d == -_CUBE_CENTER  # plane at z = 128


def test_simple_axis_x_and_y():
    state = Vispy3DVolumeViewerState()
    state.cut_enabled = True
    state.cut_mode = 'Simple'
    state.cut_depth = 0.5
    state.cut_axis = 'X'
    a, b, c, _ = cutting_plane_from_state(state)
    assert (round(a, 6), round(b, 6), round(c, 6)) == (1.0, 0.0, 0.0)
    state.cut_axis = 'Y'
    a, b, c, _ = cutting_plane_from_state(state)
    assert (round(a, 6), round(b, 6), round(c, 6)) == (0.0, 1.0, 0.0)


def test_depth_endpoints_are_outside_cube():
    state = Vispy3DVolumeViewerState()
    state.cut_enabled = True
    state.cut_mode = 'Simple'
    state.cut_axis = 'Z'
    state.cut_depth = 0.0
    _, _, _, d_no_cut = cutting_plane_from_state(state)
    # depth=0 -> plane just past the +normal corner of the cube; the
    # plane crosses z = -d, which for normal +z must lie beyond z=256.
    assert -d_no_cut > 256
    state.cut_depth = 1.0
    _, _, _, d_all_cut = cutting_plane_from_state(state)
    # depth=1 -> plane just past the -normal corner: z = -d < 0.
    assert -d_all_cut < 0


def test_advanced_normal_is_unit_vector():
    state = Vispy3DVolumeViewerState()
    state.cut_enabled = True
    state.cut_mode = 'Advanced'
    state.cut_tilt = 0.7
    state.cut_rotation = 1.3
    state.cut_depth = 0.5
    a, b, c, _ = cutting_plane_from_state(state)
    assert round(a * a + b * b + c * c, 6) == 1.0


def test_advanced_known_orientation():
    # Tilt = pi/2, rotation = 0 -> normal pointing +x; same as Simple X.
    state = Vispy3DVolumeViewerState()
    state.cut_enabled = True
    state.cut_mode = 'Advanced'
    state.cut_tilt = float(np.pi / 2)
    state.cut_rotation = 0.0
    state.cut_depth = 0.5
    a, b, c, d = cutting_plane_from_state(state)
    assert (round(a, 6), round(b, 6), round(c, 6)) == (1.0, 0.0, 0.0)
    assert round(d, 6) == -_CUBE_CENTER
