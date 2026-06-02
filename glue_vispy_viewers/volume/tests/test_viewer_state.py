import numpy as np

from ..viewer_state import (Vispy3DVolumeViewerState, cutting_plane_from_state,
                            cutting_plane_polygon, _CUBE_CENTER)


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


def _set_extent(state, x=(0., 10.), y=(0., 10.), z=(0., 10.)):
    state.x_min, state.x_max = x
    state.y_min, state.y_max = y
    state.z_min, state.z_max = z


def test_polygon_none_when_disabled():
    state = Vispy3DVolumeViewerState()
    _set_extent(state)
    assert cutting_plane_polygon(state) is None


def test_polygon_simple_z_is_horizontal_square():
    # depth=0.5, axis Z -> plane at z = (z_min+z_max)/2; intersects the
    # cube as the four mid-z corners of a horizontal square.
    state = Vispy3DVolumeViewerState()
    _set_extent(state)
    state.cut_enabled = True
    state.cut_mode = 'Simple'
    state.cut_axis = 'Z'
    state.cut_depth = 0.5
    poly = cutting_plane_polygon(state)
    assert poly.shape == (4, 3)
    assert np.allclose(poly[:, 2], 5.0)
    # The four corners should be the cube's x/y extremes at z=5
    xy = sorted(map(tuple, np.round(poly[:, :2], 6).tolist()))
    assert xy == [(0., 0.), (0., 10.), (10., 0.), (10., 10.)]


def test_polygon_depth_endpoints_miss_the_box():
    state = Vispy3DVolumeViewerState()
    _set_extent(state)
    state.cut_enabled = True
    state.cut_mode = 'Simple'
    state.cut_axis = 'Z'
    state.cut_depth = 0.0
    assert cutting_plane_polygon(state) is None
    state.cut_depth = 1.0
    assert cutting_plane_polygon(state) is None


def test_polygon_tilted_lies_on_plane():
    state = Vispy3DVolumeViewerState()
    _set_extent(state, x=(-1., 1.), y=(-2., 2.), z=(0., 4.))
    state.cut_enabled = True
    state.cut_mode = 'Advanced'
    state.cut_tilt = 0.7
    state.cut_rotation = 1.3
    state.cut_depth = 0.5
    poly = cutting_plane_polygon(state)
    assert poly is not None and poly.shape[0] >= 3
    # Each vertex satisfies the plane equation in data coords; the cube
    # centre lies on the plane at depth 0.5, so signed distance from
    # centre equals (vertex - centre) . n.
    centre = np.array([0., 0., 2.])
    a = np.sin(state.cut_tilt) * np.cos(state.cut_rotation)
    b = np.sin(state.cut_tilt) * np.sin(state.cut_rotation)
    c = np.cos(state.cut_tilt)
    # Convert shader normal (in v_position coords) to data-coord normal
    # by dividing by the per-axis scale.
    n_data = np.array([a / 1.0, b / 2.0, c / 2.0])  # scales = range/CUBE_EXTENT cancel
    for p in poly:
        assert abs(np.dot(p - centre, n_data)) < 1e-6


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
