import numpy as np

from geobot.geo import cell_id, haversine_km


def test_haversine_zero_distance() -> None:
    d = haversine_km([0.0], [0.0], [0.0], [0.0])
    assert np.allclose(d, [0.0])


def test_cell_id_is_stable() -> None:
    assert cell_id(40.7, -74.0, 1.0) == "cell_130_106"
