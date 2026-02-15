import math
from typing import Iterable, List

import numpy as np


def haversine_km(
    lat1: Iterable[float], lon1: Iterable[float], lat2: Iterable[float], lon2: Iterable[float]
) -> np.ndarray:
    lat1_arr = np.asarray(list(lat1), dtype=float)
    lon1_arr = np.asarray(list(lon1), dtype=float)
    lat2_arr = np.asarray(list(lat2), dtype=float)
    lon2_arr = np.asarray(list(lon2), dtype=float)

    lat1_rad = np.radians(lat1_arr)
    lon1_rad = np.radians(lon1_arr)
    lat2_rad = np.radians(lat2_arr)
    lon2_rad = np.radians(lon2_arr)

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arcsin(np.sqrt(a))
    return 6371.0088 * c


def cell_id(lat: float, lon: float, cell_size_deg: float) -> str:
    lat_bin = int(math.floor((lat + 90.0) / cell_size_deg))
    lon_bin = int(math.floor((lon + 180.0) / cell_size_deg))
    return f"cell_{lat_bin}_{lon_bin}"


def build_cell_ids(latitudes: Iterable[float], longitudes: Iterable[float], cell_size_deg: float) -> List[str]:
    return [cell_id(float(lat), float(lon), cell_size_deg) for lat, lon in zip(latitudes, longitudes)]
