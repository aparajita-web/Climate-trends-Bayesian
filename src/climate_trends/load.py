"""
Utility functions to load NetCDF data for EDA.

Supports:
- Single file loading
- Multi-file datasets (time series, tiles, etc.)
- Optional variable selection
- Lazy loading (dask-enabled)
"""

import os
from typing import List, Optional, Union

import xarray as xr


def load_single_file(
    file_path: str,
    variables: Optional[List[str]] = None,
    decode_times: bool = True
) -> xr.Dataset:
   
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} not found")

    ds = xr.open_dataset(file_path, decode_times=decode_times)

    if variables is not None:
        ds = ds[variables]

    return ds


