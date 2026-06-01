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
import pandas as pd


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




def dataset_to_series(ds, variable_name):
    return ds[variable_name].to_dataframe().reset_index()

def dataframe_to_parquet(df, output_file):
    df.to_parquet(output_file, index=False);
    print(f"Saved parquet file: {output_file}")



def netcdf_to_parquet(ds, variable_name, output_file):

    df = ds[variable_name].to_dataframe().reset_index()

    if "valid_time" in df.columns:
        df["valid_time"] = pd.to_datetime(df["valid_time"]).astype(str)

    df.to_parquet(
        output_file,
        engine="pyarrow",
        index=False
    )

    print(f"Saved parquet file: {output_file}")

