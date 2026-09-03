
import os
from pathlib import Path
import cdsapi
import numpy as np
import pandas as pd


def download_era5(variable, years, area, output_path,overwrite=False):
    """
    Download ERA5 monthly averaged data for a given variable and area.

    Parameters
    ----------
    variable : str
        ERA5 variable name (e.g., '2m_temperature')
    years : list or iterable
        List of years (e.g., range(1950, 2025))
    area : list
        [North, West, South, East] in degrees
        For a single grid cell: same values for N=S and W=E
    output_path : str or Path
        Path to output NetCDF file
    """

    output_path = Path(output_path)

    # --- Idempotency check ---
    if output_path.exists() and not overwrite:
        print(f"[SKIP] File already exists: {output_path}")
        return output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[DOWNLOAD] {variable} -> {output_path}")

    c = cdsapi.Client()

    c.retrieve(
        "reanalysis-era5-single-levels-monthly-means",
        {
            "product_type": "monthly_averaged_reanalysis",
            "variable": variable,
            "year": [str(y) for y in years],
            "month": [f"{m:02d}" for m in range(1, 13)],
            "time": "00:00",
            "area": area,  # [N, W, S, E]
            "data_format": "netcdf",
        },
        str(output_path),
    )

    print(f"Saved to {output_path}")
    return output_path
    
    
from pathlib import Path
import pandas as pd
import xarray as xr
import cdsapi


def update_era5(variable, area, output_path):
    """
    Update an existing ERA5 monthly dataset with missing months.

    The function:
    1. Reads the latest timestamp in the existing NetCDF file.
    2. Determines the latest complete month to download.
    3. Downloads only missing months.
    4. Appends the new data to the existing NetCDF file.
    5. Does nothing if the dataset is already up to date.

    Parameters
    ----------
    variable : str
        ERA5 variable name, e.g. '2m_temperature'.

    area : list
        [North, West, South, East].

    output_path : str or Path
        Existing ERA5 NetCDF file.
    """

    output_path = Path(output_path)

    if not output_path.exists():
        raise FileNotFoundError(
            f"ERA5 file does not exist: {output_path}"
        )

    # --------------------------------------------------
    # 1. Read existing dataset
    # --------------------------------------------------

    ds = xr.open_dataset(output_path)

    latest_date = pd.Timestamp(
        ds["valid_time"].max().values
    )

    print(f"[INFO] Latest local ERA5 date: {latest_date.date()}")

    ds.close()

    # --------------------------------------------------
    # 2. Determine latest available complete month
    # --------------------------------------------------

    current_date = pd.Timestamp.today()

    # Last day of the previous month
    latest_available_month = (
        current_date.to_period("M") - 2
    ).to_timestamp("M")

    print(
        f"[INFO] Latest month to check: "
        f"{latest_available_month.strftime('%Y-%m')}"
    )

    # --------------------------------------------------
    # 3. Determine first missing month
    # --------------------------------------------------

    first_missing_month = (
        latest_date.to_period("M") + 1
    ).to_timestamp()

    if first_missing_month > latest_available_month:

        print("[UP TO DATE] ERA5 dataset is already current.")

        return output_path

    # --------------------------------------------------
    # 4. Generate missing months
    # --------------------------------------------------

    missing_months = pd.date_range(
        start=first_missing_month,
        end=latest_available_month,
        freq="MS"
    )

    print(
        f"[UPDATE] Missing months: "
        f"{missing_months[0].strftime('%Y-%m')} → "
        f"{missing_months[-1].strftime('%Y-%m')}"
    )

    # --------------------------------------------------
    # 5. Download missing data
    # --------------------------------------------------

    update_path = output_path.with_name(
        output_path.stem + "_update.nc"
    )

    c = cdsapi.Client()

    c.retrieve(
        "reanalysis-era5-single-levels-monthly-means",
        {
            "product_type": "monthly_averaged_reanalysis",
            "variable": variable,

            "year": sorted(
                set(str(d.year) for d in missing_months)
            ),

            "month": [
                f"{d.month:02d}"
                for d in missing_months
            ],

            "time": "00:00",

            "area": area,

            "data_format": "netcdf",
        },
        str(update_path),
    )

    print(f"[DOWNLOAD] Saved update to {update_path}")

    # --------------------------------------------------
    # 6. Combine old + new data
    # --------------------------------------------------

    ds_old = xr.open_dataset(output_path)
    ds_new = xr.open_dataset(update_path)

    ds_combined = xr.concat(
        [ds_old, ds_new],
        dim="valid_time"
    )

    # Remove any accidental duplicates
    _, unique_indices = np.unique(
        ds_combined["valid_time"].values,
        return_index=True
    )

    ds_combined = ds_combined.isel(
        valid_time=sorted(unique_indices)
    )

    # Sort chronologically
    ds_combined = ds_combined.sortby("valid_time")

    # --------------------------------------------------
    # 7. Save updated dataset
    # --------------------------------------------------

    temp_path = output_path.with_name(
        output_path.stem + "_new.nc"
    )

    ds_combined.to_netcdf(temp_path)

    ds_old.close()
    ds_new.close()
    ds_combined.close()

    # Replace old file only after successful write
    temp_path.replace(output_path)

    # Remove temporary download
    update_path.unlink()

    print(
        f"[SUCCESS] ERA5 dataset updated: "
        f"{output_path}"
    )

    return output_path
