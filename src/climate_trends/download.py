
import os
from pathlib import Path
import cdsapi


def download_era5(variable, years, area, output_path):
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
    if output_path.exists():
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