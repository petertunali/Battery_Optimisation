# pv_simulate.py

import numpy as np
import pandas as pd
import PySAM.Pvwattsv8 as pv
from pathlib import Path
from datetime import datetime

def simulate_pv_simple(
    weather_file: str,
    system_capacity_kw: float,
    tilt: float,
    azimuth: float,
    soiling: float = 2.0,
    shading: float = 0.0,
    snow: float = 0.0,
    mismatch: float = 2.0,
    wiring: float = 2.0,
    connections: float = 0.5,
    lid: float = 1.5,
    nameplate: float = 1.0,
    age: float = 0.0,
    availability: float = 3.0,
    dc_ac_ratio: float = 1.15,
    inv_eff: float = 96.0,
    array_type: int = 1,
    module_type: int = 0,
    gcr: float = 0.3,
    start_year: int = 2024,
) -> pd.DataFrame:
    """
    Run a PVWatts simulation and return half-hourly energy (kWh) for one year.
    Each hourly kWh output is evenly split into two 30-minute intervals.
    """
    # Sum loss categories
    total_loss = (
        soiling + shading + snow + mismatch + wiring +
        connections + lid + nameplate + age + availability
    )

    # Configure and execute PVWatts
    model = pv.default('PVWattsNone')
    model.SolarResource.solar_resource_file = str(weather_file)
    sd = model.SystemDesign
    sd.system_capacity = system_capacity_kw
    sd.dc_ac_ratio     = dc_ac_ratio
    sd.inv_eff         = inv_eff
    sd.tilt            = tilt
    sd.azimuth         = azimuth
    sd.array_type      = array_type
    sd.module_type     = module_type
    sd.gcr             = gcr
    sd.losses          = total_loss
    model.execute()

    # Hourly AC output (W) → kWh
    ac_kwh = np.array(model.Outputs.ac) / 1000.0

    # Build hourly index for the given year (freq='h' now)
    idx_hour = pd.date_range(
        datetime(start_year, 1, 1),
        periods=len(ac_kwh),
        freq='h'
    )

    # Build half-hour index and split values
    idx_half = pd.date_range(idx_hour[0], periods=len(ac_kwh)*2, freq='30min')
    half_kwh = np.repeat(ac_kwh / 2, 2)

    df_half = pd.DataFrame({'simulated_kwh': half_kwh}, index=idx_half)
    return df_half[df_half.index.year == start_year]


def simulate_total_pv(
    weather_file: str,
    roof_params: list,
    start_year: int = 2024
) -> (pd.DataFrame, dict):
    """
    Simulate multiple PV arrays (roofs) for one representative year.
    Returns:
      total_df    - half-hourly summed PV generation DataFrame for one year.
      roof_outputs - dict mapping roof name to its individual DataFrame.
    """
    total_df = None
    roof_outputs = {}

    for params in roof_params:
        name = params.get('name', f"roof_{len(roof_outputs)+1}")
        df = simulate_pv_simple(
            weather_file,
            system_capacity_kw=params.get('system_capacity_kw', 0),
            tilt=params.get('tilt', 0),
            azimuth=params.get('azimuth', 180),
            soiling=params.get('soiling', 2.0),
            shading=params.get('shading', 0.0),
            snow=params.get('snow', 0.0),
            mismatch=params.get('mismatch', 2.0),
            wiring=params.get('wiring', 2.0),
            connections=params.get('connections', 0.5),
            lid=params.get('lid', 1.5),
            nameplate=params.get('nameplate', 1.0),
            age=params.get('age', 0.0),
            availability=params.get('availability', 3.0),
            dc_ac_ratio=params.get('dc_ac_ratio', 1.15),
            inv_eff=params.get('inv_eff', 96.0),
            array_type=params.get('array_type', 1),
            module_type=params.get('module_type', 0),
            gcr=params.get('gcr', 0.3),
            start_year=start_year
        )
        roof_outputs[name] = df
        if total_df is None:
            total_df = df.copy()
        else:
            total_df['simulated_kwh'] += df['simulated_kwh']

    return total_df, roof_outputs


def simulate_multi_year_pv(
    weather_files: list,
    roof_params: list,
    repeats_per_file: int = 10,
    start_years: list = None
) -> pd.DataFrame:
    """
    Build a multi-decade half-hourly PV generation profile by repeating
    1-year simulations of representative weather files.

    For 3 files × 10 years each, returns 30 years at 17,520 * 30 = 525,600 intervals.
    Applies a continuous datetime index from the first start_year onward.
    """
    if start_years is None:
        start_years = [
            int(''.join(filter(str.isdigit, Path(wf).stem)))
            for wf in weather_files
        ]

    segments = []
    for wf, sy in zip(weather_files, start_years):
        total_df, _ = simulate_total_pv(wf, roof_params, start_year=sy)
        # tile for repeats_per_file years
        decade = pd.concat([total_df.copy() for _ in range(repeats_per_file)], ignore_index=True)
        segments.append(decade)

    full = pd.concat(segments, ignore_index=True)

    # Build a continuous 30-min datetime index (’h’ not needed here)
    full_start = datetime(start_years[0], 1, 1)
    full_index = pd.date_range(full_start, periods=len(full), freq='30min')
    full.index = full_index

    # Validate length: 525,600 for 30 years
    assert len(full) == len(weather_files) * repeats_per_file * 17520

    return full
