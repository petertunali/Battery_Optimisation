#pv.py

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
    start_year: int = 2025,
) -> pd.DataFrame:
    """
    Run a PVWatts simulation and return one year of half-hourly kWh.
    Each hourly output is evenly split into two 30‑min intervals.
    """
    # sum all loss components
    total_loss = (soiling + shading + snow + mismatch
                  + wiring + connections + lid
                  + nameplate + age + availability)

    # configure & run PVWatts
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

    # get hourly output in kWh
    ac_kwh = np.array(model.Outputs.ac) / 1000.0

    # build an hourly index for that start_year
    idx_hour = pd.date_range(
        datetime(start_year, 1, 1),
        periods=len(ac_kwh),
        freq='h'           # lowercase 'h' to avoid warnings
    )

    # split each hourly kWh into two half‑hours
    half_index = pd.date_range(idx_hour[0],
                                periods=len(ac_kwh)*2,
                                freq='30min')
    half_kwh   = np.repeat(ac_kwh/2, 2)

    df = pd.DataFrame({'simulated_kwh': half_kwh}, index=half_index)
    return df[df.index.year == start_year]


def simulate_total_pv(
    weather_file: str,
    roof_params: list,
    start_year: int = 2024
) -> (pd.DataFrame, dict):
    """
    Simulate each roof in roof_params for one year, then sum them.
    Returns (total_df, dict of individual roof dfs).
    """
    total_df = None
    roof_outputs = {}

    for params in roof_params:
        name = params.get('name', f"roof_{len(roof_outputs)+1}")
        df = simulate_pv_simple(
            weather_file=weather_file,
            system_capacity_kw=params['system_capacity_kw'],
            tilt=params['tilt'],
            azimuth=params['azimuth'],
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
    Build a full 30‑year half-hourly PV profile by:
      1) simulating each weather_file once,
      2) repeating that one‑year output `repeats_per_file` times,
      3) concatenating all segments,
      4) stamping a continuous 30‑min index from the first start_year.
    """
    # infer start_years from filenames if not provided (take first 4 digits)
    if start_years is None:
        start_years = [
            int(''.join(filter(str.isdigit, Path(wf).stem))[:4])
            for wf in weather_files
        ]

    segments = []
    for wf, sy in zip(weather_files, start_years):
        one_year_df, _ = simulate_total_pv(wf, roof_params, start_year=sy)
        # repeat that year `repeats_per_file` times (default=10)
        for _ in range(repeats_per_file):
            segments.append(one_year_df.copy())

    full = pd.concat(segments, ignore_index=True)

    # build a continuous datetime index
    idx = pd.date_range(
        datetime(start_years[0], 1, 1),
        periods=len(full),
        freq='30min'
    )
    full.index = idx

    # sanity check: 3 files × 10 repeats × 17,520 half‑hours = 525,600
    expected = len(weather_files) * repeats_per_file * 17520
    assert len(full) == expected, f"expected {expected} intervals, got {len(full)}"

    return full
