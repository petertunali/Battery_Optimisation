# pv.py - PV simulation functions for battery optimization
import numpy as np
import pandas as pd
import PySAM.Pvwattsv8 as pv
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

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
    # If system capacity is zero, return zeros
    if system_capacity_kw <= 0:
        idx = pd.date_range(
            datetime(start_year, 1, 1),
            periods=8760*2,  # 8760 hours in a year, times 2 for half-hourly
            freq='30min'
        )
        return pd.DataFrame({'simulated_kwh': np.zeros(len(idx))}, index=idx)
        
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


def allocate_pv_capacity(total_capacity_kw: float, options: list) -> list:
    """
    Allocate PV capacity across available options based on priority.
    
    Args:
        total_capacity_kw: Total additional PV capacity to allocate
        options: List of PV options with max_capacity_kw and other parameters
        
    Returns:
        allocated_pv: List of PV configurations with allocated capacity
    """
    remaining_capacity = total_capacity_kw
    allocated_pv = []
    
    # If total capacity is zero or negative, return empty list
    if total_capacity_kw <= 0:
        return allocated_pv
    
    # Try to import config for prioritization setting
    try:
        from config import PRIORITIZE_PV_ALLOCATION
        prioritize = PRIORITIZE_PV_ALLOCATION
    except ImportError:
        # Default to True if not found in config
        prioritize = True
    
    # Allocate in order of options list (prioritized approach)
    if prioritize:
        for option in options:
            option_copy = option.copy()
            # Allocate capacity to this option (limited by max capacity)
            allocation = min(remaining_capacity, option['max_capacity_kw'])
            
            if allocation > 0:
                option_copy['system_capacity_kw'] = allocation
                allocated_pv.append(option_copy)
                remaining_capacity -= allocation
            
            if remaining_capacity <= 0:
                break
    else:
        # If prioritization is disabled, distribute proportionally
        # This is an alternative allocation strategy
        total_max = sum(opt['max_capacity_kw'] for opt in options)
        for option in options:
            option_copy = option.copy()
            
            # If unlimited capacity (inf), allocate remaining after others
            if option['max_capacity_kw'] == float('inf'):
                continue
                
            # Calculate proportional allocation
            proportion = option['max_capacity_kw'] / total_max
            allocation = min(proportion * total_capacity_kw, option['max_capacity_kw'])
            
            if allocation > 0:
                option_copy['system_capacity_kw'] = allocation
                allocated_pv.append(option_copy)
                remaining_capacity -= allocation
        
        # Allocate any remaining to unlimited options
        for option in options:
            if option['max_capacity_kw'] == float('inf') and remaining_capacity > 0:
                option_copy = option.copy()
                option_copy['system_capacity_kw'] = remaining_capacity
                allocated_pv.append(option_copy)
                remaining_capacity = 0
    
    return allocated_pv


def load_demand_profile(csv_path, config=None):
    """
    Load demand profile from CSV file.
    
    Args:
        csv_path: Path to the CSV file
        config: Configuration parameters
        
    Returns:
        pandas.Series: Timestamp-indexed Series with consumption values
    """
    # Try to load column names from config
    timestamp_col = "Date and Time"  # Default
    consumption_col = "Consumption (kWh)"  # Default
    
    if config is not None:
        try:
            timestamp_col = config.DEMAND_FILE_CONFIG["columns"]["timestamp"]
            consumption_col = config.DEMAND_FILE_CONFIG["columns"]["consumption"]
        except (AttributeError, KeyError):
            pass  # Use defaults if not found
    
    # Load the CSV file
    raw = pd.read_csv(csv_path, parse_dates=[timestamp_col], dayfirst=True)
    
    # Drop rows with NaN timestamps right away
    raw = raw.dropna(subset=[timestamp_col])
    
    # Get the consumption column - try to handle various names
    if consumption_col not in raw.columns:
        # Try to find the consumption column
        for col in raw.columns:
            if 'consum' in col.lower() or 'demand' in col.lower():
                consumption_col = col
                break
        else:
            # If no consumption column found, use the second column
            consumption_col = raw.columns[1]
    
    # Create a Series with timestamp index and consumption values
    s = pd.Series(raw[consumption_col].values, index=raw[timestamp_col])
    
    # Check for duplicate timestamps
    dup_count = s.index.duplicated().sum()
    if dup_count:
        s = s[~s.index.duplicated(keep='first')]
    
    # Build the expected half-hour index for entire year (no Feb 29)
    year = s.index.min().year
    start = pd.Timestamp(year, 1, 1, 0, 0)
    end = pd.Timestamp(year, 12, 31, 23, 30)
    expected = pd.date_range(start, end, freq="30min")
    expected = expected[~((expected.month==2) & (expected.day==29))]
    
    # Reindex to ensure complete coverage
    s = s.reindex(expected)
    missing = s.isna().sum()
    if missing:
        s = s.fillna(0.0)
    
    # Final sanity check
    assert len(s) == 17520, f"Got {len(s)} points, expected 17520"
    return s


def create_30_year_profile(one_year_series, years=30, start_year=2025, apply_growth=False):
    """
    Create a 30-year profile from a 1-year series.
    
    Args:
        one_year_series: Base year demand series
        years: Number of years to create
        start_year: Starting year for the profile
        apply_growth: If True, apply demand growth according to config settings
        
    Returns:
        Pandas Series with the multi-year profile
    """
    all_data = []
    
    # Try to import demand growth parameters from config
    growth_enabled = False
    growth_pattern = 'decade'
    growth_pct = 0
    
    if apply_growth:
        try:
            from config import DEMAND_GROWTH
            growth_enabled = DEMAND_GROWTH.get('enabled', False)
            growth_pattern = DEMAND_GROWTH.get('pattern', 'decade')
            growth_pct = DEMAND_GROWTH.get('percent_per_decade', 0)
        except ImportError:
            pass
    
    for year_offset in range(years):
        # Copy the data for this year
        year_data = one_year_series.copy()
        
        # Apply demand growth if enabled
        if growth_enabled:
            decade = year_offset // 10  # Calculate which decade we're in
            
            if growth_pattern == 'decade':
                # Apply growth by decade (e.g., years 0-9 = 1x, years 10-19 = 1.1x, etc.)
                growth_multiplier = 1 + (decade * growth_pct / 100)
                year_data = year_data * growth_multiplier
                
            elif growth_pattern == 'annual':
                # Apply compounding annual growth
                annual_rate = (1 + growth_pct / 100) ** (1/10) - 1
                growth_multiplier = (1 + annual_rate) ** year_offset
                year_data = year_data * growth_multiplier
        
        # Create index for this specific year
        target_year = start_year + year_offset
        year_start = pd.Timestamp(target_year, 1, 1, 0, 0)
        year_end = pd.Timestamp(target_year, 12, 31, 23, 30)
        year_range = pd.date_range(start=year_start, end=year_end, freq="30min")
        
        # Remove Feb 29 if it's a leap year
        year_range = year_range[~((year_range.month == 2) & (year_range.day == 29))]
        
        # Make sure it has the right number of points
        assert len(year_range) == len(one_year_series), f"Year {target_year} has {len(year_range)} points, expected {len(one_year_series)}"
        
        # Assign the new index and add to our list
        year_data.index = year_range
        all_data.append(year_data)
    
    # Concatenate all years
    return pd.concat(all_data)