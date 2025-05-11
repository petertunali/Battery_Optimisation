import pandas as pd
import numpy as np

def simulate_battery_dispatch(
    pv_gen: pd.Series,
    demand: pd.Series,
    battery_kwh: float,
    battery_kw: float = None,
    roundtrip_eff: float = 0.9,      # 90% round-trip efficiency
    min_soc_pct: float = 0.05,       # 95% depth of discharge
    annual_deg_rate: float = 0.01,   # 1% annual degradation
    grid_emission_rate: float = 0.81  # 0.81 kg CO2e per kWh
) -> (pd.DataFrame, dict):
    """
    Simulate half‑hourly battery dispatch over a multi‑year profile with simple degradation model.
    
    Parameters
    ----------
    pv_gen : pd.Series
        Half‑hourly PV generation [kWh].
    demand : pd.Series
        Half‑hourly load [kWh].
    battery_kwh : float
        Usable energy capacity [kWh].
    battery_kw : float, optional
        Power rating [kW]. Defaults to 0.5·battery_kwh.
    roundtrip_eff : float
        Round‑trip efficiency (fraction).
    min_soc_pct : float
        Minimum state‑of‑charge as fraction of capacity.
    annual_deg_rate : float
        Annual degradation rate (fraction).
    grid_emission_rate : float
        kgCO2e emitted per kWh imported from grid.
    
    Returns
    -------
    df : pd.DataFrame
        Half‑hourly columns:
          - pv_gen, demand, pv_used
          - battery_charge, battery_discharge, battery_soc
          - pv_export, grid_import_peak, grid_import_offpeak
          - battery_capacity
    totals : dict
        Aggregated metrics including battery cycling metrics.
    """
    # 1) Align series
    pv, dem = pv_gen.align(demand, join='inner')

    # Special case for no battery (handles zero capacity)
    if battery_kwh <= 0:
        # Create dataframe with zeros for battery columns
        df = pd.DataFrame({
            'pv_gen': pv.values,
            'demand': dem.values,
            'pv_used': np.minimum(pv.values, dem.values),
            'battery_charge': np.zeros_like(pv.values),
            'battery_discharge': np.zeros_like(pv.values),
            'battery_soc': np.zeros_like(pv.values),
            'pv_export': np.maximum(pv.values - dem.values, 0),
            'grid_import_peak': np.zeros_like(pv.values),
            'grid_import_offpeak': np.zeros_like(pv.values),
            'battery_capacity': np.zeros_like(pv.values)
        }, index=pv.index)
        
        # For each timestamp, determine if peak or offpeak and set grid import accordingly
        for i, ts in enumerate(pv.index):
            m, d, h = ts.month, ts.day, ts.hour
            if (m < 4 or (m == 4 and d <= 1)) or (m > 10 or (m == 10 and d >= 2)):
                peak_start, peak_end = 14, 20
            else:
                peak_start, peak_end = 15, 21
            is_peak = peak_start <= h < peak_end
            
            deficit = max(dem.iloc[i] - pv.iloc[i], 0)
            if is_peak:
                df.iloc[i, df.columns.get_loc('grid_import_peak')] = deficit
            else:
                df.iloc[i, df.columns.get_loc('grid_import_offpeak')] = deficit
        
        # Calculate totals
        totals = {
            'total_demand': df['demand'].sum(),
            'total_pv_used': df['pv_used'].sum(),
            'total_battery_discharge': 0.0,
            'total_grid_import_peak': df['grid_import_peak'].sum(),
            'total_grid_import_offpeak': df['grid_import_offpeak'].sum(),
            'total_pv_export': df['pv_export'].sum(),
            'total_grid_emissions': (df['grid_import_peak'].sum() + df['grid_import_offpeak'].sum()) * grid_emission_rate,
            'renewable_fraction': df['pv_used'].sum() / df['demand'].sum() if df['demand'].sum() > 0 else 0.0,
            'grid_fraction': (df['grid_import_peak'].sum() + df['grid_import_offpeak'].sum()) / df['demand'].sum() if df['demand'].sum() > 0 else 0.0,
            'self_consumption_rate': df['pv_used'].sum() / (df['pv_used'].sum() + df['pv_export'].sum()) if (df['pv_used'].sum() + df['pv_export'].sum()) > 0 else 0.0,
            'initial_battery_capacity': 0.0,
            'final_battery_capacity': 0.0,
            'total_battery_degradation_pct': 0.0,
            'battery_total_cycles': 0.0,
            'battery_average_dod': 0.0,
            'final_degradation_pct': 0.0
        }
        
        return df, totals

    # 2) Default power rating = 0.5C if not given
    if battery_kw is None:
        battery_kw = 0.5 * battery_kwh

    # half‑hour interval in hours
    delta_h = (pv.index[1] - pv.index[0]).total_seconds() / 3600.0

    # split round‐trip efficiency
    eff_chg = np.sqrt(roundtrip_eff)
    eff_dis = np.sqrt(roundtrip_eff)

    # intervals per year (half‑hourly)
    ints_per_yr = 365 * 24 * 2
    
    # degradation per interval (linear 1% per year)
    deg_per_interval = annual_deg_rate / ints_per_yr
    
    # Total simulation years
    total_years = len(pv) / ints_per_yr
    
    # 3) initialize variables
    soc = battery_kwh * min_soc_pct  # Start at minimum SOC
    initial_capacity = battery_kwh    # Initial battery capacity
    capacity = initial_capacity       # Current battery capacity
    
    # Initialize tracking variables
    total_discharge = 0.0
    total_cycles = 0.0

    # 4) prepare storage
    cols = {c: [] for c in [
        'pv_gen','demand','pv_used',
        'battery_charge','battery_discharge','battery_soc',
        'pv_export','grid_import_peak','grid_import_offpeak',
        'battery_capacity'
    ]}

    # 5) loop through each timestep
    for i, (ts, pv_val, dem_val) in enumerate(zip(pv.index, pv, dem)):
        # 5a) Apply gradual battery degradation
        capacity *= (1.0 - deg_per_interval)
        
        # Cap SOC to current capacity
        soc = min(soc, capacity)

        # 5b) peak/off‑peak?
        m, d, h = ts.month, ts.day, ts.hour
        if (m < 4 or (m == 4 and d <= 1)) or (m > 10 or (m == 10 and d >= 2)):
            peak_start, peak_end = 14, 20
        else:
            peak_start, peak_end = 15, 21
        is_peak = peak_start <= h < peak_end

        # 5c) PV → load
        pv_used = min(pv_val, dem_val)
        surplus = pv_val - pv_used
        deficit = dem_val - pv_used

        # 5d) charge battery
        max_usable_capacity = capacity * (1 - min_soc_pct)
        if eff_chg > 0:
            charge = min(surplus, battery_kw * delta_h, (capacity - soc) / eff_chg)
        else:
            charge = 0
        
        soc += charge * eff_chg
        surplus -= charge

        # 5e) export any leftover PV
        export = surplus

        # 5f) discharge battery to cover deficit
        avail = (soc - capacity * min_soc_pct) * eff_dis
        discharge = min(deficit, battery_kw * delta_h, avail)
        
        if eff_dis > 0:
            soc -= discharge / eff_dis
        deficit -= discharge
        
        # Track discharge for cycle counting
        if capacity > 0:
            total_discharge += discharge
            # Update cycle count (equivalent full cycles)
            total_cycles = total_discharge / initial_capacity

        # 5g) import remaining from grid
        grid = deficit
        peak_imp = grid if is_peak else 0.0
        off_imp = grid if not is_peak else 0.0

        # 5h) record results
        cols['pv_gen'].append(pv_val)
        cols['demand'].append(dem_val)
        cols['pv_used'].append(pv_used)
        cols['battery_charge'].append(charge)
        cols['battery_discharge'].append(discharge)
        cols['battery_soc'].append(soc)
        cols['pv_export'].append(export)
        cols['grid_import_peak'].append(peak_imp)
        cols['grid_import_offpeak'].append(off_imp)
        cols['battery_capacity'].append(capacity)

    # 6) build DataFrame
    df = pd.DataFrame(cols, index=pv.index)

    # 7) aggregate totals
    total_demand = df['demand'].sum()
    total_pv_used = df['pv_used'].sum()
    total_batt_discharge = df['battery_discharge'].sum()
    total_imp_peak = df['grid_import_peak'].sum()
    total_imp_offpeak = df['grid_import_offpeak'].sum()
    total_import = total_imp_peak + total_imp_offpeak
    total_export = df['pv_export'].sum()
    total_emissions = total_import * grid_emission_rate
    
    # Calculate final degradation percentage
    final_degradation_pct = (initial_capacity - capacity) / initial_capacity * 100 if initial_capacity > 0 else 0

    # 8) compute shares
    renewable_supplied = total_pv_used + total_batt_discharge
    renewable_fraction = renewable_supplied / total_demand if total_demand > 0 else 0.0
    grid_fraction = total_import / total_demand if total_demand > 0 else 0.0
    self_consumption_rate = total_pv_used / (total_pv_used + total_export) if (total_pv_used + total_export) > 0 else 0.0

    totals = {
        'total_demand': total_demand,
        'total_pv_used': total_pv_used,
        'total_battery_discharge': total_batt_discharge,
        'total_grid_import_peak': total_imp_peak,
        'total_grid_import_offpeak': total_imp_offpeak,
        'total_pv_export': total_export,
        'total_grid_emissions': total_emissions,
        'renewable_fraction': renewable_fraction,
        'grid_fraction': grid_fraction,
        'self_consumption_rate': self_consumption_rate,
        'initial_battery_capacity': initial_capacity,
        'final_battery_capacity': capacity,
        'final_degradation_pct': final_degradation_pct,
        'battery_total_cycles': total_cycles,
        'battery_average_dod': 0.95  # Assuming average DoD is close to max (95% in this case)
    }

    return df, totals