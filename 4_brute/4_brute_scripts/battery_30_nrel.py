import pandas as pd
import numpy as np
import math

def simulate_battery_dispatch(
    pv_gen: pd.Series,
    demand: pd.Series,
    battery_kwh: float,
    battery_kw: float = None,
    roundtrip_eff: float = 0.9,      # 90% round-trip efficiency
    min_soc_pct: float = 0.05,       # 95% depth of discharge
    grid_emission_rate: float = 0.81, # 0.81 kg CO2e per kWh
    # NREL battery parameters
    calendar_life_years: float = 15.0,  # Calendar life in years
    cycle_life_cycles: float = 3500.0,  # Cycle life at reference DoD
    reference_dod: float = 0.8,        # Reference depth of discharge
    temp_ref_degC: float = 25.0,       # Reference temperature in Celsius
    operating_temp_degC: float = 25.0  # Operating temperature in Celsius
) -> (pd.DataFrame, dict):
    """
    Simulate half‑hourly battery dispatch over a multi‑year profile with NREL-style degradation,
    accounting for both calendar aging and cycle aging.
    
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
    grid_emission_rate : float
        kgCO2e emitted per kWh imported from grid.
    calendar_life_years : float
        Calendar life in years at reference temperature.
    cycle_life_cycles : float
        Cycle life in equivalent full cycles at reference DoD.
    reference_dod : float
        Reference depth of discharge for cycle life rating.
    temp_ref_degC : float
        Reference temperature for calendar aging in Celsius.
    operating_temp_degC : float
        Operating temperature in Celsius.
    
    Returns
    -------
    df : pd.DataFrame
        Half‑hourly columns:
          - pv_gen, demand, pv_used
          - battery_charge, battery_discharge, battery_soc
          - pv_export, grid_import_peak, grid_import_offpeak
          - battery_capacity, calendar_degradation, cycle_degradation
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
            'battery_capacity': np.zeros_like(pv.values),
            'calendar_degradation': np.zeros_like(pv.values),
            'cycle_degradation': np.zeros_like(pv.values)
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
            'calendar_degradation_pct': 0.0,
            'cycle_degradation_pct': 0.0,
            'battery_total_cycles': 0.0,
            'battery_average_dod': 0.0
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
    
    # Total simulation years
    total_years = len(pv) / ints_per_yr
    
    # 3) initialize variables
    soc = battery_kwh * min_soc_pct  # Start at minimum SOC
    initial_capacity = battery_kwh    # Initial battery capacity
    capacity = initial_capacity       # Current battery capacity
    
    # NREL degradation model parameters
    # Simplified temperature factor (25°C reference)
    # For every 10°C above reference, calendar aging doubles
    temp_factor = 2**((operating_temp_degC - temp_ref_degC)/10)
    
    # Calendar degradation per interval
    cal_deg_per_interval = (1 - math.exp(-math.log(2)/calendar_life_years)) / ints_per_yr * temp_factor
    
    # Cycle degradation parameters
    # For every equivalent full cycle (EFC), this much capacity is lost
    cycle_deg_per_efc = 1 / cycle_life_cycles
    
    # Initialize tracking variables
    calendar_degradation = 0.0
    cycle_degradation = 0.0
    total_discharge = 0.0
    total_cycles = 0.0
    cycle_history = []  # Track depth of each cycle for weighted degradation

    # 4) prepare storage
    cols = {c: [] for c in [
        'pv_gen','demand','pv_used',
        'battery_charge','battery_discharge','battery_soc',
        'pv_export','grid_import_peak','grid_import_offpeak',
        'battery_capacity','calendar_degradation','cycle_degradation'
    ]}

    # 5) loop through each timestep
    for i, (ts, pv_val, dem_val) in enumerate(zip(pv.index, pv, dem)):
        # 5a) Calendar degradation (happens every interval)
        # NREL model: Higher SOC accelerates calendar degradation
        if capacity > 0:
            soc_factor = 1.0 + 0.5 * max(0, (soc / capacity - min_soc_pct)) / max(0.001, (1 - min_soc_pct))  # SOC stress factor with safeguards
        else:
            soc_factor = 1.0  # Default when capacity is zero
            
        cal_deg_this_interval = cal_deg_per_interval * soc_factor * capacity
        calendar_degradation += cal_deg_this_interval
        capacity -= cal_deg_this_interval
        
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
            
        soc_before_charge = soc
        soc += charge * eff_chg
        surplus -= charge

        # 5e) export any leftover PV
        export = surplus

        # 5f) discharge battery to cover deficit
        avail = (soc - capacity * min_soc_pct) * eff_dis
        discharge = min(deficit, battery_kw * delta_h, avail)
        soc_before_discharge = soc
        
        if eff_dis > 0:
            soc -= discharge / eff_dis
        deficit -= discharge
        
        # 5g) Track cycle degradation 
        # Only count if we have meaningful capacity
        if capacity > 0.01 * initial_capacity and capacity > 0:
            # Calculate normalized DoD for this half-cycle
            if discharge > 0:
                dod = (soc_before_discharge - soc) / capacity  # Now safe since we checked capacity > 0
                total_discharge += discharge
                
                # NREL-style cycle degradation: Weighted by DoD relative to reference
                # Higher DoD causes more damage per energy throughput
                if reference_dod > 0:
                    dod_factor = (dod / reference_dod)**1.2
                else:
                    dod_factor = 0
                
                # Calculate equivalent full cycles (EFC) for this discharge
                efc_this_discharge = discharge / capacity * dod_factor  # Now safe since we checked capacity > 0
                
                # Calculate capacity loss from this cycle
                cycle_deg_this_discharge = efc_this_discharge * cycle_deg_per_efc * capacity
                cycle_degradation += cycle_deg_this_discharge
                capacity -= cycle_deg_this_discharge
                
                # Update cycle count
                total_cycles += discharge / capacity  # Now safe since we checked capacity > 0
                
                # Add to cycle history for tracking
                cycle_history.append((dod, discharge))

        # 5h) import remaining from grid
        grid = deficit
        peak_imp = grid if is_peak else 0.0
        off_imp = grid if not is_peak else 0.0

        # 5i) record results
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
        cols['calendar_degradation'].append(calendar_degradation)
        cols['cycle_degradation'].append(cycle_degradation)

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
    
    # Calculate final degradation percentage with safety checks
    initial_deg = 0.0
    if initial_capacity > 0:
        final_deg = (initial_capacity - capacity) / initial_capacity * 100
        cal_deg_pct = calendar_degradation / initial_capacity * 100
        cyc_deg_pct = cycle_degradation / initial_capacity * 100
    else:
        final_deg = 0.0
        cal_deg_pct = 0.0
        cyc_deg_pct = 0.0

    # 8) compute shares with safety checks
    renewable_supplied = total_pv_used + total_batt_discharge
    renewable_fraction = renewable_supplied / total_demand if total_demand > 0 else 0.0
    grid_fraction = total_import / total_demand if total_demand > 0 else 0.0
    
    # Self-consumption rate with safety check
    denominator = total_pv_used + total_export
    self_consumption_rate = total_pv_used / denominator if denominator > 0 else 0.0
    
    # Average cycling depth with safety check
    if cycle_history and sum(discharge for _, discharge in cycle_history) > 0:
        avg_dod = sum(dod * discharge for dod, discharge in cycle_history) / sum(discharge for _, discharge in cycle_history)
    else:
        avg_dod = 0.0

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
        'total_battery_degradation_pct': final_deg,
        'calendar_degradation_pct': cal_deg_pct,
        'cycle_degradation_pct': cyc_deg_pct,
        'battery_total_cycles': total_cycles,
        'battery_average_dod': avg_dod
    }

    return df, totals