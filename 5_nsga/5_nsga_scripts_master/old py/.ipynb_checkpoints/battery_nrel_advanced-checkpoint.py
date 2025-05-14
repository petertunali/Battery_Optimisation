# battery_advanced.py - Advanced battery control strategy with peak/off-peak awareness
import pandas as pd
import numpy as np
import PySAM.BatteryStateful as battery_model

def simulate_battery_dispatch(
    pv_gen: pd.Series,
    demand: pd.Series,
    battery_kwh: float,
    battery_kw: float = None,
    roundtrip_eff: float = 0.9,
    min_soc_pct: float = 0.05,  # 95% depth of discharge
    annual_deg_rate: float = 0.01,  # Only used as fallback
    grid_emission_rate: float = 0.79,  # 0.79 kg CO2e per kWh
    peak_reserve_soc: float = 0.8,  # Reserve level for peak periods (80%)
    peak_reserve_hours: int = 2,   # Hours before peak to start reserving capacity
    off_peak_min_soc: float = 0.6  # Minimum SOC during off-peak (60%)
) -> (pd.DataFrame, dict):
    """
    Simulate half‑hourly battery dispatch with advanced peak-time awareness.
    Battery reserves capacity for peak time periods and maintains a higher
    minimum SOC during off-peak periods.
    
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
        Round‑trip efficiency (fraction). Default: 0.9 (90%).
    min_soc_pct : float
        Absolute minimum state‑of‑charge as fraction of capacity. Default: 0.05 (95% DoD).
    annual_deg_rate : float
        Fallback degradation rate if NREL model fails. Default: 0.01 (1% per year).
    grid_emission_rate : float
        kgCO2e emitted per kWh imported from grid. Default: 0.79 kg CO2e/kWh.
    peak_reserve_soc : float
        Target SOC to reserve for peak periods. Default: 0.8 (80%).
    peak_reserve_hours : int
        Hours before peak period to start reserving capacity. Default: 2.
    off_peak_min_soc : float
        Minimum SOC allowed during off-peak periods. Default: 0.6 (60%).
        
    Returns
    -------
    df : pd.DataFrame
        Half‑hourly columns for all energy flows and battery states.
    totals : dict
        Aggregated metrics including renewable fraction and cycle count.
    """
    # 1) Align series
    pv, dem = pv_gen.align(demand, join='inner')

    # 2) Default power rating = 0.5C if not given
    if battery_kw is None:
        battery_kw = 0.5 * battery_kwh

    # Set up half‑hour interval in hours
    delta_h = (pv.index[1] - pv.index[0]).total_seconds() / 3600.0
    
    # Split round‐trip efficiency
    eff_chg = np.sqrt(roundtrip_eff)
    eff_dis = np.sqrt(roundtrip_eff)

    # Intervals per year (half‑hourly)
    ints_per_yr = 365 * 24 * 2
    
    # 3) Initialize PySAM battery model if battery capacity > 0
    use_nrel_model = battery_kwh > 0
    if use_nrel_model:
        try:
            # Create PySAM battery model
            bat = battery_model.default('BatteryStateful')
            
            # Set battery parameters
            bat.Battery.batt_chem = 1  # 1 = Lithium-ion
            bat.Battery.batt_computed_design_capacity = battery_kwh
            bat.Battery.batt_power_discharge_max_kwac = battery_kw
            bat.Battery.batt_power_charge_max_kwac = battery_kw
            bat.Battery.batt_minimum_state_of_charge = min_soc_pct * 100  # Convert to percentage
            bat.Battery.batt_efficiency_cutoff_lookback = 1  # Hours
            
            # Set up degradation model
            bat.Battery.batt_calendar_choice = 1  # Enable calendar degradation
            bat.Battery.batt_cycle_choice = 1     # Enable cycle degradation
            
            # Default values for lithium-ion
            bat.Battery.batt_calendar_q0 = 1.02   # Initial capacity
            bat.Battery.batt_calendar_a = 0.2     # Calendar life parameter
            bat.Battery.batt_calendar_b = 2.25    # Calendar life parameter
            bat.Battery.batt_calendar_c = 4300    # Calendar life parameter
            
            # Cycle life parameters
            bat.Battery.batt_cycle_q0 = 1.05      # Initial capacity
            bat.Battery.batt_cycle_k = 0.00038    # Cycle life parameter
            bat.Battery.batt_cycle_c = 1.06       # Cycle life parameter
            bat.Battery.batt_cycle_k_temp = 0.005 # Temperature factor
            bat.Battery.batt_cycle_k_soc = 0.08   # SOC factor
            
            # Track initial capacity for degradation calculation
            initial_capacity = bat.Battery.batt_computed_design_capacity
            print(f"NREL PySAM battery model initialized with {initial_capacity} kWh capacity")
        except Exception as e:
            print(f"Warning: Could not initialize NREL battery model: {e}")
            print("Falling back to simple degradation model.")
            use_nrel_model = False
    
    # 4) Initialize SOC at minimum level
    soc = battery_kwh * min_soc_pct

    # 5) Prepare storage
    cols = {c: [] for c in [
        'pv_gen', 'demand', 'pv_used',
        'battery_charge', 'battery_discharge', 'battery_soc',
        'pv_export', 'grid_import_peak', 'grid_import_offpeak',
        'is_peak', 'time_to_peak'  # Add these columns to track peak periods
    ]}
    
    # For tracking battery cycles and degradation
    total_charge = 0
    total_discharge = 0
    cycle_count = 0
    
    # Create arrays for tracking capacity and degradation over time
    years = len(pv) // ints_per_yr + 1
    capacity_over_time = np.ones(years) * battery_kwh
    
    # Pre-compute peak periods for the entire timespan
    peak_periods = []
    for ts in pv.index:
        m, d, h = ts.month, ts.day, ts.hour
        if (m < 4 or (m == 4 and d <= 1)) or (m > 10 or (m == 10 and d >= 2)):
            peak_start, peak_end = 14, 20  # Winter peak: 2pm to 8pm
        else:
            peak_start, peak_end = 15, 21  # Summer peak: 3pm to 9pm
        
        is_peak = peak_start <= h < peak_end
        peak_periods.append(is_peak)
    
    # Calculate time to next peak period for each timestep
    time_to_peak = []
    for i, ts in enumerate(pv.index):
        # If current timestep is already in a peak period
        if peak_periods[i]:
            time_to_peak.append(0)
            continue
            
        # Find the next peak period
        hours_to_peak = float('inf')
        for j in range(i+1, min(i+49, len(peak_periods))):  # Look ahead max 24 hours (48 half-hour intervals)
            if peak_periods[j]:
                hours_to_peak = (j - i) * delta_h
                break
                
        time_to_peak.append(hours_to_peak)
    
    # 6) Main simulation loop - process each timestep
    for i, (ts, pv_val, dem_val) in enumerate(zip(pv.index, pv, dem)):
        # 6a) Get current battery capacity with degradation
        year = i // ints_per_yr
        
        if use_nrel_model and i % (ints_per_yr // 12) == 0:  # Update monthly to save computation
            try:
                # Update degradation based on cycles and calendar aging
                cycles_so_far = total_charge / battery_kwh if battery_kwh > 0 else 0
                days_so_far = i * delta_h / 24
                
                # Update the NREL model
                bat.Battery.batt_cycle_number = cycles_so_far
                bat.Battery.batt_calendar_day = days_so_far
                bat.execute()
                
                # Get degraded capacity
                degraded_percent = bat.Battery.batt_capacity_percent
                cur_capacity = battery_kwh * degraded_percent
                # Store for reporting
                capacity_over_time[year] = cur_capacity
            except Exception as e:
                # If NREL model fails, fall back to simple model
                print(f"Warning: NREL battery model failed: {e}")
                print("Falling back to simple degradation model.")
                use_nrel_model = False
                cur_capacity = battery_kwh * ((1 - annual_deg_rate)**year)
                capacity_over_time[year] = cur_capacity
        elif not use_nrel_model:
            # Use simple degradation model as fallback
            cur_capacity = battery_kwh * ((1 - annual_deg_rate)**year)
            capacity_over_time[year] = cur_capacity
        else:
            # Use last computed capacity between updates
            cur_capacity = capacity_over_time[year]
            
        # Ensure SOC doesn't exceed current capacity
        soc = min(soc, cur_capacity)

        # 6b) Determine peak/off‑peak periods
        is_peak = peak_periods[i]
        hours_to_peak = time_to_peak[i]
        
        # Determine effective minimum SOC based on peak/off-peak strategy
        if is_peak:
            # During peak, can discharge down to absolute minimum
            effective_min_soc = cur_capacity * min_soc_pct
        elif hours_to_peak <= peak_reserve_hours:
            # Approaching peak time, try to charge to target level
            effective_min_soc = cur_capacity * off_peak_min_soc
            # Also try to charge up to the peak reserve level if below it
            if soc < cur_capacity * peak_reserve_soc:
                # Allow charging even when there's deficit
                effective_min_soc = cur_capacity * off_peak_min_soc
            else:
                # Already at or above target, maintain it
                effective_min_soc = cur_capacity * off_peak_min_soc
        else:
            # Normal off-peak operation, maintain higher minimum
            effective_min_soc = cur_capacity * off_peak_min_soc

        # 6c) Direct PV → load
        pv_used = min(pv_val, dem_val)
        surplus = pv_val - pv_used
        deficit = dem_val - pv_used

        # 6d) Charge battery if there's surplus or we're approaching peak time
        charge = 0
        approaching_peak = not is_peak and hours_to_peak <= peak_reserve_hours
        priority_charging = approaching_peak and soc < cur_capacity * peak_reserve_soc
        
        if surplus > 0:
            # We have PV surplus, use it to charge
            charge = min(
                surplus,
                battery_kw * delta_h,
                (cur_capacity - soc) / eff_chg
            )
            surplus -= charge
        elif priority_charging:
            # We're approaching peak time and need to charge from grid
            charge_needed = (cur_capacity * peak_reserve_soc - soc) / eff_chg
            grid_charge = min(
                charge_needed,
                battery_kw * delta_h
            )
            charge = grid_charge
            deficit += grid_charge  # This will be counted as grid import
        
        soc += charge * eff_chg
        total_charge += charge

        # 6e) Export any leftover PV
        export = surplus

        # 6f) Discharge battery to cover deficit, but only if appropriate
        discharge = 0
        if deficit > 0:
            # Calculate available energy (respecting the effective minimum SOC)
            avail = (soc - effective_min_soc) * eff_dis
            
            if is_peak or (avail > 0 and not approaching_peak):
                # During peak always try to discharge if available
                # During off-peak, only discharge if we have excess above the effective minimum
                discharge = min(
                    deficit,
                    battery_kw * delta_h,
                    avail
                )
                
            deficit -= discharge
            soc -= discharge / eff_dis
            total_discharge += discharge

        # 6g) Import remaining from grid
        grid = deficit
        peak_imp = grid if is_peak else 0.0
        off_imp = grid if not is_peak else 0.0

        # 6h) Record results
        cols['pv_gen'].append(pv_val)
        cols['demand'].append(dem_val)
        cols['pv_used'].append(pv_used)
        cols['battery_charge'].append(charge)
        cols['battery_discharge'].append(discharge)
        cols['battery_soc'].append(soc)
        cols['pv_export'].append(export)
        cols['grid_import_peak'].append(peak_imp)
        cols['grid_import_offpeak'].append(off_imp)
        cols['is_peak'].append(is_peak)
        cols['time_to_peak'].append(hours_to_peak)

    # 7) Build DataFrame
    df = pd.DataFrame(cols, index=pv.index)

    # 8) Aggregate totals
    total_demand = df['demand'].sum()
    total_pv_used = df['pv_used'].sum()
    total_batt_discharge = df['battery_discharge'].sum()
    total_imp_peak = df['grid_import_peak'].sum()
    total_imp_offpeak = df['grid_import_offpeak'].sum()
    total_import = total_imp_peak + total_imp_offpeak
    total_export = df['pv_export'].sum()
    total_emissions = total_import * grid_emission_rate
    
    # 9) Calculate battery metrics
    battery_cycles = 0
    final_capacity = battery_kwh
    final_degradation_pct = 0
    
    if battery_kwh > 0:
        # Calculate equivalent full cycles
        battery_cycles = total_charge / battery_kwh
        # Get final capacity (from last year with data)
        final_capacity = next((cap for cap in reversed(capacity_over_time) if cap > 0), battery_kwh)
        # Calculate degradation percentage
        final_degradation_pct = (1 - (final_capacity / battery_kwh)) * 100

    # 10) Compute energy shares
    renewable_supplied = total_pv_used + total_batt_discharge
    renewable_fraction = renewable_supplied / total_demand if total_demand else 0.0
    grid_fraction = total_import / total_demand if total_demand else 0.0
    self_consumption_rate = total_pv_used / (total_pv_used + total_export) \
                             if (total_pv_used + total_export) else 0.0

    # 11) Get NREL model degradation details if available
    nrel_model_details = {}
    if use_nrel_model:
        try:
            nrel_model_details = {
                'cycle_degradation_pct': bat.Battery.batt_cycle_degradation * 100,
                'calendar_degradation_pct': bat.Battery.batt_calendar_degradation * 100,
                'capacity_percent': bat.Battery.batt_capacity_percent * 100,
            }
        except Exception as e:
            print(f"Warning: Could not retrieve NREL model details: {e}")

    # Calculate peak vs off-peak statistics
    peak_hours = df['is_peak'].sum() * delta_h
    offpeak_hours = (len(df) - df['is_peak'].sum()) * delta_h
    peak_discharge = df.loc[df['is_peak'], 'battery_discharge'].sum()
    offpeak_discharge = df.loc[~df['is_peak'], 'battery_discharge'].sum()
    peak_charge = df.loc[df['is_peak'], 'battery_charge'].sum()
    offpeak_charge = df.loc[~df['is_peak'], 'battery_charge'].sum()
    peak_demand = df.loc[df['is_peak'], 'demand'].sum()
    offpeak_demand = df.loc[~df['is_peak'], 'demand'].sum()
    
    peak_pct_supplied_by_battery = peak_discharge / peak_demand if peak_demand > 0 else 0
    
    peak_statistics = {
        'peak_hours': peak_hours,
        'offpeak_hours': offpeak_hours,
        'peak_discharge_kwh': peak_discharge,
        'offpeak_discharge_kwh': offpeak_discharge,
        'peak_charge_kwh': peak_charge,
        'offpeak_charge_kwh': offpeak_charge,
        'peak_demand_kwh': peak_demand,
        'offpeak_demand_kwh': offpeak_demand,
        'peak_pct_supplied_by_battery': peak_pct_supplied_by_battery
    }

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
        'battery_cycles': battery_cycles,
        'final_capacity_kwh': final_capacity,
        'final_degradation_pct': final_degradation_pct,
        'nrel_model_used': use_nrel_model,
        **nrel_model_details,
        **peak_statistics
    }

    return df, totals