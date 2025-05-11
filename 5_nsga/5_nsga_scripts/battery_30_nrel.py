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
    grid_emission_rate: float = 0.81  # 0.81 kg CO2e per kWh
) -> (pd.DataFrame, dict):
    """
    Simulate half‑hourly battery dispatch over a multi‑year profile using NREL's
    PySAM battery model for more realistic degradation modeling.
    
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
        Minimum state‑of‑charge as fraction of capacity. Default: 0.05 (95% DoD).
    annual_deg_rate : float
        Fallback degradation rate if NREL model fails. Default: 0.01 (1% per year).
    grid_emission_rate : float
        kgCO2e emitted per kWh imported from grid. Default: 0.81 kg CO2e/kWh.
        
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
        'pv_export', 'grid_import_peak', 'grid_import_offpeak'
    ]}
    
    # For tracking battery cycles and degradation
    total_charge = 0
    total_discharge = 0
    cycle_count = 0
    
    # Create arrays for tracking capacity and degradation over time
    years = len(pv) // ints_per_yr + 1
    capacity_over_time = np.ones(years) * battery_kwh
    
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
        m, d, h = ts.month, ts.day, ts.hour
        if (m < 4 or (m == 4 and d <= 1)) or (m > 10 or (m == 10 and d >= 2)):
            peak_start, peak_end = 14, 20
        else:
            peak_start, peak_end = 15, 21
        is_peak = peak_start <= h < peak_end

        # 6c) Direct PV → load
        pv_used = min(pv_val, dem_val)
        surplus = pv_val - pv_used
        deficit = dem_val - pv_used

        # 6d) Charge battery with surplus PV
        charge = min(surplus,
                     battery_kw * delta_h if battery_kw else 0,
                     (cur_capacity - soc) / eff_chg if cur_capacity > 0 else 0)
        soc += charge * eff_chg
        surplus -= charge
        total_charge += charge

        # 6e) Export any leftover PV
        export = surplus

        # 6f) Discharge battery to cover deficit
        avail = (soc - cur_capacity * min_soc_pct) * eff_dis if cur_capacity > 0 else 0
        discharge = min(deficit,
                        battery_kw * delta_h if battery_kw else 0,
                        avail)
        soc -= discharge / eff_dis
        deficit -= discharge
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
        **nrel_model_details
    }

    return df, totals