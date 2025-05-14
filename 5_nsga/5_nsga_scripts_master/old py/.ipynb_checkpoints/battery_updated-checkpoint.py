# battery_updated.py
import pandas as pd
import numpy as np
print("Using advanced peak/off-peak simulate_battery_dispatch()", __name__)


def simulate_battery_dispatch(
    pv_gen: pd.Series,
    demand: pd.Series,
    battery_kwh: float,
    battery_kw: float = None,
    roundtrip_eff: float = 0.9,
    min_soc_pct: float = 0.05,     # 95% depth of discharge = 5% min SOC
    annual_deg_rate: float = 0.01, # 1% capacity fade per year
    grid_emission_rate: float = 0.79,
    peak_reserve_soc: float = 0.8,   # 80% SOC reserved for peak
    peak_reserve_hours: int = 2,     # start reserving within 2 h of peak
    off_peak_min_soc: float = 0.6    # 60% min SOC off-peak
) -> (pd.DataFrame, dict):
    """
    Simulate half-hourly battery dispatch with simple degradation
    plus peak/off-peak awareness.
    """
    # 1) Align series
    pv, dem = pv_gen.align(demand, join='inner')

    # 2) Default power rating = 0.5C
    if battery_kw is None:
        battery_kw = 0.5 * battery_kwh

    # 3) Timesteps and efficiency
    delta_h = (pv.index[1] - pv.index[0]).total_seconds() / 3600.0
    eff_chg = np.sqrt(roundtrip_eff)
    eff_dis = np.sqrt(roundtrip_eff)
    ints_per_yr = 365 * 24 * 2

    # 4) Initial SOC at minimum
    soc = battery_kwh * min_soc_pct
    total_charge = 0.0
    total_discharge = 0.0

    # 5) Precompute peak flags & time_to_peak
    peak_periods = []
    for ts in pv.index:
        m, d, h = ts.month, ts.day, ts.hour
        # Winter: Oct–Mar 14–20, Summer: Apr–Sep 15–21
        if (m <= 3 or m >= 10):
            peak_periods.append(14 <= h < 20)
        else:
            peak_periods.append(15 <= h < 21)

    time_to_peak = []
    for i in range(len(pv.index)):
        if peak_periods[i]:
            time_to_peak.append(0.0)
        else:
            # look ahead up to 48 steps (24 h)
            hrs = float('inf')
            for j in range(i+1, min(i+49, len(pv.index))):
                if peak_periods[j]:
                    hrs = (j - i) * delta_h
                    break
            time_to_peak.append(hrs)

    # 6) Storage for outputs
    cols = {
        'pv_gen': [], 'demand': [], 'pv_used': [],
        'battery_charge': [], 'battery_discharge': [], 'battery_soc': [],
        'pv_export': [], 'grid_import_peak': [], 'grid_import_offpeak': [],
        'is_peak': [], 'time_to_peak': [], 'grid_to_battery': []  # Added grid_to_battery
    }

    # 7) Main loop
    for i, (ts, pv_val, dem_val) in enumerate(zip(pv.index, pv, dem)):
        # 7a) degradation
        year = i // ints_per_yr
        cur_capacity = battery_kwh * ((1 - annual_deg_rate)**year)
        soc = min(soc, cur_capacity)

        # 7b) flags
        is_peak = peak_periods[i]
        hrs_to_peak = time_to_peak[i]
        approaching_peak = (not is_peak) and (hrs_to_peak <= peak_reserve_hours)

        # 7c) PV → load
        pv_used = min(pv_val, dem_val)
        surplus = pv_val - pv_used
        deficit = dem_val - pv_used

        # 7d) charge battery
        charge = 0.0
        if surplus > 0:
            charge = min(surplus,
                         battery_kw * delta_h,
                         (cur_capacity - soc) / eff_chg)
            surplus -= charge
        elif approaching_peak and soc < cur_capacity * peak_reserve_soc:
            # grid‐charge up to reserve
            want = (cur_capacity * peak_reserve_soc - soc) / eff_chg
            charge = min(want, battery_kw * delta_h)
            deficit += charge  # counts as grid import
        soc += charge * eff_chg
        total_charge += charge

        # Calculate grid-to-battery component explicitly
        grid_to_battery = 0.0
        if charge > 0:
            # If there was PV surplus, some charging came from PV
            grid_to_battery = max(0, charge - surplus)

        # 7e) export leftover PV
        export = surplus

        # 7f) discharge battery
        discharge = 0.0
        if deficit > 0:
            effective_min = (cur_capacity * min_soc_pct) if is_peak else (cur_capacity * off_peak_min_soc)
            avail = max(0.0, (soc - effective_min) * eff_dis)
            if is_peak or (avail > 0 and not approaching_peak):
                discharge = min(deficit,
                                battery_kw * delta_h,
                                avail)
            soc -= discharge / eff_dis
            deficit -= discharge
            total_discharge += discharge

        # 7g) grid import
        grid_imp = deficit
        peak_imp = grid_imp if is_peak else 0.0
        off_imp  = grid_imp if not is_peak else 0.0

        # 7h) record
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
        cols['time_to_peak'].append(hrs_to_peak)
        cols['grid_to_battery'].append(grid_to_battery)  # Add grid-to-battery

    # 8) Build DataFrame
    df = pd.DataFrame(cols, index=pv.index)

    # 9) Totals
    total_demand = df['demand'].sum()
    total_pv_used = df['pv_used'].sum()
    total_grid_peak = df['grid_import_peak'].sum()
    total_grid_off  = df['grid_import_offpeak'].sum()
    total_export    = df['pv_export'].sum()
    total_import    = total_grid_peak + total_grid_off
    total_emissions = total_import * grid_emission_rate
    
    # Calculate grid-to-battery totals
    total_grid_to_battery = df['grid_to_battery'].sum()
    peak_grid_to_battery = df.loc[df['is_peak'], 'grid_to_battery'].sum()
    offpeak_grid_to_battery = df.loc[~df['is_peak'], 'grid_to_battery'].sum()

    # 10) battery metrics
    cycles = total_charge / battery_kwh if battery_kwh > 0 else 0.0
    final_deg_pct = (1 - ((cur_capacity)/battery_kwh)) * 100 if battery_kwh > 0 else 0.0

    # 11) shares
    renewable_supplied = total_pv_used + total_discharge
    renewable_frac = renewable_supplied / total_demand if total_demand else 0.0
    self_consume   = total_pv_used / (total_pv_used + total_export) if (total_pv_used + total_export) else 0.0

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

    totals = {
        'total_demand': total_demand,
        'total_pv_used': total_pv_used,
        'total_battery_discharge': total_discharge,
        'total_grid_import_peak': total_grid_peak,
        'total_grid_import_offpeak': total_grid_off,
        'total_pv_export': total_export,
        'total_grid_emissions': total_emissions,
        'renewable_fraction': renewable_frac,
        'self_consumption_rate': self_consume,
        'battery_cycles': cycles,
        'final_degradation_pct': final_deg_pct,
        'total_grid_to_battery': total_grid_to_battery,
        'peak_grid_to_battery': peak_grid_to_battery,
        'offpeak_grid_to_battery': offpeak_grid_to_battery,
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

    return df, totals