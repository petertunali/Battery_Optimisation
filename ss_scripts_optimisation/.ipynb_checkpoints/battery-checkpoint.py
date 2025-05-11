import pandas as pd
import numpy as np

def simulate_battery_dispatch(
    pv_gen: pd.Series,
    demand: pd.Series,
    battery_kwh: float,
    battery_kw: float = None,
    roundtrip_eff: float = 0.9,
    min_soc_pct: float = 0.2,
    annual_deg_rate: float = 0.01
) -> (pd.DataFrame, dict):
    """
    Simulate battery dispatch with annual capacity degradation and separate
    peak vs off-peak imports. Returns half-hourly flows + aggregated totals.

    Defaults to a 0.25C power rating (i.e., 25% of energy capacity per hour).
    """
    # Align PV & demand
    pv, dem = pv_gen.align(demand, join='inner')

    # Default power rating = 0.25C
    if battery_kw is None:
        battery_kw = 0.25 * battery_kwh

    # Time interval in hours (expect 0.5)
    delta_h = (pv.index[1] - pv.index[0]).total_seconds() / 3600.0

    # Efficiencies
    eff_chg = np.sqrt(roundtrip_eff)
    eff_dis = np.sqrt(roundtrip_eff)

    # Intervals per year for half-hourly data
    intervals_per_year = int(365 * 24 * 2)  # 17,520

    # Initial SOC at minimum capacity
    soc = battery_kwh * min_soc_pct

    cols = {
        'pv_gen': [], 'demand': [], 'pv_used': [],
        'battery_charge': [], 'battery_discharge': [], 'battery_soc': [],
        'pv_export': [], 'grid_import_peak': [], 'grid_import_offpeak': []
    }

    # Iterate with index for degradation
    for i, (ts, pv_val, dem_val) in enumerate(zip(pv.index, pv.values, dem.values)):
        # Year index for degradation
        year = i // intervals_per_year
        current_capacity = battery_kwh * ((1 - annual_deg_rate) ** year)
        soc = min(soc, current_capacity)

        # Determine peak/off-peak window
        m, d, h = ts.month, ts.day, ts.hour
        if ((m < 4) or (m == 4 and d <= 1)) or ((m > 10) or (m == 10 and d >= 2)):
            peak_hours = (14, 20)
        else:
            peak_hours = (15, 21)
        is_peak = (h >= peak_hours[0] and h < peak_hours[1])

        # 1) Use PV for demand
        pv_used = min(pv_val, dem_val)
        surplus = pv_val - pv_used
        deficit = dem_val - pv_used

        # 2) Charge battery
        max_charge = min(surplus, battery_kw * delta_h, (current_capacity - soc) / eff_chg)
        charge = max_charge
        soc += charge * eff_chg
        surplus -= charge

        # 3) Export any remaining surplus
        export = surplus

        # 4) Discharge battery
        avail = (soc - current_capacity * min_soc_pct) * eff_dis
        max_discharge = min(deficit, battery_kw * delta_h, avail)
        discharge = max_discharge
        soc -= discharge / eff_dis
        deficit -= discharge

        # 5) Import remainder
        grid = deficit
        peak_imp = grid if is_peak else 0.0
        off_imp  = grid if not is_peak else 0.0

        # Record
        cols['pv_gen'].append(pv_val)
        cols['demand'].append(dem_val)
        cols['pv_used'].append(pv_used)
        cols['battery_charge'].append(charge)
        cols['battery_discharge'].append(discharge)
        cols['battery_soc'].append(soc)
        cols['pv_export'].append(export)
        cols['grid_import_peak'].append(peak_imp)
        cols['grid_import_offpeak'].append(off_imp)

    df = pd.DataFrame(cols, index=pv.index)
    totals = {
        'total_grid_import_peak': df['grid_import_peak'].sum(),
        'total_grid_import_offpeak': df['grid_import_offpeak'].sum(),
        'total_pv_export': df['pv_export'].sum()
    }
    return df, totals
