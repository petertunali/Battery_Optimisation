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
    Simulate battery dispatch with 0.25C max power and 1%/yr degradation.
    Returns half-hourly flows + aggregated totals for the simulated horizon.
    """
    pv, dem = pv_gen.align(demand, join='inner')
    if battery_kw is None:
        battery_kw = 0.25 * battery_kwh

    delta_h = (pv.index[1] - pv.index[0]).total_seconds() / 3600.0
    eff_chg = np.sqrt(roundtrip_eff)
    eff_dis = np.sqrt(roundtrip_eff)
    intervals_per_year = 365 * 24 * 2  # 17,520

    soc = battery_kwh * min_soc_pct
    cols = {
        'pv_gen': [], 'demand': [], 'pv_used': [],
        'battery_charge': [], 'battery_discharge': [], 'battery_soc': [],
        'pv_export': [], 'grid_import_peak': [], 'grid_import_offpeak': []
    }

    for i, (ts, pv_val, dem_val) in enumerate(zip(pv.index, pv, dem)):
        year = i // intervals_per_year
        current_cap = battery_kwh * ((1 - annual_deg_rate) ** year)
        soc = min(soc, current_cap)

        m, d, h = ts.month, ts.day, ts.hour
        if ((m < 4) or (m == 4 and d <= 1)) or ((m > 10) or (m == 10 and d >= 2)):
            peak_hours = (14, 20)
        else:
            peak_hours = (15, 21)
        is_peak = peak_hours[0] <= h < peak_hours[1]

        pv_used = min(pv_val, dem_val)
        surplus = pv_val - pv_used
        deficit = dem_val - pv_used

        charge    = min(surplus, battery_kw * delta_h, (current_cap - soc) / eff_chg)
        soc      += charge * eff_chg
        surplus  -= charge
        export    = surplus

        avail     = (soc - current_cap * min_soc_pct) * eff_dis
        discharge = min(deficit, battery_kw * delta_h, avail)
        soc      -= discharge / eff_dis
        deficit -= discharge

        grid = deficit
        peak_imp = grid if is_peak else 0.0
        off_imp  = grid if not is_peak else 0.0

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
        'total_grid_import_peak':    df['grid_import_peak'].sum(),
        'total_grid_import_offpeak': df['grid_import_offpeak'].sum(),
        'total_pv_export':           df['pv_export'].sum()
    }
    return df, totals
