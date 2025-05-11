import pandas as pd
import numpy as np

def simulate_battery_dispatch(
    pv_gen: pd.Series,
    demand: pd.Series,
    battery_kwh: float,
    battery_kw: float = None,
    roundtrip_eff: float = 0.9,
    min_soc_pct: float = 0.2,
    annual_deg_rate: float = 0.01,
    grid_emission_rate: float = 1.0    # kgCO2e per kWh
) -> (pd.DataFrame, dict):
    """
    Simulate battery dispatch and compute energy & emissions shares.

    Returns
    -------
    df : pd.DataFrame
      Half‑hourly columns including:
        pv_gen, demand, pv_used, battery_charge, battery_discharge,
        battery_soc, pv_export, grid_import_peak, grid_import_offpeak
    totals : dict
      Aggregates including:
        total_grid_import_peak,
        total_grid_import_offpeak,
        total_pv_export,
        total_demand,
        total_pv_used,
        total_battery_discharge,
        renewable_fraction,   # (pv_used + battery_discharge) / total_demand
        grid_fraction,        # total_grid_import / total_demand
        self_consumption_rate,# total_pv_used / (total_pv_used + total_pv_export)
        total_emissions       # total_grid_import * grid_emission_rate
    """
    # Align series
    pv, dem = pv_gen.align(demand, join='inner')

    # Default 0.25C if not specified
    if battery_kw is None:
        battery_kw = 0.25 * battery_kwh

    delta_h = (pv.index[1] - pv.index[0]).total_seconds() / 3600.0
    eff_chg = np.sqrt(roundtrip_eff)
    eff_dis = np.sqrt(roundtrip_eff)
    intervals_per_year = 365 * 24 * 2  # half‑hourly

    soc = battery_kwh * min_soc_pct
    cols = {k: [] for k in [
        'pv_gen','demand','pv_used',
        'battery_charge','battery_discharge','battery_soc',
        'pv_export','grid_import_peak','grid_import_offpeak'
    ]}

    for i, (ts, pv_val, dem_val) in enumerate(zip(pv.index, pv, dem)):
        # degrade capacity each year
        year = i // intervals_per_year
        current_cap = battery_kwh * ((1 - annual_deg_rate)**year)
        soc = min(soc, current_cap)

        # peak window logic
        m, d, h = ts.month, ts.day, ts.hour
        if ((m < 4) or (m == 4 and d <= 1)) or ((m > 10) or (m == 10 and d >= 2)):
            peak_h = (14, 20)
        else:
            peak_h = (15, 21)
        is_peak = peak_h[0] <= h < peak_h[1]

        # PV → Demand
        pv_used = min(pv_val, dem_val)
        surplus = pv_val - pv_used
        deficit = dem_val - pv_used

        # Surplus → Battery
        charge = min(surplus, battery_kw*delta_h, (current_cap - soc)/eff_chg)
        soc += charge*eff_chg
        surplus -= charge

        # Export leftover PV
        export = surplus

        # Battery → Deficit
        avail = (soc - current_cap*min_soc_pct)*eff_dis
        discharge = min(deficit, battery_kw*delta_h, avail)
        soc -= discharge/eff_dis
        deficit -= discharge

        # Grid import remainder
        grid = deficit
        peak_imp = grid if is_peak else 0.0
        off_imp  = grid if not is_peak else 0.0

        # Record
        for k,v in [
            ('pv_gen',pv_val), ('demand',dem_val), ('pv_used',pv_used),
            ('battery_charge', charge), ('battery_discharge', discharge),
            ('battery_soc', soc),   ('pv_export', export),
            ('grid_import_peak', peak_imp),
            ('grid_import_offpeak', off_imp)
        ]:
            cols[k].append(v)

    df = pd.DataFrame(cols, index=pv.index)

    # Aggregates
    total_demand            = df['demand'].sum()
    total_pv_used           = df['pv_used'].sum()
    total_batt_discharge    = df['battery_discharge'].sum()
    total_import_peak       = df['grid_import_peak'].sum()
    total_import_offpeak    = df['grid_import_offpeak'].sum()
    total_import            = total_import_peak + total_import_offpeak
    total_export            = df['pv_export'].sum()
    total_emissions         = total_import * grid_emission_rate

    # Shares
    renewable_supplied      = total_pv_used + total_batt_discharge
    renewable_fraction      = renewable_supplied / total_demand if total_demand else 0.0
    grid_fraction           = total_import / total_demand if total_demand else 0.0
    self_consumption_rate   = total_pv_used / (total_pv_used + total_export) \
                                if (total_pv_used + total_export) else 0.0

    totals = {
      'total_grid_import_peak':    total_import_peak,
      'total_grid_import_offpeak': total_import_offpeak,
      'total_pv_export':           total_export,
      'total_demand':              total_demand,
      'total_pv_used':             total_pv_used,
      'total_battery_discharge':   total_batt_discharge,
      'total_grid_emissions':      total_emissions,
      'renewable_fraction':        renewable_fraction,
      'grid_fraction':             grid_fraction,
      'self_consumption_rate':     self_consumption_rate
    }

    return df, totals
