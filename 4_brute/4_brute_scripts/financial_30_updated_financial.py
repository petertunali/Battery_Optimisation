import numpy as np
import numpy_financial as nf

def compute_financials(
    totals: dict,
    battery_kwh: float,
    pv_kw: float,
    feed_in_tariff: float = 0.033,
    import_price_peak: float = 0.39710,
    import_price_offpeak: float = 0.13530,
    pv_cost_per_kw: float = None,  # Will be calculated if None
    pv_installation_cost: float = 5000.0,
    battery_cost_per_kwh: float = None,  # Will be calculated if None
    battery_installation_cost_per_kwh: float = 174.0,
    battery_power_ratio: float = 0.5,  # Power rating as fraction of capacity
    pv_maintenance_per_kw_day: float = 0.13,
    battery_maintenance_per_kw_day: float = 0.12,
    years: int = 30,
    discount_rate: float = 0.07,
    baseline_import_cost: float = 0.0,  # Total baseline cost over project life
    true_baseline_cost: float = 9424.48 * 30  # No PV, no battery cost ($9,424.48/year × 30 years)
) -> dict:
    """
    Compute lifecycle CAPEX, OPEX, revenues/costs, IRR & NPC over `years`.
    
    PV cost follows formula: y = 1047.3e^(-0.002*x) with minimum $750/kW
    Battery cost follows formula: y = 977.54e^(-0.004*x) with minimum $600/kWh
    
    Parameters
    ----------
    totals : dict
        Dispatch simulation results
    battery_kwh : float
        Battery capacity in kWh
    pv_kw : float
        PV capacity in kW
    feed_in_tariff : float
        Rate for exported energy ($/kWh)
    import_price_peak : float
        Peak electricity rate ($/kWh)
    import_price_offpeak : float
        Off-peak electricity rate ($/kWh)
    pv_cost_per_kw : float
        Per-kW cost of PV ($ per kW)
    pv_installation_cost : float
        Fixed installation cost for PV ($)
    battery_cost_per_kwh : float
        Per-kWh cost of battery ($ per kWh)
    battery_installation_cost_per_kwh : float
        Per-kWh installation cost for battery ($ per kWh)
    battery_power_ratio : float
        Battery power rating as fraction of capacity (e.g., 0.5 = 0.5C)
    pv_maintenance_per_kw_day : float
        Daily maintenance cost per kW of PV ($ per kW per day)
    battery_maintenance_per_kw_day : float
        Daily maintenance cost per kW of battery power ($ per kW per day)
    years : int
        Project lifetime in years
    discount_rate : float
        Discount rate for NPV calculations (decimal)
    baseline_import_cost : float
        Cost of importing all electricity with existing PV, but no battery ($ over project life)
    true_baseline_cost : float
        Cost of importing all electricity with no PV and no battery ($ over project life)
        
    Returns
    -------
    dict
        Financial metrics including CAPEX, OPEX, revenues, costs, IRR, NPV
    """
    # 1) Calculate cost per kW/kWh using exponential formulas if not provided
    # PV cost formula with $750 floor
    if pv_cost_per_kw is None:
        pv_cost_per_kw = max(1047.3 * np.exp(-0.002 * pv_kw), 750.0) if pv_kw > 0 else 0.0
    
    # Battery cost formula with $600 floor
    if battery_cost_per_kwh is None:
        battery_cost_per_kwh = max(977.54 * np.exp(-0.004 * battery_kwh), 600.0) if battery_kwh > 0 else 0.0
    
    # 2) CAPEX
    capex_pv = pv_kw * pv_cost_per_kw + (pv_installation_cost if pv_kw > 0 else 0)
    capex_batt = battery_kwh * (battery_cost_per_kwh + battery_installation_cost_per_kwh)
    capex_total = capex_pv + capex_batt
    
    # 3) Revenues & gross import cost
    export_rev = totals.get('total_pv_export', 0.0) * feed_in_tariff
    import_cost = (
        totals.get('total_grid_import_peak', 0.0) * import_price_peak +
        totals.get('total_grid_import_offpeak', 0.0) * import_price_offpeak
    )
    
    # 4) OPEX (maintenance costs for both PV and battery over the years)
    # PV maintenance is per kW of PV
    pv_opex = pv_maintenance_per_kw_day * pv_kw * 365 * years if pv_kw > 0 else 0
    
    # Battery maintenance is per kW of battery power (not kWh)
    battery_kw = battery_kwh * battery_power_ratio
    battery_opex = battery_maintenance_per_kw_day * battery_kw * 365 * years if battery_kwh > 0 else 0
    
    opex_total = pv_opex + battery_opex
    
    # 5) Calculate savings in two different ways for clarity
    # a) Savings compared to baseline with existing PV (i.e., adding battery only)
    pv_battery_savings = baseline_import_cost - import_cost + export_rev
    
    # b) Savings compared to true baseline (no PV, no battery)
    total_savings = true_baseline_cost - import_cost + export_rev
    
    # 6) Net present cost (incremental)
    net_cost = capex_total + opex_total - pv_battery_savings
    
    # 7) Build annualized cash-flows
    annual_export = export_rev / years
    annual_import_savings = (baseline_import_cost - import_cost) / years
    annual_opex = opex_total / years
    annual_net = annual_import_savings + annual_export - annual_opex
    cash_flows = [-capex_total] + [annual_net] * years
    
    # 8) IRR - catch exceptions for cases with no valid IRR
    try:
        irr = nf.irr(cash_flows)
        if np.isnan(irr):
            irr = None
    except:
        irr = None
    
    # 9) Discounted NPV (using the specified discount rate)
    npv = sum(cf / (1 + discount_rate)**i for i, cf in enumerate(cash_flows))
    
    # 10) Additional metrics
    # Calculate payback period (simple)
    annual_savings = annual_import_savings + annual_export
    if capex_total > 0 and annual_savings > annual_opex:
        payback_years = capex_total / (annual_savings - annual_opex)
    else:
        payback_years = float('inf')
    
    # Calculate final degradation percentage
    # Based on the battery_30.py simulation with annual_deg_rate
    final_cycles = totals.get('total_battery_discharge', 0.0) / battery_kwh if battery_kwh > 0 else 0
    annual_degradation = 0.01  # 1% per year as used in battery_30.py
    final_degradation = (1 - (1 - annual_degradation) ** years) * 100 if battery_kwh > 0 else 0
    
    return {
        'pv_cost_per_kw': pv_cost_per_kw,
        'battery_cost_per_kwh': battery_cost_per_kwh,
        'capex_pv': capex_pv,
        'capex_battery': capex_batt,
        'capex_total': capex_total,
        'export_revenue_total': export_rev,
        'import_cost_total': import_cost,
        'opex_total': opex_total,
        'net_cost': net_cost,
        'annual_cash_flow': cash_flows,
        'irr': irr,
        'npv': npv,
        'payback_years': payback_years,
        'battery_total_cycles': final_cycles,
        'battery_final_degradation_pct': final_degradation,
        'pv_battery_savings': pv_battery_savings,  # Savings from adding battery to existing PV
        'total_savings': total_savings,  # Savings compared to no PV, no battery
        'annual_import_savings': annual_import_savings,
        'annual_export_revenue': annual_export,
        'annual_opex': annual_opex,
        'annual_net_benefit': annual_net
    }