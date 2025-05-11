import numpy as np
import numpy_financial as nf

def compute_financials(
    totals: dict,
    battery_kwh: float,
    pv_kw: float,
    feed_in_tariff: float = 0.033,
    import_price_peak: float = 0.39710,
    import_price_offpeak: float = 0.13530,
    pv_installation_cost: float = 5000.0,
    battery_installation_cost_per_kwh: float = 174.0,
    years: int = 30,
    discount_rate: float = 0.07,
    baseline_import_cost: float = 0.0
) -> dict:
    """
    Compute lifecycle CAPEX, OPEX, revenues/costs, IRR & NPC over `years`.
    Subtracts `baseline_import_cost` from the gross import bill to get *incremental* cost.
    
    PV cost follows formula: y = 1047.3e^(-0.002*x) with minimum $750/kW
    Battery cost follows formula: y = 977.54e^(-0.004*x) with minimum $600/kWh
    """
    # 1) Calculate cost per kW/kWh using exponential formulas
    # PV cost formula with $750 floor
    pv_cost_per_kw = max(1047.3 * np.exp(-0.002 * pv_kw), 750.0) if pv_kw > 0 else 0.0
    
    # Battery cost formula with $600 floor
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
    # PV maintenance is per kW of PV - NREL reference: $0.08 USD per kW per day
    pv_maintenance_per_day = 0.13
    pv_opex = pv_maintenance_per_day * pv_kw * 365 * years if pv_kw > 0 else 0
    
    # Battery maintenance is per kW of battery power (not kWh) - NREL reference: $0.085 USD per kW per day
    battery_maintenance_per_day = 0.12
    battery_kw = battery_kwh * 0.5  # 0.5C rate
    battery_opex = battery_maintenance_per_day * battery_kw * 365 * years if battery_kwh > 0 else 0
    
    opex_total = pv_opex + battery_opex
    
    # 5) Incremental import cost
    import_cost_inc = import_cost - baseline_import_cost
    
    # 6) Net present cost (incremental)
    net_cost = capex_total + opex_total + import_cost_inc - export_rev
    
    # 7) Build annualized cash-flows
    annual_export = export_rev / years
    annual_import_inc = import_cost_inc / years
    annual_opex = opex_total / years
    annual_net = annual_export - annual_import_inc - annual_opex
    cash_flows = [-capex_total] + [annual_net] * years
    
    # 8) IRR
    irr = nf.irr(cash_flows) or 0.0
    
    # 9) Discounted NPV (using the specified discount rate)
    npv = sum(cf / (1 + discount_rate)**i for i, cf in enumerate(cash_flows))
    
    # 10) Calculate final degradation percentage
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
        'import_cost_total': import_cost_inc,
        'opex_total': opex_total,
        'net_cost': net_cost,
        'annual_cash_flow': cash_flows,
        'irr': irr,
        'npv': npv,
        'battery_total_cycles': final_cycles,
        'battery_final_degradation_pct': final_degradation
    }