import numpy as np
import numpy_financial as nf

def compute_financials(
    totals: dict,
    battery_kwh: float,
    pv_kw: float,
    pv_cost_per_kw: float = 1000.0,
    pv_installation_cost: float = 5000.0,
    battery_cost_per_kwh: float = None,  # Now calculated with formula
    battery_installation_cost_per_kwh: float = 174.0,
    feed_in_tariff: float = 0.033,
    import_price_peak: float = 0.39710,
    import_price_offpeak: float = 0.13530,
    pv_maintenance_per_kw_day: float = 0.13,  # Updated to 0.13 per kW per day
    battery_maintenance_per_kw_day: float = 0.12,  # Updated to 0.12 per kW per day
    battery_power_ratio: float = 0.5,  # Power rating as fraction of capacity (0.5C)
    years: int = 30,
    discount_rate: float = 0.07,
    baseline_import_cost: float = 0.0,  # No PV, no battery cost (e.g., $9,424.48/year × 30)
    baseline_pv_only_cost: float = 0.0  # PV-only, no battery cost (e.g., $8,246.44/year × 30)
) -> dict:
    """
    Compute lifecycle CAPEX, OPEX, revenues/costs, IRR & NPV over `years`.
    Now properly compares to both no-PV baseline and PV-only baseline.
    
    Parameters:
    -----------
    totals : dict
        Energy flow totals from battery simulation
    battery_kwh : float
        Battery energy capacity in kWh
    pv_kw : float
        PV system capacity in kW
    baseline_import_cost : float
        Total 30-year cost with no PV, no battery (original baseline)
    baseline_pv_only_cost : float
        Total 30-year cost with PV but no battery (for calculating battery-only benefits)
    """
    # Calculate battery power rating (kW)
    battery_kw = battery_kwh * battery_power_ratio
    
    # Calculate battery cost per kWh using the formula: y = 977.54 * e^(-0.004*x)
    # With minimum of $600/kWh
    if battery_cost_per_kwh is None:
        if battery_kwh > 0:
            battery_cost_per_kwh = 977.54 * np.exp(-0.004 * battery_kwh)
            battery_cost_per_kwh = max(600.0, battery_cost_per_kwh)  # Minimum $600/kWh
        else:
            battery_cost_per_kwh = 0.0  # No cost for 0 kWh battery
    
    # 1) CAPEX
    capex_pv = pv_kw * pv_cost_per_kw + (pv_installation_cost if pv_kw > 0 and pv_cost_per_kw > 0 else 0)
    capex_batt = battery_kwh * (battery_cost_per_kwh + battery_installation_cost_per_kwh)
    capex_total = capex_pv + capex_batt
    
    # 2) Calculate current system's energy costs
    export_rev = totals.get('total_pv_export', 0.0) * feed_in_tariff
    current_import_cost = (
        totals.get('total_grid_import_peak', 0.0) * import_price_peak +
        totals.get('total_grid_import_offpeak', 0.0) * import_price_offpeak
    )
    current_net_cost = current_import_cost - export_rev  # Current net energy cost
    
    # 3) OPEX (maintenance costs based on kW rating, not kWh capacity)
    pv_opex = pv_maintenance_per_kw_day * pv_kw * 365 * years if pv_kw > 0 else 0
    battery_opex = battery_maintenance_per_kw_day * battery_kw * 365 * years if battery_kw > 0 else 0
    opex_total = pv_opex + battery_opex
    
    # 4) Calculate savings vs. both baselines
    savings_vs_no_pv = baseline_import_cost - current_net_cost
    savings_vs_pv_only = baseline_pv_only_cost - current_net_cost
    
    # 5) Net present cost calculation for existing PV + new battery
    if pv_cost_per_kw == 0 and pv_installation_cost == 0:
        # We're only evaluating adding a battery to existing PV
        net_cost = capex_batt + battery_opex - savings_vs_pv_only
    else:
        # We're evaluating a full new PV+battery system
        net_cost = capex_total + opex_total - savings_vs_no_pv
    
    # 6) Build annualized cash-flows for IRR calculation
    if pv_cost_per_kw == 0 and pv_installation_cost == 0:
        # For adding battery to existing PV
        annual_savings = savings_vs_pv_only / years
        annual_opex = battery_opex / years
        cash_flows = [-capex_batt] + [annual_savings - annual_opex] * years
    else:
        # For full new PV+battery system
        annual_savings = savings_vs_no_pv / years
        annual_opex = opex_total / years
        cash_flows = [-capex_total] + [annual_savings - annual_opex] * years
    
    # 7) IRR
    try:
        irr = nf.irr(cash_flows)
    except:
        irr = None  # IRR calculation failed
    
    # 8) Discounted NPV (using the specified discount rate)
    npv = sum(cf / (1 + discount_rate)**i for i, cf in enumerate(cash_flows))
    
    # Calculate battery cost for reporting
    actual_battery_cost = battery_cost_per_kwh if battery_kwh > 0 else 0
    
    # Annual bill metrics for clarity
    annual_import_cost = current_import_cost / years
    annual_export_rev = export_rev / years
    annual_net_bill = annual_import_cost - annual_export_rev
    annual_no_pv_bill = baseline_import_cost / years
    annual_pv_only_bill = baseline_pv_only_cost / years
    
    return {
        'capex_pv': capex_pv,
        'capex_battery': capex_batt,
        'capex_total': capex_total,
        'battery_cost_per_kwh': actual_battery_cost,
        'export_revenue_total': export_rev,
        'annual_export_revenue': annual_export_rev,
        'import_cost_total': current_import_cost,
        'annual_import_cost': annual_import_cost,
        'annual_bill': annual_net_bill,
        'annual_no_pv_bill': annual_no_pv_bill,
        'annual_pv_only_bill': annual_pv_only_bill,
        'total_energy_bill': current_net_cost,
        'savings_vs_no_pv': savings_vs_no_pv,
        'savings_vs_pv_only': savings_vs_pv_only,
        'opex_total': opex_total,
        'net_cost': net_cost,
        'annual_cash_flow': cash_flows,
        'irr': irr,
        'npv': npv,
        'simple_payback_years': capex_batt / (savings_vs_pv_only / years) if savings_vs_pv_only > 0 else float('inf')
    }