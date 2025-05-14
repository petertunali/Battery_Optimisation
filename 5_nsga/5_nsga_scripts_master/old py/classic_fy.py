# fin.py - financial calculations for the optimization
import numpy as np
import pandas as pd
from scipy import optimize

def calculate_lcoe(total_generation, capital_cost, annual_maintenance, maintenance_inflation_rate, discount_rate, years=30):
    """
    Calculate Levelized Cost of Energy (LCOE)
    
    Parameters:
    -----------
    total_generation : float
        Total energy generation over the project lifetime (kWh)
    capital_cost : float
        Total initial capital cost ($)
    annual_maintenance : float
        First year maintenance cost ($/year)
    maintenance_inflation_rate : float
        Annual inflation rate for maintenance costs
    discount_rate : float
        Discount rate for present value calculations
    years : int
        Project lifetime in years
        
    Returns:
    --------
    lcoe : float
        Levelized Cost of Energy ($/kWh)
    """
    if total_generation <= 0:
        return float('inf')
    
    # Calculate discounted total costs
    total_costs = capital_cost
    
    # Add discounted maintenance costs
    for year in range(1, years + 1):
        # Apply maintenance inflation
        year_maintenance = annual_maintenance * (1 + maintenance_inflation_rate)**(year-1)
        # Discount to present value
        total_costs += year_maintenance / (1 + discount_rate)**year
    
    # Calculate LCOE
    lcoe = total_costs / total_generation
    return lcoe

def compute_financials(
    totals,
    battery_kwh,
    pv_kw,
    additional_pv_cost=0.0,
    battery_cost_per_kwh=None,
    battery_installation_cost_per_kwh=174.0,
    feed_in_tariff=0.033,
    peak_rate=0.39710,
    offpeak_rate=0.13530,
    pv_maintenance_per_day=0.68,  # $250/year = $0.68/day
    battery_maintenance_per_kw_day=0.0,  # No battery maintenance
    battery_power_ratio=0.5,
    years=30,
    discount_rate=0.07,
    escalation_rate=0.03,  # 3% annual escalation
    maintenance_inflation_rate=0.03,  # 3% annual maintenance inflation (same as escalation)
    baseline_import_cost=0.0,
    baseline_pv_only_cost=0.0
):
    """
    Compute lifecycle CAPEX, OPEX, revenues/costs, IRR & NPV over 'years'.
    Includes electricity price escalation and properly accounts for incremental analysis.
    
    Parameters:
    -----------
    totals : dict
        Energy flow totals from battery simulation
    battery_kwh : float
        Battery energy capacity in kWh
    pv_kw : float
        Total PV system capacity in kW
    additional_pv_cost : float
        Cost of additional PV (not including existing system)
    escalation_rate : float
        Annual electricity price escalation rate (default: 3%)
    maintenance_inflation_rate : float
        Annual inflation rate for maintenance costs (default: 3%)
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
    capex_pv = additional_pv_cost  # Cost of additional PV (existing system already paid for)
    capex_batt = battery_kwh * (battery_cost_per_kwh + battery_installation_cost_per_kwh) if battery_kwh > 0 else 0.0
    capex_total = capex_pv + capex_batt
    
    # 2) Calculate annual maintenance costs
    annual_pv_maintenance = pv_maintenance_per_day * 365  # $250/year ($0.68/day) - flat fee
    annual_battery_maintenance = battery_maintenance_per_kw_day * battery_kw * 365 if battery_kw > 0 else 0.0
    annual_maintenance = annual_pv_maintenance + annual_battery_maintenance
    
    # 3) Calculate year-by-year electricity costs with escalation
    # Extract energy values
    total_grid_import_peak = totals['total_grid_import_peak']
    total_grid_import_offpeak = totals['total_grid_import_offpeak']
    total_pv_export = totals['total_pv_export']
    
    # Calculate average annual values
    annual_grid_import_peak = total_grid_import_peak / years
    annual_grid_import_offpeak = total_grid_import_offpeak / years
    annual_pv_export = total_pv_export / years
    
    # Get baseline annual costs
    annual_no_pv_cost = baseline_import_cost / years
    annual_pv_only_cost = baseline_pv_only_cost / years
    
    # Calculate year-by-year costs with price escalation
    annual_costs = []
    for year in range(years):
        # Apply escalation
        year_escalation = (1 + escalation_rate)**year
        year_peak_rate = peak_rate * year_escalation
        year_offpeak_rate = offpeak_rate * year_escalation
        year_export_rate = feed_in_tariff * year_escalation
        
        # Apply maintenance inflation
        year_maintenance = annual_maintenance * (1 + maintenance_inflation_rate)**year
        
        # Calculate costs for this year
        year_import_cost = (
            annual_grid_import_peak * year_peak_rate + 
            annual_grid_import_offpeak * year_offpeak_rate
        )
        year_export_revenue = annual_pv_export * year_export_rate
        year_electricity_cost = year_import_cost - year_export_revenue
        
        # Escalate baseline costs too
        year_no_pv_cost = annual_no_pv_cost * year_escalation
        year_pv_only_cost = annual_pv_only_cost * year_escalation
        
        annual_costs.append({
            'year': year + 1,
            'import_cost': year_import_cost,
            'export_revenue': year_export_revenue,
            'electricity_cost': year_electricity_cost,
            'maintenance': year_maintenance,
            'total_cost': year_electricity_cost + year_maintenance,
            'no_pv_cost': year_no_pv_cost,
            'pv_only_cost': year_pv_only_cost,
            'savings_vs_no_pv': year_no_pv_cost - year_electricity_cost - year_maintenance,
            'savings_vs_pv_only': year_pv_only_cost - year_electricity_cost - year_maintenance
        })
    
    # 4) Calculate NPV and IRR
    # For IRR calculation, we need the initial investment and annual cash flows
    initial_investment = -capex_total
    
    # Cash flows compared to "PV-only" scenario (incremental analysis)
    cash_flows = [initial_investment]
    for year_costs in annual_costs:
        cash_flows.append(year_costs['savings_vs_pv_only'])
    
    # Calculate IRR
    try:
        def npv_func(rate):
            return sum(cf / (1 + rate)**(i) for i, cf in enumerate(cash_flows))
        
        try:
            irr_value = optimize.newton(npv_func, 0.05)
        except:
            try:
                irr_value = optimize.brentq(npv_func, -0.9999, 2.0)
            except:
                irr_value = None
    except:
        irr_value = None
    
    # Calculate NPV
    npv_value = sum(cf / (1 + discount_rate)**(i) for i, cf in enumerate(cash_flows))
    
    # Calculate total costs over the project lifetime
    total_import_cost = sum(year_data['import_cost'] for year_data in annual_costs)
    total_export_revenue = sum(year_data['export_revenue'] for year_data in annual_costs)
    total_electricity_cost = sum(year_data['electricity_cost'] for year_data in annual_costs)
    total_maintenance = sum(year_data['maintenance'] for year_data in annual_costs)
    total_cost = total_electricity_cost + total_maintenance
    total_savings_vs_no_pv = sum(year_data['savings_vs_no_pv'] for year_data in annual_costs)
    total_savings_vs_pv_only = sum(year_data['savings_vs_pv_only'] for year_data in annual_costs)
    
    # Simple payback period
    if annual_costs[0]['savings_vs_pv_only'] > 0:
        simple_payback = capex_total / annual_costs[0]['savings_vs_pv_only']
    else:
        simple_payback = float('inf')

    # Estimate total generation over the project lifetime
    if 'total_pv_used' in totals and 'total_pv_export' in totals:
        total_generation = totals['total_pv_used'] + totals['total_pv_export']
    else:
        # Fallback if direct generation data isn't available
        total_generation = annual_pv_export * years * 30  # Rough estimate

    # Calculate LCOE
    lcoe = calculate_lcoe(
        total_generation=total_generation,
        capital_cost=capex_total,
        annual_maintenance=annual_maintenance,
        maintenance_inflation_rate=maintenance_inflation_rate,
        discount_rate=discount_rate,
        years=years
    )
    
    # Return financial metrics
    return {
        'capex_pv': capex_pv,
        'capex_battery': capex_batt,
        'capex_total': capex_total,
        'battery_cost_per_kwh': battery_cost_per_kwh,
        'annual_maintenance': annual_maintenance,
        'total_maintenance': total_maintenance,
        'total_import_cost': total_import_cost,
        'total_export_revenue': total_export_revenue,
        'total_electricity_cost': total_electricity_cost,
        'total_cost': total_cost,
        'total_savings_vs_no_pv': total_savings_vs_no_pv,
        'total_savings_vs_pv_only': total_savings_vs_pv_only,
        'annual_costs': annual_costs,
        'cash_flows': cash_flows,
        'npv': npv_value,
        'net_cost': -npv_value,  # Convert NPV to NPC
        'irr': irr_value,
        'simple_payback': simple_payback,
        'lcoe': lcoe  # Add this line
    }