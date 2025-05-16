# fin.py - financial calculations for the optimization
import numpy as np
import pandas as pd
from scipy import optimize
import math

def calculate_pv_cost(capacity_kw: float, cost_multiplier: float = 1.0) -> float:
    """
    Calculate PV cost using the configured formula with economies of scale.
    
    Args:
        capacity_kw: PV capacity in kW
        cost_multiplier: Multiplier for special installations (e.g., 1.25 for ground-mounted)
        
    Returns:
        total_cost: Total cost in dollars
    """
    try:
        # Try to import config for cost parameters
        from config import PV_COST_FORMULA
        base_cost = PV_COST_FORMULA['base_cost']
        exponent = PV_COST_FORMULA['exponent']
        minimum = PV_COST_FORMULA['minimum']
    except (ImportError, KeyError):
        # Default values if not in config
        base_cost = 1047.3  # $/kW
        exponent = -0.002   # For economies of scale
        minimum = 750       # Minimum cost per kW
    
    if capacity_kw <= 0:
        return 0.0
    
    # Apply economies of scale formula
    cost_per_kw = base_cost * np.exp(exponent * capacity_kw)
    
    # Apply minimum cost
    cost_per_kw = max(minimum, cost_per_kw)
    
    # Apply cost multiplier
    cost_per_kw *= cost_multiplier
    
    # Calculate total cost
    total_cost = cost_per_kw * capacity_kw
    
    return total_cost

def calculate_battery_cost(capacity_kwh: float) -> float:
    """
    Calculate battery cost using the configured formula with economies of scale.
    
    Args:
        capacity_kwh: Battery capacity in kWh
        
    Returns:
        total_cost: Total cost in dollars (including installation)
    """
    try:
        # Try to import config for cost parameters
        from config import BATTERY_COST_FORMULA
        base_cost = BATTERY_COST_FORMULA['base_cost']
        exponent = BATTERY_COST_FORMULA['exponent']
        minimum = BATTERY_COST_FORMULA['minimum']
        installation = BATTERY_COST_FORMULA['installation']
    except (ImportError, KeyError):
        # Default values if not in config
        base_cost = 977.54  # $/kWh
        exponent = -0.004   # For economies of scale
        minimum = 600       # Minimum cost per kWh
        installation = 174  # Installation cost per kWh
    
    if capacity_kwh <= 0:
        return 0.0
    
    # Apply economies of scale formula
    cost_per_kwh = base_cost * np.exp(exponent * capacity_kwh)
    
    # Apply minimum cost
    cost_per_kwh = max(minimum, cost_per_kwh)
    
    # Add installation cost
    total_per_kwh = cost_per_kwh + installation
    
    # Calculate total cost
    total_cost = total_per_kwh * capacity_kwh
    
    return total_cost

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
    additional_pv_kw,
    config=None,
    feed_in_tariff=None,
    peak_rate=None,
    offpeak_rate=None,
    discount_rate=None,
    escalation_rate=None,
    maintenance_inflation_rate=None,
    project_lifetime=None
):
    """
    Compute lifecycle CAPEX, OPEX, revenues/costs, IRR, NPV & PI.
    Includes electricity price escalation and properly accounts for incremental analysis.
    
    Parameters:
    -----------
    totals : dict
        Energy flow totals from battery simulation
    battery_kwh : float
        Battery energy capacity in kWh
    additional_pv_kw : float
        Additional PV capacity in kW (not including existing system)
    config : object
        Configuration object with parameters
    """
    # Get parameters from config if available
    if config is not None:
        # Electricity rates
        if feed_in_tariff is None and hasattr(config, 'ELECTRICITY_RATES'):
            feed_in_tariff = config.ELECTRICITY_RATES.get('export', 0.033)  # Default: 3.3c/kWh
        if peak_rate is None and hasattr(config, 'ELECTRICITY_RATES'):
            peak_rate = config.ELECTRICITY_RATES.get('peak', 0.3971)  # Default: 39.71c/kWh
        if offpeak_rate is None and hasattr(config, 'ELECTRICITY_RATES'):
            offpeak_rate = config.ELECTRICITY_RATES.get('offpeak', 0.1353)  # Default: 13.53c/kWh
        
        # Financial parameters
        if discount_rate is None and hasattr(config, 'DISCOUNT_RATE'):
            discount_rate = config.DISCOUNT_RATE  # Default: 7%
        if escalation_rate is None and hasattr(config, 'ELECTRICITY_PRICE_ESCALATION'):
            escalation_rate = config.ELECTRICITY_PRICE_ESCALATION  # Default: 3%
        if maintenance_inflation_rate is None and hasattr(config, 'MAINTENANCE_INFLATION'):
            maintenance_inflation_rate = config.MAINTENANCE_INFLATION  # Default: 3%
        
        # Project parameters
        if project_lifetime is None and hasattr(config, 'PROJECT_LIFETIME'):
            project_lifetime = config.PROJECT_LIFETIME  # Default: 30 years
        
        # Baseline costs
        baseline_no_pv_cost = getattr(config, 'ANNUAL_NO_PV_COST', 9424.48)
        baseline_pv_only_cost = getattr(config, 'ANNUAL_PV_ONLY_COST', 8246.44)
    else:
        # Default values if config not provided
        baseline_no_pv_cost = 9424.48
        baseline_pv_only_cost = 8246.44
    
    # Set defaults for any missing parameters
    feed_in_tariff = 0.033 if feed_in_tariff is None else feed_in_tariff
    peak_rate = 0.3971 if peak_rate is None else peak_rate
    offpeak_rate = 0.1353 if offpeak_rate is None else offpeak_rate
    discount_rate = 0.07 if discount_rate is None else discount_rate
    escalation_rate = 0.03 if escalation_rate is None else escalation_rate
    maintenance_inflation_rate = 0.03 if maintenance_inflation_rate is None else maintenance_inflation_rate
    project_lifetime = 30 if project_lifetime is None else project_lifetime
    
    # Get battery power ratio from config if available
    battery_power_ratio = 0.5  # Default: 0.5C
    if config is not None and hasattr(config, 'BATTERY_POWER_RATIO'):
        battery_power_ratio = config.BATTERY_POWER_RATIO
    
    # Calculate battery power rating (kW)
    battery_kw = battery_kwh * battery_power_ratio if battery_kwh > 0 else 0
    
    # 1) CAPEX
    # Calculate PV cost using the specialized function
    pv_cost_multiplier = 1.0  # Default multiplier
    if additional_pv_kw > 0:
        # If adding PV, determine cost based on which option is being used
        if config is not None and hasattr(config, 'PV_OPTIONS') and config.PRIORITIZE_PV_ALLOCATION:
            # Allocate PV and get cost using the prioritized options
            from pv import allocate_pv_capacity
            allocated_pv = allocate_pv_capacity(additional_pv_kw, config.PV_OPTIONS)
            capex_pv = sum(
                calculate_pv_cost(pv['system_capacity_kw'], pv.get('cost_multiplier', 1.0)) 
                for pv in allocated_pv
            )
        else:
            # Simple calculation without allocation
            capex_pv = calculate_pv_cost(additional_pv_kw, pv_cost_multiplier)
    else:
        capex_pv = 0.0
    
    # Calculate battery cost
    capex_batt = calculate_battery_cost(battery_kwh) if battery_kwh > 0 else 0.0
    
    # Total capital expenditure
    capex_total = capex_pv + capex_batt
    
    # 2) Calculate annual maintenance costs
    # Get maintenance costs from config if available
    pv_maintenance_per_year = 250.0  # Default: $250/year
    battery_maintenance_per_year = 0.0  # Default: $0/year
    
    if config is not None:
        if hasattr(config, 'ANNUAL_MAINTENANCE_PV'):
            pv_maintenance_per_year = config.ANNUAL_MAINTENANCE_PV
        if hasattr(config, 'ANNUAL_MAINTENANCE_BATTERY'):
            battery_maintenance_per_year = config.ANNUAL_MAINTENANCE_BATTERY
    
    # Calculate maintenance based on capacity
    annual_pv_maintenance = pv_maintenance_per_year if additional_pv_kw > 0 else 0.0
    annual_battery_maintenance = battery_maintenance_per_year if battery_kwh > 0 else 0.0
    annual_maintenance = annual_pv_maintenance + annual_battery_maintenance
    
    # 3) Calculate year-by-year electricity costs with escalation
    # Extract energy values for the entire project lifetime
    total_grid_import_peak = totals['total_grid_import_peak'] 
    total_grid_import_offpeak = totals['total_grid_import_offpeak']
    total_pv_export = totals['total_pv_export']
    
    # Calculate average annual values
    annual_grid_import_peak = total_grid_import_peak / project_lifetime
    annual_grid_import_offpeak = total_grid_import_offpeak / project_lifetime
    annual_pv_export = total_pv_export / project_lifetime
    
    # Calculate total baseline costs over project lifetime
    baseline_no_pv_lifetime = baseline_no_pv_cost * sum((1 + escalation_rate)**year for year in range(project_lifetime))
    baseline_pv_only_lifetime = baseline_pv_only_cost * sum((1 + escalation_rate)**year for year in range(project_lifetime))
    
    # Get baseline annual costs
    annual_no_pv_cost = baseline_no_pv_cost
    annual_pv_only_cost = baseline_pv_only_cost
    
    # Calculate year-by-year costs with price escalation
    annual_costs = []
    for year in range(project_lifetime):
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
    
    # Zero edge case check
    if capex_total == 0:
        # No investment case (0 additional PV, 0 battery)
        irr_value = None  # IRR is undefined for no investment
        npv_value = 0.0   # NPV is zero for no investment
        profitability_index = None  # PI is undefined for no investment
        simple_payback = 0.0  # No payback needed for no investment
    else:
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
        
        # Calculate Profitability Index (PI)
        profitability_index = npv_value / capex_total if capex_total > 0 else None
        
        # Simple payback period
        if annual_costs[0]['savings_vs_pv_only'] > 0:
            simple_payback = capex_total / annual_costs[0]['savings_vs_pv_only']
        else:
            simple_payback = float('inf')
    
    # Calculate total costs over the project lifetime
    total_import_cost = sum(year_data['import_cost'] for year_data in annual_costs)
    total_export_revenue = sum(year_data['export_revenue'] for year_data in annual_costs)
    total_electricity_cost = sum(year_data['electricity_cost'] for year_data in annual_costs)
    total_maintenance = sum(year_data['maintenance'] for year_data in annual_costs)
    total_cost = total_electricity_cost + total_maintenance
    total_savings_vs_no_pv = sum(year_data['savings_vs_no_pv'] for year_data in annual_costs)
    total_savings_vs_pv_only = sum(year_data['savings_vs_pv_only'] for year_data in annual_costs)
    
    # Estimate total generation over the project lifetime
    if 'total_pv_used' in totals and 'total_pv_export' in totals:
        total_generation = totals['total_pv_used'] + totals['total_pv_export']
    else:
        # Fallback if direct generation data isn't available
        total_generation = annual_pv_export * project_lifetime * 30  # Rough estimate

    # Calculate LCOE
    lcoe = calculate_lcoe(
        total_generation=total_generation,
        capital_cost=capex_total,
        annual_maintenance=annual_maintenance,
        maintenance_inflation_rate=maintenance_inflation_rate,
        discount_rate=discount_rate,
        years=project_lifetime
    )
    
    # Return financial metrics
    return {
        'capex_pv': capex_pv,
        'capex_battery': capex_batt,
        'capex_total': capex_total,
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
        'profitability_index': profitability_index,
        'simple_payback': simple_payback,
        'lcoe': lcoe,
        'initial_investment': capex_total  # Added for clarity in results
    }