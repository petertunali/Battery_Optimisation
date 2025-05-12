# obj.py - objective function for optimization
import pandas as pd
import numpy as np
from pv import simulate_multi_year_pv
from battery import simulate_battery_dispatch
from fin import compute_financials

# These will be set by the main script
BASE_IMPORT_COST = 0.0      # No PV, no battery cost
PV_ONLY_COST = 0.0          # PV-only, no battery cost
WEATHER_FILES = []          # Weather files list
START_YEARS = []            # Starting years for simulation
EXISTING_PV_CAPACITY = 10.0 # Default existing PV capacity
PV_OPTIONS = []             # PV installation options

# Electricity pricing
BASE_PEAK_RATE = 0.39710     # Base peak rate
BASE_OFFPEAK_RATE = 0.13530  # Base off-peak rate
BASE_EXPORT_RATE = 0.033     # Base export rate
ESCALATION_RATE = 0.03       # Annual price escalation rate (3%)

def calculate_pv_cost(capacity_kw, cost_multiplier=1.0):
    """
    Calculate PV cost using the formula: y = 1047.3 * e^(-0.002*x) with minimum $750
    
    Args:
        capacity_kw: PV capacity in kW
        cost_multiplier: Multiplier for special installations (e.g., 1.2 for ground-mounted)
        
    Returns:
        cost_per_kw: Cost per kW in dollars
    """
    if capacity_kw <= 0:
        return 0.0
    
    # Apply economies of scale formula
    cost_per_kw = 1047.3 * np.exp(-0.002 * capacity_kw)
    
    # Apply minimum cost of $750/kW
    cost_per_kw = max(750.0, cost_per_kw)
    
    # Apply cost multiplier
    return cost_per_kw * cost_multiplier

def allocate_pv_capacity(total_capacity_kw, options):
    """
    Allocate PV capacity across available options based on priority.
    
    Args:
        total_capacity_kw: Total additional PV capacity to allocate
        options: List of PV options with max_capacity_kw and other parameters
        
    Returns:
        allocated_pv: List of PV configurations with allocated capacity
    """
    remaining_capacity = total_capacity_kw
    allocated_pv = []
    
    for option in options:
        option_copy = option.copy()
        # Allocate capacity to this option (limited by max capacity)
        allocation = min(remaining_capacity, option['max_capacity_kw'])
        
        if allocation > 0:
            option_copy['system_capacity_kw'] = allocation
            allocated_pv.append(option_copy)
            remaining_capacity -= allocation
        
        if remaining_capacity <= 0:
            break
    
    return allocated_pv

def evaluate_solution(params, pv_profile, demand_profile):
    """
    Evaluate a solution for NSGA-II optimization.
    
    Parameters
    ----------
    params : dict
        Dictionary containing:
        - 'battery_kwh': Battery size in kWh
        - 'additional_pv_kw': Additional PV capacity in kW
    pv_profile : pd.DataFrame
        Existing PV generation profile with 'simulated_kwh' column.
    demand_profile : pd.Series
        Energy demand profile.
        
    Returns
    -------
    list
        [–IRR, -NPV] for NSGA-II minimization.
    """
    # Extract parameters
    battery_kwh = params['battery_kwh']
    additional_pv_kw = params['additional_pv_kw']
    
    # Apply default battery power if not specified (0.5C rate)
    battery_kw = params.get('battery_kw', battery_kwh * 0.5)
    
    # Only simulate additional PV if there's any
    if additional_pv_kw > 0:
        # Allocate capacity to different PV options
        allocated_pv = allocate_pv_capacity(additional_pv_kw, PV_OPTIONS)
        
        # Create configuration for simulation with existing PV
        existing_pv = {
            'name': 'existing_system',
            'system_capacity_kw': EXISTING_PV_CAPACITY,
            'tilt': 10.0,
            'azimuth': 18.0,
            'shading': 43.0,
            'array_type': 1
        }
        all_pv = [existing_pv] + allocated_pv
        
        # Simulate combined PV generation
        combined_pv_profile = simulate_multi_year_pv(
            weather_files=WEATHER_FILES,
            roof_params=all_pv,
            repeats_per_file=10,
            start_years=START_YEARS
        )
        
        # Update total PV capacity
        total_pv_capacity = EXISTING_PV_CAPACITY + additional_pv_kw
        total_pv_profile = combined_pv_profile
    else:
        # No additional PV, use existing profile and capacity
        total_pv_profile = pv_profile
        total_pv_capacity = EXISTING_PV_CAPACITY
    
    # Extract PV generation series
    if hasattr(total_pv_profile, 'columns'):
        gen = total_pv_profile['simulated_kwh']
    else:
        gen = total_pv_profile
    
    # Simulate battery dispatch
    dispatch_df, totals = simulate_battery_dispatch(
        pv_gen=gen,
        demand=demand_profile,
        battery_kwh=battery_kwh,
        battery_kw=battery_kw,
        roundtrip_eff=0.9,           # 90% round trip efficiency
        min_soc_pct=0.05,            # 95% depth of discharge (5% min SOC)
        annual_deg_rate=0.01,        # Fallback rate if NREL model fails
        grid_emission_rate=0.81      # 0.81 kg CO2e/kWh
    )
    
    # Calculate PV capital costs
    pv_capital_cost = 0
    if additional_pv_kw > 0:
        # Calculate costs for each allocated PV component
        for pv_config in allocated_pv:
            capacity = pv_config['system_capacity_kw']
            cost_multiplier = pv_config.get('cost_multiplier', 1.0)
            cost_per_kw = calculate_pv_cost(capacity, cost_multiplier)
            pv_capital_cost += capacity * cost_per_kw
        
        # Add fixed installation cost
        pv_capital_cost += 1000.0
    
    # Compute financial metrics with updated parameters
    fin = compute_financials(
        totals,
        battery_kwh=battery_kwh,
        pv_kw=total_pv_capacity,                 
        additional_pv_cost=pv_capital_cost,    # Total cost for additional PV
        battery_cost_per_kwh=None,             # Use internal formula
        battery_installation_cost_per_kwh=174.0,
        battery_power_ratio=0.5,               # Power rating as fraction of capacity
        pv_maintenance_per_day=0.68,           # $250/year = $0.68/day
        battery_maintenance_per_kw_day=0.0,    # No battery maintenance as requested
        discount_rate=0.07,
        peak_rate=BASE_PEAK_RATE,
        offpeak_rate=BASE_OFFPEAK_RATE,
        feed_in_tariff=BASE_EXPORT_RATE,
        escalation_rate=ESCALATION_RATE,       # 3% annual escalation
        baseline_import_cost=BASE_IMPORT_COST,  # No PV, no battery cost
        baseline_pv_only_cost=PV_ONLY_COST      # PV-only, no battery cost
    )
    
    # Return objectives to minimize: negative IRR, negative NPV
    irr = fin['irr'] or 0.0  # Handle None case
    npv = -fin['net_cost']   # Convert NPC to NPV
    
    return [-irr, -npv]