import pandas as pd
from battery_30 import simulate_battery_dispatch
from financial_30 import compute_financials
import numpy as np
from pv_simulate_30 import simulate_multi_year_pv

# These will be set by the main script
BASE_IMPORT_COST = 0.0  # No PV, no battery cost
PV_ONLY_COST = 0.0      # PV-only, no battery cost
WEATHER_FILES = []      # Will be set by main script
START_YEARS = []        # Will be set by main script

# Existing PV system parameters
EXISTING_PV_CAPACITY = 10.0  # Your current PV system size in kW

# Define the roof parameters for additional PV
# Type A - Existing roof configuration
TYPE_A_ROOF_PARAMS = {
    'name': 'existing_roof_extension',
    'tilt': 12.5,
    'azimuth': 37.0,
    'shading': 6.18,
    'array_type': 1,  # Fixed roof mount
}

# Type B - Optimal angle installation
TYPE_B_ROOF_PARAMS = {
    'name': 'optimal_angle_installation',
    'tilt': 30.0,
    'azimuth': 5.0,
    'shading': 0.0,
    'array_type': 0,  # Fixed open rack
}

def evaluate_solution(
    params: dict,
    pv_profile: pd.DataFrame,
    demand_profile: pd.Series
) -> list:
    """
    Evaluate a solution for NSGA-II optimization with additional PV capacity.
    
    Parameters
    ----------
    params : dict
        Dictionary containing:
        - 'battery_kwh': Battery size in kWh
        - 'additional_pv_kw': Additional PV capacity in kW
        - 'allocation_factor': Percentage (0-1) to allocate to Type A (existing roof)
                               remainder goes to Type B (optimal angle)
    pv_profile : pd.DataFrame
        Existing PV generation profile with 'simulated_kwh' column.
    demand_profile : pd.Series
        Energy demand profile.
        
    Returns
    -------
    list
        [–IRR, -NPV, emissions] for NSGA-II minimization.
    """
    # Extract parameters
    battery_kwh = params['battery_kwh']
    additional_pv_kw = params['additional_pv_kw']
    allocation_factor = params['allocation_factor']
    
    # Calculate the allocation of additional PV capacity
    type_a_capacity = min(additional_pv_kw * allocation_factor, 26.40)  # Capped at max capacity
    type_b_capacity = additional_pv_kw - type_a_capacity
    
    # Apply default battery power if not specified (0.5C rate)
    battery_kw = params.get('battery_kw', battery_kwh * 0.5)
    
    # Only simulate additional PV if there's any
    if additional_pv_kw > 0:
        # Create roof parameters for simulation
        additional_roof_params = []
        
        # Add Type A if capacity > 0
        if type_a_capacity > 0:
            type_a_params = TYPE_A_ROOF_PARAMS.copy()
            type_a_params['system_capacity_kw'] = type_a_capacity
            additional_roof_params.append(type_a_params)
        
        # Add Type B if capacity > 0
        if type_b_capacity > 0:
            type_b_params = TYPE_B_ROOF_PARAMS.copy()
            type_b_params['system_capacity_kw'] = type_b_capacity
            additional_roof_params.append(type_b_params)
        
        # Simulate additional PV generation
        additional_pv_profile = simulate_multi_year_pv(
            weather_files=WEATHER_FILES,
            roof_params=additional_roof_params,
            repeats_per_file=10,
            start_years=START_YEARS
        )
        
        # Combine existing and additional PV generation
        total_pv_kwh = pv_profile['simulated_kwh'] + additional_pv_profile['simulated_kwh']
        total_pv_profile = pd.DataFrame({'simulated_kwh': total_pv_kwh})
        
        # Update total PV capacity
        total_pv_capacity = EXISTING_PV_CAPACITY + additional_pv_kw
    else:
        # No additional PV, use existing profile and capacity
        total_pv_profile = pv_profile
        total_pv_capacity = EXISTING_PV_CAPACITY
    
    # Extract PV generation series
    if hasattr(total_pv_profile, 'columns'):
        gen = total_pv_profile['simulated_kwh']
    else:
        gen = total_pv_profile
    
    # Simulate battery dispatch with updated parameters
    dispatch_df, totals = simulate_battery_dispatch(
        pv_gen=gen,
        demand=demand_profile,
        battery_kwh=battery_kwh,
        battery_kw=battery_kw,
        roundtrip_eff=0.9,           # 90% round trip efficiency
        min_soc_pct=0.05,            # 95% depth of discharge (5% min SOC)
        annual_deg_rate=0.01,        # 1% degradation per year
        grid_emission_rate=0.81      # 0.81 kg CO2e/kWh
    )
    
    # Calculate PV capital costs
    pv_cost_per_kw = 1500.0  # Base cost per kW for PV
    pv_type_a_cost = type_a_capacity * pv_cost_per_kw  # Existing roof installation
    pv_type_b_cost = type_b_capacity * (pv_cost_per_kw * 1.15)  # 15% premium for optimal-angle installation
    pv_installation_cost = 1000.0 if additional_pv_kw > 0 else 0.0  # Fixed installation overhead
    total_pv_cost = pv_type_a_cost + pv_type_b_cost + pv_installation_cost
    
    # Compute financial metrics with updated parameters
    fin = compute_financials(
        totals,
        battery_kwh=battery_kwh,
        pv_kw=total_pv_capacity,                 
        pv_cost_per_kw=total_pv_cost/additional_pv_kw if additional_pv_kw > 0 else 0.0,  # Effective cost per kW
        pv_installation_cost=0.0,              # Already included in total_pv_cost
        battery_cost_per_kwh=None,             # Use formula: 977.54 * e^(-0.004*x) with $600 minimum
        battery_installation_cost_per_kwh=174.0,
        battery_power_ratio=0.5,               # Power rating as fraction of capacity
        pv_maintenance_per_kw_day=0.13,        # $0.13 per kW per day for PV
        battery_maintenance_per_kw_day=0.12,   # $0.12 per kW per day for battery
        discount_rate=0.07,
        baseline_import_cost=BASE_IMPORT_COST,  # No PV, no battery cost
        baseline_pv_only_cost=PV_ONLY_COST      # PV-only, no battery cost
    )
    
    # Return objectives to minimize: negative IRR, negative NPV, and CO2 emissions
    irr = fin['irr'] or 0.0  # Handle None case
    npv = -fin['net_cost']   # Convert NPC to NPV
    emissions = totals['total_grid_emissions']  # Total CO2 emissions
    
    return [-irr, -npv, emissions]