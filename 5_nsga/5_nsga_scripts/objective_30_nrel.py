import pandas as pd
from battery_30_nrel import simulate_battery_dispatch
from financial_30 import compute_financials
# This will be set by the main script during initialization
BASE_IMPORT_COST = 0.0
def evaluate_solution(
    params: dict,
    pv_profile: pd.DataFrame,
    demand_profile: pd.Series
) -> list:
    """
    Evaluate a solution for NSGA-II optimization with NREL battery model.
    
    Parameters
    ----------
    params : dict
        Dictionary containing 'battery_kwh' and 'pv_kw' values.
        May also contain 'battery_kw' if not using default power.
    pv_profile : pd.DataFrame or pd.Series
        PV generation profile with 'simulated_kwh' column or as Series.
    demand_profile : pd.Series
        Energy demand profile.
        
    Returns
    -------
    list
        [–IRR, -NPV] for NSGA-II minimization.
    """
    # Extract PV generation series
    if hasattr(pv_profile, 'columns'):
        gen = pv_profile['simulated_kwh']
    else:
        gen = pv_profile
    # Apply default battery power if not specified (0.5C rate)
    battery_kwh = params['battery_kwh']
    battery_kw = params.get('battery_kw', battery_kwh * 0.5)
    
    # Simulate battery dispatch with updated parameters
    dispatch_df, totals = simulate_battery_dispatch(
        pv_gen=gen,
        demand=demand_profile,
        battery_kwh=battery_kwh,
        battery_kw=battery_kw,
        roundtrip_eff=0.9,           # 90% round trip efficiency
        min_soc_pct=0.05,            # 95% depth of discharge (5% min SOC)
        annual_deg_rate=0.01,        # 1% degradation per year (fallback)
        grid_emission_rate=0.81      # 0.81 kg CO2e/kWh
    )
    # Compute financial metrics with updated parameters
    # For existing PV system, we use the PV size but set the cost to zero
    fin = compute_financials(
        totals,
        battery_kwh=battery_kwh,
        pv_kw=params['pv_kw'],                 # Keep PV size for operational calculations
        pv_cost_per_kw=0.0,                    # No PV capital cost (already installed)
        pv_installation_cost=0.0,              # No PV installation cost (already installed)
        battery_cost_per_kwh=None,             # Use formula: 977.54 * e^(-0.004*x) with $600 minimum
        battery_installation_cost_per_kwh=174.0,
        battery_power_ratio=0.5,               # Power rating as fraction of capacity
        pv_maintenance_per_kw_day=0.13,        # $0.13 per kW per day for PV
        battery_maintenance_per_kw_day=0.12,   # $0.12 per kW per day for battery
        discount_rate=0.07,
        baseline_import_cost=BASE_IMPORT_COST
    )
    # Return objectives to minimize: negative IRR and negative NPV
    irr = fin['irr'] or 0.0  # Handle None case
    npv = -fin['net_cost']   # Convert NPC to NPV
    
    return [-irr, -npv]