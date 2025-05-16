# obj.py - objective function for optimization
import pandas as pd
import numpy as np
from pymoo.core.problem import Problem

def evaluate_solution(params, pv_profile, demand_profile, config=None):
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
    config : object, optional
        Configuration parameters
        
    Returns
    -------
    list
        [–IRR, -NPV] for NSGA-II minimization.
    """
    # Import required modules - import here to avoid circular dependencies
    from battery import simulate_battery_dispatch
    from pv import simulate_multi_year_pv, allocate_pv_capacity
    from fin import compute_financials
    
    # Extract parameters
    battery_kwh = params['battery_kwh']
    additional_pv_kw = params['additional_pv_kw']
    
    # Get configuration parameters or use defaults
    if config is not None:
        # PV parameters
        existing_pv = config.EXISTING_PV
        pv_options = config.PV_OPTIONS
        weather_files = config.DESIRED_WEATHER_FILES
        # Time periods for different weather files
        start_years = getattr(config, 'START_YEARS', [2025, 2040, 2050])
        # Battery parameters
        battery_eff = config.BATTERY_EFF
        min_soc = config.MIN_SOC
        battery_power_ratio = config.BATTERY_POWER_RATIO
        battery_degradation = config.BATTERY_DEGRADATION
        grid_emissions = config.GRID_EMISSIONS
        # Baseline costs
        baseline_no_pv_cost = config.ANNUAL_NO_PV_COST
        baseline_pv_only_cost = config.ANNUAL_PV_ONLY_COST
    else:
        # Default parameters if config is not available
        existing_pv = {
            'name': 'existing_system',
            'system_capacity_kw': 10.0,
            'tilt': 10.0,
            'azimuth': 18.0,
            'shading': 43.0,
            'array_type': 1
        }
        pv_options = []
        weather_files = []
        start_years = [2025, 2040, 2050]
        battery_eff = 0.9
        min_soc = 0.05
        battery_power_ratio = 0.5
        battery_degradation = 0.01
        grid_emissions = 0.79
        baseline_no_pv_cost = 9424.48
        baseline_pv_only_cost = 8246.44
    
    # Apply default battery power if not specified
    battery_kw = params.get('battery_kw', battery_kwh * battery_power_ratio)
    
    # Only simulate additional PV if there's any
    if additional_pv_kw > 0:
        # Allocate capacity to different PV options
        allocated_pv = allocate_pv_capacity(additional_pv_kw, pv_options)
        
        # Create configuration for simulation with existing PV
        all_pv = [existing_pv] + allocated_pv
        
        # Simulate combined PV generation
        combined_pv_profile = simulate_multi_year_pv(
            weather_files=weather_files,
            roof_params=all_pv,
            repeats_per_file=10,
            start_years=start_years
        )
        
        # Update total PV capacity
        total_pv_capacity = existing_pv['system_capacity_kw'] + additional_pv_kw
        total_pv_profile = combined_pv_profile
    else:
        # No additional PV, use existing profile and capacity
        total_pv_profile = pv_profile
        total_pv_capacity = existing_pv['system_capacity_kw']
    
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
        roundtrip_eff=battery_eff,
        min_soc_pct=min_soc,
        annual_deg_rate=battery_degradation,
        grid_emission_rate=grid_emissions,
        config=config  # Pass config for advanced options
    )
    
    # Compute financial metrics
    fin = compute_financials(
        totals,
        battery_kwh=battery_kwh,
        additional_pv_kw=additional_pv_kw,
        config=config  # Pass config for financial parameters
    )
    
    # Return objectives to minimize: negative IRR, negative NPV
    irr = fin['irr'] or 0.0  # Handle None case
    npv = fin['npv']
    
    return [-irr, -npv]  # Return negative values for minimization

class BatteryPVOptimizationProblem(Problem):
    """
    NSGA-II optimization problem for battery and PV system.
    """
    def __init__(self, pv_profile, demand_profile, config=None):
        """
        Initialize the optimization problem.
        
        Parameters
        ----------
        pv_profile : pd.DataFrame
            Existing PV generation profile
        demand_profile : pd.Series
            Electricity demand profile
        config : object, optional
            Configuration parameters
        """
        # Store inputs
        self.pv_profile = pv_profile
        self.demand_profile = demand_profile
        self.config = config
        
        # Define variable ranges
        if config is not None:
            # Get max battery size if specified
            max_battery = getattr(config, 'MAX_BATTERY_SIZE', 100)
            # Get total available PV capacity
            max_pv = sum(opt['max_capacity_kw'] for opt in config.PV_OPTIONS 
                         if opt['max_capacity_kw'] != float('inf'))
            # Add unlimited capacity if available
            if any(opt['max_capacity_kw'] == float('inf') for opt in config.PV_OPTIONS):
                max_pv = max(max_pv, 100)  # Allow up to 100 kW if unlimited is available
        else:
            # Default ranges if config not provided
            max_battery = 100
            max_pv = 100
        
        # Define variable bounds: [battery_kwh, additional_pv_kw]
        xl = [0.0, 0.0]  # Lower bounds
        xu = [max_battery, max_pv]  # Upper bounds
        
        # Initialize problem with 2 variables, 2 objectives, 0 constraints
        super().__init__(n_var=2, n_obj=2, n_constr=0, xl=xl, xu=xu)
    
    def _evaluate(self, x, out, *args, **kwargs):
        """
        Evaluate solutions for the NSGA-II algorithm.
        
        Parameters
        ----------
        x : numpy.ndarray
            Array of solution variables [battery_kwh, additional_pv_kw]
        out : dict
            Output dictionary for objectives and constraints
        """
        # Initialize arrays for objectives
        n_solutions = x.shape[0]
        f1 = np.zeros(n_solutions)  # -IRR
        f2 = np.zeros(n_solutions)  # -NPV
        
        # Evaluate each solution
        for i in range(n_solutions):
            battery_kwh = float(x[i, 0])
            additional_pv_kw = float(x[i, 1])
            
            # Create solution dictionary
            solution = {
                'battery_kwh': battery_kwh,
                'additional_pv_kw': additional_pv_kw
            }
            
            # Evaluate the solution
            objectives = evaluate_solution(
                solution, 
                self.pv_profile, 
                self.demand_profile,
                self.config
            )
            
            # Store objectives
            f1[i] = objectives[0]  # -IRR
            f2[i] = objectives[1]  # -NPV
        
        # Set outputs
        out["F"] = np.column_stack([f1, f2])