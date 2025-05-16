#results.py

# results.py - Results processing and visualization module for battery optimization

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import os
from pathlib import Path
import time
from datetime import datetime, timedelta
import json

def setup_results_directory(base_results_dir):
    """
    Create a new numbered results directory for this optimization run.
    
    Args:
        base_results_dir: Base directory path for results
        
    Returns:
        Path: Path to the new results directory
    """
    # Create base results directory if it doesn't exist
    if not os.path.exists(base_results_dir):
        os.makedirs(base_results_dir)
    
    # Generate a unique run identifier
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(base_results_dir) / f"run_{timestamp}"
    run_dir.mkdir(exist_ok=True)
    
    # Create subdirectories for organization
    (run_dir / "plots").mkdir(exist_ok=True)
    (run_dir / "data").mkdir(exist_ok=True)
    
    return run_dir

def calculate_present_value(annual_amount, discount_rate, years=30, escalation_rate=0.03):
    """
    Calculate present value of an escalating annual amount.
    
    Args:
        annual_amount: Annual amount in first year
        discount_rate: Discount rate for NPV calculation
        years: Project lifetime in years
        escalation_rate: Annual escalation rate
        
    Returns:
        float: Present value of the stream
    """
    total_pv = 0
    for year in range(years):
        escalated_amount = annual_amount * (1 + escalation_rate)**year
        present_value = escalated_amount / (1 + discount_rate)**(year+1)
        total_pv += present_value
    return total_pv

def visualize_battery_operation(dispatch_df, date_str, output_file=None):
    """
    Create a diagnostic visualization of battery operation for a single day.
    
    Args:
        dispatch_df: DataFrame with battery dispatch data
        date_str: Date string to visualize (YYYY-MM-DD)
        output_file: Path to save the figure
        
    Returns:
        dict: Summary statistics for the day
    """
    # Extract data for the specified date
    day_data = dispatch_df[dispatch_df.index.date == pd.Timestamp(date_str).date()]
    
    if len(day_data) == 0:
        print(f"No data available for {date_str}")
        return None
    
    # Create plot
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    
    # Energy flows
    ax1 = axes[0]
    ax1.plot(day_data.index.strftime('%H:%M'), day_data['pv_gen'], 'orange', label='PV Generation')
    ax1.plot(day_data.index.strftime('%H:%M'), day_data['demand'], 'blue', label='Demand')
    ax1.fill_between(day_data.index.strftime('%H:%M'), 0, day_data['is_peak'], alpha=0.2, color='pink', label='Peak Period')
    ax1.set_title('Energy Generation and Demand')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Battery charging/discharging
    ax2 = axes[1]
    ax2.plot(day_data.index.strftime('%H:%M'), day_data['battery_charge'], 'green', label='Battery Charging')
    ax2.plot(day_data.index.strftime('%H:%M'), day_data['battery_discharge'], 'red', label='Battery Discharging')
    if 'grid_to_battery' in day_data.columns:
        ax2.plot(day_data.index.strftime('%H:%M'), day_data['grid_to_battery'], 'purple', label='Grid→Battery')
    ax2.fill_between(day_data.index.strftime('%H:%M'), 0, day_data['is_peak'], alpha=0.2, color='pink')
    ax2.set_title('Battery Charging/Discharging')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Battery SOC and grid imports
    ax3 = axes[2]
    ax3.plot(day_data.index.strftime('%H:%M'), day_data['battery_soc'], 'green', label='Battery SOC')
    ax3.plot(day_data.index.strftime('%H:%M'), day_data['grid_import_peak'] + day_data['grid_import_offpeak'], 
             'red', label='Grid Import')
    soc_max = day_data['battery_soc'].max() * 1.1  # Add some headroom
    ax3.set_ylim(0, soc_max)
    
    # Add target SOC lines
    if 'battery_soc' in day_data and day_data['battery_soc'].max() > 0:
        capacity = day_data['battery_soc'].max() / 0.95  # Estimate capacity from max SOC
        ax3.axhline(y=capacity * 0.8, color='blue', linestyle='--', alpha=0.5, label='80% Target')
        ax3.axhline(y=capacity * 0.6, color='orange', linestyle='--', alpha=0.5, label='60% Min (Off-Peak)')
        ax3.axhline(y=capacity * 0.05, color='red', linestyle='--', alpha=0.5, label='5% Absolute Min')
    
    ax3.fill_between(day_data.index.strftime('%H:%M'), 0, day_data['is_peak'], alpha=0.2, color='pink')
    ax3.set_title('Battery State of Charge and Grid Imports')
    ax3.set_xlabel('Time of Day')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Rotate x-axis labels for better readability
    for ax in axes:
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300)
    
    # Show plot if not in a notebook environment
    if not 'ipykernel' in sys.modules:
        plt.show()
    
    plt.close()
    
    # Compile summary statistics
    summary_stats = {
        'Date': date_str,
        'Total PV Generation': day_data['pv_gen'].sum(),
        'Total Demand': day_data['demand'].sum(),
        'Peak Hours': day_data['is_peak'].sum() * 0.5,
        'Battery Charge': day_data['battery_charge'].sum(),
        'Battery Discharge': day_data['battery_discharge'].sum(),
        'Grid Import (Peak)': day_data['grid_import_peak'].sum(),
        'Grid Import (Off-Peak)': day_data['grid_import_offpeak'].sum(),
        'PV Export': day_data['pv_export'].sum()
    }
    
    if 'grid_to_battery' in day_data.columns:
        summary_stats['Grid-to-Battery Charging'] = day_data['grid_to_battery'].sum()
    
    # Print summary
    print(f"Daily Summary for {date_str}:")
    for key, value in summary_stats.items():
        if isinstance(value, float):
            print(f"  • {key}: {value:.2f}")
        else:
            print(f"  • {key}: {value}")
            
    return summary_stats

def create_pareto_plot(df, run_dir):
    """
    Create a visualization of the Pareto front of solutions.
    
    Args:
        df: DataFrame with optimization results
        run_dir: Directory to save results
        
    Returns:
        dict: Dictionary with indices of important solutions
    """
    # Find best IRR and NPV indices
    best_irr_idx = df['irr'].idxmax()
    best_npv_idx = df['npv'].idxmax()
    
    # Calculate balanced solution (closest to utopia point)
    irr_min, irr_max = df['irr'].min(), df['irr'].max()
    npv_min, npv_max = df['npv'].min(), df['npv'].max()
    irr_norm = (df['irr'] - irr_min) / (irr_max - irr_min)
    npv_norm = (df['npv'] - npv_min) / (npv_max - npv_min)
    
    # Calculate distance from ideal point [1,1]
    df['distance'] = np.sqrt((1-irr_norm)**2 + (1-npv_norm)**2)
    balanced_idx = df['distance'].idxmin()
    
    # Create enhanced visualization
    plt.figure(figsize=(14, 10))
    
    # Create primary scatter plot colored by battery size
    scatter = plt.scatter(df['irr']*100, df['npv']/1000, 
                         c=df['battery_kwh'], s=60, 
                         cmap='viridis', alpha=0.8, edgecolors='gray')
    
    # Add Pareto front line connecting the points in IRR order
    sorted_df = df.sort_values('irr')
    plt.plot(sorted_df['irr']*100, sorted_df['npv']/1000, 'k--', alpha=0.6)
    
    # Add PV size labels for every 5th point
    step = max(1, len(df) // 10)  # Show about 10 labels
    for i, (_, row) in enumerate(sorted_df.iterrows()):
        if i % step == 0:
            plt.annotate(f"{row['additional_pv_kw']:.1f} kW", 
                         xy=(row['irr']*100, row['npv']/1000),
                         fontsize=8, alpha=0.7,
                         xytext=(5, 0), textcoords='offset points')
    
    # Highlight the key solutions
    plt.scatter(df.loc[best_irr_idx, 'irr']*100, df.loc[best_irr_idx, 'npv']/1000, 
                s=180, color='blue', edgecolor='black', zorder=5, label='Best IRR')
    plt.scatter(df.loc[best_npv_idx, 'irr']*100, df.loc[best_npv_idx, 'npv']/1000, 
                s=180, color='green', edgecolor='black', zorder=5, label='Best NPV')
    plt.scatter(df.loc[balanced_idx, 'irr']*100, df.loc[balanced_idx, 'npv']/1000, 
                s=180, color='red', edgecolor='black', zorder=5, label='Balanced Solution')
    
    # Annotate best points
    plt.annotate(f"Best IRR: {df.loc[best_irr_idx, 'irr']*100:.1f}%\nBatt: {df.loc[best_irr_idx, 'battery_kwh']:.1f} kWh\nPV: {df.loc[best_irr_idx, 'additional_pv_kw']:.1f} kW",
                 xy=(df.loc[best_irr_idx, 'irr']*100, df.loc[best_irr_idx, 'npv']/1000),
                 xytext=(15, -30), textcoords='offset points',
                 arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'),
                 fontsize=10, fontweight='bold')
    
    plt.annotate(f"Best NPV: ${df.loc[best_npv_idx, 'npv']/1000:.0f}k\nBatt: {df.loc[best_npv_idx, 'battery_kwh']:.1f} kWh\nPV: {df.loc[best_npv_idx, 'additional_pv_kw']:.1f} kW",
                 xy=(df.loc[best_npv_idx, 'irr']*100, df.loc[best_npv_idx, 'npv']/1000),
                 xytext=(-70, 30), textcoords='offset points',
                 arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'),
                 fontsize=10, fontweight='bold')
    
    plt.annotate(f"Balanced:\nBatt: {df.loc[balanced_idx, 'battery_kwh']:.1f} kWh\nPV: {df.loc[balanced_idx, 'additional_pv_kw']:.1f} kW\nIRR: {df.loc[balanced_idx, 'irr']*100:.1f}%\nNPV: ${df.loc[balanced_idx, 'npv']/1000:.0f}k",
                 xy=(df.loc[balanced_idx, 'irr']*100, df.loc[balanced_idx, 'npv']/1000),
                 xytext=(20, 30), textcoords='offset points',
                 arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'),
                 fontsize=10, fontweight='bold')
    
    # Add colorbar for battery size
    cbar = plt.colorbar(scatter, label='Battery Size (kWh)')
    
    # Add legend for reference lines
    from matplotlib.lines import Line2D
    custom_lines = [
        Line2D([0], [0], color='k', linestyle='--', alpha=0.6, label='Pareto Front'),
    ]
    plt.legend(handles=custom_lines, loc='upper right')
    
    # Configure plot formatting
    plt.xlabel('Internal Rate of Return (%)', fontsize=12)
    plt.ylabel('Net Present Value ($000s)', fontsize=12)
    plt.title('Financial Trade-off Analysis: IRR vs NPV', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    
    # Save the figure
    plot_path = run_dir / "plots" / "pareto_front_irr_npv.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Pareto front visualization saved to {plot_path}")
    
    # Return indices of important solutions
    return {
        'best_irr_idx': best_irr_idx,
        'best_npv_idx': best_npv_idx,
        'balanced_idx': balanced_idx
    }

def create_convergence_plot(callback_data, run_dir):
    """
    Create visualization showing convergence of the NSGA-II algorithm.
    
    Args:
        callback_data: Data from optimization callback
        run_dir: Directory to save results
        
    Returns:
        None
    """
    # Create convergence dataframe
    convergence_df = pd.DataFrame({
        'Generation': callback_data['gen'],
        'Best_IRR': [irr * 100 for irr in callback_data['best_irr']],
        'Best_NPV': callback_data['best_npv'],
        'Battery_IRR': callback_data['batt_irr'],
        'Battery_NPV': callback_data['batt_npv'],
        'PV_IRR': callback_data['pv_irr'],
        'PV_NPV': callback_data['pv_npv'],
        'Time_Elapsed': callback_data['time_elapsed']
    })
    
    # Save convergence data
    data_path = run_dir / "data" / "convergence_data.csv"
    convergence_df.to_csv(data_path, index=False)
    print(f"✅ Convergence data saved to {data_path}")
    
    # Create convergence plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # Plot IRR convergence
    ax1.plot(callback_data['gen'], [irr * 100 for irr in callback_data['best_irr']], 'b-', linewidth=2)
    ax1.set_ylabel('Best IRR (%)', fontsize=12)
    ax1.set_title('NSGA-II Convergence: IRR and NPV', fontsize=14)
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Plot NPV convergence
    ax2.plot(callback_data['gen'], callback_data['best_npv'], 'g-', linewidth=2)
    ax2.set_xlabel('Generation', fontsize=12)
    ax2.set_ylabel('Best NPV ($)', fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    # Add annotations for final values
    ax1.annotate(f"Final IRR: {callback_data['best_irr'][-1]*100:.2f}%", 
                 xy=(callback_data['gen'][-1], callback_data['best_irr'][-1]*100),
                 xytext=(10, 0), textcoords='offset points', fontsize=10,
                 arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'))
    
    ax2.annotate(f"Final NPV: ${callback_data['best_npv'][-1]:,.0f}", 
                 xy=(callback_data['gen'][-1], callback_data['best_npv'][-1]),
                 xytext=(10, 0), textcoords='offset points', fontsize=10,
                 arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'))
    
    plt.tight_layout()
    
    # Save convergence plot
    plot_path = run_dir / "plots" / "convergence_plot.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Convergence plot saved to {plot_path}")

def analyze_solution(solution_params, pv_profile, demand_profile, config, run_dir, solution_name):
    """
    Simulate and analyze a specific solution.
    
    Args:
        solution_params: Parameters of the solution to analyze
        pv_profile: PV generation profile
        demand_profile: Demand profile
        config: Configuration parameters
        run_dir: Directory to save results
        solution_name: Name for this solution (best_irr, best_npv, balanced)
        
    Returns:
        tuple: (dispatch_df, totals) - Battery dispatch dataframe and summary metrics
    """
    from pv import simulate_multi_year_pv, allocate_pv_capacity
    from battery import simulate_battery_dispatch
    from fin import compute_financials
    import sys
    
    # Extract parameters
    battery_kwh = solution_params['battery_kwh']
    additional_pv_kw = solution_params['additional_pv_kw']
    
    print(f"\nAnalyzing {solution_name} solution:")
    print(f"Battery size: {battery_kwh:.2f} kWh")
    print(f"Additional PV: {additional_pv_kw:.2f} kW")
    
    # Configure base PV and additional PV
    allocated_pv = allocate_pv_capacity(additional_pv_kw, config.PV_OPTIONS)
    all_pv = [config.EXISTING_PV.copy()] + allocated_pv
    
    # Show PV allocation
    print("\nPV allocation:")
    allocation_text = []
    allocation_text.append(f"Existing PV: {config.EXISTING_PV['system_capacity_kw']:.2f} kW")
    
    for pv in allocated_pv:
        allocation_text.append(f"{pv['name']}: {pv['system_capacity_kw']:.2f} kW")
    
    allocation_text.append(f"Total system: {config.EXISTING_PV['system_capacity_kw'] + additional_pv_kw:.2f} kW")
    
    for text in allocation_text:
        print(f"  • {text}")
    
    # Simulate PV generation if there's additional PV
    if additional_pv_kw > 0:
        print("\nSimulating PV generation...")
        combined_pv = simulate_multi_year_pv(
            weather_files=config.DESIRED_WEATHER_FILES,
            roof_params=all_pv,
            repeats_per_file=10,
            start_years=getattr(config, 'START_YEARS', [2025, 2040, 2050])
        )
        pv_gen = combined_pv['simulated_kwh']
    else:
        # Use existing PV profile if no additional PV
        pv_gen = pv_profile['simulated_kwh']
    
    # Simulate battery dispatch
    print("\nSimulating battery dispatch...")
    dispatch_df, totals = simulate_battery_dispatch(
        pv_gen=pv_gen,
        demand=demand_profile,
        battery_kwh=battery_kwh,
        battery_kw=battery_kwh * config.BATTERY_POWER_RATIO,
        roundtrip_eff=config.BATTERY_EFF,
        min_soc_pct=config.MIN_SOC,
        annual_deg_rate=config.BATTERY_DEGRADATION,
        grid_emission_rate=config.GRID_EMISSIONS,
        config=config
    )
    
    # Compute financial metrics
    fin_results = compute_financials(
        totals,
        battery_kwh=battery_kwh,
        additional_pv_kw=additional_pv_kw,
        config=config
    )
    
    # Save allocation details
    allocation_df = pd.DataFrame({'Component': [item.split(':')[0] for item in allocation_text],
                                  'Capacity_kW': [float(item.split(':')[1].split('kW')[0]) for item in allocation_text]})
    allocation_df.to_csv(run_dir / "data" / f"{solution_name}_pv_allocation.csv", index=False)
    
    # Save dispatch data to CSV
    dispatch_df.to_csv(run_dir / "data" / f"{solution_name}_dispatch.csv")
    print(f"✅ Dispatch data saved to {run_dir/'data'}/{solution_name}_dispatch.csv")
    
    # Calculate annual summaries
    dispatch_df['year'] = dispatch_df.index.year
    annual_summary = dispatch_df.groupby('year').agg({
        'pv_gen': 'sum',
        'demand': 'sum',
        'pv_used': 'sum',
        'battery_charge': 'sum',
        'battery_discharge': 'sum',
        'pv_export': 'sum',
        'grid_import_peak': 'sum',
        'grid_import_offpeak': 'sum'
    })
    
    # Calculate derived annual metrics
    annual_summary['renewable_fraction'] = (annual_summary['pv_used'] + annual_summary['battery_discharge']) / annual_summary['demand']
    annual_summary['self_consumption'] = annual_summary['pv_used'] / (annual_summary['pv_used'] + annual_summary['pv_export'])
    annual_summary['grid_import_total'] = annual_summary['grid_import_peak'] + annual_summary['grid_import_offpeak']
    annual_summary['grid_dependence'] = annual_summary['grid_import_total'] / annual_summary['demand']
    
    # Save annual summary
    annual_summary.to_csv(run_dir / "data" / f"{solution_name}_annual_summary.csv")
    print(f"✅ Annual summary saved to {run_dir/'data'}/{solution_name}_annual_summary.csv")
    
    # Save financial cashflows
    cashflow_df = pd.DataFrame(fin_results['annual_costs'])
    cashflow_df.to_csv(run_dir / "data" / f"{solution_name}_cashflows.csv", index=False)
    print(f"✅ Cashflow data saved to {run_dir/'data'}/{solution_name}_cashflows.csv")
    
    # Create and save summary of key metrics
    metrics = {
        'Battery Size (kWh)': battery_kwh,
        'Additional PV (kW)': additional_pv_kw,
        'Total System PV (kW)': config.EXISTING_PV['system_capacity_kw'] + additional_pv_kw,
        'Initial Investment ($)': fin_results['capex_total'],
        'NPV ($)': fin_results['npv'],
        'IRR (%)': fin_results['irr'] * 100 if fin_results['irr'] else 0,
        'Profitability Index': fin_results['profitability_index'],
        'Simple Payback (years)': fin_results['simple_payback'],
        'LCOE ($/kWh)': fin_results['lcoe'],
        'Renewable Fraction (%)': totals['renewable_fraction'] * 100,
        'Self-Consumption Rate (%)': totals['self_consumption_rate'] * 100,
        'Battery Cycles': totals['battery_cycles'],
        'Final Degradation (%)': totals['final_degradation_pct'],
        'Total Grid Emissions (kg CO2e)': totals['total_grid_emissions'],
        'Peak Hours': totals['peak_hours'],
        'Battery Discharge During Peak (kWh)': totals['peak_discharge_kwh'],
        'Battery Discharge During Off-Peak (kWh)': totals['offpeak_discharge_kwh'],
        'Grid-to-Battery Charging (kWh)': totals['total_grid_to_battery'],
        'Peak Demand Supplied by Battery (%)': totals['peak_pct_supplied_by_battery'] * 100
    }
    
    # Calculate peak shifting financial benefit
    BASE_PEAK_RATE = config.ELECTRICITY_RATES['peak']
    BASE_OFFPEAK_RATE = config.ELECTRICITY_RATES['offpeak']
    annual_offpeak_charge_cost = totals['offpeak_grid_to_battery'] * BASE_OFFPEAK_RATE / 30
    annual_peak_discharge_value = totals['offpeak_grid_to_battery'] * 0.9 * BASE_PEAK_RATE / 30
    annual_arbitrage_benefit = annual_peak_discharge_value - annual_offpeak_charge_cost
    
    metrics['Annual Peak-Shifting Benefit ($)'] = annual_arbitrage_benefit
    metrics['30-year PV of Peak-Shifting ($)'] = calculate_present_value(annual_arbitrage_benefit, config.DISCOUNT_RATE)
    
    # Save metrics to CSV
    pd.DataFrame([metrics]).to_csv(run_dir / "data" / f"{solution_name}_summary.csv", index=False)
    print(f"✅ Summary metrics saved to {run_dir/'data'}/{solution_name}_summary.csv")
    
    # Create daily profile visualizations
    print("\nCreating daily profile visualizations...")
    summer_day = '2025-01-15'  # Mid-summer in Southern Hemisphere
    winter_day = '2025-07-15'  # Mid-winter in Southern Hemisphere
    
    summer_stats = visualize_battery_operation(dispatch_df, summer_day, 
                                              run_dir / "plots" / f"{solution_name}_summer_day.png")
    winter_stats = visualize_battery_operation(dispatch_df, winter_day, 
                                             run_dir / "plots" / f"{solution_name}_winter_day.png")
    
    # Save daily stats
    if summer_stats and winter_stats:
        daily_stats = pd.DataFrame([summer_stats, winter_stats])
        daily_stats.to_csv(run_dir / "data" / f"{solution_name}_daily_stats.csv", index=False)
        print(f"✅ Daily statistics saved to {run_dir/'data'}/{solution_name}_daily_stats.csv")
    
    return dispatch_df, totals, fin_results

def create_solution_comparison(df, key_indices, run_dir):
    """
    Create a comprehensive comparison of key solutions.
    
    Args:
        df: DataFrame with all solutions
        key_indices: Dictionary with indices of key solutions
        run_dir: Directory to save results
        
    Returns:
        None
    """
    # Extract key solutions
    best_irr = df.loc[key_indices['best_irr_idx']].to_dict()
    best_npv = df.loc[key_indices['best_npv_idx']].to_dict()
    balanced = df.loc[key_indices['balanced_idx']].to_dict()
    
    # Create comparison table
    comparison = pd.DataFrame({
        'Metric': [
            'Battery Size (kWh)',
            'Additional PV (kW)',
            'Total System PV (kW)',
            'IRR (%)',
            'NPV ($)',
            'Profitability Index',
            'Payback Period (years)',
            'Total Investment ($)',
            'Renewable Fraction (%)',
            'Self-Consumption Rate (%)',
            'Grid Independence (%)'
        ],
        'Best IRR': [
            best_irr['battery_kwh'],
            best_irr['additional_pv_kw'],
            best_irr.get('total_system_pv', best_irr['additional_pv_kw'] + 10.0),  # Add existing 10kW if not present
            best_irr['irr'] * 100,
            best_irr['npv'],
            best_irr.get('pi', None),
            best_irr.get('payback', None),
            best_irr.get('total_investment', None),
            best_irr.get('renewable_fraction', 0) * 100 if isinstance(best_irr.get('renewable_fraction', 0), (int, float)) else None,
            best_irr.get('self_consumption', 0) * 100 if isinstance(best_irr.get('self_consumption', 0), (int, float)) else None,
            best_irr.get('grid_independence', 0) * 100 if isinstance(best_irr.get('grid_independence', 0), (int, float)) else None
        ],
        'Best NPV': [
            best_npv['battery_kwh'],
            best_npv['additional_pv_kw'],
            best_npv.get('total_system_pv', best_npv['additional_pv_kw'] + 10.0),
            best_npv['irr'] * 100,
            best_npv['npv'],
            best_npv.get('pi', None),
            best_npv.get('payback', None),
            best_npv.get('total_investment', None),
            best_npv.get('renewable_fraction', 0) * 100 if isinstance(best_npv.get('renewable_fraction', 0), (int, float)) else None,
            best_npv.get('self_consumption', 0) * 100 if isinstance(best_npv.get('self_consumption', 0), (int, float)) else None,
            best_npv.get('grid_independence', 0) * 100 if isinstance(best_npv.get('grid_independence', 0), (int, float)) else None
        ],
        'Balanced': [
            balanced['battery_kwh'],
            balanced['additional_pv_kw'],
            balanced.get('total_system_pv', balanced['additional_pv_kw'] + 10.0),
            balanced['irr'] * 100,
            balanced['npv'],
            balanced.get('pi', None),
            balanced.get('payback', None),
            balanced.get('total_investment', None),
            balanced.get('renewable_fraction', 0) * 100 if isinstance(balanced.get('renewable_fraction', 0), (int, float)) else None,
            balanced.get('self_consumption', 0) * 100 if isinstance(balanced.get('self_consumption', 0), (int, float)) else None,
            balanced.get('grid_independence', 0) * 100 if isinstance(balanced.get('grid_independence', 0), (int, float)) else None
        ]
    })
    
    # Save comparison table
    comparison.to_csv(run_dir / "data" / "solution_comparison.csv", index=False)
    print(f"✅ Solution comparison saved to {run_dir/'data'}/solution_comparison.csv")
    
    return comparison

def create_summary_report(results, config, run_dir):
    """
    Create a text summary report of optimization results.
    
    Args:
        results: Results dictionary
        config: Configuration parameters
        run_dir: Directory to save results
        
    Returns:
        None
    """
    # Extract results
    best_irr = results['best_irr']
    best_npv = results['best_npv']
    balanced = results['balanced']
    
    # Create report
    report = []
    report.append("BATTERY AND PV OPTIMIZATION SUMMARY REPORT")
    report.append("===========================================")
    report.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    
    report.append("1. SYSTEM CONFIGURATION")
    report.append("----------------------")
    report.append(f"Existing PV system: {config.EXISTING_PV['system_capacity_kw']:.2f} kW")
    report.append(f"Project lifetime: {config.PROJECT_LIFETIME} years")
    report.append(f"Discount rate: {config.DISCOUNT_RATE*100:.1f}%")
    report.append(f"Electricity rates: Peak ${config.ELECTRICITY_RATES['peak']:.4f}/kWh, Off-peak ${config.ELECTRICITY_RATES['offpeak']:.4f}/kWh")
    report.append(f"Export rate: ${config.ELECTRICITY_RATES['export']:.4f}/kWh")
    report.append("")
    
    report.append("2. OPTIMIZATION RESULTS")
    report.append("----------------------")
    report.append("A. Best IRR Solution:")
    report.append(f"   - Battery: {best_irr['solution']['battery_kwh']:.2f} kWh")
    report.append(f"   - Additional PV: {best_irr['solution']['additional_pv_kw']:.2f} kW")
    report.append(f"   - Total system PV: {config.EXISTING_PV['system_capacity_kw'] + best_irr['solution']['additional_pv_kw']:.2f} kW")
    report.append(f"   - IRR: {best_irr['financials']['irr']*100:.2f}%")
    report.append(f"   - NPV: ${best_irr['financials']['npv']:,.2f}")
    report.append(f"   - Initial investment: ${best_irr['financials']['capex_total']:,.2f}")
    report.append(f"   - Payback period: {best_irr['financials']['simple_payback']:.2f} years")
    report.append(f"   - Renewable fraction: {best_irr['totals']['renewable_fraction']*100:.2f}%")
    report.append("")
    
    report.append("B. Best NPV Solution:")
    report.append(f"   - Battery: {best_npv['solution']['battery_kwh']:.2f} kWh")
    report.append(f"   - Additional PV: {best_npv['solution']['additional_pv_kw']:.2f} kW")
    report.append(f"   - Total system PV: {config.EXISTING_PV['system_capacity_kw'] + best_npv['solution']['additional_pv_kw']:.2f} kW")
    report.append(f"   - IRR: {best_npv['financials']['irr']*100:.2f}%")
    report.append(f"   - NPV: ${best_npv['financials']['npv']:,.2f}")
    report.append(f"   - Initial investment: ${best_npv['financials']['capex_total']:,.2f}")
    report.append(f"   - Payback period: {best_npv['financials']['simple_payback']:.2f} years")
    report.append(f"   - Renewable fraction: {best_npv['totals']['renewable_fraction']*100:.2f}%")
    report.append("")
    
    report.append("C. Balanced Solution:")
    report.append(f"   - Battery: {balanced['solution']['battery_kwh']:.2f} kWh")
    report.append(f"   - Additional PV: {balanced['solution']['additional_pv_kw']:.2f} kW")
    report.append(f"   - Total system PV: {config.EXISTING_PV['system_capacity_kw'] + balanced['solution']['additional_pv_kw']:.2f} kW")
    report.append(f"   - IRR: {balanced['financials']['irr']*100:.2f}%")
    report.append(f"   - NPV: ${balanced['financials']['npv']:,.2f}")
    report.append(f"   - Initial investment: ${balanced['financials']['capex_total']:,.2f}")
    report.append(f"   - Payback period: {balanced['financials']['simple_payback']:.2f} years")
    report.append(f"   - Renewable fraction: {balanced['totals']['renewable_fraction']*100:.2f}%")
    report.append("")
    
    report.append("3. RECOMMENDATIONS")
    report.append("------------------")
    
    # Determine recommendation
    if best_npv['financials']['npv'] < 0:
        report.append("Based on the financial analysis, no investment is recommended at this time.")
        report.append("The NPV is negative for all configurations, indicating that the project would not be economically viable.")
    else:
        # Find solution with highest PI if available, otherwise use highest NPV
        pi_values = []
        if best_irr['financials'].get('profitability_index') is not None:
            pi_values.append((best_irr['financials']['profitability_index'], "Best IRR"))
        if best_npv['financials'].get('profitability_index') is not None:
            pi_values.append((best_npv['financials']['profitability_index'], "Best NPV"))
        if balanced['financials'].get('profitability_index') is not None:
            pi_values.append((balanced['financials']['profitability_index'], "Balanced"))
        
        if pi_values:
            best_pi = max(pi_values, key=lambda x: x[0] if x[0] is not None else 0)
            report.append(f"The recommended solution is the {best_pi[1]} configuration, which offers:")
            if best_pi[1] == "Best IRR":
                report.append(f"- The highest return rate at {best_irr['financials']['irr']*100:.2f}%")
                report.append(f"- Initial investment: ${best_irr['financials']['capex_total']:,.2f}")
                report.append(f"- NPV: ${best_irr['financials']['npv']:,.2f}")
            elif best_pi[1] == "Best NPV":
                report.append(f"- The highest absolute return at ${best_npv['financials']['npv']:,.2f}")
                report.append(f"- Initial investment: ${best_npv['financials']['capex_total']:,.2f}")
                report.append(f"- IRR: {best_npv['financials']['irr']*100:.2f}%")
            else:
                report.append(f"- A balanced return with IRR of {balanced['financials']['irr']*100:.2f}% and NPV of ${balanced['financials']['npv']:,.2f}")
                report.append(f"- Initial investment: ${balanced['financials']['capex_total']:,.2f}")
        else:
            # If PI not available, use NPV to recommend
            report.append(f"The recommended solution is the Best NPV configuration, which offers:")
            report.append(f"- The highest absolute return at ${best_npv['financials']['npv']:,.2f}")
            report.append(f"- Initial investment: ${best_npv['financials']['capex_total']:,.2f}")
            report.append(f"- IRR: {best_npv['financials']['irr']*100:.2f}%")
    
    report.append("")
    report.append("For detailed results, please refer to the CSV files and visualizations in the results directory.")
    
    # Save report
    with open(run_dir / "summary_report.txt", "w") as f:
        f.write("\n".join(report))
    
    print(f"✅ Summary report saved to {run_dir}/summary_report.txt")

def process_optimization_results(df, callback_data, pv_profile, demand_profile, config, run_dir):
    """
    Process optimization results and generate all visualizations and reports.
    
    Args:
        df: DataFrame with optimization results
        callback_data: Data from optimization callback
        pv_profile: PV generation profile
        demand_profile: Demand profile
        config: Configuration parameters
        run_dir: Directory to save results
        
    Returns:
        dict: Dictionary with key results
    """
    print("\nProcessing optimization results...")
    
    # Ensure column names are lowercase for consistency
    df.columns = [col.lower() for col in df.columns]
    
    # Create Pareto front visualization
    key_indices = create_pareto_plot(df, run_dir)
    
    # Create convergence plot
    create_convergence_plot(callback_data, run_dir)
    
    # Extract key solutions
    best_irr_solution = {
        'battery_kwh': df.loc[key_indices['best_irr_idx'], 'battery_kwh'],
        'additional_pv_kw': df.loc[key_indices['best_irr_idx'], 'additional_pv_kw']
    }
    
    best_npv_solution = {
        'battery_kwh': df.loc[key_indices['best_npv_idx'], 'battery_kwh'],
        'additional_pv_kw': df.loc[key_indices['best_npv_idx'], 'additional_pv_kw']
    }
    
    balanced_solution = {
        'battery_kwh': df.loc[key_indices['balanced_idx'], 'battery_kwh'],
        'additional_pv_kw': df.loc[key_indices['balanced_idx'], 'additional_pv_kw']
    }
    
    # Save all optimization results to CSV
    df.to_csv(run_dir / "data" / "all_solutions.csv", index=False)
    print(f"✅ All solutions saved to {run_dir/'data'}/all_solutions.csv")
    
    # Analyze and save best IRR solution
    best_irr_dispatch, best_irr_totals, best_irr_fin = analyze_solution(
        best_irr_solution, pv_profile, demand_profile, config, run_dir, "best_irr"
    )
    
    # Analyze and save best NPV solution
    best_npv_dispatch, best_npv_totals, best_npv_fin = analyze_solution(
        best_npv_solution, pv_profile, demand_profile, config, run_dir, "best_npv"
    )
    
    # Analyze and save balanced solution
    balanced_dispatch, balanced_totals, balanced_fin = analyze_solution(
        balanced_solution, pv_profile, demand_profile, config, run_dir, "balanced"
    )
    
    # Create solution comparison
    create_solution_comparison(df, key_indices, run_dir)
    
    # Compile results
    results = {
        'best_irr': {
            'solution': best_irr_solution,
            'totals': best_irr_totals,
            'financials': best_irr_fin
        },
        'best_npv': {
            'solution': best_npv_solution,
            'totals': best_npv_totals,
            'financials': best_npv_fin
        },
        'balanced': {
            'solution': balanced_solution,
            'totals': balanced_totals,
            'financials': balanced_fin
        }
    }
    
    # Create summary report
    create_summary_report(results, config, run_dir)
    
    return results

def run_results_analysis(df, callback, pv_profile, demand_profile, config, run_dir):
    """
    Main entry point for results analysis to be called from notebook.
    
    Args:
        df: DataFrame with optimization results
        callback: Callback object from optimization
        pv_profile: PV generation profile
        demand_profile: Demand profile
        config: Configuration parameters
        run_dir: Directory to save results
        
    Returns:
        dict: Dictionary with key results
    """
    import sys
    
    # Ensure necessary directories exist
    (run_dir / "plots").mkdir(exist_ok=True)
    (run_dir / "data").mkdir(exist_ok=True)
    
    # Process results
    results = process_optimization_results(
        df, callback.data, pv_profile, demand_profile, config, run_dir
    )
    
    print("\nResults analysis complete!")
    return results