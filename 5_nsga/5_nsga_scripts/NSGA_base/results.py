# improved_results.py - Enhanced version of results.py without redundant simulations

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import os
from pathlib import Path
import time
from datetime import datetime, timedelta
import json
import sys

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

def create_solution_comparison(df, key_indices, config, run_dir):
    """
    Create a comprehensive comparison of key solutions.
    
    Args:
        df: DataFrame with all solutions
        key_indices: Dictionary with indices of key solutions
        config: Configuration parameters
        run_dir: Directory to save results
        
    Returns:
        DataFrame: Comparison table
    """
    # Extract key solutions
    best_irr = df.loc[key_indices['best_irr_idx']].to_dict()
    best_npv = df.loc[key_indices['best_npv_idx']].to_dict()
    balanced = df.loc[key_indices['balanced_idx']].to_dict()
    
    # Calculate PV allocations for reporting
    from pv import allocate_pv_capacity
    
    # Function to get detailed allocation for a solution
    def get_pv_allocation(additional_pv_kw, pv_options):
        allocation = allocate_pv_capacity(additional_pv_kw, pv_options)
        allocation_text = []
        for pv in allocation:
            if pv['system_capacity_kw'] > 0:
                allocation_text.append(f"{pv['name']}: {pv['system_capacity_kw']:.2f} kW")
        return ", ".join(allocation_text)
    
    # Get allocations
    best_irr_allocation = get_pv_allocation(best_irr['additional_pv_kw'], config.PV_OPTIONS)
    best_npv_allocation = get_pv_allocation(best_npv['additional_pv_kw'], config.PV_OPTIONS)
    balanced_allocation = get_pv_allocation(balanced['additional_pv_kw'], config.PV_OPTIONS)
    
    # Create comparison table
    comparison = pd.DataFrame({
        'Metric': [
            'Battery Size (kWh)',
            'Additional PV (kW)',
            'PV Allocation',
            'Total System PV (kW)',
            'IRR (%)',
            'NPV ($)',
            'Profitability Index',
            'Total Investment ($)',
            'Solar Fraction (%)',
            'Self-Consumption (%)'
        ],
        'Best IRR': [
            best_irr['battery_kwh'],
            best_irr['additional_pv_kw'],
            best_irr_allocation,
            config.EXISTING_PV['system_capacity_kw'] + best_irr['additional_pv_kw'],
            best_irr['irr'] * 100,
            best_irr['npv'],
            best_irr.get('pi', "N/A"),
            best_irr.get('total_investment', "N/A"),
            best_irr.get('renewable_fraction', "N/A"),
            best_irr.get('self_consumption_rate', "N/A")
        ],
        'Best NPV': [
            best_npv['battery_kwh'],
            best_npv['additional_pv_kw'],
            best_npv_allocation,
            config.EXISTING_PV['system_capacity_kw'] + best_npv['additional_pv_kw'],
            best_npv['irr'] * 100,
            best_npv['npv'],
            best_npv.get('pi', "N/A"),
            best_npv.get('total_investment', "N/A"),
            best_npv.get('renewable_fraction', "N/A"),
            best_npv.get('self_consumption_rate', "N/A")
        ],
        'Balanced': [
            balanced['battery_kwh'],
            balanced['additional_pv_kw'],
            balanced_allocation,
            config.EXISTING_PV['system_capacity_kw'] + balanced['additional_pv_kw'],
            balanced['irr'] * 100,
            balanced['npv'],
            balanced.get('pi', "N/A"),
            balanced.get('total_investment', "N/A"),
            balanced.get('renewable_fraction', "N/A"),
            balanced.get('self_consumption_rate', "N/A")
        ]
    })
    
    # Save comparison table
    comparison.to_csv(run_dir / "data" / "solution_comparison.csv", index=False)
    print(f"✅ Solution comparison saved to {run_dir/'data'}/solution_comparison.csv")
    
    return comparison

def create_annual_cashflows(solution, config, run_dir, solution_name):
    """
    Create estimated annual cashflows for the solution.
    
    Args:
        solution: Solution parameters
        config: Configuration parameters
        run_dir: Directory to save results
        solution_name: Name of the solution
        
    Returns:
        DataFrame: Annual cashflows
    """
    # Import finance calculations
    from fin import calculate_pv_cost, calculate_battery_cost
    
    # Extract parameters
    battery_kwh = solution['battery_kwh']
    additional_pv_kw = solution['additional_pv_kw']
    
    # Calculate initial investment
    battery_cost = calculate_battery_cost(battery_kwh, config) if battery_kwh > 0 else 0
    
    # Calculate PV cost based on allocation
    from pv import allocate_pv_capacity
    pv_cost = 0
    if additional_pv_kw > 0:
        allocated_pv = allocate_pv_capacity(additional_pv_kw, config.PV_OPTIONS)
        for pv in allocated_pv:
            if pv['system_capacity_kw'] > 0:
                cost_multiplier = pv.get('cost_multiplier', 1.0)
                system_cost = calculate_pv_cost(pv['system_capacity_kw'], cost_multiplier)
                pv_cost += system_cost
    
    total_capex = battery_cost + pv_cost
    
    # Create annual cashflow estimates
    years = list(range(config.PROJECT_LIFETIME + 1))
    cashflows = [-total_capex]  # Year 0 is negative capex
    
    # Estimate annual benefits
    # Note: These are rough estimates as we don't have actual simulation data
    estimated_annual_savings = config.ANNUAL_NO_PV_COST - config.ANNUAL_PV_ONLY_COST
    estimated_battery_benefit = battery_kwh * 50 if battery_kwh > 0 else 0  # Rough estimate of $50/kWh annual benefit
    estimated_additional_pv_benefit = additional_pv_kw * 150  # Rough estimate of $150/kW annual benefit
    
    total_annual_benefit = estimated_battery_benefit + estimated_additional_pv_benefit
    
    # Maintenance costs (simplified)
    annual_maintenance = -(battery_cost * 0.01 + pv_cost * 0.005)  # 1% of battery cost, 0.5% of PV cost
    
    # Generate simple cashflows with escalation
    for year in range(1, config.PROJECT_LIFETIME + 1):
        # Apply escalation to benefits
        escalated_benefit = total_annual_benefit * (1 + config.ELECTRICITY_PRICE_ESCALATION) ** (year - 1)
        # Apply inflation to maintenance
        inflated_maintenance = annual_maintenance * (1 + config.MAINTENANCE_INFLATION) ** (year - 1)
        # Year's cashflow is benefit plus maintenance
        year_cashflow = escalated_benefit + inflated_maintenance
        cashflows.append(year_cashflow)
    
    # Create cashflow dataframe
    cashflow_df = pd.DataFrame({
        'Year': years,
        'Cashflow': cashflows
    })
    
    # Calculate cumulative NPV
    discount_rate = config.DISCOUNT_RATE
    cashflow_df['Discounted_Cashflow'] = [cf / ((1 + discount_rate) ** year) for year, cf in enumerate(cashflows)]
    cashflow_df['Cumulative_NPV'] = cashflow_df['Discounted_Cashflow'].cumsum()
    
    # Save cashflows
    cashflow_df.to_csv(run_dir / "data" / f"{solution_name}_cashflows.csv", index=False)
    print(f"✅ Estimated cashflows saved to {run_dir/'data'}/{solution_name}_cashflows.csv")
    
    return cashflow_df

def create_summary_report(df, key_indices, config, run_dir):
    """
    Create a text summary report of optimization results.
    
    Args:
        df: DataFrame with optimization results
        key_indices: Dictionary with indices of key solutions
        config: Configuration parameters
        run_dir: Directory to save results
        
    Returns:
        None
    """
    # Extract results
    best_irr = df.loc[key_indices['best_irr_idx']].to_dict()
    best_npv = df.loc[key_indices['best_npv_idx']].to_dict()
    balanced = df.loc[key_indices['balanced_idx']].to_dict()
    
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
    report.append(f"   - Battery: {best_irr['battery_kwh']:.2f} kWh")
    report.append(f"   - Additional PV: {best_irr['additional_pv_kw']:.2f} kW")
    report.append(f"   - Total system PV: {config.EXISTING_PV['system_capacity_kw'] + best_irr['additional_pv_kw']:.2f} kW")
    report.append(f"   - IRR: {best_irr['irr']*100:.2f}%")
    report.append(f"   - NPV: ${best_irr['npv']:,.2f}")
    if 'pi' in best_irr:
        report.append(f"   - Profitability Index: {best_irr['pi']:.2f}")
    report.append("")
    
    report.append("B. Best NPV Solution:")
    report.append(f"   - Battery: {best_npv['battery_kwh']:.2f} kWh")
    report.append(f"   - Additional PV: {best_npv['additional_pv_kw']:.2f} kW")
    report.append(f"   - Total system PV: {config.EXISTING_PV['system_capacity_kw'] + best_npv['additional_pv_kw']:.2f} kW")
    report.append(f"   - IRR: {best_npv['irr']*100:.2f}%")
    report.append(f"   - NPV: ${best_npv['npv']:,.2f}")
    if 'pi' in best_npv:
        report.append(f"   - Profitability Index: {best_npv['pi']:.2f}")
    report.append("")
    
    report.append("C. Balanced Solution:")
    report.append(f"   - Battery: {balanced['battery_kwh']:.2f} kWh")
    report.append(f"   - Additional PV: {balanced['additional_pv_kw']:.2f} kW")
    report.append(f"   - Total system PV: {config.EXISTING_PV['system_capacity_kw'] + balanced['additional_pv_kw']:.2f} kW")
    report.append(f"   - IRR: {balanced['irr']*100:.2f}%")
    report.append(f"   - NPV: ${balanced['npv']:,.2f}")
    if 'pi' in balanced:
        report.append(f"   - Profitability Index: {balanced['pi']:.2f}")
    report.append("")
    
    report.append("3. RECOMMENDATIONS")
    report.append("------------------")
    
    # Determine recommendation
    if best_npv['npv'] < 0:
        report.append("Based on the financial analysis, no investment is recommended at this time.")
        report.append("The NPV is negative for all configurations, indicating that the project would not be economically viable.")
    else:
        # Use PI for recommendation if available
        if 'pi' in best_irr and 'pi' in best_npv and 'pi' in balanced:
            pi_values = [
                (best_irr['pi'], "Best IRR"),
                (best_npv['pi'], "Best NPV"),
                (balanced['pi'], "Balanced")
            ]
            best_pi = max(pi_values, key=lambda x: x[0] if x[0] is not None else 0)
            report.append(f"The recommended solution is the {best_pi[1]} configuration, which offers:")
            if best_pi[1] == "Best IRR":
                report.append(f"- The highest return rate at {best_irr['irr']*100:.2f}%")
                report.append(f"- NPV: ${best_irr['npv']:,.2f}")
            elif best_pi[1] == "Best NPV":
                report.append(f"- The highest absolute return at ${best_npv['npv']:,.2f}")
                report.append(f"- IRR: {best_npv['irr']*100:.2f}%")
            else:
                report.append(f"- A balanced return with IRR of {balanced['irr']*100:.2f}% and NPV of ${balanced['npv']:,.2f}")
        else:
            # Provide general recommendations without PI
            report.append(f"The recommended configuration depends on investment priorities:")
            report.append(f"- For highest return rate: Choose the Best IRR solution ({best_irr['irr']*100:.2f}%)")
            report.append(f"- For highest absolute return: Choose the Best NPV solution (${best_npv['npv']:,.2f})")
            report.append(f"- For a balanced approach: Choose the Balanced solution")
    
    report.append("")
    report.append("For detailed results, please refer to the CSV files and visualizations in the results directory.")
    
    # Save report
    with open(run_dir / "summary_report.txt", "w") as f:
        f.write("\n".join(report))
    
    print(f"✅ Summary report saved to {run_dir}/summary_report.txt")

def process_optimization_results(df, callback_data, config, run_dir):
    """
    Process optimization results and generate all visualizations and reports
    without re-running simulations.
    
    Args:
        df: DataFrame with optimization results
        callback_data: Data from optimization callback
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
    best_irr_solution = df.loc[key_indices['best_irr_idx']].to_dict()
    best_npv_solution = df.loc[key_indices['best_npv_idx']].to_dict()
    balanced_solution = df.loc[key_indices['balanced_idx']].to_dict()
    
    # Save all optimization results to CSV
    df.to_csv(run_dir / "data" / "all_solutions.csv", index=False)
    print(f"✅ All solutions saved to {run_dir/'data'}/all_solutions.csv")
    
    # Create solution comparison table
    comparison = create_solution_comparison(df, key_indices, config, run_dir)
    
    # Create estimated cashflows for each key solution
    best_irr_cashflows = create_annual_cashflows(best_irr_solution, config, run_dir, "best_irr")
    best_npv_cashflows = create_annual_cashflows(best_npv_solution, config, run_dir, "best_npv")
    balanced_cashflows = create_annual_cashflows(balanced_solution, config, run_dir, "balanced")
    
    # Create summary report
    create_summary_report(df, key_indices, config, run_dir)
    
    # Return key results
    results = {
        'best_irr': best_irr_solution,
        'best_npv': best_npv_solution,
        'balanced': balanced_solution,
        'comparison': comparison,
        'cashflows': {
            'best_irr': best_irr_cashflows,
            'best_npv': best_npv_cashflows,
            'balanced': balanced_cashflows
        }
    }
    
    return results

def run_results_analysis(df, callback, config, run_dir):
    """
    Main entry point for results analysis to be called from notebook.
    Processes results without re-running simulations.
    
    Args:
        df: DataFrame with optimization results
        callback: Callback object from optimization
        config: Configuration parameters
        run_dir: Directory to save results
        
    Returns:
        dict: Dictionary with key results
    """
    # Ensure necessary directories exist
    (run_dir / "plots").mkdir(exist_ok=True)
    (run_dir / "data").mkdir(exist_ok=True)
    
    # Process results
    results = process_optimization_results(
        df=df, 
        callback_data=callback.data, 
        config=config, 
        run_dir=run_dir
    )
    
    print("\nResults analysis complete!")
    print(f"All results have been saved to: {run_dir}")
    
    # Display paths to key result files
    print("\nKey result files:")
    print(f"- Summary report: {run_dir/'summary_report.txt'}")
    print(f"- Pareto front visualization: {run_dir/'plots'/'pareto_front_irr_npv.png'}")
    print(f"- Solution comparison: {run_dir/'data'/'solution_comparison.csv'}")
    print(f"- All solutions: {run_dir/'data'/'all_solutions.csv'}")
    
    return results