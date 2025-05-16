#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Combined Solar PV and Consumption Analysis
-----------------------------------------
This script combines various analysis cells into a single workflow
that generates and saves all visualizations to an output folder.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import calendar
import os

# Create output directory if it doesn't exist
OUTPUT_DIR = './output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Starting analysis...")

# ===============================================================
# Data Loading Functions
# ===============================================================

def load_pv_data():
    """Load the PV generation data with simulated and actual values"""
    print("Loading PV generation data...")
    df = pd.read_csv(
        '../data/PV_Generation_excel.csv',
        parse_dates=['Date and Time'],
        index_col='Date and Time',
        dayfirst=True
    )
    df.columns = df.columns.str.strip()
    return df

def load_consumption_with_brewery():
    """Load the consumption data that includes brewery"""
    print("Loading consumption data with brewery...")
    df = pd.read_csv(
        '../data/PV_Generation_excel_2024.csv',
        parse_dates=['Date and Time'],
        index_col='Date and Time',
        dayfirst=True
    )
    df.columns = df.columns.str.strip().str.strip("'\"")
    return df

def load_consumption_without_brewery():
    """Load the consumption data without brewery"""
    print("Loading consumption data without brewery...")
    df = pd.read_csv(
        '../data/demand_without.csv',
        parse_dates=['Date and Time'],
        index_col='Date and Time',
        dayfirst=True
    )
    df.columns = df.columns.str.strip().str.strip("'\"")
    return df

# ===============================================================
# SECTION 1: PV Generation Analysis
# ===============================================================

def analyze_pv_generation():
    """Run the PV generation analysis"""
    print("\nAnalyzing PV generation...")
    df = load_pv_data()
    
    # Extract simulated vs actual series
    sim = df['PV Generated (kWh)'].astype(float)
    act = df['PV Actual (kWh)'].astype(float)

    # ─── 1) MONTHLY TOTALS (table + line chart) ─────────────────────────────
    monthly_sim = sim.resample('ME').sum()
    monthly_act = act.resample('ME').sum()

    print("=== Monthly PV Generation Totals ===")
    monthly_table = pd.DataFrame({
        'Simulated (kWh)': monthly_sim,
        'Actual    (kWh)': monthly_act
    })
    print(monthly_table.to_string(), "\n")
    
    # Save table to CSV
    monthly_table.to_csv(f"{OUTPUT_DIR}/monthly_pv_totals.csv")

    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(monthly_sim.index, monthly_sim.values, marker='o', label='Simulated')
    ax.plot(monthly_act.index, monthly_act.values, marker='o', label='Actual')
    ax.set_title('Monthly PV Generation Totals')
    ax.set_ylabel('Generation (kWh)')
    ax.set_xticks(monthly_sim.index)
    ax.set_xticklabels(monthly_sim.index.strftime('%b'), rotation=0)
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/monthly_pv_totals.png", dpi=300)
    plt.close()

    # ─── 2) DAILY TOTALS (whole-year) ───────────────────────────────────────
    daily_sim = sim.resample('D').sum()
    daily_act = act.resample('D').sum()

    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(daily_sim.index, daily_sim.values, label='Simulated', alpha=0.8)
    ax.plot(daily_act.index, daily_act.values, label='Actual', alpha=0.8)
    ax.set_title('Daily PV Generation (Year-long)')
    ax.set_ylabel('Generation (kWh)')
    ax.set_xlim(daily_sim.index.min(), daily_sim.index.max())
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    ax.legend()
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/daily_pv_generation.png", dpi=300)
    plt.close()

    # ─── 3) 7-DAY ROLLING AVERAGE OF DAILY TOTALS ──────────────────────────
    rolling_sim = daily_sim.rolling(window=7, min_periods=1).mean()
    rolling_act = daily_act.rolling(window=7, min_periods=1).mean()

    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(rolling_sim.index, rolling_sim.values, label='Simulated (7-day MA)')
    ax.plot(rolling_act.index, rolling_act.values, label='Actual    (7-day MA)')
    ax.set_title('7-Day Rolling Average of Daily PV Generation')
    ax.set_ylabel('Generation (kWh)')
    ax.set_xlim(rolling_sim.index.min(), rolling_sim.index.max())
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    ax.legend()
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/rolling_pv_generation.png", dpi=300)
    plt.close()

    # ─── 4) ENHANCED MONTHLY TABLE: AVERAGES, TOTALS, DIFFERENCES & PCTs ───
    monthly_avg_sim = daily_sim.resample('ME').mean()
    monthly_avg_act = daily_act.resample('ME').mean()

    enhanced_table = pd.DataFrame({
        'Avg Sim (kWh/day)': monthly_avg_sim,
        'Avg Act (kWh/day)': monthly_avg_act,
        'Sum Sim (kWh/month)': monthly_sim,
        'Sum Act (kWh/month)': monthly_act
    })

    # add difference columns
    enhanced_table['Diff (kWh/day)'] = enhanced_table['Avg Sim (kWh/day)'] - enhanced_table['Avg Act (kWh/day)']
    enhanced_table['Diff (kWh/month)'] = enhanced_table['Sum Sim (kWh/month)'] - enhanced_table['Sum Act (kWh/month)']
    enhanced_table['Pct Diff'] = enhanced_table['Sum Act (kWh/month)'] / enhanced_table['Sum Sim (kWh/month)'] - 1

    # add absolute percent difference
    enhanced_table['Abs Pct Diff'] = enhanced_table['Pct Diff'].abs()

    # Format index for display and save original index for the CSV
    enhanced_table_display = enhanced_table.copy()
    enhanced_table_display.index = enhanced_table_display.index.to_period('M').strftime('%b')
    
    print("=== Enhanced Monthly PV Generation Summary ===")
    print(enhanced_table_display.to_string(), "\n")
    
    # Save enhanced table to CSV (with original datetime index)
    enhanced_table.to_csv(f"{OUTPUT_DIR}/enhanced_monthly_pv.csv")

    # seasonal mean differences (daily)
    winter_months = ['Jan', 'Feb']
    summer_months = ['Oct', 'Nov', 'Dec']
    
    winter_indices = [i for i, month in enumerate(enhanced_table_display.index) if month in winter_months]
    summer_indices = [i for i, month in enumerate(enhanced_table_display.index) if month in summer_months]
    
    winter = enhanced_table_display.iloc[winter_indices]['Diff (kWh/day)'].mean()
    summer = enhanced_table_display.iloc[summer_indices]['Diff (kWh/day)'].mean()
    
    print(f"Average difference in Jan & Feb: {winter:.2f} kWh/day")
    print(f"Average difference in Oct–Dec:  {summer:.2f} kWh/day")

    # average % difference for selected months
    selected_months = ['Jan', 'Feb', 'Oct', 'Nov', 'Dec']
    selected_indices = [i for i, month in enumerate(enhanced_table_display.index) if month in selected_months]
    
    avg_pct = enhanced_table_display.iloc[selected_indices]['Pct Diff'].mean()
    avg_pct_abs = enhanced_table_display.iloc[selected_indices]['Abs Pct Diff'].mean()
    
    print(f"\nAverage percentage difference (Actual vs Simulated) for {', '.join(selected_months)}: {avg_pct:.2%}")
    print(f"Average absolute percentage difference for {', '.join(selected_months)}: {avg_pct_abs:.2%}")
    
    # Create text file with summary statistics
    with open(f"{OUTPUT_DIR}/pv_summary_stats.txt", 'w') as f:
        f.write("=== PV Generation Summary Statistics ===\n\n")
        f.write(f"Average difference in Jan & Feb: {winter:.2f} kWh/day\n")
        f.write(f"Average difference in Oct–Dec:  {summer:.2f} kWh/day\n\n")
        f.write(f"Average percentage difference (Actual vs Simulated) for {', '.join(selected_months)}: {avg_pct:.2%}\n")
        f.write(f"Average absolute percentage difference for {', '.join(selected_months)}: {avg_pct_abs:.2%}\n")

    # ─── MONTHLY PROFILES ─────────────────────────────────────────────────
    # Month labels
    month_names = [calendar.month_abbr[m] for m in range(1, 13)]
    
    # Plot a 3×4 grid of average daily profiles
    fig, axes = plt.subplots(3, 4, figsize=(16, 12), sharey=True)
    for idx, month in enumerate(range(1, 13)):
        ax = axes.flat[idx]
        # filter for this month
        sim_m = sim[sim.index.month == month]
        act_m = act[act.index.month == month]
        
        # average half-hour profile
        prof_sim = sim_m.groupby(sim_m.index.strftime('%H:%M')).mean().sort_index()
        prof_act = act_m.groupby(act_m.index.strftime('%H:%M')).mean().sort_index()
        
        # convert to datetime on an arbitrary day
        times = pd.to_datetime(prof_sim.index, format='%H:%M')
        
        ax.plot(times, prof_sim.values, label='Sim', linewidth=1)
        ax.plot(times, prof_act.values, label='Act', linewidth=1, alpha=0.8)
        ax.set_title(month_names[month-1])
        
        # ticks at 00:00, 06:00, 12:00, 18:00, 24:00
        ax.xaxis.set_major_locator(mdates.HourLocator(byhour=[0, 6, 12, 18, 24]))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.tick_params(axis='x', rotation=45)
    
    # Only need one legend
    axes.flat[0].legend(loc='upper right', fontsize='small')
    fig.suptitle('Average Daily PV Generation Profile by Month', y=0.95)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(f"{OUTPUT_DIR}/monthly_pv_profiles.png", dpi=300)
    plt.close()

    return daily_sim, daily_act

# ===============================================================
# SECTION 2: Consumption Analysis with Brewery
# ===============================================================

def analyze_consumption_with_brewery():
    """Run the consumption analysis with brewery data"""
    print("\nAnalyzing consumption with brewery...")
    df = load_consumption_with_brewery()
    
    # Find consumption column
    cons_col = [c for c in df.columns if 'consum' in c.lower()][0]
    cons = df[cons_col].astype(float)
    
    # ─── MONTHLY TOTALS ───────────────────────────────────────────────────
    monthly = cons.resample('ME').sum()
    
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(monthly.index, monthly.values, marker='o', linestyle='-')
    ax.set_title('Monthly Total Consumption Demand with Brewery Production')
    ax.set_xlabel('Month')
    ax.set_ylabel('Total Consumption (kWh)')
    ax.set_xticks(monthly.index)
    ax.set_xticklabels(monthly.index.strftime('%b'), rotation=0, ha='center')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/monthly_consumption_with_brewery.png", dpi=300)
    plt.close()
    
    # Save monthly totals to CSV
    monthly_df = pd.DataFrame(monthly)
    monthly_df.columns = ['Total Consumption (kWh)']
    monthly_df.to_csv(f"{OUTPUT_DIR}/monthly_consumption_with_brewery.csv")
    
    # ─── DAILY CONSUMPTION ─────────────────────────────────────────────────
    daily_sum = cons.resample('D').sum()
    
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(daily_sum.index, daily_sum.values, linewidth=1)
    ax.set_title('Daily Consumption (With Brewery)')
    ax.set_ylabel('Consumption (kWh)')
    ax.set_xlim(daily_sum.index.min(), daily_sum.index.max())
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/daily_consumption_with_brewery.png", dpi=300)
    plt.close()
    
    # ─── 7-DAY ROLLING AVERAGE ───────────────────────────────────────────
    rolling7 = daily_sum.rolling(window=7, min_periods=1).mean()
    
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(rolling7.index, rolling7, linewidth=1)
    ax.set_xlim(rolling7.index.min(), rolling7.index.max())
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    ax.set_xlabel('Month')
    ax.set_ylabel('Consumption (kWh)')
    ax.set_title('7-Day Rolling Daily Consumption (With Brewery)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/rolling_consumption_with_brewery.png", dpi=300)
    plt.close()
    
    # ─── MONTHLY AVERAGE DAILY CONSUMPTION TABLE ─────────────────────────
    monthly_avg_daily = daily_sum.resample('ME').mean()
    table = monthly_avg_daily.rename('Avg Daily Consumption (kWh)').to_frame()
    
    # Save both formatted and raw tables
    table_display = table.copy()
    table_display.index = table_display.index.to_period('M').strftime('%b %Y')
    print("=== Average Daily Consumption per Month (With Brewery) ===")
    print(table_display.to_string())
    
    # Save table to CSV
    table.to_csv(f"{OUTPUT_DIR}/monthly_avg_daily_consumption_with_brewery.csv")
    
    # Write table to text file
    with open(f"{OUTPUT_DIR}/monthly_avg_daily_consumption_with_brewery.txt", 'w') as f:
        f.write("=== Average Daily Consumption per Month (With Brewery) ===\n\n")
        f.write(table_display.to_string())
    
    # ─── WEEKDAY AND HOURLY PROFILE ───────────────────────────────────────
    # daily totals for weekday bar chart
    daily_totals = cons.resample('D').sum()
    avg_by_wd = daily_totals.groupby(daily_totals.index.dayofweek).mean()
    avg_by_wd.index = [calendar.day_name[d] for d in avg_by_wd.index]
    
    # half-hour profile as strings
    profile = cons.groupby(cons.index.strftime('%H:%M')).mean().sort_index()
    
    # convert to datetime on an arbitrary date for a continuous axis
    times = pd.to_datetime(profile.index, format='%H:%M')
    
    # extend x‐axis to midnight next day
    start = times.min().normalize()
    end = start + pd.Timedelta(days=1)
    
    # Plot both panels
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Top: weekday bar chart
    ax1.bar(avg_by_wd.index, avg_by_wd.values, color='skyblue')
    ax1.set_title('Average Daily Consumption by Weekday (Brewery Data)')
    ax1.set_ylabel('Consumption (kWh)')
    ax1.tick_params(axis='x', rotation=45)
    
    # Bottom: half-hour line profile, end‐to‐end with 00:00 at both ends
    ax2.plot(times, profile.values, marker='o', linestyle='-')
    ax2.set_title('Average Load Profile (Half-Hour Resolution)')
    ax2.set_xlabel('Time of Day')
    ax2.set_ylabel('Consumption (kWh)')
    ax2.set_xlim(start, end)
    ax2.xaxis.set_major_locator(mdates.HourLocator(byhour=range(0, 25, 2)))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.setp(ax2.get_xticklabels(), rotation=45)
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/weekday_hourly_profile_with_brewery.png", dpi=300)
    plt.close()
    
    # Save weekday averages to CSV
    avg_by_wd_df = pd.DataFrame(avg_by_wd)
    avg_by_wd_df.columns = ['Avg Daily Consumption (kWh)']
    avg_by_wd_df.to_csv(f"{OUTPUT_DIR}/weekday_avg_consumption_with_brewery.csv")
    
    # Save hourly profile to CSV
    profile_df = pd.DataFrame(profile)
    profile_df.columns = ['Avg Consumption (kWh)']
    profile_df.to_csv(f"{OUTPUT_DIR}/hourly_profile_with_brewery.csv")
    
    return daily_sum

# ===============================================================
# SECTION 3: Consumption Analysis without Brewery
# ===============================================================

def analyze_consumption_without_brewery():
    """Run the consumption analysis without brewery data"""
    print("\nAnalyzing consumption without brewery...")
    df = load_consumption_without_brewery()
    
    # Extract first column (consumption without brewery)
    col = df.columns[0]
    cons = df[col]
    
    # ─── MONTHLY CONSUMPTION TOTALS ───────────────────────────────────────
    monthly = cons.resample('ME').sum()
    
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(monthly.index, monthly.values, marker='o', linestyle='-')
    ax.set_title('Monthly Consumption Demand (Without Brewery)')
    ax.set_ylabel('kWh')
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/monthly_consumption_without_brewery.png", dpi=300)
    plt.close()
    
    # Save monthly totals to CSV
    monthly_df = pd.DataFrame(monthly)
    monthly_df.columns = ['Total Consumption (kWh)']
    monthly_df.to_csv(f"{OUTPUT_DIR}/monthly_consumption_without_brewery.csv")
    
    # ─── DAILY CONSUMPTION ─────────────────────────────────────────────────
    daily = cons.resample('D').sum()
    
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(daily.index, daily.values, linestyle='-')
    ax.set_title('Daily Consumption (Without Brewery)')
    ax.set_ylabel('kWh')
    ax.set_xlim(daily.index.min(), daily.index.max())
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/daily_consumption_without_brewery.png", dpi=300)
    plt.close()
    
    # ─── 7-DAY ROLLING AVERAGE ───────────────────────────────────────────
    rolling7 = daily.rolling(window=7, min_periods=1).mean()
    
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(rolling7.index, rolling7.values, linestyle='-')
    ax.set_title('7-Day Rolling Daily Consumption (Without Brewery)')
    ax.set_ylabel('kWh')
    ax.set_xlim(rolling7.index.min(), rolling7.index.max())
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/rolling_consumption_without_brewery.png", dpi=300)
    plt.close()
    
    # ─── MONTHLY AVERAGE DAILY CONSUMPTION TABLE ─────────────────────────
    monthly_avg_daily = daily.resample('ME').mean()
    table = monthly_avg_daily.rename('Avg Daily Consumption (kWh)').to_frame()
    
    # Save both formatted and raw tables
    table_display = table.copy()
    table_display.index = table_display.index.to_period('M').strftime('%b %Y')
    print("=== Average Daily Consumption per Month (Without Brewery) ===")
    print(table_display.to_string())
    
    # Save table to CSV
    table.to_csv(f"{OUTPUT_DIR}/monthly_avg_daily_consumption_without_brewery.csv")
    
    # Write table to text file
    with open(f"{OUTPUT_DIR}/monthly_avg_daily_consumption_without_brewery.txt", 'w') as f:
        f.write("=== Average Daily Consumption per Month (Without Brewery) ===\n\n")
        f.write(table_display.to_string())
        
    return daily

# ===============================================================
# SECTION 4: Comparative Analysis
# ===============================================================

def analyze_generation_vs_consumption(daily_pv_sim, daily_pv_act, daily_cons_with_brewery):
    """Run comparative analysis between PV generation and consumption"""
    print("\nAnalyzing PV generation vs consumption...")
    
    # Load data if not provided
    if daily_pv_sim is None or daily_pv_act is None:
        df_pv = load_pv_data()
        daily_pv_sim = df_pv['PV Generated (kWh)'].astype(float).resample('D').sum()
        daily_pv_act = df_pv['PV Actual (kWh)'].astype(float).resample('D').sum()
    
    if daily_cons_with_brewery is None:
        df_cons = load_consumption_with_brewery()
        cons_col = [c for c in df_cons.columns if 'consum' in c.lower()][0]
        daily_cons_with_brewery = df_cons[cons_col].astype(float).resample('D').sum()
    
    # Create a combined dataframe for analysis
    combined_daily = pd.DataFrame({
        'PV Simulated': daily_pv_sim,
        'PV Actual': daily_pv_act,
        'Consumption': daily_cons_with_brewery
    })
    
    # Fill NaN values if there are any misalignments
    combined_daily = combined_daily.fillna(0)
    
    # Calculate net energy (generation - consumption)
    combined_daily['Net Energy (Actual)'] = combined_daily['PV Actual'] - combined_daily['Consumption']
    combined_daily['Net Energy (Simulated)'] = combined_daily['PV Simulated'] - combined_daily['Consumption']
    
    # Monthly aggregation
    monthly_combined = combined_daily.resample('ME').sum()
    
    # Plot comparison of monthly totals
    fig, ax = plt.subplots(figsize=(12, 6))
    
    width = 0.35
    x = range(len(monthly_combined.index))
    
    ax.bar([i - width/2 for i in x], monthly_combined['PV Actual'], width, label='PV Generation (Actual)')
    ax.bar([i + width/2 for i in x], monthly_combined['Consumption'], width, label='Consumption')
    
    ax.set_title('Monthly PV Generation vs Consumption')
    ax.set_ylabel('Energy (kWh)')
    ax.set_xticks(x)
    ax.set_xticklabels(monthly_combined.index.strftime('%b'), rotation=45)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/monthly_pv_vs_consumption.png", dpi=300)
    plt.close()
    
    # Plot net energy balance
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.bar(monthly_combined.index, monthly_combined['Net Energy (Actual)'], label='Net Energy (Actual)')
    ax.axhline(y=0, color='r', linestyle='-', alpha=0.3)
    
    ax.set_title('Monthly Net Energy Balance (Generation - Consumption)')
    ax.set_ylabel('Net Energy (kWh)')
    ax.set_xticklabels(monthly_combined.index.strftime('%b'), rotation=45)
    
    # Add text labels for net energy values
    for i, v in enumerate(monthly_combined['Net Energy (Actual)']):
        ax.text(i, v + (5 if v >= 0 else -15), f"{v:.0f}", ha='center')
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/monthly_net_energy.png", dpi=300)
    plt.close()
    
    # Save the combined data
    monthly_combined.to_csv(f"{OUTPUT_DIR}/monthly_combined_analysis.csv")
    
    # Create a summary table with percentages of self-consumption
    summary = pd.DataFrame({
        'PV Generation (kWh)': monthly_combined['PV Actual'],
        'Consumption (kWh)': monthly_combined['Consumption'],
        'Net Energy (kWh)': monthly_combined['Net Energy (Actual)']
    })
    
    # Calculate self-consumption percentage
    summary['Self-Consumption %'] = (
        (monthly_combined['PV Actual'] - monthly_combined['Net Energy (Actual)'].clip(lower=0)) / 
        monthly_combined['PV Actual'] * 100
    ).fillna(0).round(1)
    
    # Calculate PV Coverage of Consumption
    summary['PV Coverage %'] = (
        (monthly_combined['PV Actual'] - monthly_combined['Net Energy (Actual)'].clip(lower=0)) / 
        monthly_combined['Consumption'] * 100
    ).fillna(0).clip(upper=100).round(1)
    
    # Add yearly totals
    yearly_totals = summary.sum()
    yearly_totals['Self-Consumption %'] = (
        (yearly_totals['PV Generation (kWh)'] - yearly_totals['Net Energy (kWh)'].clip(lower=0)) / 
        yearly_totals['PV Generation (kWh)'] * 100
    ).round(1)
    yearly_totals['PV Coverage %'] = (
        (yearly_totals['PV Generation (kWh)'] - yearly_totals['Net Energy (kWh)'].clip(lower=0)) / 
        yearly_totals['Consumption (kWh)'] * 100
    ).clip(upper=100).round(1)
    
    summary.loc['Year Total'] = yearly_totals
    
    # Format for display
    summary_display = summary.copy()
    summary_display.index = summary_display.index.strftime('%b') if hasattr(summary_display.index, 'strftime') else summary_display.index
    
    print("=== PV Generation vs Consumption Summary ===")
    print(summary_display.to_string())
    print("\nPositive Net Energy means excess generation sent to grid")
    print("Negative Net Energy means energy imported from grid")
    
    # Save summary to CSV and text file
    summary.to_csv(f"{OUTPUT_DIR}/pv_consumption_summary.csv")
    
    with open(f"{OUTPUT_DIR}/pv_consumption_summary.txt", 'w') as f:
        f.write("=== PV Generation vs Consumption Summary ===\n\n")
        f.write(summary_display.to_string())
        f.write("\n\nPositive Net Energy means excess generation sent to grid\n")
        f.write("Negative Net Energy means energy imported from grid\n")
    
    # Create hourly heatmaps
    create_hourly_heatmaps()

def create_hourly_heatmaps():
    """Create hourly heatmaps of PV generation and consumption"""
    print("\nCreating hourly heatmaps...")
    
    # Load data
    df_cons = load_consumption_with_brewery()
    df_gen = load_pv_data()
    
    # Resample to hourly means 
    hourly_cons = df_cons['Consumption (kWh)'].resample('h').mean()
    hourly_sim = df_gen['PV Generated (kWh)'].resample('h').mean()
    
    # Pivot into hour×day arrays
    def make_hourly_pivot(series):
        tmp = series.to_frame('v')
        tmp['hour'] = tmp.index.hour
        tmp['day'] = tmp.index.dayofyear
        return tmp.pivot(index='hour', columns='day', values='v')
        
    pivot_cons = make_hourly_pivot(hourly_cons)
    pivot_sim = make_hourly_pivot(hourly_sim)
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
    
    im1 = ax1.imshow(pivot_sim, aspect='auto', origin='lower')
    ax1.set_title('Hourly Heatmap of Simulated PV Generation')
    ax1.set_ylabel('Hour of Day')
    cbar1 = fig.colorbar(im1, ax=ax1, pad=0.02)
    cbar1.set_label('kWh')
    
    im2 = ax2.imshow(pivot_cons, aspect='auto', origin='lower')
    ax2.set_title('Hourly Heatmap of Consumption')
    ax2.set_ylabel('Hour of Day')
    ax2.set_xlabel('Day of Year')
    cbar2 = fig.colorbar(im2, ax=ax2, pad=0.02)
    cbar2.set_label('kWh')
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/hourly_heatmaps.png", dpi=300)
    plt.close()

# ===============================================================
# Main execution
# ===============================================================

def main():
    """Run all analysis sections"""
    print("Starting Solar PV and Consumption Analysis...")
    
    # Run individual analysis sections
    daily_pv_sim, daily_pv_act = analyze_pv_generation()
    daily_cons_with_brewery = analyze_consumption_with_brewery()
    analyze_consumption_without_brewery()
    
    # Run comparative analysis
    analyze_generation_vs_consumption(daily_pv_sim, daily_pv_act, daily_cons_with_brewery)
    
    print(f"\nAnalysis complete! All outputs saved to '{OUTPUT_DIR}' directory.")

if __name__ == "__main__":
    main()