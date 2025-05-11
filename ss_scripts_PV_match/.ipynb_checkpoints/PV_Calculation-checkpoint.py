import os
import PySAM.Pvwattsv8 as pv
import pandas as pd
import numpy as np
from datetime import datetime

def load_data(csv_path: str) -> pd.DataFrame:
    """
    Load measured generation from a CSV with:
      • Column A: datetime strings 'YYYY-MM-DD HH:MM'
      • Column B: measured generation (kWh)
    """
    df = pd.read_csv(
        csv_path,
        usecols=[0, 1],
        names=['datetime', 'measured_kwh'],
        header=0,
        parse_dates=['datetime']
    )
    df.set_index('datetime', inplace=True)
    return df

def simulate_pv(
    weather_file: str,
    system_capacity_kw: float = 10.0,
    losses_pct: float = 20.28,
    start_year: int = 2024
) -> pd.DataFrame:
    """
    Run PVWatts simulation with an overall loss factor.
    Returns an hourly DataFrame with column 'simulated_kwh'.
    """
    model = pv.default('PVWattsNone')
    model.SolarResource.solar_resource_file = weather_file

    # System design
    model.SystemDesign.system_capacity = system_capacity_kw
    model.SystemDesign.dc_ac_ratio    = 1.15
    model.SystemDesign.inv_eff        = 96
    model.SystemDesign.tilt           = 10
    model.SystemDesign.azimuth        = 18
    model.SystemDesign.array_type     = 1      # roof-mount
    model.SystemDesign.module_type    = 0      # standard
    model.SystemDesign.gcr            = 0.3    # ground coverage ratio
    model.SystemDesign.losses         = losses_pct

    # Run
    model.execute()

    # Collect output (Watts → kWh) and timestamp index
    ac_kwh = np.array(model.Outputs.ac) / 1000.0
    start  = datetime(start_year, 1, 1)
    idx    = pd.date_range(start, periods=len(ac_kwh), freq='h')

    return pd.DataFrame({'simulated_kwh': ac_kwh}, index=idx)

def main():
    # Locate data directory relative to this script’s location
    script_dir   = os.path.dirname(os.path.realpath(__file__))
    project_dir  = os.path.abspath(os.path.join(script_dir, '..'))
    data_dir     = os.path.join(project_dir, 'data')

    weather_file = os.path.join(data_dir, 'Bonfire_2025.epw')
    measured_csv = os.path.join(data_dir, 'PV_Generation_excel.csv')

    # Load & simulate
    measured  = load_data(measured_csv)
    simulated = simulate_pv(weather_file, 10.0, 20.28, 2024)

    # Quick sanity prints
    print("Measured head:\n", measured.head(), "\n")
    print("Simulated head:\n", simulated.head(), "\n")

    # Optional monthly diff
    meas_m = measured.resample('ME').sum()
    sim_m  = simulated.resample('ME').sum()
    diff   = (sim_m - meas_m) / meas_m * 100
    print("Monthly diff %:\n", diff)

if __name__ == '__main__':
    main()
