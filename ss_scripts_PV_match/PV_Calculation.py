# PV_Calculation.py
# Minimal PVWatts simulation script using PySAM
# Place this in the `scripts/` folder of your Battery_Optimisation project

import os
from datetime import datetime
import PySAM.Pvwattsv8 as pv
import pandas as pd
import numpy as np


def simulate_pv(
    weather_file: str,
    system_capacity_kw: float = 10.0,
    soiling: float = 2.0,
    shading: float = 3.0,
    snow: float = 0.0,
    mismatch: float = 2.0,
    wiring: float = 2.0,
    connections: float = 0.5,
    lid: float = 1.5,            # light-induced degradation
    nameplate: float = 1.0,
    age: float = 0.0,
    availability: float = 3.0,
    dc_ac_ratio: float = 1.15,
    inv_eff: float = 96.0,
    tilt: float = 10.0,
    azimuth: float = 18.0,
    array_type: int = 1,
    module_type: int = 0,
    gcr: float = 0.3,
    start_year: int = 2024,
) -> pd.DataFrame:
    """
    Run a PVWatts simulation and return hourly kWh results.
    Losses are specified individually; total loss is the sum of all categories.
    """
    # Calculate total system losses
    total_loss_pct = (
        soiling + shading + snow + mismatch + wiring + connections
        + lid + nameplate + age + availability
    )

    # Instantiate PVWatts model
    model = pv.default('PVWattsNone')
    model.SolarResource.solar_resource_file = weather_file

    # Apply system design parameters
    design = model.SystemDesign
    design.system_capacity = system_capacity_kw
    design.dc_ac_ratio    = dc_ac_ratio
    design.inv_eff        = inv_eff
    design.tilt           = tilt
    design.azimuth        = azimuth
    design.array_type     = array_type
    design.module_type    = module_type
    design.gcr            = gcr
    design.losses         = total_loss_pct

    # Execute simulation
    model.execute()

    # Convert outputs (W → kWh) and index
    ac_kwh = np.array(model.Outputs.ac) / 1000.0
    start = datetime(start_year, 1, 1)
    idx = pd.date_range(start, periods=len(ac_kwh), freq='H')

    # Return DataFrame with only simulated output
    return pd.DataFrame({'simulated_kwh': ac_kwh}, index=idx)


def main():
    # --- Config ---
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    data_dir = os.path.join(project_root, 'data')
    outputs_dir = os.path.join(project_root, 'outputs')

    # Create output subfolder
    subfolder = 'case1'  # <--- change this as needed
    out_folder = os.path.join(outputs_dir, subfolder)
    os.makedirs(out_folder, exist_ok=True)

    # File paths
    weather_file = os.path.join(data_dir, 'Bonfire_2025.epw')
    output_csv = os.path.join(out_folder, 'pv_simulation_output.csv')

    # Run simulation with individual losses
    df = simulate_pv(
        weather_file,
        system_capacity_kw=10.0,
        soiling=2.0,
        shading=3.0,
        snow=0.0,
        mismatch=2.0,
        wiring=2.0,
        connections=0.5,
        lid=1.5,
        nameplate=1.0,
        age=0.0,
        availability=3.0,
        dc_ac_ratio=1.15,
        inv_eff=96.0,
        tilt=10.0,
        azimuth=18.0,
        array_type=1,
        module_type=0,
        gcr=0.3,
        start_year=2024,
    )

    # Save to CSV (only simulated_kwh column)
    df.to_csv(output_csv)
    print(f"Simulation complete. Results saved to: {output_csv}")


if __name__ == '__main__':
    main()
