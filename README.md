Latest edit 04/05/2025 8:52 P.M.

The current optimisation utilises NSGA-II. The run script is in Notebooks_optimisation in the <NSGA_II_times_3.ipynb> Juypter notebook.

The scripts it currently runs is in the scripts_optimisation_times_3 folder. <pv_simulat.py> <battery.py> <financial.py> <objective.py>.

This optimsiation tool is powered by Py-SAM from NREL. It will match an ideal size of PV system (based on a current 10 kW) and a range of other kW sizes (based on site - easily adjustable - just remove the arrays in <NSGA_II_times_3.ipynb>

Project Structure/Folder Structure

├── data/ - Current stored Data
│ ├── Bonfire_2025.epw (base file - for years 2025 - 2035)
│ ├── Bonfire_2040_4_5.epw (for years 2035 - 2045)
│ ├── Bonfire_2050_4_5.epw (for years 2045 - 2055)
│ ├── PV_Generation_excel.csv - Presented in half an hour blocks (in this version, time stamp at the start), and converted to 2025 for simplicity as 2024 (when data is orienated) (DEMAND DATA IS ALSO IN HERE - including external load of Cider and Beer production)
│ └── … (if any other inputs, I will put here)
│
├── scripts_optimisation/ reuseable scripts
│ ├── pv_simulate.py - PVWatts to half‑hourly & multi‑year tiling
│ ├── battery.py - battery dispatch + degradation model
│ ├── financial.py - CAPEX/OPEX/IRR/NPC calculations
│ ├── objective.py - PV/battery→financials for NSGA and ranking
│ └── run_nsga.py - script to launch NSGA‑II & save Pareto front
│
├── notebooks/ - Editing but also current running tool
│ ├── nsga_II.ipynb
│ └── nsga_II_times_3.ipynb - current one utilised
│
├── outputs_optimisation/ ← results (e.g. Pareto solutions)
│ └── pareto_solutions.csv
│
├── requirements.txt ← Python dependencies
└── README.md ← this file

# Please look at requirements to see what packages to install
pip install -r requirements.txt



Prepare data
Weather
Place your EPW files in data/, named:

Current files as listed above
Bonfire_2025.epw

Bonfire_2040_4_5.epw

Bonfire_2050_4_5.epw

Measured PV generation
Half‑hourly CSV with columns:

yyyy‑mm‑dd hh:mm (timestamp at start of interval)

kWh (generation)
Save as data/PV_Generation_excel.csv.

Demand profile
Can put in third column in above CSV





Module descriptions
pv_simulate.py

simulate_pv_simple(…) - 1‑year hourly PVWatts → half‑hourly kWh

simulate_multi_year_pv(…) - tiles three EPW runs (2025/2040/2050) One for each region

battery.py

simulate_battery_dispatch(…) - dispatch logic over half‑hourly PV & demand, SOC tracking, 1% annual degradation, peak/off‑peak import, export

financial.py

compute_financials(…) → CAPEX (PV + battery), OPEX, export revenue, import cost, IRR (via numpy_financial) and Net Present Cost - also times by 10 as only 3 years was done

objective.py

evaluate_solution(params, pv_profile, demand_profile) → runs dispatch + financials, returns [-IRR, NPC] for NSGA minimisation

run_nsga.py - which is currently just my juypter notebook

Defines SolarBatteryProblem (1 decision var: battery_kwh), calls pymoo NSGA‑II, saves Pareto front to outputs_optimisation/pareto_solutions.csv

