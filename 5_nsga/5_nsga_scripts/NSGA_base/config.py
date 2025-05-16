# config.py
from pathlib import Path

# Default data directory path
DATA_DIR = Path("/Users/petertunali/Documents/GitHub/Battery_Optimisation/data")

# Weather files to use in order (will be searched for in DATA_DIR)
DESIRED_WEATHER_FILES = [
    "Bonfire_2025.epw",
    "Bonfire_2040_4_5.epw",
    "Bonfire_2050_4_5.epw"
]

# Demand file configuration with exact column names
DEMAND_FILE_CONFIG = {
    "primary_file": "PV_Generation_excel.csv",
    "columns": {
        "timestamp": "Date and Time",         # Column A - Timestamp
        "pv_generation": "PV Generated (kWh)", # Column B - PV generation with units
        "consumption": "Consumption (kWh)"    # Column C - Consumption with units
    },
    "date_format": "%d/%m/%Y %H:%M"           # Format for date parsing
}

# Alternative demand files (backup options)
ALTERNATIVE_DEMAND_FILES = [
    "Energy_Demand_and_Supply_2024.csv"
]

# Define existing PV system
EXISTING_PV = {
    'name': 'existing_system',
    'system_capacity_kw': 10.0,
    'tilt': 10.0,
    'azimuth': 18.0,
    'shading': 43.0,
    'array_type': 1  # Roof-mounted
}

# Define new PV system options based on priority
PV_OPTIONS = [
    {
        'name': 'accommodation_block',
        'max_capacity_kw': 33.0,
        'tilt': 20.0,
        'azimuth': 40.0,
        'shading': 0.0,
        'array_type': 1,  # Roof-mounted
        'cost_multiplier': 1.0
    },
    {
        'name': 'small_shed',
        'max_capacity_kw': 10.0,
        'tilt': 20.0,
        'azimuth': 20.0,
        'shading': 20.0,
        'array_type': 1,  # Roof-mounted
        'cost_multiplier': 1.0
    },
    {
        'name': 'ground_mounted',
        'max_capacity_kw': float('inf'),  # Unlimited
        'tilt': 30.0,
        'azimuth': 5.0,
        'shading': 0.0,
        'array_type': 0,  # Ground-mounted
        'cost_multiplier': 1.25  # 25% cost increase
    }
]

# Financial parameters
DISCOUNT_RATE = 0.07  # 7% - Sensitivity analysis: try 5%, 9%, 11%
ELECTRICITY_RATES = {
    'peak': 0.3971,  # $/kWh - Sensitivity analysis: try ±20%
    'offpeak': 0.1353,  # $/kWh - Sensitivity analysis: try ±20%
    'export': 0.033,  # $/kWh - Sensitivity analysis: try 0.05, 0.08, 0.10
}
ANNUAL_MAINTENANCE_PV = 250  # $ per year
ANNUAL_MAINTENANCE_BATTERY = 0  # $ per year
MAINTENANCE_INFLATION = 0.03  # 3% per year - Sensitivity analysis: try 2%, 4%, 5%
ELECTRICITY_PRICE_ESCALATION = 0.03  # 3% per year - Sensitivity analysis: try 2%, 4%, 5%

# Battery parameters
BATTERY_EFF = 0.9  # 90% roundtrip efficiency
MIN_SOC = 0.05  # 5% minimum state of charge
BATTERY_POWER_RATIO = 0.5  # Power rating is 0.5C (half of energy capacity)
BATTERY_DEGRADATION = 0.01  # 1% annual degradation

# Battery control parameters (for peak/off-peak behavior)
BATTERY_CONTROL = {
    'peak_reserve_soc': 0.05,    # Set to MIN_SOC - no special reserve for basic mode
    'peak_reserve_hours': 0,     # Don't start reserving ahead of peak in basic mode
    'off_peak_min_soc': 0.05     # Same as regular MIN_SOC for basic mode
}

# Battery rebate scenarios (for sensitivity analysis)
BATTERY_REBATES = {
    'enabled': False,  # Set to True to enable rebates
    'fixed_amount': 0,  # Fixed rebate amount ($)
    'percentage': 0,    # Percentage rebate (0-1)
    'cap': 0            # Maximum rebate cap ($)
}

# NSGA-II parameters
POPULATION_SIZE = 50
N_GENERATIONS = 40

# Capital cost formulas and parameters
PV_COST_FORMULA = {
    'base_cost': 1047.3,  # $/kW - Sensitivity: try ±15%
    'exponent': -0.002,   # For economies of scale: cost * e^(exponent*capacity)
    'minimum': 750        # Minimum cost per kW - Sensitivity: try 650, 850
}

BATTERY_COST_FORMULA = {
    'base_cost': 977.54,  # $/kWh - Sensitivity: try ±20%
    'exponent': -0.004,   # For economies of scale: cost * e^(exponent*capacity)
    'minimum': 600,       # Minimum cost per kWh - Sensitivity: try 500, 700
    'installation': 174   # Installation cost per kWh - Sensitivity: try 150, 200
}

# Weather file scenarios (for sensitivity analysis)
WEATHER_SCENARIOS = {
    'baseline': DESIRED_WEATHER_FILES,
    'rcp8_5': [
        "Bonfire_2025.epw",
        "Bonfire_2040_RCP8_5.epw", 
        "Bonfire_2050_RCP8_5.epw"
    ]
}

# Baseline costs (from bills)
ANNUAL_NO_PV_COST = 9424.48  # Annual electricity cost with no PV, no battery
ANNUAL_PV_ONLY_COST = 8246.44  # Annual electricity cost with PV only, no battery, includes export incomes (which is already very minimal)

# Existing PV is a sunk cost - don't include in financial calculations
EXISTING_PV_IS_SUNK_COST = True

# Project parameters
PROJECT_LIFETIME = 30  # Years to analyze - Sensitivity: try 20, 25 years

# Grid parameters
GRID_EMISSIONS = 0.79  # kg CO2e per kWh imported - Sensitivity: try future scenarios with lower emissions

# Peak times (for time-of-use tariffs) - Properly labeled for Australian seasons
PEAK_PERIODS = {
    'AEDT_period': {'start_hour': 14, 'end_hour': 20},  # 2pm-8pm during daylight saving time
    'AEST_period': {'start_hour': 15, 'end_hour': 21},  # 3pm-9pm during standard time
}

# Month classifications by time zone
AEDT_MONTHS = [10, 11, 12, 1, 2, 3]  # October to March - Daylight saving time
AEST_MONTHS = [4, 5, 6, 7, 8, 9]     # April to September - Standard time

# For clarity, also define the seasons (even though they align with time zones in this case)
SUMMER_MONTHS = [12, 1, 2]           # December to February
AUTUMN_MONTHS = [3, 4, 5]            # March to May
WINTER_MONTHS = [6, 7, 8]            # June to August
SPRING_MONTHS = [9, 10, 11]          # September to November

# Prioritization flag for additional PV allocation
PRIORITIZE_PV_ALLOCATION = True  # If True, fill roof space in order of PV_OPTIONS

# Demand growth scenarios for sensitivity analysis
DEMAND_GROWTH = {
    'enabled': False,  # Set to True to enable demand growth
    'pattern': 'decade',  # 'decade' = increases every 10 years, 'annual' = yearly increase
    'percent_per_decade': 0,  # Base case: 0% increase (1x multiplier)
    # Sensitivity scenarios
    'scenarios': {
        'low_growth': 5,      # 5% increase per decade
        'medium_growth': 10,  # 10% increase per decade
        'high_growth': 20     # 20% increase per decade
    }
}