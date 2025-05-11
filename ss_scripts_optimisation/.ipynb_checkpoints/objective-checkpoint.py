# objective.py

import pandas as pd
from battery   import simulate_battery_dispatch
from financial import compute_financials

def evaluate_solution(
    params: dict,
    pv_profile: pd.Series,
    demand_profile: pd.Series
) -> list:
    """
    Evaluate a candidate for NSGA-II:

    params must include:
      - 'pv_kw'        : float
      - 'battery_kwh'  : float

    pv_profile : pd.Series or DataFrame with column 'simulated_kwh'
    demand_profile : pd.Series

    Returns [ -IRR, NPC ] for minimisation.
    """
    # pull out the kWh series
    if hasattr(pv_profile, 'columns'):
        gen_series = pv_profile['simulated_kwh']
    else:
        gen_series = pv_profile

    # dispatch battery
    dispatch_df, totals = simulate_battery_dispatch(
        pv_gen=gen_series,
        demand=demand_profile,
        battery_kwh=params['battery_kwh'],
        battery_kw=params.get('battery_kw', None)
    )

    # compute financials
    fin = compute_financials(
        totals,
        battery_kwh=params['battery_kwh'],
        pv_kw=params['pv_kw']
    )

    # return objectives: negate IRR so NSGA-II maximises IRR, minimise NPC
    irr = fin['irr'] or 0.0
    npc = fin['net_cost']
    return [-irr, npc]
