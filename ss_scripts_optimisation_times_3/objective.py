import pandas as pd
from battery   import simulate_battery_dispatch
from financial import compute_financials

def evaluate_solution(
    params: dict,
    pv_profile: pd.Series,
    demand_profile: pd.Series
) -> list:
    """
    Returns [–IRR, NPC] for NSGA-II.  Scales a 3-yr dispatch to 30 yr.
    """
    if hasattr(pv_profile, 'columns'):
        gen = pv_profile['simulated_kwh']
    else:
        gen = pv_profile

    dispatch_df, totals3 = simulate_battery_dispatch(
        pv_gen=gen,
        demand=demand_profile,
        battery_kwh=params['battery_kwh'],
        battery_kw=params.get('battery_kw', None)
    )

    # scale 3 yr totals → 30 yr totals
    sim_years    = len(pv_profile) / 17520
    scale_factor = 30.0 / sim_years
    totals30 = {k: v * scale_factor for k, v in totals3.items()}

    fin = compute_financials(
        totals30,
        battery_kwh=params['battery_kwh'],
        pv_kw=params['pv_kw']
    )

    irr = fin['irr'] or 0.0
    npc = fin['net_cost']
    return [-irr, npc]
