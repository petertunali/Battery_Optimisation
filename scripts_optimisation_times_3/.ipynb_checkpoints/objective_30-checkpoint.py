from battery_30 import simulate_battery_dispatch
from financial_30 import compute_financials

# to be set by your notebook (Cell 6) to the zero‑storage import bill
BASE_IMPORT_COST = 0.0  

def evaluate_solution(
    params: dict,
    pv_profile,       # pd.Series or DataFrame with 'simulated_kwh'
    demand_profile    # pd.Series of matching half‑hourly demand
) -> list:
    """
    Returns [ -IRR, NPC ] over 30 yrs, using discounted NPV for NPC.
    """
    gen = pv_profile['simulated_kwh'] if hasattr(pv_profile, 'columns') else pv_profile

    # 1) dispatch
    _, totals = simulate_battery_dispatch(
        pv_gen      = gen,
        demand      = demand_profile,
        battery_kwh = params['battery_kwh']
    )

    # 2) financials (5% real discount) with baseline import cost built in
    fin = compute_financials(
        totals,
        battery_kwh           = params['battery_kwh'],
        pv_kw                 = params['pv_kw'],
        discount_rate         = 0.05,
        baseline_import_cost  = BASE_IMPORT_COST
    )

    # 3) pack objectives
    irr = fin['irr'] or 0.0
    npc = fin['net_cost']    # incremental NPV / NPC
    return [-irr, npc]
