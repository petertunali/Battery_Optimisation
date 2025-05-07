import numpy as np
import numpy_financial as nf

def compute_financials(
    totals: dict,
    battery_kwh: float,
    pv_kw: float,
    pv_cost_per_kw: float = 1000.0,
    battery_cost_per_kwh: float = 800.0,
    feed_in_tariff: float = 0.033,
    import_price_peak: float = 0.39710,
    import_price_offpeak: float = 0.13530,
    opex_per_day: float = 0.10,
    years: int = 30,
    discount_rate: float = None,
    baseline_import_cost: float = 0.0
) -> dict:
    """
    Compute lifecycle CAPEX, OPEX, revenues/costs, IRR & NPC over `years`.
    Subtracts `baseline_import_cost` from the gross import bill to get *incremental* cost.
    """
    # 1) CAPEX
    capex_pv   = pv_kw * pv_cost_per_kw
    capex_batt = battery_kwh * battery_cost_per_kwh
    capex_total = capex_pv + capex_batt

    # 2) Revenues & gross import cost
    export_rev  = totals.get('total_pv_export', 0.0) * feed_in_tariff
    import_cost = (
        totals.get('total_grid_import_peak',   0.0) * import_price_peak +
        totals.get('total_grid_import_offpeak',0.0) * import_price_offpeak
    )

    # 3) OPEX
    opex_total = opex_per_day * battery_kwh * 365 * years

    # 4) Incremental import cost
    import_cost_inc = import_cost - baseline_import_cost

    # 5) Net present cost (incremental)
    net_cost = capex_total + opex_total + import_cost_inc - export_rev

    # 6) Build annualized cash‑flows
    annual_export     = export_rev / years
    annual_import_inc = import_cost_inc / years
    annual_opex       = opex_total / years
    annual_net        = annual_export - annual_import_inc - annual_opex

    cash_flows = [-capex_total] + [annual_net] * years

    # 7) IRR
    irr = nf.irr(cash_flows) or 0.0

    # 8) (Discounted) NPV if requested
    npv = None
    if discount_rate is not None:
        npv = sum(cf / (1 + discount_rate)**i for i, cf in enumerate(cash_flows))

    return {
        'capex_pv':            capex_pv,
        'capex_battery':       capex_batt,
        'capex_total':         capex_total,
        'export_revenue_total':export_rev,
        'import_cost_total':   import_cost_inc,
        'opex_total':          opex_total,
        'net_cost':            net_cost,
        'annual_cash_flow':    cash_flows,
        'irr':                 irr,
        'npv':                 npv
    }
