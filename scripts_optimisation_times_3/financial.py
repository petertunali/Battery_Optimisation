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
    discount_rate: float = None
) -> dict:
    """
    Compute CAPEX, OPEX, revenues/costs, IRR & NPC over 'years' horizon.
    Assumes 'totals' has already been scaled to a full 30-year basis.
    """
    capex_pv    = pv_kw * pv_cost_per_kw
    capex_batt  = battery_kwh * battery_cost_per_kwh
    capex_total = capex_pv + capex_batt

        # subtract sunk cost of existing 10 kW PV system:
    capex_total -= 10_000.0

    export_rev   = totals['total_pv_export'] * feed_in_tariff
    import_cost  = (
        totals['total_grid_import_peak']   * import_price_peak +
        totals['total_grid_import_offpeak']* import_price_offpeak
    )
    opex_total   = opex_per_day * battery_kwh * 365 * years

    net_cost = capex_total + opex_total + import_cost - export_rev

    annual_export = export_rev / years
    annual_import = import_cost / years
    annual_opex   = opex_total / years
    annual_net    = annual_export - annual_import - annual_opex

    cash_flows = [-capex_total] + [annual_net] * years
    irr  = nf.irr(cash_flows)
    npv  = None
    if discount_rate is not None:
        npv = sum(cf / (1 + discount_rate)**i for i, cf in enumerate(cash_flows))

    return {
        'capex_pv':             capex_pv,
        'capex_battery':        capex_batt,
        'capex_total':          capex_total,
        'export_revenue_total': export_rev,
        'import_cost_total':    import_cost,
        'opex_total':           opex_total,
        'net_cost':             net_cost,
        'annual_cash_flow':     cash_flows,
        'irr':                  irr,
        'npv':                  npv
    }
