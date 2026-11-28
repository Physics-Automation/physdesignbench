"""Private paper-geometry fixture for the Halir anisotropic SWG beamsplitter."""

from __future__ import annotations

import photonforge as pf


def _rect(x0: float, x1: float, y0: float, y1: float) -> pf.Polygon:
    return pf.Polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])


def _trapezoid(x0: float, x1: float, yc: float, w0: float, w1: float) -> pf.Polygon:
    return pf.Polygon(
        [
            (x0, yc - w0 / 2),
            (x1, yc - w1 / 2),
            (x1, yc + w1 / 2),
            (x0, yc + w0 / 2),
        ]
    )


def build_submission():
    component = pf.Component("ANISOTROPIC_SWG_PAPER_AUTHOR_FIXTURE")
    silicon = (1, 0)
    residual_void = (2, 0)
    period = 0.190
    bar = 0.095
    taper_periods = 30
    mmi_periods = 74
    lead = 1.5
    taper_length = taper_periods * period
    mmi_length = mmi_periods * period
    left_taper_start = lead
    mmi_start = left_taper_start + taper_length
    mmi_end = mmi_start + mmi_length
    right_taper_end = mmi_end + taper_length
    device_end = right_taper_end + lead
    port_width = 0.50
    access_width = 1.70
    mmi_width = 3.25
    bridge_tip = 0.010
    port_centers = (-1.0, 1.0)

    for center in port_centers:
        component.add(silicon, _rect(0.0, left_taper_start, center - 0.25, center + 0.25))
        component.add(silicon, _rect(right_taper_end, device_end, center - 0.25, center + 0.25))
        component.add(
            silicon,
            _trapezoid(left_taper_start, mmi_start, center, port_width, bridge_tip),
        )
        component.add(
            silicon,
            _trapezoid(mmi_end, right_taper_end, center, bridge_tip, port_width),
        )

    for index in range(taper_periods):
        u = (index + 0.5) / taper_periods
        left_width = port_width + u * (access_width - port_width)
        right_width = access_width - u * (access_width - port_width)
        left_x = left_taper_start + index * period
        right_x = mmi_end + index * period
        left_gap_x = left_x + bar
        right_gap_x = right_x + bar
        left_bridge = port_width + ((left_gap_x + 0.0475 - left_taper_start) / taper_length) * (bridge_tip - port_width)
        right_bridge = bridge_tip + ((right_gap_x + 0.0475 - mmi_end) / taper_length) * (port_width - bridge_tip)
        for center in port_centers:
            component.add(silicon, _rect(left_x, left_x + bar, center - left_width / 2, center + left_width / 2))
            component.add(silicon, _rect(right_x, right_x + bar, center - right_width / 2, center + right_width / 2))
            component.add(residual_void, _rect(left_gap_x, left_x + period, center - left_width / 2, center - left_bridge / 2))
            component.add(residual_void, _rect(left_gap_x, left_x + period, center + left_bridge / 2, center + left_width / 2))
            component.add(residual_void, _rect(right_gap_x, right_x + period, center - right_width / 2, center - right_bridge / 2))
            component.add(residual_void, _rect(right_gap_x, right_x + period, center + right_bridge / 2, center + right_width / 2))

    for index in range(mmi_periods):
        x0 = mmi_start + index * period
        component.add(silicon, _rect(x0, x0 + bar, -mmi_width / 2, mmi_width / 2))
        component.add(residual_void, _rect(x0 + bar, x0 + period, -mmi_width / 2, mmi_width / 2))

    return component
