#!/usr/bin/env python3
"""Credential-owning worker for one evaluator-constructed anisotropic-SWG FDTD run."""

from __future__ import annotations

import argparse
import ctypes
import json
import math
import os
import re
import signal
import socket
import struct
import tempfile
from pathlib import Path
from typing import Any

MAX_MESSAGE_BYTES = 4 * 1024 * 1024
PR_SET_DUMPABLE = 4
PROFILE_PATH = Path("/opt/picbench/evaluator/runtime/task_profile.json")
RESULT_CACHE_PATH = Path("/experiment/tidy3d_result.hdf5")
PAID_RECEIPT_PATH = Path(
    os.environ.get(
        "PICBENCH_PAID_USAGE_RECEIPT_PATH",
        "/experiment/paid_usage_receipt.json",
    )
)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
SPEED_OF_LIGHT_UM_PER_S = 299792458000000.0
_PRIVATE_IDENTIFIERS: set[str] = set()
MINIMUM_HEADROOM_ENV = "PICBENCH_MINIMUM_FLEXCREDIT_HEADROOM_FRACTION"


def _write_paid_receipt(*, estimate: float, start_accepted: bool) -> None:
    if os.environ.get("PICBENCH_EVALUATION_MODE", "final") != "probe":
        return
    raw_probe_id = os.environ.get("PICBENCH_PROBE_ID", "")
    submission_sha256 = os.environ.get("PICBENCH_SUBMISSION_SHA256", "")
    if (
        not raw_probe_id.isdecimal()
        or int(raw_probe_id) < 1
        or SHA256_RE.fullmatch(submission_sha256) is None
    ):
        raise RuntimeError("trusted probe accounting identity is unavailable")
    payload = {
        "schema_version": 1,
        "probe_id": int(raw_probe_id),
        "submission_sha256": submission_sha256,
        "estimated_flexcredit": estimate,
        "start_attempted": True,
        "start_accepted": start_accepted,
    }
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{PAID_RECEIPT_PATH.name}.", dir=PAID_RECEIPT_PATH.parent
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True, allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, PAID_RECEIPT_PATH)
    finally:
        temporary.unlink(missing_ok=True)


def _require_probe_budget_capacity(estimate: float) -> None:
    if os.environ.get("PICBENCH_EVALUATION_MODE", "final") != "probe":
        return
    try:
        paid_remaining = int(os.environ["PICBENCH_PAID_LEARNER_PROBES_REMAINING"])
        flexcredit_remaining = float(
            os.environ["PICBENCH_LEARNER_THERMAL_REMAINING_FLEXCREDIT"]
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("trusted probe accounting capacity is invalid") from exc
    if paid_remaining <= 0:
        raise RuntimeError("paid probe budget exhausted before cloud start")
    if not math.isfinite(flexcredit_remaining) or estimate > flexcredit_remaining:
        raise RuntimeError("probe FlexCredit budget exhausted before cloud start")


def _effective_flexcredit_limit(maximum: float) -> float:
    """Apply the optional trusted release headroom without changing native runs."""
    raw_fraction = os.environ.get(MINIMUM_HEADROOM_ENV)
    if raw_fraction is None:
        return maximum
    try:
        fraction = float(raw_fraction)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("trusted minimum FlexCredit headroom fraction is invalid") from exc
    if (
        not math.isfinite(fraction)
        or not 0.0 <= fraction < 1.0
        or not math.isfinite(maximum)
        or maximum <= 0.0
    ):
        raise RuntimeError("trusted minimum FlexCredit headroom fraction is invalid")
    return maximum * (1.0 - fraction)


def _protect_process() -> None:
    if ctypes.CDLL(None, use_errno=True).prctl(PR_SET_DUMPABLE, 0, 0, 0, 0) != 0:
        raise RuntimeError("could not disable FDTD worker dumpability")


def _receive(sock: socket.socket) -> dict[str, Any]:
    header = sock.recv(4, socket.MSG_WAITALL)
    if len(header) != 4:
        raise RuntimeError("missing trusted request header")
    length = struct.unpack("!I", header)[0]
    if length <= 0 or length > MAX_MESSAGE_BYTES:
        raise RuntimeError("trusted request exceeded the IPC limit")
    body = sock.recv(length, socket.MSG_WAITALL)
    if len(body) != length:
        raise RuntimeError("truncated trusted request")
    payload = json.loads(body)
    if not isinstance(payload, dict):
        raise RuntimeError("trusted request must be an object")
    return payload


def _send(sock: socket.socket, payload: dict[str, Any]) -> None:
    body = json.dumps(payload, separators=(",", ":"), allow_nan=False).encode("utf-8")
    if len(body) > MAX_MESSAGE_BYTES:
        raise RuntimeError("trusted response exceeded the IPC limit")
    sock.sendall(struct.pack("!I", len(body)) + body)


def _redact_error(exc: BaseException) -> str:
    message = f"{type(exc).__name__}: {exc}"
    for name in ("TIDY3D_API_KEY", "SIMCLOUD_APIKEY"):
        secret = os.environ.get(name, "")
        if secret:
            message = message.replace(secret, "[redacted]")
    for identifier in _PRIVATE_IDENTIFIERS:
        message = message.replace(identifier, "[redacted-task-id]")
    return message[:2000]


def _polygon_area(points: list[list[float]]) -> float:
    return abs(sum(a[0] * b[1] - b[0] * a[1] for a, b in zip(points, points[1:] + points[:1])) / 2.0)


def _validate_request(request: dict[str, Any], profile: dict[str, Any]) -> dict[str, Any]:
    if set(request) != {"schema_version", "device_length_um", "polygons"} or request.get("schema_version") != 1:
        raise RuntimeError("trusted geometry request has an invalid schema")
    contract = profile["layout_contract"]
    length = float(request["device_length_um"])
    tolerance = float(contract["coordinate_tolerance_um"])
    if not math.isfinite(length) or not float(contract["device_length_min_um"]) - tolerance <= length <= float(contract["device_length_max_um"]) + tolerance:
        raise RuntimeError("trusted geometry request has an invalid device length")
    allowed = {tuple(int(value) for value in pair) for pair in contract["allowed_layers"]}
    bounds = contract["bounds_um"]
    maximum_vertices = int(contract["maximum_vertex_count"])
    raw = request.get("polygons")
    if not isinstance(raw, list) or not raw:
        raise RuntimeError("trusted geometry request contains no polygons")
    by_layer: dict[str, list[dict[str, Any]]] = {f"{layer}/{datatype}": [] for layer, datatype in allowed}
    areas = {key: 0.0 for key in by_layer}
    vertex_count = 0
    for polygon in raw:
        if not isinstance(polygon, dict) or set(polygon) != {"layer", "points_um"}:
            raise RuntimeError("trusted geometry polygon has an invalid schema")
        if not isinstance(polygon["layer"], list) or len(polygon["layer"]) != 2:
            raise RuntimeError("trusted geometry polygon has an invalid layer")
        layer = tuple(int(value) for value in polygon["layer"])
        if layer not in allowed:
            raise RuntimeError("trusted geometry polygon uses a forbidden layer")
        points = polygon["points_um"]
        if not isinstance(points, list) or len(points) < 4:
            raise RuntimeError("trusted geometry polygon has too few points")
        vertices = [[float(x), float(y)] for x, y in points]
        if not all(math.isfinite(value) for point in vertices for value in point):
            raise RuntimeError("trusted geometry contains a non-finite coordinate")
        vertex_count += len(vertices)
        if vertex_count > maximum_vertices:
            raise RuntimeError("trusted geometry exceeds the vertex limit")
        xs = [point[0] for point in vertices]
        ys = [point[1] for point in vertices]
        if min(xs) < float(bounds["x_min"]) - tolerance or max(xs) > length + tolerance or min(ys) < float(bounds["y_min"]) - tolerance or max(ys) > float(bounds["y_max"]) + tolerance:
            raise RuntimeError("trusted geometry polygon violates bounds")
        key = f"{layer[0]}/{layer[1]}"
        rules = contract["layers"][key]
        if min(max(xs) - min(xs), max(ys) - min(ys)) < float(rules["minimum_bbox_feature_um"]) - tolerance:
            raise RuntimeError("trusted geometry polygon has a sub-minimum feature")
        area = _polygon_area(vertices)
        if not math.isfinite(area) or area < float(rules["minimum_polygon_area_um2"]):
            raise RuntimeError("trusted geometry polygon has invalid area")
        by_layer[key].append({"points_um": vertices})
        areas[key] += area
    for key, rules in contract["layers"].items():
        count = len(by_layer[key])
        if not int(rules["minimum_polygon_count"]) <= count <= int(rules["maximum_polygon_count"]):
            raise RuntimeError(f"trusted geometry layer {key} has an invalid polygon count")
        if not float(rules["minimum_total_area_um2"]) <= areas[key] <= float(rules["maximum_total_area_um2"]):
            raise RuntimeError(f"trusted geometry layer {key} has an invalid total area")
    return {"device_length_um": length, "layers": by_layer}


def _medium(td: Any, index: float) -> Any:
    return td.Medium(permittivity=float(index) ** 2)


def _polyslab(td: Any, points: list[list[float]], z0: float, z1: float) -> Any:
    return td.PolySlab(vertices=[(float(x), float(y)) for x, y in points], axis=2, slab_bounds=(z0, z1))


def _spectral_wavelengths_um(profile: dict[str, Any]) -> list[float]:
    objective = profile["objective"]
    start = float(objective["wavelength_start_um"])
    stop = float(objective["wavelength_stop_um"])
    step = float(objective["wavelength_step_um"])
    count = int(round((stop - start) / step))
    wavelengths = [start + index * step for index in range(count + 1)]
    if not wavelengths or abs(wavelengths[-1] - stop) > 1e-9:
        raise RuntimeError("trusted wavelength grid is inconsistent")
    return wavelengths


def build_simulation(request: dict[str, Any], profile: dict[str, Any]) -> Any:
    import tidy3d as td

    physics = profile["trusted_physics"]
    contract = profile["layout_contract"]
    length = float(request["device_length_um"])
    extension = float(physics["access_extension_um"])
    x_min, x_max = -extension, length + extension
    x_center, x_size = (x_min + x_max) / 2.0, x_max - x_min
    bounds = contract["bounds_um"]
    y_margin = float(physics["simulation_y_margin_um"])
    y_min, y_max = float(bounds["y_min"]) - y_margin, float(bounds["y_max"]) + y_margin
    y_center, y_size = (y_min + y_max) / 2.0, y_max - y_min
    z_size = float(physics["simulation_z_size_um"])
    z0 = float(physics["device_bottom_um"])
    z1 = float(physics["device_top_um"])
    indices = physics["indices"]

    silicon_polygons = [polygon["points_um"] for polygon in request["layers"]["1/0"]]
    void_polygons = [polygon["points_um"] for polygon in request["layers"]["2/0"]]
    port_width = float(contract["port_landings"]["width_um"])
    half = port_width / 2.0
    centers = [float(value) for value in contract["port_landings"]["centers_um"]]
    for start, stop in ((x_min, 0.0), (length, x_max)):
        for center in centers:
            silicon_polygons.append([[start, center - half], [stop, center - half], [stop, center + half], [start, center + half]])

    structures = [
        td.Structure(
            geometry=td.GeometryGroup(geometries=tuple(_polyslab(td, points, z0, z1) for points in silicon_polygons)),
            medium=_medium(td, indices["si"]),
            name="silicon_device",
        ),
        td.Structure(
            geometry=td.GeometryGroup(geometries=tuple(_polyslab(td, points, z0, float(physics["air_void_top_um"])) for points in void_polygons)),
            medium=_medium(td, indices["air"]),
            name="residual_bottom_air_voids",
        ),
    ]
    wavelengths = _spectral_wavelengths_um(profile)
    frequencies = sorted(SPEED_OF_LIGHT_UM_PER_S / wavelength for wavelength in wavelengths)
    source_wavelength = float(physics["source_center_wavelength_um"])
    source_frequency = SPEED_OF_LIGHT_UM_PER_S / source_wavelength
    mode_spec = td.ModeSpec(num_modes=int(physics["mode_count"]), sort_spec=td.ModeSortSpec(**physics["mode_sort"]), precision="double")
    plane_size = (0.0, float(physics["mode_plane_y_size_um"]), float(physics["mode_plane_z_size_um"]))
    input_y = max(centers)
    source = td.ModeSource(
        center=(float(physics["source_offset_um"]), input_y, 0.0), size=plane_size,
        source_time=td.GaussianPulse(freq0=source_frequency, fwidth=float(physics["source_fwidth_hz"])),
        direction="+", mode_spec=mode_spec, mode_index=0,
        num_freqs=int(physics["source_num_freqs"]), broadband_method=str(physics["source_broadband_method"]),
        name="input_port_1",
    )
    monitors = [
        td.ModeMonitor(center=(float(physics["input_monitor_offset_um"]), input_y, 0.0), size=plane_size, freqs=frequencies, mode_spec=mode_spec, name="in_mode"),
        td.ModeMonitor(center=(length + float(physics["output_monitor_offset_um"]), max(centers), 0.0), size=plane_size, freqs=frequencies, mode_spec=mode_spec, name="out_3"),
        td.ModeMonitor(center=(length + float(physics["output_monitor_offset_um"]), min(centers), 0.0), size=plane_size, freqs=frequencies, mode_spec=mode_spec, name="out_4"),
    ]
    mesh_margin = float(physics["mesh_override_margin_um"])
    refine = td.MeshOverrideStructure(
        geometry=td.Box(
            center=(length / 2.0, y_center, 0.0),
            size=(length + 2.0 * mesh_margin, float(bounds["y_max"]) - float(bounds["y_min"]) + 2.0 * mesh_margin, 0.5),
        ),
        dl=tuple(float(value) for value in physics["mesh_override_dl_um"]), name="swg_fine_mesh",
    )
    return td.Simulation(
        center=(x_center, y_center, 0.0), size=(x_size, y_size, z_size), medium=_medium(td, indices["sio2"]),
        structures=structures, sources=[source], monitors=monitors,
        grid_spec=td.GridSpec.auto(wavelength=min(wavelengths), min_steps_per_wvl=int(physics["min_steps_per_wvl"]), override_structures=[refine]),
        boundary_spec=td.BoundarySpec.all_sides(boundary=td.PML()), run_time=float(physics["run_time_s"]),
        shutoff=float(physics["shutoff"]), normalize_index=0,
    )


def _amplitudes(simulation_data: Any, monitor: str, direction: str, frequencies: list[float]) -> list[complex]:
    import numpy as np

    data = simulation_data[monitor].amps.sel(direction=direction, mode_index=0)
    values: list[complex] = []
    for frequency in frequencies:
        selected = data.sel(f=frequency, method="nearest") if "f" in data.dims else data
        raw = np.asarray(selected.values).reshape(-1)
        if raw.size != 1:
            raise RuntimeError(f"{monitor} did not produce exactly one mode amplitude per frequency")
        values.append(complex(raw[0]))
    return values


def _wrap_degrees(angle: float) -> float:
    return (angle + 180.0) % 360.0 - 180.0


def _extract_metrics(simulation_data: Any, profile: dict[str, Any]) -> tuple[float, dict[str, float], list[dict[str, float]]]:
    import numpy as np

    wavelengths = _spectral_wavelengths_um(profile)
    frequencies = [SPEED_OF_LIGHT_UM_PER_S / wavelength for wavelength in wavelengths]
    incidents = _amplitudes(simulation_data, "in_mode", "+", frequencies)
    reflected = _amplitudes(simulation_data, "in_mode", "-", frequencies)
    out3_values = _amplitudes(simulation_data, "out_3", "+", frequencies)
    out4_values = _amplitudes(simulation_data, "out_4", "+", frequencies)
    tolerance = float(profile["evaluation"]["physical_bounds"]["numerical_power_tolerance"])
    envelope = profile["baseline"]["reference_operating_envelope"]
    el_limit = float(envelope["excess_loss_db_max"])
    imbalance_limit = float(envelope["imbalance_db_max"])
    phase_limit = float(envelope["phase_error_deg_max"])
    spectrum: list[dict[str, float]] = []
    for wavelength, incident, reflected_value, out3, out4 in zip(wavelengths, incidents, reflected, out3_values, out4_values, strict=True):
        p_incident = max(abs(incident) ** 2, float(np.finfo(float).tiny))
        t3 = float(abs(out3) ** 2 / p_incident)
        t4 = float(abs(out4) ** 2 / p_incident)
        total = t3 + t4
        if not all(math.isfinite(value) and value > 0.0 for value in (t3, t4, total)) or total > 1.0 + tolerance:
            raise RuntimeError("trusted output mode powers are non-physical")
        if total > 1.0:
            t3 /= total
            t4 /= total
            total = 1.0
        excess_loss = -10.0 * math.log10(total)
        imbalance = abs(10.0 * math.log10(t3 / t4))
        phase = math.degrees(np.angle(out3 / out4))
        phase_error = min(abs(_wrap_degrees(phase - 90.0)), abs(_wrap_degrees(phase + 90.0)))
        reflection = float(abs(reflected_value) ** 2 / p_incident)
        penalty = max(max(excess_loss, 0.0) / el_limit, imbalance / imbalance_limit, phase_error / phase_limit)
        spectrum.append({
            "wavelength_nm": wavelength * 1000.0,
            "T3": t3,
            "T4": t4,
            "excess_loss_db": excess_loss,
            "imbalance_db": imbalance,
            "phase_error_deg": phase_error,
            "reflection_db": -10.0 * math.log10(max(reflection, float(np.finfo(float).tiny))),
            "normalized_penalty": penalty,
        })
    worst = max(spectrum, key=lambda point: point["normalized_penalty"])
    center = min(spectrum, key=lambda point: abs(point["wavelength_nm"] - 1550.0))
    metric = float(worst["normalized_penalty"])
    secondary = {
        "T3_1550": center["T3"], "T4_1550": center["T4"],
        "excess_loss_db_1550": center["excess_loss_db"],
        "imbalance_db_1550": center["imbalance_db"],
        "phase_error_deg_1550": center["phase_error_deg"],
        "reflection_db_1550": center["reflection_db"],
        "worst_excess_loss_db": max(point["excess_loss_db"] for point in spectrum),
        "worst_imbalance_db": max(point["imbalance_db"] for point in spectrum),
        "worst_phase_error_deg": max(point["phase_error_deg"] for point in spectrum),
        "worst_wavelength_nm": worst["wavelength_nm"],
    }
    return metric, secondary, spectrum


def run_fdtd(request: dict[str, Any]) -> dict[str, Any]:
    from tidy3d import web

    _PRIVATE_IDENTIFIERS.clear()
    profile = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
    maximum = float(profile["evaluation"]["max_flexcredit_per_run"])
    effective_maximum = _effective_flexcredit_limit(maximum)
    validated = _validate_request(request, profile)
    simulation = build_simulation(validated, profile)
    task_id = None
    try:
        task_id = web.upload(simulation=simulation, task_name=profile["trusted_physics"]["cloud_task_name"])
        task_id_text = "" if task_id is None else str(task_id).strip()
        if not task_id_text:
            raise RuntimeError("Tidy3D did not return a task ID")
        _PRIVATE_IDENTIFIERS.add(task_id_text)
        estimate = web.estimate_cost(task_id)
        if estimate is None:
            raise RuntimeError("Tidy3D did not return a cost estimate")
        estimate = float(estimate)
        if not math.isfinite(estimate) or estimate < 0.0 or estimate > maximum:
            raise RuntimeError(f"estimated FlexCredit {estimate!r} exceeds trusted limit {maximum}")
        if estimate > effective_maximum:
            raise RuntimeError(
                f"estimated FlexCredit {estimate!r} exceeds trusted headroom "
                f"limit {effective_maximum!r} under profile cap {maximum!r}"
            )
        if os.environ.get("PICBENCH_ESTIMATE_ONLY", "").strip() == "1":
            raise RuntimeError(f"estimate-only FlexCredit {estimate!r}; cloud start intentionally blocked")
        _require_probe_budget_capacity(estimate)
        _write_paid_receipt(estimate=estimate, start_accepted=False)
        web.start(task_id)
        _write_paid_receipt(estimate=estimate, start_accepted=True)
        web.monitor(task_id)
        destination = RESULT_CACHE_PATH
        destination.unlink(missing_ok=True)
        try:
            simulation_data = web.load(task_id, path=str(destination))
            metric, secondary, spectrum = _extract_metrics(simulation_data, profile)
        finally:
            destination.unlink(missing_ok=True)
        return {"source": "evaluator_owned_tidy3d_fdtd", "metric": metric, "secondary_metrics": secondary, "spectrum": spectrum, "estimated_flexcredit": estimate, "cost_guard_pass": True}
    except BaseException:
        if task_id is not None and str(task_id).strip():
            try:
                web.abort(task_id)
            except Exception:
                pass
        raise


def _interrupt(_signum: int, _frame: Any) -> None:
    raise InterruptedError("FDTD worker interrupted")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fd", type=int, required=True)
    args = parser.parse_args()
    _protect_process()
    signal.signal(signal.SIGTERM, _interrupt)
    signal.signal(signal.SIGINT, _interrupt)
    sock = socket.socket(fileno=args.fd)
    try:
        request = _receive(sock)
        try:
            response = {"ok": True, "result": run_fdtd(request)}
        except Exception as exc:
            response = {"ok": False, "error": _redact_error(exc)}
        _send(sock, response)
    finally:
        sock.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
