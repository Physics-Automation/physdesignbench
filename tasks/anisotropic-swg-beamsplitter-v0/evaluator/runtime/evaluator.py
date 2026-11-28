"""Trusted GDS validation, scoring, and publication for the anisotropic SWG task."""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
import tempfile
from pathlib import Path
from typing import Any

import gdstk

TASK_ID = "AnisotropicSWGBeamsplitterImplementationOptimization"
TASK_SLUG = "anisotropic-swg-beamsplitter-v0"
LOG_ARTIFACTS = Path("/logs/artifacts")
MAX_PROBE_HISTORY_BYTES = 256 * 1024
PROBE_HISTORY_STATE_PATH = Path(
    os.environ.get(
        "PICBENCH_PROBE_HISTORY_PATH",
        "/var/lib/picbench/anisotropic-swg-beamsplitter-v0/state/probe_history.json",
    )
)


class EvaluationError(RuntimeError):
    """A submission or trusted result violated the benchmark contract."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any], *, mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.chmod(mode)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def unavailable_probe_history(reason: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "source": "trusted_probe_sidecar_state",
        "role": "diagnostic_probe_provenance",
        "score_authority": False,
        "available": False,
        "reason": reason,
        "attempts": [],
        "probes": [],
        "totals": {"accepted_probe_count": 0},
    }


def probe_attempt_record_is_valid(record: Any) -> bool:
    expected = {
        "probe_id",
        "submission_sha256",
        "measurement_probes_remaining_after_allocation",
        "allocated_at_utc",
        "evaluator_launch_attempted_at_utc",
        "evaluator_process_started_at_utc",
        "evaluator_finished_at_utc",
        "timeout_observed",
        "process_returncode",
        "response_write_attempted_at_utc",
        "response_write_completed_at_utc",
        "http_status_sent",
    }
    if not isinstance(record, dict) or set(record) != expected:
        return False
    probe_id = record["probe_id"]
    remaining = record["measurement_probes_remaining_after_allocation"]
    digest = record["submission_sha256"]
    if (
        not isinstance(probe_id, int)
        or isinstance(probe_id, bool)
        or probe_id < 1
        or not isinstance(remaining, int)
        or isinstance(remaining, bool)
        or remaining < 0
        or not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
        or not isinstance(record["allocated_at_utc"], str)
        or not record["allocated_at_utc"]
        or len(record["allocated_at_utc"]) > 32
        or not isinstance(record["timeout_observed"], bool)
    ):
        return False
    for field in (
        "evaluator_launch_attempted_at_utc",
        "evaluator_process_started_at_utc",
        "evaluator_finished_at_utc",
        "response_write_attempted_at_utc",
        "response_write_completed_at_utc",
    ):
        value = record[field]
        if value is not None and (
            not isinstance(value, str) or not value or len(value) > 32
        ):
            return False
    returncode = record["process_returncode"]
    if returncode is not None and (
        not isinstance(returncode, int) or isinstance(returncode, bool)
    ):
        return False
    http_status = record["http_status_sent"]
    return http_status is None or (
        isinstance(http_status, int)
        and not isinstance(http_status, bool)
        and 100 <= http_status <= 599
    )


def read_probe_history_state() -> dict[str, Any]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(PROBE_HISTORY_STATE_PATH, flags)
    except FileNotFoundError:
        return unavailable_probe_history("probe_history_not_available")
    except OSError:
        return unavailable_probe_history("probe_history_unreadable")
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_PROBE_HISTORY_BYTES:
            return unavailable_probe_history("probe_history_unsafe")
        raw = os.read(descriptor, MAX_PROBE_HISTORY_BYTES + 1)
    finally:
        os.close(descriptor)
    try:
        history = json.loads(raw.decode("utf-8"))
    except Exception:
        return unavailable_probe_history("probe_history_malformed_json")
    if (
        not isinstance(history, dict)
        or history.get("schema_version") != 1
        or history.get("source") != "trusted_probe_sidecar_state"
        or history.get("role") != "diagnostic_probe_provenance"
        or history.get("score_authority") is not False
        or not isinstance(history.get("available"), bool)
        or not isinstance(history.get("attempts"), list)
        or not all(probe_attempt_record_is_valid(record) for record in history["attempts"])
        or len({record["probe_id"] for record in history["attempts"]}) != len(history["attempts"])
        or not isinstance(history.get("probes"), list)
        or not isinstance(history.get("totals"), dict)
    ):
        return unavailable_probe_history("probe_history_schema_invalid")
    return history


def stage_probe_history_artifact() -> None:
    atomic_json(LOG_ARTIFACTS / "probe_history.json", read_probe_history_state())


def load_profile(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("task_id") != TASK_ID:
        raise EvaluationError("trusted task profile has the wrong task_id")
    if payload.get("task_mode") != "implementation_optimization":
        raise EvaluationError("trusted task profile has the wrong task_mode")
    return payload


def _layer_key(layer: int, datatype: int) -> str:
    return f"{layer}/{datatype}"


def _vertical_intervals(points: list[list[float]], x_probe: float) -> list[tuple[float, float]]:
    intersections: list[float] = []
    for first, second in zip(points, points[1:] + points[:1]):
        x0, y0 = first
        x1, y1 = second
        if (x0 <= x_probe < x1) or (x1 <= x_probe < x0):
            intersections.append(y0 + (x_probe - x0) * (y1 - y0) / (x1 - x0))
    intersections.sort()
    if len(intersections) % 2:
        raise EvaluationError("silicon polygon has an ambiguous port-plane intersection")
    return [
        (intersections[index], intersections[index + 1])
        for index in range(0, len(intersections), 2)
        if intersections[index + 1] > intersections[index]
    ]


def _merged_intervals(polygons: list[dict[str, Any]], x_probe: float, tolerance: float) -> list[tuple[float, float]]:
    intervals: list[tuple[float, float]] = []
    for polygon in polygons:
        if polygon["x_bounds_um"][0] - tolerance <= x_probe <= polygon["x_bounds_um"][1] + tolerance:
            intervals.extend(_vertical_intervals(polygon["points_um"], x_probe))
    merged: list[list[float]] = []
    for start, stop in sorted(intervals):
        if not merged or start > merged[-1][1] + tolerance:
            merged.append([start, stop])
        else:
            merged[-1][1] = max(merged[-1][1], stop)
    return [(start, stop) for start, stop in merged]


def _validate_port_landings(
    polygons: list[dict[str, Any]],
    device_length: float,
    contract: dict[str, Any],
) -> list[dict[str, Any]]:
    landing = contract["port_landings"]
    if landing.get("axis") != "x":
        raise EvaluationError("trusted landing contract must use x-normal ports")
    expected_centers = [float(value) for value in landing["centers_um"]]
    expected_width = float(landing["width_um"])
    width_tolerance = float(landing["width_tolerance_um"])
    center_tolerance = float(landing["center_tolerance_um"])
    coordinate_tolerance = float(contract["coordinate_tolerance_um"])
    landing_length = float(landing["minimum_landing_um"])
    offsets = (0.1, landing_length / 2.0, landing_length - 0.1)

    def check_plane(sections: list[dict[str, Any]], x_probe: float, side: str) -> list[dict[str, float]]:
        # Do not merge a physical slit using the coordinate-tolerance allowance.
        intervals = _merged_intervals(sections, x_probe, 0.0)
        if len(intervals) != len(expected_centers):
            raise EvaluationError(f"{side} landing has the wrong number of silicon ports")
        measured = []
        for (start, stop), expected_center in zip(intervals, expected_centers):
            width = stop - start
            center = (start + stop) / 2.0
            if abs(width - expected_width) > width_tolerance:
                raise EvaluationError(f"{side} landing has an invalid port width")
            if abs(center - expected_center) > center_tolerance:
                raise EvaluationError(f"{side} landing has an invalid port center")
            measured.append({"center_y_um": center, "width_um": width})
        return measured

    checks: list[dict[str, Any]] = []
    for side, samples in (
        ("input", offsets),
        ("output", tuple(device_length - value for value in offsets)),
    ):
        x0, x1 = (0.0, landing_length) if side == "input" else (device_length - landing_length, device_length)
        # Match Boolean precision so a few floating-point ULPs at a clipped
        # boundary do not become a fictitious uncovered interval.
        x0, x1 = round(x0, 6), round(x1, 6)
        bounds = contract["bounds_um"]
        window = gdstk.rectangle(
            (x0, float(bounds["y_min"]) - coordinate_tolerance),
            (x1, float(bounds["y_max"]) + coordinate_tolerance),
        )
        try:
            clipped = gdstk.boolean(
                [gdstk.Polygon(record["points_um"]) for record in polygons],
                window, "and", precision=1e-6,
            )
        except Exception as exc:
            raise EvaluationError(f"{side} landing geometry could not be resolved") from exc
        sections = []
        breaks = {x0, x1}
        for polygon in clipped:
            points = [[round(float(x), 6), round(float(y), 6)] for x, y in polygon.points]
            xs = [point[0] for point in points]
            sections.append({"points_um": points, "x_bounds_um": [min(xs), max(xs)]})
            breaks.update(xs)
        ordered = sorted(breaks)
        for left, right in zip(ordered, ordered[1:]):
            # The canonical polygon sections are affine between consecutive
            # vertices. Check both one-sided limits, plus interior topology;
            # this covers the entire landing rather than a fixed sample grid.
            for x_probe in (math.nextafter(left, right), (left + right) / 2.0, math.nextafter(right, left)):
                check_plane(sections, x_probe, side)
        for x_probe in samples:
            measured = check_plane(sections, x_probe, side)
            checks.append({"side": side, "x_um": x_probe, "ports": measured})
    return checks


def _validate_y_mirror_symmetry(
    by_layer: dict[str, list[dict[str, Any]]],
    contract: dict[str, Any],
) -> dict[str, Any]:
    """Require each submitted mask union to equal its reflection about y=axis."""
    symmetry = contract.get("y_mirror_symmetry")
    if not isinstance(symmetry, dict) or symmetry.get("required") is not True:
        raise EvaluationError("trusted y-mirror symmetry contract is invalid")
    try:
        axis_y = float(symmetry["axis_y_um"])
        precision = float(symmetry["boolean_precision_um"])
        maximum_area = float(symmetry["maximum_symmetric_difference_area_um2"])
        layer_keys = [_layer_key(*map(int, pair)) for pair in symmetry["applies_to_layers"]]
    except (KeyError, TypeError, ValueError) as exc:
        raise EvaluationError("trusted y-mirror symmetry contract is invalid") from exc
    if (
        not math.isfinite(axis_y)
        or not math.isfinite(precision)
        or precision <= 0.0
        or not math.isfinite(maximum_area)
        or maximum_area < 0.0
        or set(layer_keys) != set(by_layer)
        or len(layer_keys) != len(set(layer_keys))
    ):
        raise EvaluationError("trusted y-mirror symmetry contract is invalid")

    difference_by_layer: dict[str, float] = {}
    for key in layer_keys:
        try:
            canonical = gdstk.boolean(
                [gdstk.Polygon(record["points_um"]) for record in by_layer[key]],
                [],
                "or",
                precision=precision,
            )
            reflected = [
                gdstk.Polygon(
                    [(float(x), 2.0 * axis_y - float(y)) for x, y in polygon.points]
                )
                for polygon in canonical
            ]
            difference = gdstk.boolean(
                canonical,
                reflected,
                "xor",
                precision=precision,
            )
        except Exception as exc:
            raise EvaluationError(f"layer {key} y-mirror symmetry could not be resolved") from exc
        difference_area = sum(abs(float(polygon.area())) for polygon in difference)
        if not math.isfinite(difference_area) or difference_area > maximum_area:
            raise EvaluationError(
                f"layer {key} violates the required strict y-mirror symmetry"
            )
        difference_by_layer[key] = difference_area
    return {
        "required": True,
        "axis_y_um": axis_y,
        "boolean_precision_um": precision,
        "maximum_symmetric_difference_area_um2": maximum_area,
        "symmetric_difference_area_um2_by_layer": difference_by_layer,
        "passed": True,
    }


def validate_gds(gds_path: Path, profile: dict[str, Any]) -> dict[str, Any]:
    """Validate canonical silicon/void masks and return bounded numeric geometry."""
    if not gds_path.is_file() or gds_path.is_symlink():
        raise EvaluationError("restricted builder did not produce a regular final_pic.gds")
    if not 0 < gds_path.stat().st_size <= 8 * 1024 * 1024:
        raise EvaluationError("final_pic.gds has an invalid size")
    try:
        library = gdstk.read_gds(str(gds_path))
    except Exception as exc:
        raise EvaluationError(f"could not parse final_pic.gds: {exc}") from exc
    tops = library.top_level()
    if len(tops) != 1:
        raise EvaluationError(f"GDS must have exactly one top cell; found {len(tops)}")
    if tops[0].get_labels(apply_repetitions=True):
        raise EvaluationError("GDS labels are forbidden")

    contract = profile["layout_contract"]
    allowed = {_layer_key(*map(int, pair)) for pair in contract["allowed_layers"]}
    required = {_layer_key(*map(int, pair)) for pair in contract["required_layers"]}
    layer_rules = contract["layers"]
    bounds = contract["bounds_um"]
    tolerance = float(contract["coordinate_tolerance_um"])
    maximum_vertices = int(contract["maximum_vertex_count"])
    by_layer: dict[str, list[dict[str, Any]]] = {key: [] for key in allowed}
    areas = {key: 0.0 for key in allowed}
    vertex_count = 0
    global_x_min = math.inf
    global_x_max = -math.inf
    global_y_min = math.inf
    global_y_max = -math.inf

    raw_polygons = tops[0].get_polygons(apply_repetitions=True, include_paths=True)
    for index, polygon in enumerate(raw_polygons):
        key = _layer_key(int(polygon.layer), int(polygon.datatype))
        if key not in allowed:
            raise EvaluationError(f"polygon {index} uses forbidden layer/datatype {key}")
        points = [[float(x), float(y)] for x, y in polygon.points]
        if len(points) < 4 or not all(math.isfinite(value) for point in points for value in point):
            raise EvaluationError(f"polygon {index} has invalid vertices")
        vertex_count += len(points)
        if vertex_count > maximum_vertices:
            raise EvaluationError("GDS exceeds maximum_vertex_count")
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
        if (
            x0 < float(bounds["x_min"]) - tolerance
            or x1 > float(bounds["x_max"]) + tolerance
            or y0 < float(bounds["y_min"]) - tolerance
            or y1 > float(bounds["y_max"]) + tolerance
        ):
            raise EvaluationError(f"polygon {index} lies outside allowed bounds")
        rules = layer_rules[key]
        if min(x1 - x0, y1 - y0) < float(rules["minimum_bbox_feature_um"]) - tolerance:
            raise EvaluationError(f"polygon {index} has a sub-minimum bounding-box feature")
        area = abs(float(polygon.area()))
        if not math.isfinite(area) or area < float(rules["minimum_polygon_area_um2"]):
            raise EvaluationError(f"polygon {index} has invalid or sub-minimum area")
        record = {
            "layer": [int(polygon.layer), int(polygon.datatype)],
            "points_um": points,
            "x_bounds_um": [x0, x1],
            "y_bounds_um": [y0, y1],
            "area_um2": area,
        }
        by_layer[key].append(record)
        areas[key] += area
        global_x_min = min(global_x_min, x0)
        global_x_max = max(global_x_max, x1)
        global_y_min = min(global_y_min, y0)
        global_y_max = max(global_y_max, y1)

    if not raw_polygons or any(not by_layer[key] for key in required):
        raise EvaluationError("GDS is empty or omits a required mask layer")
    for key, rules in layer_rules.items():
        count = len(by_layer[key])
        if not int(rules["minimum_polygon_count"]) <= count <= int(rules["maximum_polygon_count"]):
            raise EvaluationError(f"layer {key} has an invalid polygon count")
        if not float(rules["minimum_total_area_um2"]) <= areas[key] <= float(rules["maximum_total_area_um2"]):
            raise EvaluationError(f"layer {key} has an invalid total area")
    if abs(global_x_min - float(bounds["x_min"])) > tolerance:
        raise EvaluationError("submitted device must start at x=0")
    device_length = global_x_max
    if not float(contract["device_length_min_um"]) - tolerance <= device_length <= float(contract["device_length_max_um"]) + tolerance:
        raise EvaluationError("submitted device length lies outside the allowed range")
    silicon_key = _layer_key(*map(int, contract["port_landings"]["layer"]))
    port_checks = _validate_port_landings(by_layer[silicon_key], device_length, contract)
    y_mirror_symmetry = _validate_y_mirror_symmetry(by_layer, contract)
    return {
        "schema_version": 1,
        "gds_sha256": sha256_file(gds_path),
        "top_cell": tops[0].name,
        "mask_semantics": contract["mask_semantics"],
        "polygon_count": len(raw_polygons),
        "polygon_count_by_layer": {key: len(value) for key, value in by_layer.items()},
        "area_um2_by_layer": areas,
        "vertex_count": vertex_count,
        "device_length_um": device_length,
        "x_extent_um": [global_x_min, global_x_max],
        "y_extent_um": [global_y_min, global_y_max],
        "port_checks": port_checks,
        "y_mirror_symmetry": y_mirror_symmetry,
        "polygons": [record for key in sorted(by_layer) for record in by_layer[key]],
    }


def validate_fdtd_result(result: dict[str, Any], profile: dict[str, Any]) -> dict[str, Any]:
    if result.get("source") != "evaluator_owned_tidy3d_fdtd" or result.get("cost_guard_pass") is not True:
        raise EvaluationError("FDTD result has an invalid authority or cost marker")
    estimate = float(result["estimated_flexcredit"])
    maximum = float(profile["evaluation"]["max_flexcredit_per_run"])
    if not math.isfinite(estimate) or not 0.0 <= estimate <= maximum:
        raise EvaluationError("trusted FDTD estimate did not pass the cost guard")
    reported_secondary = result.get("secondary_metrics")
    raw_spectrum = result.get("spectrum")
    if not isinstance(reported_secondary, dict) or not isinstance(raw_spectrum, list):
        raise EvaluationError("trusted FDTD result omits broadband metrics")
    objective = profile["objective"]
    start = float(objective["wavelength_start_um"])
    stop = float(objective["wavelength_stop_um"])
    step = float(objective["wavelength_step_um"])
    expected_wavelengths = [1000.0 * (start + index * step) for index in range(int(round((stop - start) / step)) + 1)]
    if len(raw_spectrum) != len(expected_wavelengths):
        raise EvaluationError("trusted FDTD spectrum has the wrong wavelength count")
    envelope = profile["baseline"]["reference_operating_envelope"]
    el_limit = float(envelope["excess_loss_db_max"])
    imbalance_limit = float(envelope["imbalance_db_max"])
    phase_limit = float(envelope["phase_error_deg_max"])
    tolerance = float(profile["evaluation"]["physical_bounds"]["numerical_power_tolerance"])
    spectrum: list[dict[str, float]] = []
    try:
        for raw, expected_wavelength in zip(raw_spectrum, expected_wavelengths, strict=True):
            point = {name: float(raw[name]) for name in ("wavelength_nm", "T3", "T4", "excess_loss_db", "imbalance_db", "phase_error_deg", "reflection_db", "normalized_penalty")}
            if not all(math.isfinite(value) for value in point.values()) or abs(point["wavelength_nm"] - expected_wavelength) > 1e-6:
                raise EvaluationError("trusted FDTD spectrum contains an invalid wavelength or value")
            total = point["T3"] + point["T4"]
            if point["T3"] <= 0.0 or point["T4"] <= 0.0 or total > 1.0 + tolerance or not 0.0 <= point["phase_error_deg"] <= 90.0:
                raise EvaluationError("trusted broadband output metrics are non-physical")
            expected_el = -10.0 * math.log10(min(total, 1.0))
            expected_imbalance = abs(10.0 * math.log10(point["T3"] / point["T4"]))
            expected_penalty = max(max(expected_el, 0.0) / el_limit, expected_imbalance / imbalance_limit, point["phase_error_deg"] / phase_limit)
            if abs(point["excess_loss_db"] - expected_el) > 1e-8 or abs(point["imbalance_db"] - expected_imbalance) > 1e-8 or abs(point["normalized_penalty"] - expected_penalty) > 1e-8:
                raise EvaluationError("trusted broadband splitter metric is inconsistent")
            spectrum.append(point)
        score = float(result["metric"])
    except (KeyError, TypeError, ValueError) as exc:
        raise EvaluationError("trusted FDTD result has an invalid broadband schema") from exc
    expected_score = max(point["normalized_penalty"] for point in spectrum)
    if not math.isfinite(score) or abs(score - expected_score) > 1e-8:
        raise EvaluationError("trusted worst-band splitter score is inconsistent")
    worst = max(spectrum, key=lambda point: point["normalized_penalty"])
    center = min(spectrum, key=lambda point: abs(point["wavelength_nm"] - 1550.0))
    values = {
        "T3_1550": center["T3"], "T4_1550": center["T4"],
        "excess_loss_db_1550": center["excess_loss_db"], "imbalance_db_1550": center["imbalance_db"],
        "phase_error_deg_1550": center["phase_error_deg"], "reflection_db_1550": center["reflection_db"],
        "worst_excess_loss_db": max(point["excess_loss_db"] for point in spectrum),
        "worst_imbalance_db": max(point["imbalance_db"] for point in spectrum),
        "worst_phase_error_deg": max(point["phase_error_deg"] for point in spectrum),
        "worst_wavelength_nm": worst["wavelength_nm"],
    }
    try:
        if any(abs(float(reported_secondary[name]) - value) > 1e-8 for name, value in values.items()):
            raise EvaluationError("trusted broadband secondary metrics are inconsistent")
    except (KeyError, TypeError, ValueError) as exc:
        raise EvaluationError("trusted FDTD result has an invalid secondary metric schema") from exc
    return {"metric": score, "secondary_metrics": values, "spectrum": spectrum, "estimated_flexcredit": estimate}


def _render_geometry(manifest: dict[str, Any], destination: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon

    colors = {"1/0": "#d34a43", "2/0": "#48a9a6"}
    figure, axis = plt.subplots(figsize=(12.0, 3.4))
    axis.set_facecolor("#e8ecef")
    for polygon in manifest["polygons"]:
        key = _layer_key(*polygon["layer"])
        axis.add_patch(Polygon(polygon["points_um"], closed=True, facecolor=colors[key], edgecolor="none", alpha=0.86))
    axis.set_xlim(-0.4, float(manifest["device_length_um"]) + 0.4)
    axis.set_ylim(-1.9, 1.9)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("x / propagation (um)")
    axis.set_ylabel("y / lateral (um)")
    axis.set_title("Evaluator-imported silicon (red) and residual-void (cyan) masks")
    figure.tight_layout()
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=180)
    plt.close(figure)


def _render_response(measurement: dict[str, Any], destination: Path) -> None:
    import matplotlib.pyplot as plt

    spectrum = measurement["spectrum"]
    wavelengths = [point["wavelength_nm"] for point in spectrum]
    figure, axis = plt.subplots(figsize=(8.2, 4.4))
    axis.plot(wavelengths, [max(point["excess_loss_db"], 0.0) for point in spectrum], marker="o", markersize=3, label="Excess loss / 1 dB")
    axis.plot(wavelengths, [point["imbalance_db"] for point in spectrum], marker="o", markersize=3, label="Imbalance / 1 dB")
    axis.plot(wavelengths, [point["phase_error_deg"] / 5.0 for point in spectrum], marker="o", markersize=3, label="Phase error / 5 deg")
    axis.axhline(1.0, color="#c9423a", linestyle="--", linewidth=1.2, label="Published envelope")
    axis.set_xlabel("Wavelength (nm)")
    axis.set_ylabel("Normalized envelope ratio")
    axis.set_title(f"Worst-band splitter penalty: {measurement['metric']:.4f}")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8, ncol=2)
    figure.tight_layout()
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=180)
    plt.close(figure)


def _artifact_record(path: Path) -> dict[str, Any]:
    return {"size": path.stat().st_size, "sha256": sha256_file(path)}


def _validate_mode(mode: str, probe_id: int | None) -> None:
    if mode not in {"final", "probe"}:
        raise EvaluationError("evaluation mode must be final or probe")
    if mode == "final" and probe_id is not None:
        raise EvaluationError("final evaluation cannot carry a probe ID")
    if mode == "probe" and (not isinstance(probe_id, int) or isinstance(probe_id, bool) or probe_id < 1):
        raise EvaluationError("probe evaluation requires a positive probe ID")


def _identity(mode: str, probe_id: int | None, submission_sha256: str, returncode: int) -> dict[str, Any]:
    label = "final" if mode == "final" else f"probe-{probe_id:06d}"
    return {"schema_version": 1, "mode": mode, "evaluation_id": f"anisotropic-swg-{label}-{submission_sha256[:16]}", "probe_id": probe_id, "evaluator_returncode": returncode}


def publish_success(
    *, output_dir: Path, artifact_dir: Path, verifier_dir: Path | None = None,
    submission_sha256: str, submission_size: int = 0, profile_path: Path,
    geometry_manifest: dict[str, Any], fdtd_result: dict[str, Any],
    evaluation_mode: str = "final", probe_id: int | None = None,
) -> dict[str, Any]:
    _validate_mode(evaluation_mode, probe_id)
    profile = load_profile(profile_path)
    final_gds = output_dir / "final_pic.gds"
    if not final_gds.is_file() or final_gds.is_symlink() or sha256_file(final_gds) != geometry_manifest.get("gds_sha256"):
        raise EvaluationError("successful publication requires the validated canonical final_pic.gds")
    measurement = validate_fdtd_result(fdtd_result, profile)
    score = measurement["metric"]
    baseline = float(profile["baseline"]["reference_value"])
    paper_margin = baseline - score
    framework_reward = 1.0 / (1.0 + score)
    identity = _identity(evaluation_mode, probe_id, submission_sha256, 0)
    identity.update(submission_sha256=submission_sha256, submission_size=int(submission_size))
    secondary = measurement["secondary_metrics"]
    payload: dict[str, Any] = {
        "result_schema_version": 1,
        "task_id": profile["task_id"], "task_version": profile["task_version"], "task_mode": profile["task_mode"],
        "evaluation_type": profile["evaluation"]["evaluation_type"], "status": "score_valid",
        "learner_reported_metrics_trusted": False, "learner_code_executed_by_root": False,
        "learner_code_execution": "restricted_non_root_builder", "valid": True, "design_valid": True,
        "simulation_valid": True, "evidence_valid": True, "gate_pass": True, "score_eligible": True,
        "candidate_runs": 1, "candidate_subruns": 1,
        "physical_input_count": 2, "simulated_input_count": 1,
        "derived_input_count": 1, "symmetry_gate_pass": True,
        "input_coverage_pass": True,
        "baseline_rerun": False, "objective_name": profile["objective"]["name"], "task_metric": score,
        "task_metric_source": "private_evaluator_simulation_data", "trusted_metric_source": "evaluator_owned_tidy3d_fdtd",
        "official_score": score, "official_metric": score, "objective_value": score,
        "reference_baseline": baseline, "paper_margin": paper_margin,
        "reward": framework_reward, **secondary,
        "spectrum": measurement["spectrum"],
        "estimated_flexcredit": measurement["estimated_flexcredit"], "cost_guard_pass": True,
        "trusted_estimated_flexcredit_values": [measurement["estimated_flexcredit"]],
        "max_estimated_flexcredit": measurement["estimated_flexcredit"],
        "max_flexcredit_per_run": float(profile["evaluation"]["max_flexcredit_per_run"]),
        "submission_sha256": submission_sha256, "gds_sha256": geometry_manifest["gds_sha256"],
        "profile_sha256": sha256_file(profile_path), "evaluator_image": os.environ.get("PICBENCH_EVALUATOR_IMAGE", ""),
        "evaluator_image_digest": os.environ.get("PICBENCH_EVALUATOR_IMAGE_DIGEST", ""),
        "geometry": {key: value for key, value in geometry_manifest.items() if key != "polygons"},
        "generated_plot_evidence_valid": True,
        "plots": {"generated_geometry": "plots/generated_geometry.png", "response": "plots/response.png"},
        "published_plots": {"generated_geometry": "plots/generated_geometry.png", "response": "plots/response.png"},
        "checks": {"schema_valid": True, "design_valid": True, "simulation_valid": True, "evidence_valid": True, "cost_guard_pass": True, "symmetry_gate_pass": True, "input_coverage_pass": True, "generated_plot_evidence_valid": True},
        "metrics": [
            {"key": "official_score", "label": "Official worst-band normalized splitter penalty J", "role": "primary", "type": "number", "value": score, "unit": "normalized_penalty", "direction": "min", "components": {"source": "private_evaluator_simulation_data", "physical_inputs": 2, "simulated_inputs": 1, "derived_input_relation": "strict_y_mirror_geometry_with_output_port_exchange"}},
            {"key": "reference_baseline", "label": "Published operating envelope", "role": "diagnostic", "type": "number", "value": baseline, "unit": "normalized_penalty", "direction": "min", "components": {"rerun_by_evaluator": False}},
            {"key": "paper_margin", "label": "Paper-envelope margin (positive is better)", "role": "diagnostic", "type": "number", "value": paper_margin, "unit": "normalized_penalty", "direction": "max", "components": {"candidate_runs": 1, "candidate_subruns": 1}},
            *[{"key": name, "label": name, "role": "diagnostic", "type": "number", "value": value, "unit": "dB" if "_db" in name else ("deg" if "_deg" in name else ("nm" if name.endswith("_nm") else "fraction")), "direction": "none", "components": {}} for name, value in secondary.items()],
        ],
        "evaluation_transaction": identity,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    score_dir = output_dir if verifier_dir is None else verifier_dir
    score_dir.mkdir(parents=True, exist_ok=True)
    _render_geometry(geometry_manifest, artifact_dir / "generated_geometry.png")
    _render_response(measurement, artifact_dir / "response.png")
    atomic_json(output_dir / "geometry_manifest.json", geometry_manifest)
    atomic_json(output_dir / "eval.json", payload)
    atomic_json(output_dir / "traces.json", payload)
    atomic_json(score_dir / "reward.json", {"reward": framework_reward, "objective_value": score, "official_score": score, "official_score_present": 1.0, "valid": 1.0, "evidence_valid": 1.0, "design_valid": 1.0, "gate_pass": 1.0})
    (score_dir / "reward.txt").write_text(f"{framework_reward}\n", encoding="utf-8")
    manifest = {
        "schema_version": 1, "complete": True, "mode": evaluation_mode, "task_slug": TASK_SLUG,
        "task_id": profile["task_id"], "evaluation_id": identity["evaluation_id"], "probe_id": probe_id,
        "evaluator_returncode": 0, "submission": {"size": int(submission_size), "sha256": submission_sha256},
        "submission_sha256": submission_sha256, "profile_sha256": payload["profile_sha256"],
        "gds_sha256": geometry_manifest["gds_sha256"], "evaluator_image": payload["evaluator_image"],
        "evaluator_image_digest": payload["evaluator_image_digest"],
        "artifacts": {name: _artifact_record(path) for name, path in {
            "eval.json": output_dir / "eval.json", "traces.json": output_dir / "traces.json",
            "geometry_manifest.json": output_dir / "geometry_manifest.json", "final_pic.gds": final_gds,
            "plots/generated_geometry.png": artifact_dir / "generated_geometry.png", "plots/response.png": artifact_dir / "response.png",
        }.items()},
    }
    atomic_json(output_dir / "evaluation_manifest.json", manifest)
    stage_probe_history_artifact()
    return payload


def publish_failure(
    *, output_dir: Path, artifact_dir: Path, verifier_dir: Path, error: str,
    submission_sha256: str, submission_size: int, profile_path: Path,
    design_valid: bool, evaluation_mode: str, probe_id: int | None,
) -> dict[str, Any]:
    _validate_mode(evaluation_mode, probe_id)
    profile = load_profile(profile_path)
    identity = _identity(evaluation_mode, probe_id, submission_sha256, 1)
    identity.update(submission_sha256=submission_sha256, submission_size=int(submission_size))
    baseline = float(profile["baseline"]["reference_value"])
    payload = {
        "result_schema_version": 1,
        "task_id": profile["task_id"], "task_version": profile["task_version"], "task_mode": profile["task_mode"],
        "evaluation_type": profile["evaluation"]["evaluation_type"], "status": "evaluation_error",
        "learner_reported_metrics_trusted": False, "learner_code_executed_by_root": False,
        "valid": False, "design_valid": bool(design_valid), "simulation_valid": False, "evidence_valid": False,
        "gate_pass": False, "score_eligible": False, "official_score": None, "official_metric": None, "objective_value": None,
        "task_metric": None, "task_metric_source": None, "reference_baseline": baseline, "reward": 0.0,
        "generated_plot_evidence_valid": False, "plots": {}, "published_plots": {},
        "metrics": [{"key": "reference_baseline", "label": "Published operating envelope", "role": "diagnostic", "type": "number", "value": baseline, "unit": "normalized_penalty", "direction": "min", "components": {"rerun_by_evaluator": False}}],
        "submission_sha256": submission_sha256, "errors": [error[:1200]], "evaluation_transaction": identity,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    verifier_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(output_dir / "eval.json", payload)
    atomic_json(output_dir / "traces.json", payload)
    atomic_json(verifier_dir / "reward.json", {"reward": 0.0, "official_score_present": 0.0, "valid": 0.0, "evidence_valid": 0.0, "design_valid": 1.0 if design_valid else 0.0, "gate_pass": 0.0})
    (verifier_dir / "reward.txt").write_text("0\n", encoding="utf-8")
    manifest = {"schema_version": 1, "complete": True, "mode": evaluation_mode, "task_slug": TASK_SLUG, "task_id": profile["task_id"], "evaluation_id": identity["evaluation_id"], "probe_id": probe_id, "evaluator_returncode": 1, "submission": {"size": int(submission_size), "sha256": submission_sha256}, "profile_sha256": sha256_file(profile_path), "evaluator_image": os.environ.get("PICBENCH_EVALUATOR_IMAGE", ""), "evaluator_image_digest": os.environ.get("PICBENCH_EVALUATOR_IMAGE_DIGEST", ""), "artifacts": {"eval.json": _artifact_record(output_dir / "eval.json"), "traces.json": _artifact_record(output_dir / "traces.json")}}
    atomic_json(output_dir / "evaluation_manifest.json", manifest)
    stage_probe_history_artifact()
    return payload
