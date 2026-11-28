from __future__ import annotations

import importlib.util
import io
import json
import math
import shutil
from pathlib import Path

import gdstk
import pytest

REPO = Path(__file__).resolve().parents[3]
TASK = REPO / "tasks/anisotropic-swg-beamsplitter-v0"
PROFILE_PATH = TASK / "evaluator/runtime/task_profile.json"
PROBE_SERVICE_PATH = TASK / "evaluator/src/probe_service.py"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _profile() -> dict:
    return json.loads(PROFILE_PATH.read_text(encoding="utf-8"))


def _fixture_gds(tmp_path: Path) -> Path:
    pytest.importorskip("photonforge")
    fixture = _load(
        "anisotropic_fixture",
        REPO / "tests/tasks/anisotropic-swg-beamsplitter-v0/fixtures/paper_candidate_design.py",
    )
    path = tmp_path / "fixture.gds"
    fixture.build_submission().write_gds(str(path))
    return path


@pytest.fixture
def budget_service(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "anisotropic_budget_service", TASK / "evaluator/src/probe_service.py"
    )
    assert spec and spec.loader
    service = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(service)
    for name, filename in (
        ("STATE_PATH", "probe_state.json"), ("STATE_LOCK", "probe.lock"),
        ("PAID_RECEIPT_PATH", "paid_usage_receipt.json"),
    ):
        monkeypatch.setattr(service, name, tmp_path / filename)
    monkeypatch.setattr(service, "STATE_DIR", tmp_path)
    for name in (
        "PICBENCH_PROBE_BUDGET", "PICBENCH_MAX_PAID_LEARNER_PROBES",
        "PICBENCH_CUMULATIVE_THERMAL_LIMIT_FLEXCREDIT",
    ):
        monkeypatch.delenv(name, raising=False)
    return service


def _paid_probe(service, *, accepted=True, estimate=4.0):
    probe_id, remaining = service._allocate_probe()
    service._atomic_json(service.PAID_RECEIPT_PATH, {
        "schema_version": 1, "probe_id": probe_id, "submission_sha256": "a" * 64,
        "estimated_flexcredit": estimate, "start_attempted": True,
        "start_accepted": accepted,
    })
    service._record_paid_receipt(probe_id, "a" * 64)
    return probe_id, remaining


def test_trial_budget_protects_final_after_24_full_price_probes(budget_service, monkeypatch):
    service = budget_service
    service._initialize_probe_state()
    assert service._read_probe_state(30)["cost_budget"] == {
        "limit_flexcredit": 100.0, "final_reserve_flexcredit": 4.0,
        "used_flexcredit": 0.0, "learner_remaining_flexcredit": 96.0,
    }
    for _ in range(24):
        probe_id, remaining = _paid_probe(service)
    envelope = service._probe_budget_envelope(probe_id, remaining)
    assert envelope["measurement_probes_remaining"] == 6
    assert envelope["estimated_flexcredit_used"] == 96.0
    assert envelope["estimated_flexcredit_remaining"] == 0.0
    assert envelope["final_reserve"] == 4.0
    assert envelope["estimated_flexcredit_used"] + envelope["final_reserve"] == 100.0
    worker = _load("anisotropic_budget_worker", TASK / "evaluator/runtime/fdtd_worker.py")
    monkeypatch.setenv("PICBENCH_EVALUATION_MODE", "probe")
    for name, value in service._probe_accounting_environment().items():
        monkeypatch.setenv(name, value)
    with pytest.raises(RuntimeError, match="FlexCredit budget exhausted"):
        worker._require_probe_budget_capacity(0.01)
    # The independent final worker still applies the public/private 4-FC cap;
    # it neither reads nor resets the exhausted probe state.
    before = service.STATE_PATH.read_bytes()
    monkeypatch.setenv("PICBENCH_EVALUATION_MODE", "final")
    worker._require_probe_budget_capacity(4.0)
    assert service.STATE_PATH.read_bytes() == before


def test_uncertain_start_is_charged_once_even_after_restart(budget_service):
    service = budget_service
    service._initialize_probe_state()
    _paid_probe(service, accepted=False)
    service._initialize_probe_state()
    service._reconcile_pending_paid_receipt()
    state = service._read_probe_state(30)
    assert state["cost_budget"]["learner_remaining_flexcredit"] == 92.0
    assert state["task_accounting"]["uncertain_paid_start_attempts"] == 1
    assert state["paid_budget"]["used"] == 1


def test_thirty_lower_cost_probes_still_fit(budget_service):
    service = budget_service
    service._initialize_probe_state()
    for _ in range(30):
        _paid_probe(service, estimate=3.0)
    assert service._read_probe_state(30)["cost_budget"]["learner_remaining_flexcredit"] == 6.0
    with pytest.raises(RuntimeError, match="probe budget exhausted"):
        service._allocate_probe()


@pytest.mark.parametrize("name,value", [
    ("PICBENCH_PROBE_BUDGET", "31"),
    ("PICBENCH_MAX_PAID_LEARNER_PROBES", "31"),
    ("PICBENCH_CUMULATIVE_THERMAL_LIMIT_FLEXCREDIT", "101"),
    ("PICBENCH_CUMULATIVE_THERMAL_LIMIT_FLEXCREDIT", "3.99"),
    ("PICBENCH_CUMULATIVE_THERMAL_LIMIT_FLEXCREDIT", "nan"),
])
def test_invalid_trial_caps_fail_closed(budget_service, monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    with pytest.raises(RuntimeError):
        budget_service._initialize_probe_state()


def test_old_zero_reserve_state_is_rejected(budget_service):
    service = budget_service
    service._initialize_probe_state()
    state = service._read_probe_state(30)
    state["cost_budget"]["final_reserve_flexcredit"] = 0.0
    state["cost_budget"]["learner_remaining_flexcredit"] = 100.0
    service._atomic_json(service.STATE_PATH, state)
    with pytest.raises(RuntimeError, match="probe state is invalid"):
        service._initialize_probe_state()


def test_receipt_cannot_consume_final_reserve(budget_service):
    service = budget_service
    service._initialize_probe_state()
    for _ in range(24):
        _paid_probe(service)
    before = service._read_probe_state(30)["cost_budget"]
    with pytest.raises(RuntimeError, match="exceeded its trusted limit"):
        _paid_probe(service, estimate=0.01)
    assert service._read_probe_state(30)["cost_budget"] == before
    assert service.PAID_RECEIPT_PATH.exists()  # Preserve failed accounting evidence.


def test_probe_budget_uses_durable_measurement_and_paid_receipt_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    service = _load("anisotropic_probe_budget", PROBE_SERVICE_PATH)
    state_dir = tmp_path / "state"
    monkeypatch.setattr(service, "STATE_DIR", state_dir)
    monkeypatch.setattr(service, "STATE_PATH", state_dir / "probe_state.json")
    monkeypatch.setattr(service, "STATE_LOCK", state_dir / "probe.lock")
    monkeypatch.setattr(
        service,
        "PAID_RECEIPT_PATH",
        state_dir / "paid_usage_receipt.json",
    )
    monkeypatch.setenv("PICBENCH_PROBE_BUDGET", "3")
    monkeypatch.setenv("PICBENCH_MAX_PAID_LEARNER_PROBES", "2")
    monkeypatch.setenv("PICBENCH_CUMULATIVE_THERMAL_LIMIT_FLEXCREDIT", "5")
    service._initialize_probe_state()

    probe_id, remaining = service._allocate_probe()
    free_budget = service._probe_budget_envelope(probe_id, remaining)
    assert free_budget == {
        "measurement_probes_used": 1,
        "measurement_probes_remaining": 2,
        "paid_probes_used": 0,
        "paid_probes_remaining": 2,
        "estimated_flexcredit_used": 0.0,
        "estimated_flexcredit_remaining": 1.0,
        "billed_flexcredit_used": None,
        "per_run_cap": 4.0,
        "cumulative_cap": 5.0,
        "final_reserve": 4.0,
        "budget_authority": "private_evaluator_sidecar",
    }

    service._atomic_json(
        service.PAID_RECEIPT_PATH,
        {
            "schema_version": 1,
            "probe_id": probe_id,
            "submission_sha256": "a" * 64,
            "estimated_flexcredit": 0.75,
            "start_attempted": True,
            "start_accepted": True,
        },
    )
    service._record_paid_receipt(probe_id, "a" * 64)
    assert not service.PAID_RECEIPT_PATH.exists()
    paid_budget = service._probe_budget_envelope(probe_id, remaining)
    assert paid_budget["paid_probes_used"] == 1
    assert paid_budget["paid_probes_remaining"] == 1
    assert paid_budget["estimated_flexcredit_used"] == pytest.approx(0.75)
    assert paid_budget["estimated_flexcredit_remaining"] == pytest.approx(0.25)

    second_id, second_remaining = service._allocate_probe()
    service._record_paid_receipt(second_id, "b" * 64)
    second_budget = service._probe_budget_envelope(second_id, second_remaining)
    assert second_budget["measurement_probes_used"] == 2
    assert second_budget["measurement_probes_remaining"] == 1
    assert second_budget["paid_probes_used"] == 1
    assert second_budget["paid_probes_remaining"] == 1
    assert second_budget["estimated_flexcredit_used"] == pytest.approx(0.75)


def test_probe_feedback_is_useful_bounded_and_does_not_copy_adversarial_details():
    service = _load("anisotropic_probe_feedback", PROBE_SERVICE_PATH)
    secret = "sk-private-token /opt/picbench/evaluator traceback hidden_threshold=0.001"
    feedback = service._public_result(
        {
            "status": secret * 100,
            "valid": False,
            "design_valid": False,
            "simulation_valid": False,
            "gate_pass": False,
            "task_metric": 0.7,
            "errors": [secret * 1000],
            "geometry": {"private_path": secret, "internal_rule": {"threshold": 0.001}},
            "spectrum": [{"provider_payload": secret}] * 10000,
            "private_task_id": secret,
        }
    )

    assert feedback["task_slug"] == "anisotropic-swg-beamsplitter-v0"
    assert feedback["blocking_stage"] == "public_geometry_contract"
    assert feedback["public_error_codes"] == ["geometry_contract_failed"]
    assert feedback["summary"]["gate_pass"] is False
    assert feedback["summary"]["task_metric"] == pytest.approx(0.7)
    encoded = json.dumps(feedback, sort_keys=True)
    assert len(encoded.encode()) < service.MAX_PUBLIC_FEEDBACK_BYTES
    assert secret not in encoded
    assert "traceback" not in encoded.casefold()
    assert "hidden_threshold" not in encoded
    assert "errors" not in feedback
    assert "geometry" not in feedback
    assert "spectrum" not in feedback
    assert max(len(value) for value in feedback.values() if isinstance(value, list)) <= 1
    assert all(len(value) <= 160 for value in feedback.values() if isinstance(value, str))


def test_probe_handler_exception_http_payload_is_bounded_and_redacted(monkeypatch):
    service = _load("anisotropic_probe_exception", PROBE_SERVICE_PATH)
    secret = (
        "SECRET /opt/private/results provider-task-123 sk-private-value "
        "Traceback hidden_threshold=0.001 "
    )

    def fail_allocate():
        raise RuntimeError(secret * 2000)

    monkeypatch.setattr(service, "_allocate_probe", fail_allocate)
    statuses: list[int] = []
    response_headers: list[tuple[str, str]] = []
    handler = object.__new__(service.Handler)
    handler.path = "/v1/probe"
    handler.headers = {"Content-Length": "8"}
    handler.rfile = io.BytesIO(b"design()")
    handler.wfile = io.BytesIO()
    handler.send_response = statuses.append
    handler.send_header = lambda name, value: response_headers.append((name, value))
    handler.end_headers = lambda: None

    handler.do_POST()

    body = handler.wfile.getvalue()
    payload = json.loads(body.decode("utf-8"))
    assert statuses == [409]
    assert payload == {
        "error": "probe_request_failed",
        "category": "evaluation_failed",
    }
    assert payload["category"] in {
        "evaluation_timeout",
        "invalid_evaluator_result",
        "invalid_request",
        "probe_budget_exhausted",
        "evaluation_failed",
    }
    assert int(dict(response_headers)["Content-Length"]) == len(body)
    assert len(body) <= 64 * 1024
    decoded = body.decode("utf-8")
    for forbidden in (
        secret,
        "/opt/private",
        "provider-task-123",
        "sk-private-value",
        "Traceback",
        "hidden_threshold",
    ):
        assert forbidden not in decoded


def test_paper_fixture_passes_two_layer_geometry_and_port_contract(tmp_path: Path):
    evaluator = _load("anisotropic_evaluator", TASK / "evaluator/runtime/evaluator.py")
    geometry = evaluator.validate_gds(_fixture_gds(tmp_path), _profile())

    assert geometry["polygon_count_by_layer"] == {"1/0": 202, "2/0": 314}
    assert geometry["device_length_um"] == pytest.approx(28.46, abs=0.002)
    assert geometry["area_um2_by_layer"]["1/0"] == pytest.approx(44.2015)
    assert geometry["area_um2_by_layer"]["2/0"] == pytest.approx(32.4805)
    assert len(geometry["port_checks"]) == 6
    assert all(len(check["ports"]) == 2 for check in geometry["port_checks"])
    assert geometry["y_mirror_symmetry"]["passed"] is True
    assert geometry["y_mirror_symmetry"][
        "symmetric_difference_area_um2_by_layer"
    ] == {"1/0": 0.0, "2/0": 0.0}


@pytest.mark.parametrize("layer", [1, 2])
def test_one_sided_mask_mutation_is_rejected_by_strict_y_mirror_gate(tmp_path: Path, layer: int):
    evaluator = _load("anisotropic_evaluator_asymmetric", TASK / "evaluator/runtime/evaluator.py")
    path = _fixture_gds(tmp_path)
    library = gdstk.read_gds(str(path))
    library.top_level()[0].add(
        gdstk.rectangle((12.0, 1.70), (12.05, 1.75), layer=layer, datatype=0)
    )
    library.write_gds(str(path))

    with pytest.raises(evaluator.EvaluationError, match="strict y-mirror symmetry"):
        evaluator.validate_gds(path, _profile())


def test_forbidden_layer_is_rejected(tmp_path: Path):
    evaluator = _load("anisotropic_evaluator_bad_layer", TASK / "evaluator/runtime/evaluator.py")
    library = gdstk.Library()
    cell = library.new_cell("BAD")
    cell.add(gdstk.rectangle((0, -1.25), (28, -0.75), layer=9, datatype=0))
    path = tmp_path / "bad.gds"
    library.write_gds(path)
    with pytest.raises(evaluator.EvaluationError, match="forbidden layer"):
        evaluator.validate_gds(path, _profile())


def _landing_gds(tmp_path: Path, defect: str | None = None, side: str = "input", length: float = 28.0) -> Path:
    library = gdstk.Library()
    cell = library.new_cell("LANDING_CHECK")
    for center in (-1.0, 1.0):
        strip = gdstk.rectangle((0, center - .25), (length, center + .25))
        if defect in ("gap", "tiny_gap", "hole"):
            right = .201 if defect == "tiny_gap" else .3
            half_height = .05 if defect == "hole" else .3
            cut = gdstk.rectangle((.2, center - half_height), (right, center + half_height))
            pieces = gdstk.boolean(strip, cut, "not", precision=1e-6, layer=1)
        elif defect in ("peak", "within_tolerance"):
            # Only the vertex, not either adjacent segment's midpoint, exceeds
            # the declared width tolerance; sparse sampling cannot certify it.
            peak = .28 if defect == "peak" else .265
            if center < 0:
                points = [(0, center - .25), (length, center - .25),
                    (length, center + .25), (.4, center + .25), (.3, center + peak),
                    (.2, center + .25), (0, center + .25)]
            else:
                points = [(0, center - .25), (.2, center - .25),
                    (.3, center - peak), (.4, center - .25),
                    (length, center - .25), (length, center + .25),
                    (0, center + .25)]
            pieces = [gdstk.Polygon(points, layer=1)]
        else:
            strip.layer = 1
            pieces = [strip]
        for piece in pieces:
            if side == "output":
                piece.mirror((length / 2, 0), (length / 2, 1))
            cell.add(piece)
    # Meet the unchanged mask census using interior polygons; this fixture tests
    # geometric admission only, not optical performance or a candidate solution.
    for index in range(49):
        x = 2 + .2 * index
        cell.add(gdstk.rectangle((x, -.95), (x + .1, -.85), layer=1))
        cell.add(gdstk.rectangle((x, .85), (x + .1, .95), layer=1))
    for index in range(50):
        x = 2 + .2 * index
        cell.add(gdstk.rectangle((x, -.2), (x + .04, .2), layer=2))
    path = tmp_path / "landing.gds"
    library.write_gds(str(path))
    return path


@pytest.mark.parametrize("variation", [None, "within_tolerance"])
@pytest.mark.parametrize("length", [28.0, 28.46])
def test_complete_gds_accepts_continuous_declared_landings(tmp_path: Path, variation: str | None, length: float):
    evaluator = _load("anisotropic_valid_landings", TASK / "evaluator/runtime/evaluator.py")
    geometry = evaluator.validate_gds(_landing_gds(tmp_path, variation, length=length), _profile())
    assert len(geometry["port_checks"]) == 6


@pytest.mark.parametrize("side", ["input", "output"])
@pytest.mark.parametrize("defect", ["gap", "tiny_gap", "hole", "peak"])
def test_complete_gds_rejects_defects_between_sparse_landing_planes(tmp_path: Path, defect: str, side: str):
    evaluator = _load("anisotropic_landing_defect", TASK / "evaluator/runtime/evaluator.py")
    with pytest.raises(evaluator.EvaluationError, match="landing"):
        evaluator.validate_gds(_landing_gds(tmp_path, defect, side), _profile())


def _fdtd_result(estimate: float = 0.5) -> dict:
    profile = _profile()
    objective = profile["objective"]
    start = float(objective["wavelength_start_um"])
    stop = float(objective["wavelength_stop_um"])
    step = float(objective["wavelength_step_um"])
    t3, t4 = 0.47, 0.46
    excess_loss = -10.0 * math.log10(t3 + t4)
    imbalance = abs(10.0 * math.log10(t3 / t4))
    spectrum = []
    for index in range(int(round((stop - start) / step)) + 1):
        wavelength_nm = 1000.0 * (start + index * step)
        phase_error = 6.0 if index == int(round((stop - start) / step)) else 1.2
        spectrum.append({
            "wavelength_nm": wavelength_nm,
            "T3": t3,
            "T4": t4,
            "excess_loss_db": excess_loss,
            "imbalance_db": imbalance,
            "phase_error_deg": phase_error,
            "reflection_db": 28.0,
            "normalized_penalty": max(excess_loss, imbalance, phase_error / 5.0),
        })
    worst = max(spectrum, key=lambda point: point["normalized_penalty"])
    center = min(spectrum, key=lambda point: abs(point["wavelength_nm"] - 1550.0))
    return {
        "source": "evaluator_owned_tidy3d_fdtd",
        "metric": worst["normalized_penalty"],
        "secondary_metrics": {
            "T3_1550": center["T3"],
            "T4_1550": center["T4"],
            "excess_loss_db_1550": center["excess_loss_db"],
            "imbalance_db_1550": center["imbalance_db"],
            "phase_error_deg_1550": center["phase_error_deg"],
            "reflection_db_1550": center["reflection_db"],
            "worst_excess_loss_db": excess_loss,
            "worst_imbalance_db": imbalance,
            "worst_phase_error_deg": 6.0,
            "worst_wavelength_nm": worst["wavelength_nm"],
        },
        "spectrum": spectrum,
        "estimated_flexcredit": estimate,
        "cost_guard_pass": True,
    }


def test_metric_recomputed_from_trusted_output_powers():
    evaluator = _load("anisotropic_evaluator_metric", TASK / "evaluator/runtime/evaluator.py")
    validated = evaluator.validate_fdtd_result(_fdtd_result(), _profile())
    assert validated["metric"] == pytest.approx(1.2)
    assert validated["secondary_metrics"]["worst_wavelength_nm"] == pytest.approx(1750.0)
    bad = _fdtd_result()
    bad["metric"] += 0.1
    with pytest.raises(evaluator.EvaluationError, match="inconsistent"):
        evaluator.validate_fdtd_result(bad, _profile())


def test_broadband_response_plot_renders(tmp_path: Path):
    pytest.importorskip("matplotlib")
    evaluator = _load("anisotropic_evaluator_plot", TASK / "evaluator/runtime/evaluator.py")
    destination = tmp_path / "response.png"
    evaluator._render_response(evaluator.validate_fdtd_result(_fdtd_result(), _profile()), destination)
    assert destination.read_bytes().startswith(b"\x89PNG")


def test_over_cap_estimate_fails_closed():
    evaluator = _load("anisotropic_evaluator_cost", TASK / "evaluator/runtime/evaluator.py")
    profile = _profile()
    over_cap = float(profile["evaluation"]["max_flexcredit_per_run"]) + 0.1
    with pytest.raises(evaluator.EvaluationError, match="cost guard"):
        evaluator.validate_fdtd_result(_fdtd_result(over_cap), profile)


def test_private_worker_constructs_one_source_three_monitor_simulation(tmp_path: Path):
    evaluator = _load("anisotropic_evaluator_geometry", TASK / "evaluator/runtime/evaluator.py")
    worker = _load("anisotropic_worker", TASK / "evaluator/runtime/fdtd_worker.py")
    profile = _profile()
    geometry = evaluator.validate_gds(_fixture_gds(tmp_path), profile)
    request = {
        "schema_version": 1,
        "device_length_um": geometry["device_length_um"],
        "polygons": [
            {"layer": polygon["layer"], "points_um": polygon["points_um"]}
            for polygon in geometry["polygons"]
        ],
    }
    validated = worker._validate_request(request, profile)
    simulation = worker.build_simulation(validated, profile)
    assert [source.name for source in simulation.sources] == ["input_port_1"]
    assert [monitor.name for monitor in simulation.monitors] == ["in_mode", "out_3", "out_4"]
    assert len(simulation.structures) == 2
    assert float(simulation.grid_spec.wavelength) == pytest.approx(1.250)
    assert len(simulation.monitors[0].freqs) == 51
    assert int(simulation.sources[0].num_freqs) == 7
    assert float(simulation.run_time) == pytest.approx(3e-12)
    assert float(simulation.shutoff) == pytest.approx(1e-5)
    assert all(
        int(grid.min_steps_per_wvl) == 15
        for grid in (simulation.grid_spec.grid_x, simulation.grid_spec.grid_y, simulation.grid_spec.grid_z)
    )
    refine = simulation.grid_spec.override_structures[0]
    assert tuple(float(value) for value in refine.dl) == pytest.approx((0.019, 0.025, 0.02))
    assert tuple(float(value) for value in refine.geometry.size) == pytest.approx((28.96, 4.3, 0.5))


def test_success_publication_is_manifest_bound(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    evaluator = _load("anisotropic_evaluator_publish", TASK / "evaluator/runtime/evaluator.py")
    profile = _profile()
    geometry = evaluator.validate_gds(_fixture_gds(tmp_path), profile)
    output = tmp_path / "artifacts"
    plots = output / "plots"
    verifier = tmp_path / "verifier"
    output.mkdir()
    shutil.copy2(tmp_path / "fixture.gds", output / "final_pic.gds")
    monkeypatch.setattr(evaluator, "LOG_ARTIFACTS", output)
    monkeypatch.setattr(evaluator, "_render_geometry", lambda _m, path: path.write_bytes(b"geometry"))
    monkeypatch.setattr(evaluator, "_render_response", lambda _m, path: path.write_bytes(b"response"))

    payload = evaluator.publish_success(
        output_dir=output,
        artifact_dir=plots,
        verifier_dir=verifier,
        submission_sha256="a" * 64,
        submission_size=123,
        profile_path=PROFILE_PATH,
        geometry_manifest=geometry,
        fdtd_result=_fdtd_result(),
    )
    manifest = json.loads((output / "evaluation_manifest.json").read_text())
    reward = json.loads((verifier / "reward.json").read_text())
    assert payload["score_eligible"] is True
    assert payload["candidate_runs"] == 1
    assert payload["candidate_subruns"] == 1
    assert payload["physical_input_count"] == 2
    assert payload["simulated_input_count"] == 1
    assert payload["derived_input_count"] == 1
    assert payload["symmetry_gate_pass"] is True
    assert payload["input_coverage_pass"] is True
    assert payload["checks"]["symmetry_gate_pass"] is True
    assert payload["checks"]["input_coverage_pass"] is True
    assert payload["baseline_rerun"] is False
    assert payload["paper_margin"] == pytest.approx(1.0 - payload["official_score"])
    assert payload["task_metric"] == payload["official_score"] == payload["objective_value"]
    assert "improvement_db" not in payload
    assert manifest["complete"] is True
    assert manifest["gds_sha256"] == geometry["gds_sha256"]
    assert reward["official_score"] == pytest.approx(payload["official_score"])
    assert reward["reward"] == pytest.approx(1.0 / (1.0 + payload["official_score"]))
    assert all(isinstance(value, (int, float)) for value in reward.values())


def test_failure_publication_explicitly_disables_score_eligibility(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    evaluator = _load("anisotropic_evaluator_failure", TASK / "evaluator/runtime/evaluator.py")
    output = tmp_path / "artifacts"
    verifier = tmp_path / "verifier"
    monkeypatch.setattr(evaluator, "LOG_ARTIFACTS", output)

    payload = evaluator.publish_failure(
        output_dir=output,
        artifact_dir=output / "plots",
        verifier_dir=verifier,
        error="synthetic failure",
        submission_sha256="f" * 64,
        submission_size=12,
        profile_path=PROFILE_PATH,
        design_valid=False,
        evaluation_mode="final",
        probe_id=None,
    )
    reward = json.loads((verifier / "reward.json").read_text())

    assert payload["score_eligible"] is False
    assert payload["task_metric"] is payload["official_score"] is None
    assert payload["reward"] == 0.0
    assert all(isinstance(value, (int, float)) for value in reward.values())


def test_worker_cost_guard_precedes_single_start_and_deletes_private_hdf5():
    source = (TASK / "evaluator/runtime/fdtd_worker.py").read_text(encoding="utf-8")
    assert source.count("web.start(task_id)") == 1
    assert source.index("web.estimate_cost(task_id)") < source.index("web.start(task_id)")
    assert source.count("destination.unlink(missing_ok=True)") >= 2
    assert "web.abort(task_id)" in source
