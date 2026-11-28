from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib


REPO = Path(__file__).resolve().parents[3]
TASK = REPO / "tasks/anisotropic-swg-beamsplitter-v0"
CERTIFICATE = Path(__file__).with_name("author_candidate_cost_estimate_v2_2026-08-27.json")
V2_CLOUD_CERTIFICATE = Path(__file__).with_name(
    "author_candidate_cloud_v2_certification_2026-08-27.json"
)
V1_CERTIFICATE = Path(__file__).with_name("author_candidate_cost_estimate_v1_2026-08-26.json")
LEGACY_CERTIFICATE = Path(__file__).with_name("author_candidate_cost_estimate_2026-08-26.json")


def test_public_and_private_profiles_have_one_explicit_boundary():
    public = json.loads((TASK / "environment/task-deps/task_profile.json").read_text())
    private = json.loads((TASK / "evaluator/runtime/task_profile.json").read_text())
    trusted = private.pop("trusted_physics")
    assert private == public
    assert trusted["model"] == "anisotropic_swg_beamsplitter"
    assert public["task_mode"] == "implementation_optimization"
    assert public["evaluation"]["candidate_runs_per_evaluation"] == 1
    assert public["evaluation"]["candidate_subruns_per_evaluation"] == 1
    assert public["evaluation"]["max_flexcredit_per_run"] == 4.0
    assert public["baseline"]["rerun_by_evaluator"] is False
    assert public["task_version"] == "v4-y-mirror-input-coverage"
    assert "instruction.md" in public["public_files"]
    assert public["layout_contract"]["y_mirror_symmetry"]["required"] is True
    assert public["evaluation"]["input_coverage"] == {
        "physical_input_count": 2,
        "simulated_input_count": 1,
        "simulated_input_center_y_um": 1.0,
        "derived_input_center_y_um": -1.0,
        "derivation": "strict_y_mirror_geometry_with_output_port_exchange",
        "eligibility_requires_symmetry": True,
    }
    assert public["objective"]["primary_metric_field"] == "official_score"
    assert public["objective"]["direction"] == "min"
    assert "MAX_FLEXCREDIT_PER_CANDIDATE_EVALUATION = 4.0" in (
        TASK / "evaluator/src/probe_service.py"
    ).read_text()
    assert public["metrics"]["primary"]["field"] == "official_score"
    assert public["baseline"]["reference_value"] == 1.0
    assert public["objective"]["wavelength_start_um"] == 1.250
    assert public["objective"]["wavelength_stop_um"] == 1.750
    assert public["objective"]["wavelength_step_um"] == 0.010
    assert public["evaluation"]["max_flexcredit_per_run"] == 4.0


def test_task_uses_separate_read_only_evaluator_and_source_only_submission():
    config = tomllib.loads((TASK / "task.toml").read_text())
    assert config["schema_version"] == "1.3"
    assert config["metadata"]["task_mode"] == "implementation_optimization"
    assert config["verifier"]["environment_mode"] == "separate"
    assert config["verifier"]["environment"]["docker_read_only"] is True
    assert config["artifacts"] == [
        "/submission/design.py",
        {
            "source": "/workspace/.benchmark/probe-results",
            "destination": "workspace",
            "service": "main",
        },
        {"source": "/var/lib/picbench/anisotropic-swg-beamsplitter-v0/state/probe_history.json", "service": "evaluator"},
        {"source": "/var/lib/picbench/anisotropic-swg-beamsplitter-v0/state/probe_state.json", "service": "evaluator"},
        {"source": "/var/lib/picbench/anisotropic-swg-beamsplitter-v0/state/paid_usage_receipt.json", "service": "evaluator"},
    ]


def test_learner_image_has_no_private_evaluator_or_credentials():
    dockerfile = (TASK / "environment/Dockerfile").read_text()
    compose = (TASK / "environment/docker-compose.yaml").read_text()
    learner, evaluator = compose.split("\n  evaluator:", 1)
    assert "COPY evaluator/" not in dockerfile
    assert "USER agent" in dockerfile
    assert "TIDY3D_API_KEY" not in learner
    assert "agenticpic-layout-token" in learner
    assert "TIDY3D_API_KEY: ${TIDY3D_API_KEY:-}" in evaluator
    assert "read_only: true" in evaluator


def test_canonical_builder_and_evaluator_paths_are_shared_by_probe_and_final():
    probe = (TASK / "evaluator/src/probe_service.py").read_text()
    probe_client = (TASK / "environment/run_probe").read_text()
    final = (TASK / "tests/test.sh").read_text()
    builder = (TASK / "evaluator/builder/restricted_builder.py").read_text()
    assert 'EVALUATE = Path("/opt/picbench/evaluator/bin/evaluate")' in probe
    assert "/opt/picbench/evaluator/bin/evaluate" in final
    assert "_enter_user_and_network_namespaces" in builder
    assert "_install_landlock" in builder
    assert "_install_no_network_seccomp" in builder
    assert "_probe_budget_envelope(probe_id, remaining)" in probe
    assert "response = _public_response(result, probe_id, remaining, source_sha256)" in probe
    assert '"experiments"' not in probe
    assert "top-level fields do not match the 13-field schema" in probe_client
    assert 'payload["submission_sha256"] != expected_submission_sha256' in probe_client
    assert "Path(destination).write_bytes(response_bytes)" in probe_client
    assert "--max-filesize 65536" in probe_client
    assert '"task_feedback": _public_result(payload)' in probe
    assert '"measurement_probes_used": None' not in probe_client
    assert '"paid_probes_used": probe_id' not in probe_client
    assert '"errors"' not in probe.split("def _public_result", 1)[1].split("class Handler", 1)[0]


def test_instruction_and_references_are_public_and_consistent():
    root = (TASK / "instruction.md").read_bytes()
    baked = (TASK / "environment/task-deps/instruction.md").read_bytes()
    instruction = root.decode()
    assert root == baked
    assert "max_wavelength" in instruction
    assert "1250–1750 nm" in instruction
    assert "10-nm wavelength" in " ".join(instruction.split())
    assert "Tidy3D" not in instruction and "PhotonForge" not in instruction
    assert "`(2, 0)`" in instruction
    assert (TASK / "environment/task-deps/references/paper.pdf").stat().st_size > 100_000
    paper = json.loads((TASK / "environment/task-deps/references/paper_parameters.json").read_text())
    assert paper["paper_geometry_um"]["swg_period"] == 0.190
    assert paper["paper_results"]["simulated_broadband_criteria"]["phase_error_deg_max"] == 5.0
    assert "code_baseline_result" not in paper
    guide = (TASK / "environment/task-deps/photonforge_tidy3d.md").read_text()
    assert "fixed, nondispersive material indices" in guide
    assert "does not claim equivalence" in guide


def test_probe_profile_declares_common_thirteen_field_envelope():
    profile = json.loads((TASK / "environment/task-deps/task_profile.json").read_text())
    assert len(profile["probe"]["response_top_level_fields"]) == 13
    assert profile["probe"]["task_feedback"] == "task-specific bounded diagnostics"
    assert profile["probe"]["budget_authority"] == "private_evaluator_sidecar"


def test_v2_cost_certificate_is_retained_historical_evidence():
    certificate = json.loads(CERTIFICATE.read_text())
    v1 = json.loads(V1_CERTIFICATE.read_text())
    legacy = json.loads(LEGACY_CERTIFICATE.read_text())
    profile_path = TASK / "evaluator/runtime/task_profile.json"
    profile = json.loads(profile_path.read_text())
    estimate = float(certificate["estimated_flexcredit"])
    cap = float(certificate["max_flexcredit_per_run"])
    assert cap == 1.2  # Historical estimate certificate; not a current-cap certificate.
    assert cap <= float(profile["evaluation"]["max_flexcredit_per_run"])
    assert cap >= estimate * (1.0 + float(certificate["minimum_headroom_fraction"]))
    assert certificate["mode"] == "upload_and_estimate_only"
    assert certificate["cloud_start_attempted"] is False
    assert certificate["cloud_task_started"] is False
    assert certificate["provider_task_uploaded"] is True
    assert certificate["local_simulation_construction_smoke_passed"] is True
    assert certificate["restricted_builder_smoke_passed"] is True
    assert certificate["response_plot_smoke_passed"] is True
    assert certificate["estimate_failure_transaction_complete"] is True
    assert certificate["estimate_failure_transaction_design_valid"] is True
    assert certificate["profile_sha256"] != hashlib.sha256(profile_path.read_bytes()).hexdigest()
    assert certificate["task_version"] == "v2-500nm-envelope"
    assert profile["task_version"] == "v4-y-mirror-input-coverage"
    assert v1["profile_sha256"] != certificate["profile_sha256"]
    assert legacy["profile_sha256"] != certificate["profile_sha256"]
    fixture = Path(__file__).with_name("fixtures") / "paper_candidate_design.py"
    assert certificate["submission_sha256"] == hashlib.sha256(fixture.read_bytes()).hexdigest()


def test_v2_author_score_recomputes_under_unchanged_canonical_j_formula():
    certificate = json.loads(V2_CLOUD_CERTIFICATE.read_text())
    profile_path = TASK / "evaluator/runtime/task_profile.json"
    profile = json.loads(profile_path.read_text())

    envelope = profile["baseline"]["reference_operating_envelope"]
    expected = max(
        certificate["worst_excess_loss_db"] / envelope["excess_loss_db_max"],
        certificate["worst_imbalance_db"] / envelope["imbalance_db_max"],
        certificate["worst_phase_error_deg"] / envelope["phase_error_deg_max"],
    )

    assert certificate["profile_sha256"] != hashlib.sha256(
        profile_path.read_bytes()
    ).hexdigest()
    assert certificate["task_version"] == "v2-500nm-envelope"
    assert profile["task_version"] == "v4-y-mirror-input-coverage"
    assert certificate["task_metric"] == certificate["official_score"]
    assert math.isclose(
        certificate["official_score"], expected, rel_tol=0.0, abs_tol=1e-15
    )
    assert certificate["reference_baseline"] == profile["baseline"]["reference_value"]
