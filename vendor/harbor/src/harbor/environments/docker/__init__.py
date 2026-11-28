import json
from pathlib import Path

from harbor.models.trial.config import ServiceVolumeConfig

# Shared compose file paths used by both local Docker and Daytona DinD environments.
COMPOSE_DIR = Path(__file__).parent
COMPOSE_BUILD_PATH = COMPOSE_DIR / "docker-compose-build.yaml"
COMPOSE_PREBUILT_PATH = COMPOSE_DIR / "docker-compose-prebuilt.yaml"
COMPOSE_NO_NETWORK_PATH = COMPOSE_DIR / "docker-compose-no-network.yaml"
COMPOSE_EGRESS_CONTROL_PATH = COMPOSE_DIR / "docker-compose-egress-control.yaml"
EGRESS_CONTROL_SIDECAR_CONTEXT_PATH = (
    COMPOSE_DIR / "harbor-docker-egress-control-sidecar"
)
COMPOSE_WINDOWS_KEEPALIVE_PATH = COMPOSE_DIR / "docker-compose-windows-keepalive.yaml"
RESOURCES_COMPOSE_NAME = "docker-compose-resources.json"
ENV_COMPOSE_NAME = "docker-compose-environment.json"


def write_env_compose_file(path: Path, env: dict[str, str]) -> Path:
    """Write a Compose override that injects task env into the main service."""
    compose = {"services": {"main": {"environment": env}}}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(compose, indent=2))
    return path


def write_mounts_compose_file(path: Path, mounts: list[ServiceVolumeConfig]) -> Path:
    """Write a compose override that declares services.main.volumes."""
    volumes = [
        {key: value for key, value in mount.items() if key != "preserve_permissions"}
        for mount in mounts
    ]
    compose = {"services": {"main": {"volumes": volumes}}}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(compose, indent=2))
    return path


def write_resources_compose_file(
    path: Path,
    *,
    cpu_request: int | None = None,
    cpu_limit: int | None = None,
    memory_request_mb: int | None = None,
    memory_limit_mb: int | None = None,
    gpus: int | None = None,
    seccomp_profile: Path | None = None,
    no_new_privileges: bool = False,
    read_only: bool = False,
    init: bool = False,
    cap_drop: list[str] | None = None,
    cap_add: list[str] | None = None,
    pids_limit: int | None = None,
    tmpfs: list[str] | None = None,
) -> Path:
    """Write a Compose override for main-service resources and isolation."""
    resources: dict[str, object] = {}
    limits: dict[str, str | int] = {}
    reservations: dict[str, object] = {}
    main: dict[str, object] = {}

    if cpu_limit is not None:
        main["cpus"] = float(cpu_limit)
        limits["cpus"] = str(cpu_limit)
    if memory_limit_mb is not None:
        main["mem_limit"] = f"{memory_limit_mb}m"
        limits["memory"] = f"{memory_limit_mb}M"
    if pids_limit is not None:
        # Compose treats service-level ``pids_limit`` and the deploy resource
        # limit as two representations of the same constraint.  Recent
        # Compose releases reject a project when only one representation is
        # supplied alongside other deploy limits because the normalized
        # values differ.  Emit both with the same value.
        limits["pids"] = pids_limit
    if cpu_request is not None:
        reservations["cpus"] = str(cpu_request)
    if memory_request_mb is not None:
        reservations["memory"] = f"{memory_request_mb}M"
    if gpus is not None and gpus > 0:
        reservations["devices"] = [
            {"driver": "nvidia", "count": gpus, "capabilities": ["gpu"]}
        ]

    if reservations:
        resources["reservations"] = reservations
    if limits:
        resources["limits"] = limits
    if resources:
        main["deploy"] = {"resources": resources}
    security_opt: list[str] = []
    if no_new_privileges:
        security_opt.append("no-new-privileges:true")
    if seccomp_profile is not None:
        security_opt.append(f"seccomp={seccomp_profile.as_posix()}")
    if security_opt:
        main["security_opt"] = security_opt
    if read_only:
        main["read_only"] = True
    if init:
        main["init"] = True
    if cap_drop:
        main["cap_drop"] = list(cap_drop)
    if cap_add:
        main["cap_add"] = list(cap_add)
    if pids_limit is not None:
        main["pids_limit"] = pids_limit
    if tmpfs:
        main["tmpfs"] = list(tmpfs)
    compose = {"services": {"main": main}}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(compose, indent=2))
    return path


def self_bind_mount(mount: ServiceVolumeConfig) -> ServiceVolumeConfig:
    """Return a copy of *mount* with ``source`` set equal to ``target``.

    Used by cloud providers whose docker compose "host" filesystem is the VM,
    not the user's machine. Binding ``target → target`` lets task-author
    compose files share the same dir between services without each one
    having to know the cloud provider's internal VM path layout.
    """
    new_mount: ServiceVolumeConfig = {
        "type": mount["type"],
        "source": mount["target"],
        "target": mount["target"],
    }
    if mount.get("read_only"):
        new_mount["read_only"] = True
    if mount.get("preserve_permissions"):
        new_mount["preserve_permissions"] = True
    if "bind" in mount:
        new_mount["bind"] = mount["bind"]
    if "volume" in mount:
        new_mount["volume"] = mount["volume"]
    if "image" in mount:
        new_mount["image"] = mount["image"]
    return new_mount
