#!/usr/bin/env python3
"""Public-safe restricted builder for one frozen anisotropic SWG 3-dB component submission.

This process establishes mandatory namespace, capability, Landlock, and
seccomp isolation before importing learner code. Its only output is one bounded canonical GDS file; it contains no evaluator rules, models, or credentials.
"""

from __future__ import annotations

import ctypes
import errno
import gc
import importlib
import importlib.util
import os
import platform
import resource
import socket
import stat
import sys
import sysconfig
from pathlib import Path

_APPROVED_PRELOAD_MODULES = (
    "numpy",
    "gdstk",
    "photonforge",
)
_PRIVATE_PATH_PREFIXES = (
    Path("/opt/picbench/evaluator"),
    Path("/tests"),
    Path("/submission"),
)

_CLONE_NEWUSER = 0x10000000
_CLONE_NEWNET = 0x40000000
_PR_GET_DUMPABLE = 3
_PR_SET_DUMPABLE = 4
_PR_SET_NO_NEW_PRIVS = 38
_PR_GET_NO_NEW_PRIVS = 39
_PR_CAPBSET_DROP = 24
_PR_SET_SECCOMP = 22
_SECCOMP_MODE_FILTER = 2

_LANDLOCK_CREATE_RULESET_VERSION = 1
_LANDLOCK_RULE_PATH_BENEATH = 1
_LANDLOCK_ACCESS_FS_EXECUTE = 1 << 0
_LANDLOCK_ACCESS_FS_WRITE_FILE = 1 << 1
_LANDLOCK_ACCESS_FS_READ_FILE = 1 << 2
_LANDLOCK_ACCESS_FS_READ_DIR = 1 << 3
_LANDLOCK_ACCESS_FS_REMOVE_DIR = 1 << 4
_LANDLOCK_ACCESS_FS_REMOVE_FILE = 1 << 5
_LANDLOCK_ACCESS_FS_MAKE_CHAR = 1 << 6
_LANDLOCK_ACCESS_FS_MAKE_DIR = 1 << 7
_LANDLOCK_ACCESS_FS_MAKE_REG = 1 << 8
_LANDLOCK_ACCESS_FS_MAKE_SOCK = 1 << 9
_LANDLOCK_ACCESS_FS_MAKE_FIFO = 1 << 10
_LANDLOCK_ACCESS_FS_MAKE_BLOCK = 1 << 11
_LANDLOCK_ACCESS_FS_MAKE_SYM = 1 << 12
_LANDLOCK_ACCESS_FS_REFER = 1 << 13
_LANDLOCK_ACCESS_FS_TRUNCATE = 1 << 14

_LANDLOCK_READ_DIR = _LANDLOCK_ACCESS_FS_READ_FILE | _LANDLOCK_ACCESS_FS_READ_DIR
_LANDLOCK_WRITE_DIR = (
    _LANDLOCK_READ_DIR
    | _LANDLOCK_ACCESS_FS_WRITE_FILE
    | _LANDLOCK_ACCESS_FS_REMOVE_DIR
    | _LANDLOCK_ACCESS_FS_REMOVE_FILE
    | _LANDLOCK_ACCESS_FS_MAKE_CHAR
    | _LANDLOCK_ACCESS_FS_MAKE_DIR
    | _LANDLOCK_ACCESS_FS_MAKE_REG
    | _LANDLOCK_ACCESS_FS_MAKE_SOCK
    | _LANDLOCK_ACCESS_FS_MAKE_FIFO
    | _LANDLOCK_ACCESS_FS_MAKE_BLOCK
    | _LANDLOCK_ACCESS_FS_MAKE_SYM
    | _LANDLOCK_ACCESS_FS_REFER
    | _LANDLOCK_ACCESS_FS_TRUNCATE
)
_LANDLOCK_HANDLED_FS = _LANDLOCK_ACCESS_FS_EXECUTE | _LANDLOCK_WRITE_DIR

_BPF_LD_W_ABS = 0x20
_BPF_JMP_JEQ_K = 0x15
_BPF_JMP_JGE_K = 0x35
_BPF_RET_K = 0x06
_SECCOMP_RET_KILL_PROCESS = 0x80000000
_SECCOMP_RET_ERRNO = 0x00050000
_SECCOMP_RET_ALLOW = 0x7FFF0000

_LIBC = ctypes.CDLL(None, use_errno=True)
_LIBC.syscall.restype = ctypes.c_long


class _LandlockRulesetAttr(ctypes.Structure):
    _fields_ = [("handled_access_fs", ctypes.c_uint64)]


class _LandlockPathBeneathAttr(ctypes.Structure):
    _fields_ = [
        ("allowed_access", ctypes.c_uint64),
        ("parent_fd", ctypes.c_int32),
        ("reserved", ctypes.c_uint32),
    ]


class _CapabilityHeader(ctypes.Structure):
    _fields_ = [
        ("version", ctypes.c_uint32),
        ("pid", ctypes.c_int),
    ]


class _CapabilityData(ctypes.Structure):
    _fields_ = [
        ("effective", ctypes.c_uint32),
        ("permitted", ctypes.c_uint32),
        ("inheritable", ctypes.c_uint32),
    ]


class _SockFilter(ctypes.Structure):
    _fields_ = [
        ("code", ctypes.c_ushort),
        ("jt", ctypes.c_ubyte),
        ("jf", ctypes.c_ubyte),
        ("k", ctypes.c_uint32),
    ]


class _SockFprog(ctypes.Structure):
    _fields_ = [
        ("length", ctypes.c_ushort),
        ("filter", ctypes.POINTER(_SockFilter)),
    ]


def _raise_errno(label: str) -> None:
    code = ctypes.get_errno() or errno.EPERM
    raise OSError(code, f"{label}: {os.strerror(code)}")


def _preload_approved_layout_modules() -> None:
    for module_name in _APPROVED_PRELOAD_MODULES:
        try:
            importlib.import_module(module_name)
        except Exception as exc:
            raise RuntimeError(
                f"approved layout import failed before sandbox activation: {module_name}: {exc}"
            ) from exc


def _write_proc_mapping(path: str, value: str) -> None:
    flags = os.O_WRONLY | getattr(os, "O_CLOEXEC", 0)
    fd = os.open(path, flags)
    try:
        encoded = value.encode("ascii")
        written = os.write(fd, encoded)
        if written != len(encoded):
            raise RuntimeError(f"short write while configuring {path}")
    finally:
        os.close(fd)


def _deny_setgroups() -> None:
    path = "/proc/self/setgroups"
    current = Path(path).read_text(encoding="ascii").strip()
    if current == "allow":
        _write_proc_mapping(path, "deny")
    elif current != "deny":
        raise RuntimeError(f"unexpected {path} state: {current!r}")
    if Path(path).read_text(encoding="ascii").strip() != "deny":
        raise RuntimeError("could not verify setgroups denial in nested user namespace")


def _call_unshare(flag: int, label: str) -> None:
    ctypes.set_errno(0)
    if _LIBC.unshare(ctypes.c_int(flag)) != 0:
        _raise_errno(f"required {label} namespace unavailable")


def _enter_user_and_network_namespaces(expected_uid: int, expected_gid: int) -> None:
    if expected_uid == 0 or expected_gid == 0:
        raise RuntimeError("restricted builder must start as a non-root outer UID/GID")
    if (os.getuid(), os.geteuid(), os.getgid(), os.getegid()) != (
        expected_uid,
        expected_uid,
        expected_gid,
        expected_gid,
    ):
        raise RuntimeError(
            "restricted builder outer identity mismatch: "
            f"got uid/euid/gid/egid={os.getuid()}/{os.geteuid()}/{os.getgid()}/{os.getegid()}"
        )

    original_user_namespace = os.readlink("/proc/self/ns/user")
    original_network_namespace = os.readlink("/proc/self/ns/net")
    _call_unshare(_CLONE_NEWUSER, "user")
    _deny_setgroups()
    _write_proc_mapping(
        "/proc/self/uid_map",
        f"{expected_uid} {expected_uid} 1",
    )
    _write_proc_mapping(
        "/proc/self/gid_map",
        f"{expected_gid} {expected_gid} 1",
    )
    if (os.getuid(), os.geteuid(), os.getgid(), os.getegid()) != (
        expected_uid,
        expected_uid,
        expected_gid,
        expected_gid,
    ):
        raise RuntimeError("nested user namespace did not preserve the outer UID/GID")

    _call_unshare(_CLONE_NEWNET, "network")

    isolated_user_namespace = os.readlink("/proc/self/ns/user")
    isolated_network_namespace = os.readlink("/proc/self/ns/net")
    if isolated_user_namespace == original_user_namespace:
        raise RuntimeError("user namespace isolation was not established")
    if isolated_network_namespace == original_network_namespace:
        raise RuntimeError("network namespace isolation was not established")


def _close_preload_sockets_and_assert_fd_boundary() -> None:
    for candidate in gc.get_objects():
        try:
            if isinstance(candidate, socket.socket):
                candidate.close()
        except Exception:
            continue
    gc.collect()

    leaked: list[str] = []
    for entry in Path("/proc/self/fd").iterdir():
        if not entry.name.isdigit() or int(entry.name) <= 2:
            continue
        try:
            target = os.readlink(entry)
        except FileNotFoundError:
            continue
        leaked.append(f"{entry.name}:{target}")
    if leaked:
        raise RuntimeError("restricted builder inherited unexpected open descriptors: " + ",".join(sorted(leaked)))


def _drop_all_capabilities() -> None:
    # A freshly mapped user namespace grants capabilities only inside that
    # namespace.  Remove those too, so learner code cannot configure its
    # isolated network namespace or mount new filesystems.
    for capability in range(64):
        ctypes.set_errno(0)
        result = _LIBC.prctl(
            ctypes.c_int(_PR_CAPBSET_DROP),
            ctypes.c_ulong(capability),
            0,
            0,
            0,
        )
        if result != 0 and ctypes.get_errno() not in (errno.EINVAL,):
            _raise_errno(f"could not drop capability {capability} from bounding set")

    header = _CapabilityHeader(0x20080522, 0)
    data = (_CapabilityData * 2)()
    ctypes.set_errno(0)
    if _LIBC.capset(ctypes.byref(header), ctypes.byref(data)) != 0:
        _raise_errno("could not clear restricted-builder capabilities")
    verified = (_CapabilityData * 2)()
    ctypes.set_errno(0)
    if _LIBC.capget(ctypes.byref(header), ctypes.byref(verified)) != 0:
        _raise_errno("could not verify restricted-builder capabilities")
    if any(item.effective or item.permitted or item.inheritable for item in verified):
        raise RuntimeError("restricted builder retained Linux capabilities")

    ctypes.set_errno(0)
    if _LIBC.prctl(_PR_SET_DUMPABLE, 0, 0, 0, 0) != 0:
        _raise_errno("could not disable restricted-builder dumpability")
    if _LIBC.prctl(_PR_GET_DUMPABLE, 0, 0, 0, 0) != 0:
        raise RuntimeError("restricted-builder dumpability remained enabled")
    ctypes.set_errno(0)
    if _LIBC.prctl(_PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0:
        _raise_errno("could not set restricted-builder no_new_privs")
    if _LIBC.prctl(_PR_GET_NO_NEW_PRIVS, 0, 0, 0, 0) != 1:
        raise RuntimeError("restricted builder no_new_privs verification failed")


def _install_resource_limits() -> None:
    limits = (
        (resource.RLIMIT_CORE, 0),
        (resource.RLIMIT_CPU, 120),
        (resource.RLIMIT_FSIZE, 8 * 1024 * 1024),
        (resource.RLIMIT_NOFILE, 64),
        (resource.RLIMIT_NPROC, 32),
        (resource.RLIMIT_AS, 4 * 1024 * 1024 * 1024),
    )
    for kind, ceiling in limits:
        resource.setrlimit(kind, (ceiling, ceiling))


def _landlock_syscall_numbers() -> tuple[int, int, int]:
    machine = platform.machine().lower()
    if machine not in {"x86_64", "amd64", "aarch64", "arm64"}:
        raise RuntimeError(f"unsupported architecture for Landlock enforcement: {machine}")
    # Landlock uses the generic syscall numbers on the supported 64-bit
    # architectures.
    return 444, 445, 446


def _is_private_path(path: Path, learner_path: Path, output_dir: Path) -> bool:
    if path == learner_path or path == output_dir or output_dir in path.parents:
        return False
    experiment = learner_path.parent
    if path == experiment or experiment in path.parents:
        return True
    return any(path == prefix or prefix in path.parents for prefix in _PRIVATE_PATH_PREFIXES)


def _approved_read_paths(learner_path: Path, output_dir: Path) -> list[Path]:
    candidates: set[Path] = set()
    for raw in (
        *sys.path,
        sysconfig.get_paths().get("stdlib", ""),
        sysconfig.get_paths().get("platstdlib", ""),
        sysconfig.get_paths().get("purelib", ""),
        sysconfig.get_paths().get("platlib", ""),
        "/lib",
        "/lib64",
        "/usr/lib",
        "/usr/lib64",
        "/usr/share/fonts",
        "/etc/fonts",
        "/etc/ld.so.cache",
        "/etc/localtime",
    ):
        if not raw:
            continue
        try:
            resolved = Path(raw).resolve(strict=True)
        except (OSError, RuntimeError):
            continue
        if resolved == Path("/") or _is_private_path(resolved, learner_path, output_dir):
            continue
        candidates.add(resolved)
    return sorted(candidates, key=lambda item: (len(item.parts), str(item)))


def _add_landlock_path_rule(
    add_rule_number: int,
    ruleset_fd: int,
    path: Path,
    allowed_access: int,
) -> None:
    flags = getattr(os, "O_PATH", 0) | getattr(os, "O_CLOEXEC", 0)
    if not getattr(os, "O_PATH", 0):
        raise RuntimeError("O_PATH is required for Landlock enforcement")
    parent_fd = os.open(path, flags)
    try:
        info = os.fstat(parent_fd)
        if stat.S_ISDIR(info.st_mode):
            access = allowed_access
        elif stat.S_ISREG(info.st_mode) or stat.S_ISCHR(info.st_mode):
            access = allowed_access & (
                _LANDLOCK_ACCESS_FS_READ_FILE | _LANDLOCK_ACCESS_FS_WRITE_FILE | _LANDLOCK_ACCESS_FS_TRUNCATE
            )
        else:
            raise RuntimeError(f"unsupported Landlock allowlist object: {path}")
        attribute = _LandlockPathBeneathAttr(access, parent_fd, 0)
        ctypes.set_errno(0)
        result = _LIBC.syscall(
            add_rule_number,
            ruleset_fd,
            _LANDLOCK_RULE_PATH_BENEATH,
            ctypes.byref(attribute),
            0,
        )
        if result != 0:
            _raise_errno(f"could not add Landlock rule for {path}")
    finally:
        os.close(parent_fd)


def _install_landlock(learner_path: Path, output_dir: Path) -> int:
    create_ruleset, add_rule, restrict_self = _landlock_syscall_numbers()
    ctypes.set_errno(0)
    abi = int(
        _LIBC.syscall(
            create_ruleset,
            0,
            0,
            _LANDLOCK_CREATE_RULESET_VERSION,
        )
    )
    if abi < 3:
        if abi < 0:
            _raise_errno("Landlock ABI query failed")
        raise RuntimeError(f"Landlock ABI >= 3 is required to restrict file truncation; found ABI {abi}")

    ruleset_attribute = _LandlockRulesetAttr(_LANDLOCK_HANDLED_FS)
    ctypes.set_errno(0)
    ruleset_fd = int(
        _LIBC.syscall(
            create_ruleset,
            ctypes.byref(ruleset_attribute),
            ctypes.sizeof(ruleset_attribute),
            0,
        )
    )
    if ruleset_fd < 0:
        _raise_errno("could not create mandatory Landlock ruleset")
    try:
        for path in _approved_read_paths(learner_path, output_dir):
            _add_landlock_path_rule(add_rule, ruleset_fd, path, _LANDLOCK_READ_DIR)
        _add_landlock_path_rule(
            add_rule,
            ruleset_fd,
            learner_path,
            _LANDLOCK_ACCESS_FS_READ_FILE,
        )
        _add_landlock_path_rule(
            add_rule,
            ruleset_fd,
            output_dir,
            _LANDLOCK_WRITE_DIR,
        )
        ctypes.set_errno(0)
        if _LIBC.syscall(restrict_self, ruleset_fd, 0) != 0:
            _raise_errno("could not apply mandatory Landlock ruleset")
    finally:
        os.close(ruleset_fd)

    # Runtime checks make configuration mistakes fail before learner import.
    with learner_path.open("rb") as handle:
        handle.read(1)
    try:
        Path("/proc/self/environ").read_bytes()
    except PermissionError:
        pass
    else:
        raise RuntimeError("Landlock self-test could still read /proc/self/environ")
    outside_probe = Path("/tmp") / f".picbench-builder-write-probe-{os.getpid()}"
    try:
        outside_probe.write_bytes(b"blocked")
    except PermissionError:
        pass
    else:
        try:
            outside_probe.unlink()
        finally:
            raise RuntimeError("Landlock self-test could write outside learner staging")
    return abi


def _seccomp_architecture() -> tuple[int, tuple[int, ...]]:
    machine = platform.machine().lower()
    if machine in {"x86_64", "amd64"}:
        return (
            0xC000003E,
            (
                41,
                42,
                43,
                44,
                45,
                46,
                47,
                48,
                49,
                50,
                51,
                52,
                53,
                54,
                55,
                272,
                288,
                299,
                307,
                308,
                425,
                426,
                427,
            ),
        )
    if machine in {"aarch64", "arm64"}:
        return (
            0xC00000B7,
            (
                97,
                198,
                199,
                200,
                201,
                202,
                203,
                204,
                205,
                206,
                207,
                208,
                209,
                210,
                211,
                212,
                242,
                243,
                268,
                269,
                425,
                426,
                427,
            ),
        )
    raise RuntimeError(f"unsupported architecture for seccomp enforcement: {machine}")


def _install_no_network_seccomp() -> None:
    # The outer container profile permits only the two exact unshare calls
    # needed to establish this sandbox. Once established, the inner filter
    # closes unshare and setns again in addition to all socket/io_uring calls.
    audit_arch, blocked_syscalls = _seccomp_architecture()
    instructions: list[_SockFilter] = [
        _SockFilter(_BPF_LD_W_ABS, 0, 0, 4),
        _SockFilter(_BPF_JMP_JEQ_K, 1, 0, audit_arch),
        _SockFilter(_BPF_RET_K, 0, 0, _SECCOMP_RET_KILL_PROCESS),
        _SockFilter(_BPF_LD_W_ABS, 0, 0, 0),
        # Reject the x32 syscall namespace wholesale; otherwise a submission
        # could try an alternate ABI number for a network or io_uring call.
        _SockFilter(_BPF_JMP_JGE_K, 0, 1, 0x40000000),
        _SockFilter(_BPF_RET_K, 0, 0, _SECCOMP_RET_ERRNO | errno.EPERM),
    ]
    for syscall_number in blocked_syscalls:
        instructions.extend(
            (
                _SockFilter(_BPF_JMP_JEQ_K, 0, 1, syscall_number),
                _SockFilter(
                    _BPF_RET_K,
                    0,
                    0,
                    _SECCOMP_RET_ERRNO | errno.EPERM,
                ),
            )
        )
    instructions.append(_SockFilter(_BPF_RET_K, 0, 0, _SECCOMP_RET_ALLOW))
    array_type = _SockFilter * len(instructions)
    instruction_array = array_type(*instructions)
    program = _SockFprog(len(instructions), instruction_array)
    ctypes.set_errno(0)
    if (
        _LIBC.prctl(
            _PR_SET_SECCOMP,
            _SECCOMP_MODE_FILTER,
            ctypes.byref(program),
        )
        != 0
    ):
        _raise_errno("could not install mandatory no-network seccomp filter")

    try:
        socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    except PermissionError:
        pass
    else:
        raise RuntimeError("seccomp self-test could still create a network socket")


def _enter_restricted_builder(learner_path: Path, output_dir: Path) -> None:
    expected_uid = int(os.environ["PICBENCH_EXPECTED_LEARNER_UID"])
    expected_gid = int(os.environ["PICBENCH_EXPECTED_LEARNER_GID"])
    if Path.cwd().resolve() != output_dir:
        raise RuntimeError("restricted builder working directory must be learner staging")

    # These trusted imports may use only the local license proxy.  No learner
    # source has been parsed or imported at this point.
    _preload_approved_layout_modules()
    for name in (
        "SIMCLOUD_APIKEY",
        "TIDY3D_API_KEY",
        "TIDY3D_WEB__API_ENDPOINT",
        "AGENTICPIC_LAYOUT_PROXY_ENDPOINT",
    ):
        os.environ.pop(name, None)

    _enter_user_and_network_namespaces(expected_uid, expected_gid)
    _close_preload_sockets_and_assert_fd_boundary()
    _drop_all_capabilities()
    _install_resource_limits()
    _install_landlock(learner_path, output_dir)
    _install_no_network_seccomp()


def _load_frozen_module(path: Path):
    spec = importlib.util.spec_from_file_location("_agenticpic_frozen_submission", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load frozen submission: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(spec.name, None)
        raise
    return module


def _normalize_submission(submission, component_type):
    if isinstance(submission, dict):
        raise TypeError("build_submission() must return one component directly, not a dictionary")
    if component_type is None or type(submission) is not component_type:
        raise TypeError("build_submission() must return one exact PhotonForge Component")
    return submission


learner_path = Path(sys.argv[1]).resolve()
output_dir = Path(sys.argv[2]).resolve()
output_dir.mkdir(parents=True, exist_ok=True)
_enter_restricted_builder(learner_path, output_dir)

# Capture the exact trusted class and unbound writer before learner code runs.
# Calling the captured descriptor prevents a Component subclass or instance
# attribute from replacing the canonical PhotonForge GDS export path.
photonforge = sys.modules.get("photonforge")
trusted_component_type = getattr(photonforge, "Component", None)
trusted_write_gds = getattr(trusted_component_type, "write_gds", None)
if trusted_component_type is None or not callable(trusted_write_gds):
    raise RuntimeError("trusted PhotonForge Component API is unavailable")

module = _load_frozen_module(learner_path)
builder = getattr(module, "build_submission", None)
if not callable(builder):
    raise AttributeError("design.py must expose callable build_submission()")
submitted_component = _normalize_submission(builder(), trusted_component_type)
trusted_write_gds(submitted_component, str(output_dir / "final_pic.gds"))
