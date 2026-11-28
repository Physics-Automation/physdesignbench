import asyncio


class AgentSetupTimeoutError(asyncio.TimeoutError):
    pass


class AgentTimeoutError(asyncio.TimeoutError):
    pass


class AgentCleanupError(RuntimeError):
    """Operational cleanup failed after the learner phase.

    A trusted final verifier may still measure the retained submission. This
    error does not imply a successful agent lifecycle and must not be used for
    failed source-integrity, isolation or credential checks.
    """


class AgentEvidenceIntegrityError(RuntimeError):
    """Trusted agent evidence or isolation failed; final verification is blocked."""


class VerifierTimeoutError(asyncio.TimeoutError):
    pass


class EnvironmentStartTimeoutError(asyncio.TimeoutError):
    pass
