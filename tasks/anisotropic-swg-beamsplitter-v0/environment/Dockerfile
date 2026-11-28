FROM python:3.12-slim@sha256:7a8b475003c4fe15a2cd4e55e5cfc2f3560bdc9333d624f24cdd6d4340fd7a17

ARG PIP_VERSION=26.2.1
ARG SETUPTOOLS_VERSION=84.0.0
ARG WHEEL_VERSION=0.48.0
ARG NUMPY_VERSION=2.4.6
ARG SCIPY_VERSION=1.18.1
ARG MATPLOTLIB_VERSION=3.11.1
ARG GDSTK_VERSION=1.0.1
ARG TIDY3D_VERSION=2.11.2
ARG PHOTONFORGE_VERSION=1.4.3

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       ca-certificates \
       curl \
       git \
       jq \
       poppler-utils \
       tmux \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -s /bin/bash -u 1000 agent

WORKDIR /workspace

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg

RUN python -m pip install --upgrade \
       "pip==${PIP_VERSION}" \
       "setuptools==${SETUPTOOLS_VERSION}" \
       "wheel==${WHEEL_VERSION}" \
    && python -m pip install \
       "numpy==${NUMPY_VERSION}" \
       "scipy==${SCIPY_VERSION}" \
       "matplotlib==${MATPLOTLIB_VERSION}" \
       "gdstk==${GDSTK_VERSION}" \
       "tidy3d==${TIDY3D_VERSION}" \
       "photonforge==${PHOTONFORGE_VERSION}"

RUN python -c "import importlib.metadata as md; import gdstk; import tidy3d as td; print('gdstk', gdstk.__version__); print('Tidy3D', td.__version__); print('PhotonForge', md.version('photonforge'))"

RUN mkdir -p /workspace/references /outputs /logs/agent /logs/artifacts /logs/verifier /submission

COPY baseline/design.py /workspace/design.py
COPY task-deps/task_profile.json /workspace/task_profile.json
COPY task-deps/instruction.md /workspace/instruction.md
COPY task-deps/harness_contract.md /workspace/harness_contract.md
COPY task-deps/photonforge_tidy3d.md /workspace/photonforge_tidy3d.md
COPY task-deps/references/paper_parameters.json /workspace/references/paper_parameters.json
COPY task-deps/references/paper.pdf /workspace/references/paper.pdf
COPY run_probe /usr/local/bin/run_probe
COPY benchmark_status /usr/local/bin/benchmark_status
COPY benchmark_status_lib /usr/local/libexec/picbench_benchmark_status_lib
COPY export_submission /usr/local/sbin/export_anisotropic_swg_beamsplitter_submission

RUN git config --global user.email "agent@picbench.local" \
    && git config --global user.name "PICBench Agent" \
    && git init \
    && git add design.py task_profile.json instruction.md harness_contract.md photonforge_tidy3d.md references \
    && git commit -m "Add public anisotropic SWG implementation task files" \
    && cp /root/.gitconfig /home/agent/.gitconfig \
    && chown agent:agent /home/agent/.gitconfig \
    && git config --global --add safe.directory /workspace

RUN chmod 555 /usr/local/bin/run_probe \
    && chmod 755 /usr/local/bin/benchmark_status \
    && chmod 444 /usr/local/libexec/picbench_benchmark_status_lib \
    && chmod 500 /usr/local/sbin/export_anisotropic_swg_beamsplitter_submission \
    && chown root:root /usr/local/bin/run_probe /usr/local/sbin/export_anisotropic_swg_beamsplitter_submission \
    && chown -R agent:agent /workspace /outputs /logs/agent \
    && chmod 755 /logs /workspace /outputs /logs/agent \
    && chmod 700 /logs/artifacts /logs/verifier /submission

USER agent
