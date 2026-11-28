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

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*

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

# Keep the outer identity distinct from host users and other tasks.
# Nested-user-namespace isolation and builder resource limits stay unchanged.
RUN groupadd --gid 62001 evallearner \
    && useradd --uid 62001 --gid 62001 --home-dir /nonexistent --shell /usr/sbin/nologin --no-create-home evallearner

RUN mkdir -p \
       /opt/picbench/evaluator/bin \
       /opt/picbench/evaluator/builder \
       /opt/picbench/evaluator/runtime \
       /opt/picbench/evaluator/src \
       /submission \
       /experiment \
       /logs/artifacts \
       /logs/verifier \
       /var/lib/picbench/anisotropic-swg-beamsplitter-v0/state \
       /tests

COPY evaluator/bin/evaluate /opt/picbench/evaluator/bin/evaluate
COPY evaluator/bin/probe-service /opt/picbench/evaluator/bin/probe-service
COPY evaluator/bin/probe-healthcheck /opt/picbench/evaluator/bin/probe-healthcheck
COPY evaluator/builder/restricted_builder.py /opt/picbench/evaluator/builder/restricted_builder.py
COPY evaluator/runtime/evaluator.py /opt/picbench/evaluator/runtime/evaluator.py
COPY evaluator/runtime/fdtd_worker.py /opt/picbench/evaluator/runtime/fdtd_worker.py
COPY evaluator/runtime/layout_license_proxy.py /opt/picbench/evaluator/runtime/layout_license_proxy.py
COPY evaluator/runtime/start_layout_license_proxy /opt/picbench/evaluator/runtime/start_layout_license_proxy
COPY evaluator/runtime/task_profile.json /opt/picbench/evaluator/runtime/task_profile.json
COPY evaluator/src/probe_service.py /opt/picbench/evaluator/src/probe_service.py
COPY tests/test.sh /tests/test.sh
COPY evaluator/runtime/layout_license_proxy.py /usr/local/bin/agenticpic_layout_license_proxy
COPY evaluator/runtime/start_layout_license_proxy /usr/local/bin/start_agenticpic_layout_license_proxy

RUN chown -R root:root /opt/picbench/evaluator /usr/local/bin/agenticpic_layout_license_proxy /usr/local/bin/start_agenticpic_layout_license_proxy /tests /submission /experiment /logs /var/lib/picbench \
    && chmod 711 /opt/picbench/evaluator \
    && chmod 700 /opt/picbench/evaluator/bin /opt/picbench/evaluator/runtime /opt/picbench/evaluator/src \
    && chmod 711 /opt/picbench/evaluator/builder \
    && chmod 500 /opt/picbench/evaluator/bin/evaluate /opt/picbench/evaluator/bin/probe-service /opt/picbench/evaluator/bin/probe-healthcheck /usr/local/bin/agenticpic_layout_license_proxy /usr/local/bin/start_agenticpic_layout_license_proxy /tests/test.sh \
    && chmod 444 /opt/picbench/evaluator/builder/restricted_builder.py \
    && find /opt/picbench/evaluator/runtime /opt/picbench/evaluator/src -type f -exec chmod 400 {} \; \
    && chmod 700 /tests /logs/artifacts /logs/verifier /var/lib/picbench/anisotropic-swg-beamsplitter-v0/state \
    && chmod 755 /submission \
    && chmod 711 /experiment

WORKDIR /
USER root
