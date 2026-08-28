FROM python:3.12-slim

WORKDIR /app

# hadolint ignore=DL3008
RUN --mount=type=cache,target=/var/lib/apt --mount=type=cache,target=/var/cache/apt \
    apt-get update && \
    apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
COPY .git/ .git/
COPY src/ src/

# `default` (empty extras group) => no semantic_cache/lmcache; those pull torch /
# faiss / vllm which have no arm64 wheels at the pinned versions. roundrobin
# routing needs none of them.
ARG INSTALL_OPTIONAL_DEP=default
ENV INSTALL_OPTIONAL_DEP=${INSTALL_OPTIONAL_DEP}
# Shallow clone has no router version tag; pin it so setuptools_scm succeeds.
ARG ROUTER_VERSION=0.1.12
ENV SETUPTOOLS_SCM_PRETEND_VERSION=${ROUTER_VERSION}

RUN pip install --no-cache-dir uv && \
    uv venv /opt/venv && \
    . /opt/venv/bin/activate && \
    uv pip install --upgrade --no-cache-dir pip setuptools_scm && \
    uv pip install --no-cache-dir .[$INSTALL_OPTIONAL_DEP]

ENTRYPOINT ["/opt/venv/bin/vllm-router"]
CMD []
