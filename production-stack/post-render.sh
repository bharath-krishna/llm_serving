#!/usr/bin/env bash
# Helm post-renderer: injects the LMCache MP sidecar into the nemotron engine
# Deployment (the chart has no field for it). Helm pipes its rendered manifests
# to stdin; we hand them to kustomize with patch-lmcache-sidecar.yaml and emit
# the result on stdout.
#
#   helm upgrade --install vllm vllm/vllm-stack -n vllm \
#     -f production-stack/values.yaml --version 0.1.12 \
#     --post-renderer production-stack/post-render.sh
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

cat > "$tmp/rendered.yaml"
cp "$here/kustomization.yaml" "$here/patch-lmcache-sidecar.yaml" "$tmp/"

kubectl kustomize "$tmp"
