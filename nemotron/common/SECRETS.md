# Secrets

Both nemotron variants and all four bench pods share ONE Secret. It is created
imperatively on the cluster and only *referenced* from manifests — no secret
value appears anywhere in this repo, and none ever should.

| Secret | key | consumed by | as |
|---|---|---|---|
| `vllm-auth` (ns `vllm`) | `api-key` | `nvfp4/deployment.yaml`, `bf16/deployment.yaml` | `VLLM_API_KEY` (server side) |
| | | `*/bench/bench-pod.yaml`, `*/bench/sweep-pod.yaml` | `OPENAI_API_KEY` (client bearer token) |

## Why it is shared

The public URL is the same for both variants (`nemo35-lightning.krishb.in`), so
the key must be too — a client should not have to know which variant is live.
Sharing it also means a rotation covers everything at once and bench pods need
no change, because they read the same Secret key the servers do.

vLLM treats `VLLM_API_KEY` as the fallback for `--api-key`
(`entrypoints/openai/api_server.py:266`), so the env var is equivalent to
passing the flag — but the key stays out of git and out of
`kubectl get deploy -o yaml` argv.

**Without it these endpoints are open, unauthenticated LLM APIs.**
`*.krishb.in` resolves publicly (103.76.103.148) and already takes internet
scanner traffic. That is not just inference theft: an anonymous caller can
submit max-model-len-sized requests, and sustained load on this box hard-resets
the node (`BENCHMARK-HANDOFF.md` section 3).

## Create / rotate

```bash
kubectl create secret generic vllm-auth -n vllm \
  --from-literal=api-key="sk-vllm-$(python3 -c 'import secrets;print(secrets.token_urlsafe(32))')" \
  --dry-run=client -o yaml | kubectl apply -f -

# re-roll every consumer (gemma4 shares this Secret too)
kubectl rollout restart deploy/vllm-nemotron-nvfp4 deploy/vllm-nemotron-bf16 deploy/vllm-gemma4 -n vllm
```

Bench pods pick the new key up on their next create; no action needed.

Read it back for a manual `curl`:

```bash
KEY=$(kubectl get secret vllm-auth -n vllm -o jsonpath='{.data.api-key}' | base64 -d)
curl -s https://nemo35-lightning.krishb.in/v1/models -H "Authorization: Bearer $KEY" | jq .
```

## Other secrets referenced from this repo

`registry-krishb-creds` — an imagePullSecret named only by
`production-stack/values.yaml` (the unused Helm path). Not used by any manifest
under `nemotron/`.
