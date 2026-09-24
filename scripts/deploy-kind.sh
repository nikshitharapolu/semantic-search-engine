#!/usr/bin/env bash
set -euo pipefail

cluster_name="rag-local"
image_name="semantic-search-rag:local"

for command_name in docker kind kubectl; do
  if ! command -v "$command_name" >/dev/null 2>&1; then
    echo "Missing required command: $command_name" >&2
    exit 1
  fi
done

if ! kind get clusters | grep -qx "$cluster_name"; then
  kind create cluster --name "$cluster_name"
fi

docker build -t "$image_name" .
kind load docker-image "$image_name" --name "$cluster_name"
kubectl config use-context "kind-$cluster_name"
kubectl apply -k k8s
kubectl rollout status deployment/rag-api -n rag-platform --timeout=240s
kubectl rollout status deployment/rag-ui -n rag-platform --timeout=240s

echo
echo "Kubernetes deployment is ready. In separate terminals run:"
echo "  kubectl port-forward service/rag-ui 8501:8501 -n rag-platform"
echo "  kubectl port-forward service/rag-api 8000:8000 -n rag-platform"
