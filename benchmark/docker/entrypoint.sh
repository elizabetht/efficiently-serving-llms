#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-serve}"

# Config settings via env vars
MODEL="${MODEL:-Qwen/Qwen2.5-0.5B}"
PORT="${PORT:-8000}"
TP_SIZE="${TP_SIZE:-1}"

# Bench shape/config (used when MODE=bench-serve)
BENCH_INPUT_LEN="${BENCH_INPUT_LEN:-2048}"
BENCH_OUTPUT_LEN="${BENCH_OUTPUT_LEN:-128}"
BENCH_NUM_PROMPTS="${BENCH_NUM_PROMPTS:-64}"
BENCH_MAX_CONCURRENCY="${BENCH_MAX_CONCURRENCY:-8}"

wait_for_port() {
  local host="$1"
  local port="$2"
  local retries=60

  echo "Waiting for $host:$port to be ready..."
  for i in $(seq 1 "$retries"); do
    if python3 - <<EOF
import socket, sys
s = socket.socket()
s.settimeout(1)
try:
    s.connect(("$host", $port))
    s.close()
    sys.exit(0)
except Exception:
    sys.exit(1)
EOF
    then
      echo "Port $host:$port is ready."
      return 0
    fi
    sleep 2
  done
  echo "ERROR: $host:$port did not become ready in time."
  return 1
}

if [[ "$MODE" == "serve" ]]; then
  echo "Starting vLLM server..."
  exec vllm serve "$MODEL" \
    --port "$PORT" \
    --tensor-parallel-size "$TP_SIZE"
    --gpu-memory-utilization 0.5

elif [[ "$MODE" == "bench-serve" ]]; then
  echo "Starting vLLM server in background for benchmarking..."
  vllm serve "$MODEL" \
    --port "$PORT" \
    --tensor-parallel-size "$TP_SIZE" &
  SERVER_PID=$!

  # Wait until server is live
  wait_for_port "127.0.0.1" "$PORT"

  echo "Running vllm bench serve against http://127.0.0.1:${PORT} ..."
  vllm bench serve \
    --backend vllm \
    --host "127.0.0.1" \
    --port "$PORT" \
    --model "$MODEL" \
    --dataset-name random \
    --random-input-len "$BENCH_INPUT_LEN" \
    --random-output-len "$BENCH_OUTPUT_LEN" \
    --num-prompts "$BENCH_NUM_PROMPTS" \
    --max-concurrency "$BENCH_MAX_CONCURRENCY"

  echo "Benchmarks complete, stopping server..."
  kill "$SERVER_PID" || true
  wait "$SERVER_PID" || true

else
  echo "Unknown MODE '$MODE', executing raw command: $*"
  exec "$@"
fi
