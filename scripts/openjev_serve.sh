#!/usr/bin/env bash
# Serve OpenJev FP8 on two 3090s and the decision helper in front of it.
# GPU 0 is kos1tv's policy server and is never used. GPU 1 stays free for the
# Kassow VLA. GPUs 2 and 3 are used only when no compute process is on them.
# The arm is not contacted.
set -euo pipefail

ROOT="${OPENJEV_ROOT:-/home/hme2lo/models/openjev}"
MODEL="${ROOT}/openjev-FP8"
VENV="${ROOT}/venv"
HELPER="${ROOT}/openjev-src/helper/shim.py"
PORT_VLLM="${OPENJEV_VLLM_PORT:-8000}"
PORT_HELPER="${OPENJEV_HELPER_PORT:-3000}"

if [[ ! -x "${VENV}/bin/vllm" ]]; then
  echo "missing ${VENV}/bin/vllm; the OpenJev environment is not installed" >&2
  exit 1
fi
if [[ ! -f "${MODEL}/config.json" ]]; then
  echo "missing ${MODEL}/config.json; the FP8 checkpoint is not downloaded" >&2
  exit 1
fi
if [[ ! -f "${HELPER}" ]]; then
  echo "missing ${HELPER}; download helper/shim.py from openjev/openjev" >&2
  exit 1
fi

python3 - <<'PY'
import subprocess
import sys

wanted = ("2", "3")
gpus = {}
for line in subprocess.check_output(
    ["nvidia-smi", "--query-gpu=index,uuid,memory.free", "--format=csv,noheader"],
    text=True,
).splitlines():
    index, uuid, free = [part.strip() for part in line.split(",")]
    gpus[index] = {"uuid": uuid, "free": free, "apps": []}
apps = subprocess.check_output(
    ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name", "--format=csv,noheader"],
    text=True,
).strip()
uuid_to_index = {info["uuid"]: index for index, info in gpus.items()}
if apps:
    for line in apps.splitlines():
        uuid, pid, name = [part.strip() for part in line.split(",", 2)]
        index = uuid_to_index.get(uuid)
        if index is not None:
            gpus[index]["apps"].append(f"{pid} {name}")
busy = [index for index in wanted if gpus.get(index, {}).get("apps")]
if busy or any(index not in gpus for index in wanted):
    print("Refusing to load OpenJev. GPUs 2 and 3 must both be free of compute processes.", file=sys.stderr)
    print("GPU 0 and GPU 1 are not candidates.", file=sys.stderr)
    for index in wanted:
        info = gpus.get(index)
        if info is None:
            print(f"  GPU {index}: missing", file=sys.stderr)
        elif info["apps"]:
            print(f"  GPU {index}: busy ({info['free']} free) {'; '.join(info['apps'])}", file=sys.stderr)
        else:
            print(f"  GPU {index}: free ({info['free']} free)", file=sys.stderr)
    sys.exit(1)
print("GPUs 2 and 3 are free; starting OpenJev there.")
PY

export CUDA_VISIBLE_DEVICES=2,3
# The venv ships nvcc and ninja. Without them vLLM dies while warming the sampler.
export CUDA_HOME="${VENV}/lib/python3.12/site-packages/nvidia/cu13"
export PATH="${VENV}/bin:${CUDA_HOME}/bin:${PATH}"
export VLLM_USE_FLASHINFER_SAMPLER=0
LOG_DIR="${ROOT}/logs"
mkdir -p "${LOG_DIR}"

"${VENV}/bin/vllm" serve "${MODEL}" \
  --host 127.0.0.1 \
  --served-model-name qwen \
  --port "${PORT_VLLM}" \
  --enable-prefix-caching \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90 \
  --limit-mm-per-prompt '{"image":7}' \
  --trust-remote-code \
  --max-num-seqs 4 \
  --max-logprobs 64 \
  --gdn-prefill-backend triton \
  --quantization fp8 \
  --tensor-parallel-size 2 \
  >"${LOG_DIR}/vllm.log" 2>&1 &
VLLM_PID=$!

cleanup() {
  if kill -0 "${VLLM_PID}" 2>/dev/null; then
    kill "${VLLM_PID}" 2>/dev/null || true
    wait "${VLLM_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

for _ in $(seq 1 180); do
  if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
    echo "vLLM exited before it was ready. See ${LOG_DIR}/vllm.log" >&2
    exit 1
  fi
  if curl -sf "http://127.0.0.1:${PORT_VLLM}/health" >/dev/null; then
    break
  fi
  sleep 10
done
if ! curl -sf "http://127.0.0.1:${PORT_VLLM}/health" >/dev/null; then
  echo "vLLM did not become healthy. See ${LOG_DIR}/vllm.log" >&2
  exit 1
fi

VLLM="http://127.0.0.1:${PORT_VLLM}/v1" \
TOKENIZER="${MODEL}" \
READOUT_T=0.85 \
READOUT_NOUL_T=1.829074 \
READOUT_NOUL_BIAS=0 \
READOUT_TARGETED=1 \
READOUT_INSTR_STYLE=pyrepr \
SHIM_STAGGER=1 \
  "${VENV}/bin/python" "${HELPER}" --host 127.0.0.1 --port "${PORT_HELPER}"
