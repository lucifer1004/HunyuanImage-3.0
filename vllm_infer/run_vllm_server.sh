#!/bin/bash
set -e

model_path="$1"

# Check if model path is provided
if [ -z "$model_path" ]; then
    echo "Error: Please provide model path as the first argument"
    echo "Usage: $0 /path/to/model"
    exit 1
fi

export VLLM_ENABLE_HUNYUAN_IMAGE3_TASK="${VLLM_ENABLE_HUNYUAN_IMAGE3_TASK:-1}"
TP_SIZE="${TP_SIZE:-8}"

############################################
# 3. Start vLLM service
############################################
# Uncomment the next line for nsys profiling
# nsys launch --trace-fork-before-exec true --session test -t cuda,cublas,cudnn,nvtx --cuda-graph-trace=node \
vllm serve "$model_path" \
    --trust-remote-code \
    --served-model-name vllm_hunyuan_image3 \
    --max-model-len 10000 \
    --gpu-memory-utilization 0.6 \
    --no-enable-prefix-caching \
    --no-enable-chunked-prefill \
    --max-num-batched-tokens 10000 \
    --max-num-seqs 1 \
    --enforce-eager \
    --trust-request-chat-template \
    -tp "$TP_SIZE"
