#!/bin/bash
# Run this on whatever machine serves medical-appointment's api.py --
# a cloud VM or your own Linux PC, doesn't matter. Verified against Ubuntu
# (bash + systemd); other Linux distros should work the same way as long as
# they also use systemd, which is most of them.
#
# Installs ollama, starts it in the background, and pulls/creates the models
# example.py needs. `llama3.2-medqa-ft` (the fine-tuned model, now the code's
# default OLLAMA_MODEL) is NOT in ollama's public registry -- `ollama pull`
# cannot fetch it. It must be created locally from a GGUF + Modelfile you
# transfer here yourself, e.g.:
#   scp /Users/mikelyu/Desktop/Nordic-AI-Cup-2026/medical-appointment/tools/.export/deploy_package/* \
#       user@this-vm:/path/to/medqa_deploy/
# Point MEDQA_PACKAGE_DIR below at wherever you put those files (Modelfile +
# llama3.2-medqa-ft.Q4_K_M.gguf). If you skip that transfer, this script
# still gets you a WORKING (if less accurate) server on the original
# llama3.2:3b -- see the fallback note at the end.

set -euo pipefail

MEDQA_PACKAGE_DIR="${MEDQA_PACKAGE_DIR:-./medqa_deploy}"

echo "=== 1/4: installing ollama ==="
curl -fsSL https://ollama.com/install.sh | sh

echo "=== 2/4: starting ollama serve in the background ==="
# On Ubuntu (and most systemd Linux), ollama's installer auto-starts it as a
# systemd service. That instance won't see CUDA_VISIBLE_DEVICES/
# OLLAMA_MAX_LOADED_MODELS exported below -- systemd services get their
# environment from systemd, not from this shell -- so it has to be stopped
# first and replaced with a manually-started process that does see them.
if systemctl is-active --quiet ollama 2>/dev/null; then
    echo "stopping the auto-started ollama systemd service (so our env vars actually apply)..."
    sudo systemctl stop ollama
    sudo systemctl disable ollama > /dev/null 2>&1 || true
fi

export CUDA_VISIBLE_DEVICES=0
export OLLAMA_MAX_LOADED_MODELS=2
nohup ollama serve > /tmp/ollama_serve.log 2>&1 &
disown
OLLAMA_PID=$!
echo "ollama serve started (pid $OLLAMA_PID), log at /tmp/ollama_serve.log"

echo "waiting for ollama to come up..."
for i in $(seq 1 30); do
    if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        echo "ollama ready after ${i}s"
        break
    fi
    sleep 1
    if [ "$i" -eq 30 ]; then
        echo "ollama did not come up after 30s -- check /tmp/ollama_serve.log" >&2
        exit 1
    fi
done

echo "=== 3/4: pulling models available from the public registry ==="
ollama pull nomic-embed-text
ollama pull llama3.2:3b   # safe fallback base -- keep even if the fine-tuned model below succeeds

echo "=== 4/4: creating the fine-tuned model (llama3.2-medqa-ft) from a local GGUF ==="
if [ -f "$MEDQA_PACKAGE_DIR/Modelfile" ] && [ -f "$MEDQA_PACKAGE_DIR/llama3.2-medqa-ft.Q4_K_M.gguf" ]; then
    ( cd "$MEDQA_PACKAGE_DIR" && ollama create llama3.2-medqa-ft -f Modelfile )
    echo "created llama3.2-medqa-ft"
else
    echo "!! $MEDQA_PACKAGE_DIR/Modelfile and/or llama3.2-medqa-ft.Q4_K_M.gguf not found."
    echo "!! Skipping the fine-tuned model. The server will still work on llama3.2:3b,"
    echo "!! but example.py's code default is now llama3.2-medqa-ft -- set"
    echo "!!   export OLLAMA_MODEL=llama3.2:3b"
    echo "!! in this VM's environment before starting api.py, or every LLM call will"
    echo "!! fail (missing model) and the pipeline floors at the 0.200 score."
fi

echo
echo "=== final check: what's actually registered ==="
curl -s http://localhost:11434/api/tags | python3 -m json.tool

echo
echo "Confirm the model(s) needed above are listed before starting api.py."
