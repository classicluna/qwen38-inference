#!/usr/bin/env bash
# Peak VRAM per (ctx, MTP) with the vision projector loaded, after a real 24k-token prompt + 128 tokens.
set -uo pipefail
cd /home/evank/dev/inference
B=/home/evank/llama-prism-kern/build-kern/bin
export LD_LIBRARY_PATH=/home/evank/rocm-runtime/opt/rocm/lib
V=/sys/class/drm/card1/device/mem_info_vram_used
P=$(python3 -c 'import json;print(json.dumps({"prompt":open("corpus.txt").read()[100000:195000],"n_predict":128,"temperature":0,"cache_prompt":False}))')
for spec in "$@"; do  # ctx:mtp(0|1)
  ctx=${spec%%:*}; mtp=${spec#*:}
  pgrep -af 'llama-server|llama-bench' && { echo "GPU busy"; exit 1; }
  extra=(); [ "$mtp" = 1 ] && extra=(--spec-draft-model models/MTP/mtp-Qwen3.8-27B-Q4_0.gguf --spec-type draft-mtp --spec-draft-n-max 3)
  $B/llama-server -m models/Ternary-Bonsai-2-27B-PTQ1_0.gguf --mmproj models/Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf \
    -ngl 99 -c "$ctx" -fa on -ctk f16 -ctv f16 -ub 256 -b 2048 --parallel 1 --port 8090 "${extra[@]}" \
    > results/mtp/fit-$ctx-$mtp.log 2>&1 &
  S=$!; peak=0; ok=0
  for i in $(seq 150); do curl -sf localhost:8090/health >/dev/null && { ok=1; break; }; kill -0 $S 2>/dev/null || break; sleep 2; done
  if [ $ok = 1 ]; then
    curl -s localhost:8090/completion -d "$P" >/dev/null &
    C=$!
    while kill -0 $C 2>/dev/null; do u=$(cat $V); [ $u -gt $peak ] && peak=$u; sleep 0.1; done
  fi
  gtt=$(cat /sys/class/drm/card1/device/mem_info_gtt_used)
  kill $S 2>/dev/null; wait $S 2>/dev/null
  awk -v c=$ctx -v m=$mtp -v p=$peak -v ok=$ok -v g=$gtt 'BEGIN{printf "ctx=%d mtp=%d loaded=%d peak=%.2f GiB free=%.2f GiB gtt=%.2f GiB\n",c,m,ok,p/2^30,15.98-p/2^30,g/2^30}'
  for i in $(seq 60); do [ $(cat $V) -lt 2000000000 ] && break; sleep 1; done
done
