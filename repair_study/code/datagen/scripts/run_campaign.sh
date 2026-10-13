#!/usr/bin/env bash
# One lane of the 3.3 cross-family campaign.
#
# The bracket found that the 8.47-point cost is NOT a property of the method:
# RoLlama2-7B lost 2.0%, RoGemma-7B-DPO lost nothing (+0.9%), and RoLlama3.1-8B-DPO
# lost 15.8%. Ordering families by how far the repair had to move them does not
# predict it -- the largest update produced the least damage. So the question is
# no longer "how big is the cost" but "which checkpoints pay it", and that needs
# every family, not a bracket.
#
#   run_campaign.sh <gpu> <lane-name> <job> [job ...]
#   job = BASE_ID,PREFIX,ROW,BATCH,ATTN,SEEDS   (SEEDS like "1 2" or "0 1 2")
#
# ATTN IS PER FAMILY TOO, and it is a correctness flag rather than a capacity
# one. Gemma-2 applies attention and final logit softcapping; HF computes it
# only on the eager path and SDPA silently skips it, so a Gemma-2 row scored
# without eager is quietly wrong. Verified from the configs: both RoGemma2
# checkpoints carry attn_logit_softcapping=50.0 / final=30.0; Gemma-1, Llama
# and Mistral carry none, so they pass ATTN empty and are unchanged.
#
# BATCH IS PER FAMILY. Gemma carries a 256k vocabulary, so batch 64 asks for a
# 38 GiB logits tensor and the card refuses; 16 for Gemma, 64 for the rest, and
# both arms of a pair share one value so nothing inside a comparison varies.
#
# DISK WAITS, IT DOES NOT FAIL. A merged 8B checkpoint is ~15-18 GB and several
# lanes run at once, so a lane that cannot merge yet sleeps until another lane
# deletes its checkpoint. Failing instead would lose the lane's whole queue for a
# transient condition.
#
# AND MERGES TAKE A LOCK. Checking free space without one is a race every lane
# wins: four lanes each saw 45 GB, each concluded there was room, and together
# they wrote 61 GB onto a volume with 45 -- taking it to 4.8 GB and killing a
# lane mid-ladder. The check is only meaningful while nothing else can start a
# merge, so it happens under flock and the merge finishes before the lock drops.
set -uo pipefail
GPU="$1"; LANE="$2"; shift 2
. "$HOME/setup_kd.sh"
cd "$HOME/kd-hria"
export CUDA_VISIBLE_DEVICES="$GPU"
RET="$HOME/kd-hria/datagen/retention/results"
MIN_FREE_GB="${MIN_FREE_GB:-24}"
free_gb () { df --output=avail -BG / | tail -1 | tr -dc '0-9'; }

wait_for_disk () {
  local waited=0
  while [ "$(free_gb)" -lt "$MIN_FREE_GB" ]; do
    [ "$waited" -ge 7200 ] && { echo "[$LANE] disk never freed after 2h, giving up" >&2; return 1; }
    echo "[$LANE] $(free_gb) GB free, waiting for room to merge..."
    sleep 300; waited=$((waited+300))
  done
  return 0
}

rungs () { ls -d "$RET/$1"/fs*/ 2>/dev/null | wc -l; }

for JOB in "$@"; do
  IFS=, read -r BASE_ID PREFIX ROW BATCH ATTN SEEDS <<< "$JOB"
  echo "=========== [$LANE] family $ROW (batch $BATCH, attn ${ATTN:-default}, seeds: $SEEDS) ==========="

  # The base arm needs no merge, but the FIRST touch of a family downloads
  # ~14-18 GB of weights, and an unguarded download fills the volume exactly as
  # an unguarded merge does. So the fetch happens under the same lock, before
  # the ladder, and the lock drops before the ladder's 90 minutes begin.
  exec 9>"$HOME/.campaign_merge.lock"
  flock 9
  if wait_for_disk; then
    echo "[$LANE] fetching $BASE_ID if absent ($(free_gb) GB free, holding lock)"
    python - "$BASE_ID" <<'PYFETCH'
import sys
from huggingface_hub import snapshot_download
snapshot_download(sys.argv[1], allow_patterns=["*.json","*.safetensors","tokenizer*","*.model","*.jinja"])
PYFETCH
  fi
  flock -u 9; exec 9>&-

  if [ "$(rungs "base_${ROW}_lim")" -eq 6 ]; then
    echo "[$LANE] base_${ROW}_lim already complete, skipping"
  else
    echo "[$LANE] base_${ROW}_lim"
    env SETUP="$HOME/rollm_env.sh" LIMIT=2000 BATCH="$BATCH" ${ATTN:+ATTN="$ATTN"} \
      bash datagen/retention/run_rollm_eval.sh "$BASE_ID" "base_${ROW}_lim" \
      || echo "[$LANE] BASE FAILED: $ROW"
  fi

  for SEED in $SEEDS; do
    NAME="${PREFIX}_s${SEED}"
    if [ "$(rungs "${NAME}_lim")" -eq 6 ]; then
      echo "[$LANE] ${NAME}_lim already complete, skipping"; continue
    fi
    ADAPTER="$HOME/adapters/$NAME/final"
    [ -d "$ADAPTER" ] || { echo "[$LANE] no adapter $ADAPTER, skipping" >&2; continue; }
    MERGED="$HOME/merged_models/$NAME"
    if [ ! -f "$MERGED/config.json" ]; then
      exec 9>"$HOME/.campaign_merge.lock"
      flock 9
      if wait_for_disk; then
        echo "[$LANE] merging $NAME ($(free_gb) GB free, holding merge lock)"
        python datagen/scripts/merge_lora.py "$BASE_ID" "$ADAPTER" "$MERGED" \
          && python datagen/scripts/align_stop_tokens.py "$MERGED" \
          || { echo "[$LANE] MERGE FAILED: $NAME" >&2; rm -rf "$MERGED"; }
      fi
      flock -u 9; exec 9>&-
      [ -f "$MERGED/config.json" ] || { echo "[$LANE] no checkpoint for $NAME, skipping" >&2; continue; }
    fi
    echo "[$LANE] ${NAME}_lim"
    env SETUP="$HOME/rollm_env.sh" LIMIT=2000 BATCH="$BATCH" TOKENIZER="$BASE_ID" ${ATTN:+ATTN="$ATTN"} \
      bash datagen/retention/run_rollm_eval.sh "$MERGED" "${NAME}_lim" \
      || echo "[$LANE] LADDER FAILED: $NAME"
    rm -rf "$MERGED"
    echo "[$LANE] done $NAME, $(free_gb) GB free"
  done
  # The base weights are ~14-18 GB each and eight families will not fit at once,
  # so a finished family gives its cache back. Re-downloading is bandwidth, not
  # work, and PURGE=0 keeps it for a family that is being revisited.
  if [ "${PURGE:-1}" = "1" ]; then
    python - "$BASE_ID" <<'PYPURGE'
import sys
from huggingface_hub import scan_cache_dir
c = scan_cache_dir()
revs = [r.commit_hash for repo in c.repos if repo.repo_id == sys.argv[1]
        for r in repo.revisions]
if revs:
    s = c.delete_revisions(*revs)
    print(f"  purged {sys.argv[1]}: {s.expected_freed_size_str}")
    s.execute()
PYPURGE
  fi
done
echo "[$LANE] LANE COMPLETE"
