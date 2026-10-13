"""LoRA SFT launcher for the Ro-SO smoke run.

Plain supervised finetuning on the assembled mix (build_mix output): the
model sees the full instruction (Romanian/English wrapper + schema +
document) as the prompt and learns to emit the JSON output. Loss is on the
completion only (TRL's prompt-completion format, completion_only_loss).

Hyperparameters and base model come from the training yaml (single source
of truth); CLI flags override for experiments.

Usage (from the repo root, with no vLLM server holding the card):
  python datagen/training/train_lora.py \
      --config datagen/training/lora_rollama31_8b_v2.yaml \
      --data datagen/data/sft/train_mix_4k_v2.jsonl --output runs/x
  python datagen/training/train_lora.py ... --max-steps 5   # wiring smoke test

After training: merge the adapter for vLLM with datagen/scripts/merge_lora.py
-- or let datagen/scripts/run_family.sh do the whole
train -> merge -> generate -> score chain.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import pathlib
import re

import yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent

# The module names every config has always used. Kept only as the default for
# a config that omits `lora.target_modules`; the yaml is what governs.
DEFAULT_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"]


def target_module_regex(names: list[str]) -> str:
    """Anchored regex over the module names the yaml asks for.

    peft accepts a bare list, and on wrapped architectures a bare list can match
    nothing at all -- producing an adapter over zero modules that trains, merges
    and serves while reproducing the base model exactly (see
    scripts/train_gkd.py). The anchored regex fails loudly instead.

    Built from the yaml rather than hardcoded. It was hardcoded until now while
    every config also carried a `lora.target_modules` list that was never read,
    so the configs described one thing and the runs did another. Nothing about
    the released adapters changes: for the module list every config uses, this
    returns the exact pattern that produced them.
    """
    if not names:
        raise ValueError("lora.target_modules is empty; an adapter over no "
                         "modules trains and serves while changing nothing")
    return (r"^model\.layers\.\d+\.(self_attn|mlp)\.("
            + "|".join(re.escape(n) for n in names) + r")$")


def load_mix_records(path: str | pathlib.Path) -> list[dict]:
    """Mix JSONL -> TRL prompt-completion conversational records.

    The completion is the compact JSON serialization of the gold output —
    deterministic, ensure_ascii=False so Romanian text stays readable and
    tokenizes naturally.

    The prose prefix ahead of that JSON comes from `explanation` (datagen/,
    every track) or the older `plan` (ro-so/pipelines, Track C only). This is
    the ONLY place the training target is built, so the prefix has to be
    applied here.

    A mix where NOTHING carries a prefix is refused. That is the failure this
    guard exists for: the datagen corpus renames `plan` to `explanation`, and
    reading only the old key would silently train bare JSON on every record —
    exactly the answer shape the corpus was regenerated to eliminate, and
    invisible until evaluation came back flat. A prefix on no record at all is
    always a wiring mistake now, never a deliberate mix.

    REPLAY records (`kind == "replay"`) are the one deliberate exception. They
    are general-instruction rehearsal: a prompt the base checkpoint was tuned
    on, answered by that same checkpoint, carried verbatim as prose with no
    schema, no JSON and no prefix. They exist to be rehearsed, not learned, so
    serializing them as JSON would be wrong and requiring a prefix would be
    meaningless. They are excluded from the prefix count rather than exempted
    from the guard: a mix of structured records that has lost its prefixes is
    still refused, whether or not replay is present."""
    records, n_prefixed, n_structured = [], 0, 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("kind") == "replay":
                completion = r["output"]
                if not isinstance(completion, str) or not completion.strip():
                    raise ValueError(
                        f"{path}: replay record {r.get('id')!r} carries "
                        f"{type(r['output']).__name__} output; replay is prose "
                        f"the model already produces, so it must be a non-empty "
                        f"string and must not be serialized as JSON.")
                records.append({
                    "prompt": [{"role": "user", "content": r["instruction"]}],
                    "completion": [{"role": "assistant", "content": completion}],
                })
                continue
            n_structured += 1
            body = json.dumps(r["output"], ensure_ascii=False)
            prefix = r.get("explanation") or r.get("plan")
            if prefix and prefix.strip():
                completion = f"{prefix.strip()}\n\n{body}"
                n_prefixed += 1
            else:
                completion = body
            records.append({
                "prompt": [{"role": "user", "content": r["instruction"]}],
                "completion": [{"role": "assistant", "content": completion}],
            })
    if not records:
        raise ValueError(f"no records in {path}")
    if n_structured and n_prefixed == 0:
        raise ValueError(
            f"{path}: not one record carries `explanation` or `plan`, so every "
            f"completion would be bare JSON. The benchmark asks for an "
            f"explanation first and the JSON block last; training this mix "
            f"would teach the wrong answer shape and look fine until "
            f"evaluation. Check the field name in the mix.")
    return records


def train(args):
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    # cuDNN SDPA backward produces NaN for some left-padded batch shapes on
    # the H200 — same failure documented in scripts/train_gkd.py; keep the
    # workaround identical here.
    torch.backends.cuda.enable_cudnn_sdp(False)

    cfg = yaml.safe_load(pathlib.Path(args.config).read_text(encoding="utf-8"))
    base_model = args.base_model or cfg["base_model"]
    lora_cfg, train_cfg = cfg["lora"], cfg["train"]

    records = load_mix_records(args.data)
    dataset = Dataset.from_list(records)

    # Held-out slice, built by build_mix --val-total from records NOT sampled
    # into the training mix, so there is no overlap and the training file stays
    # byte-identical to one built without it. Without this, overfitting can only
    # be inferred from training loss — which is exactly the ambiguity that left
    # "8k is worse data" and "1000 steps overfits" indistinguishable in it5.
    val_dataset = None
    if args.val_data:
        val_dataset = Dataset.from_list(load_mix_records(args.val_data))

    # Provenance banner + mlflow params. it4 silently trained on the stale 1k
    # mix because --data defaulted to it and the runbook omitted the flag; the
    # only evidence afterwards was a step count (126 = 1k at batch 16, not 4k).
    # Make the inputs impossible to miss, and recorded rather than inferred.
    data_path = pathlib.Path(args.data)
    data_sha = hashlib.sha256(data_path.read_bytes()).hexdigest()
    import trl as _trl, transformers as _tf, datasets as _ds
    stack = f"trl={_trl.__version__} transformers={_tf.__version__} datasets={_ds.__version__}"
    # Counted from the source records, not sniffed out of the serialized
    # completion — a substring test would miscount any gold whose own JSON
    # happens to contain the separator.
    n_planned = sum(
        1 for line in data_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and (json.loads(line).get("explanation")
                             or json.loads(line).get("plan")))
    print("=" * 68)
    print(f"  data       : {data_path}")
    print(f"  records    : {len(dataset)}")
    print(f"  sha256     : {data_sha}")
    # The operator's one chance to catch a wiring mistake before spending a
    # GPU-day: anything short of full coverage means some completions are bare
    # JSON, which is the answer shape this corpus exists to eliminate.
    shortfall = ("" if n_planned == len(dataset)
                 else "  <-- CHECK: the rest train as bare JSON")
    print(f"  w/ prose   : {n_planned} / {len(dataset)}{shortfall}")
    if args.val_data:
        val_path = pathlib.Path(args.val_data)
        val_sha = hashlib.sha256(val_path.read_bytes()).hexdigest()
        print(f"  val data   : {val_path}")
        print(f"  val records: {len(val_dataset)}")
        print(f"  val sha256 : {val_sha}")
    else:
        val_sha = None
        print("  val data   : (none — overfitting will not be observable)")
    print(f"  base model : {base_model}")
    print(f"  stack      : {stack}")
    print("=" * 68)

    try:
        import mlflow
        mlflow.log_params({
            "data_path": str(data_path), "data_sha256": data_sha,
            "data_records": len(dataset), "base_model": base_model,
            "data_records_with_plan": n_planned,
            "val_data_path": str(args.val_data) if args.val_data else None,
            "val_data_sha256": val_sha,
            "val_records": len(val_dataset) if val_dataset is not None else 0,
            "trl_version": _trl.__version__,
            "transformers_version_runtime": _tf.__version__,
            "datasets_version": _ds.__version__,
        })
    except Exception as e:      # never let telemetry block a training run
        print(f"(mlflow param logging skipped: {e})")

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        # Llama tokenizers ship without one (see train_gkd.py)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Gemma-2 applies logit softcapping, which HF only computes correctly under
    # the eager attention path -- SDPA silently skips it and the loss is subtly
    # wrong rather than loudly broken. Set `attn_implementation: eager` in the
    # yaml for that family. Absent (Llama, Mistral) leaves the default, so
    # existing configs behave exactly as before.
    load_kwargs = {"torch_dtype": torch.bfloat16, "device_map": "auto"}
    attn_impl = cfg.get("model", {}).get("attn_implementation")
    if attn_impl:
        load_kwargs["attn_implementation"] = attn_impl
        print(f"  attention   : {attn_impl}")
    model = AutoModelForCausalLM.from_pretrained(base_model, **load_kwargs)
    target_modules = lora_cfg.get("target_modules") or DEFAULT_TARGET_MODULES
    print(f"  lora targets: {', '.join(target_modules)}")
    model = get_peft_model(model, LoraConfig(
        r=lora_cfg["r"], lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        target_modules=target_module_regex(target_modules),
        task_type="CAUSAL_LM"))
    model.print_trainable_parameters()

    sft_args = SFTConfig(
        output_dir=args.output,
        run_name=args.run_name,
        completion_only_loss=True,          # yaml: train_on_inputs false
        max_length=cfg["data"]["max_seq_len"],
        per_device_train_batch_size=train_cfg["per_device_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation"],
        learning_rate=float(train_cfg["lr"]),
        lr_scheduler_type=train_cfg["scheduler"],
        warmup_ratio=train_cfg["warmup_ratio"],
        num_train_epochs=train_cfg["epochs"],
        max_steps=args.max_steps if args.max_steps is not None else -1,
        seed=args.seed if args.seed is not None else train_cfg["seed"],
        bf16=train_cfg["bf16"],
        logging_steps=10,
        save_steps=50,
        # Tracking only -- it does not touch the weights. Hardcoded to "mlflow"
        # this was a silent hard dependency: transformers instantiates the
        # callback and raises if the package is missing, so a clean environment
        # failed at Trainer construction rather than merely losing its logging.
        # The param logging above already degrades gracefully; this now matches.
        report_to="mlflow" if importlib.util.find_spec("mlflow") else "none",
        **({"eval_strategy": "steps", "eval_steps": 50}
           if val_dataset is not None else {}),
    )
    trainer = SFTTrainer(model=model, args=sft_args,
                         processing_class=tokenizer, train_dataset=dataset,
                         eval_dataset=val_dataset)
    trainer.train(resume_from_checkpoint=args.resume if args.resume else None)

    # periodic checkpoints land on save_steps multiples only; always save the
    # end-of-training adapter explicitly so the merge never grabs a stale step
    final_dir = pathlib.Path(args.output) / "final"
    trainer.save_model(str(final_dir))
    print(f"final adapter -> {final_dir}")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Ro-SO LoRA SFT")
    ap.add_argument("--config", default=str(ROOT / "training/lora_rollama31_8b_v2.yaml"))
    # REQUIRED, deliberately. This used to default to train_mix_1k.jsonl, and
    # it4 trained on that stale file for a full run because the flag was left
    # off -- silently, since a wrong-but-valid mix looks exactly like a right
    # one. A missing path must stop the run, not pick one.
    ap.add_argument("--data", required=True,
                    help="assembled mix from build_mix (e.g. "
                         "data/sft/train_mix_4k.jsonl) — no default on purpose")
    ap.add_argument("--val-data", default=None,
                    help="held-out slice from build_mix --val-total (e.g. "
                         "data/sft/train_mix_4k.val.jsonl). Without it, "
                         "overfitting can only be inferred from training loss")
    ap.add_argument("--output", default=str(ROOT.parent / "runs/roso_smoke"))
    ap.add_argument("--base-model", default=None,
                    help="override the yaml base_model (e.g. the plain "
                         "Instruct checkpoint for the secondary run)")
    ap.add_argument("--run-name", default=None)
    # Overrides train.seed in the yaml. Every result so far is N=1 at seed 0,
    # and the differences we care about (it6 - it4 on C_) are ~3 points, so we
    # cannot tell them from run-to-run noise without replicates.
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--max-steps", type=int, default=None,
                    help="cap optimizer steps (wiring smoke test)")
    ap.add_argument("--resume", action="store_true",
                    help="resume from latest checkpoint-* in --output")
    args = ap.parse_args(argv)
    train(args)


if __name__ == "__main__":
    main()
