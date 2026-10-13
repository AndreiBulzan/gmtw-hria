#!/usr/bin/env python3
"""
Merge a trained LoRA adapter checkpoint into its base model, producing a
standard HF model directory that vLLM / run_vllm_batch.py can load directly.

Pure CPU/GPU-light: loads base weights in bf16, applies the adapter, folds it
into the weights (merge_and_unload), saves. The tokenizer is saved alongside
from the BASE model so the merged dir is fully self-contained.

--scale interpolates between the base model and the repaired one. LoRA's update
is (alpha/r) * B @ A, which is linear in B, so multiplying every lora_B by
lambda multiplies the whole update by lambda:

    base + lambda * (repaired - base)

exactly. Doing it this way rather than subtracting two merged checkpoints
matters on this box: a full-checkpoint interpolation needs the base AND the
repaired model on disk at once, which is 30 GB of a 46 GB budget before the
output is written. Scaling the adapter needs the base and 1.8 GB of adapter.

Why sweep it at all: 3.3 measured the repair costing 8.35 points of academic
average, and two recipe levers failed to separate the two effects. Interpolation
asks whether they separate along the line between the two models instead -- if
the repair saturates at a smaller lambda than the damage does, there is a
mitigation with no extra training and no replay corpus. If they move together,
that says repair and damage are the same direction in weight space, which is a
stronger form of the dose curve's one-event finding.
"""

import json
import pathlib

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def _first_lora_module(model):
    """The first adapted module, used as the probe for the merge check."""
    for name, mod in model.named_modules():
        if getattr(mod, "lora_B", None) and len(mod.lora_B):
            return name, mod
    raise ValueError(
        "no module on this PeftModel carries a lora_B -- the adapter applied "
        "nothing, so the merge would silently reproduce the base model")


def scale_adapter(model, scale: float) -> int:
    """Multiply every lora_B by `scale`, in place. Returns the count.

    Raises on zero, because an adapter that scales nothing merges cleanly and
    serves cleanly and is simply the base model wearing a different name -- the
    same failure run_family.sh checks for after training.
    """
    n = 0
    for _, mod in model.named_modules():
        lora_b = getattr(mod, "lora_B", None)
        if not lora_b:
            continue
        for lin in lora_b.values():
            lin.weight.data.mul_(scale)
            n += 1
    if n == 0:
        raise ValueError("--scale was given but no lora_B was found to scale")
    return n


def merge_lora(base_model_path: str, adapter_path: str, output_dir: str,
               scale: float = 1.0) -> None:
    print(f"Loading base model {base_model_path}...")
    model = AutoModelForCausalLM.from_pretrained(base_model_path, torch_dtype=torch.bfloat16)

    print(f"Applying adapter {adapter_path}...")
    model = PeftModel.from_pretrained(model, adapter_path)

    # Probe one adapted weight so the merge can be checked against arithmetic we
    # did ourselves, rather than trusting that peft's scaling semantics are what
    # this script assumes they are. Costs one small matmul.
    probe_name, probe = _first_lora_module(model)
    adapter_name = next(iter(probe.lora_A))
    A = probe.lora_A[adapter_name].weight.detach().float()
    B = probe.lora_B[adapter_name].weight.detach().float()
    expected = probe.base_layer.weight.detach().float() + scale * probe.scaling[adapter_name] * (B @ A)
    flat_name = probe_name.removeprefix("base_model.model.")

    if scale != 1.0:
        n = scale_adapter(model, scale)
        print(f"Scaling adapter by lambda={scale} ({n} lora_B matrices)...")

    print("Merging adapter into base weights...")
    model = model.merge_and_unload()

    got = model.get_submodule(flat_name).weight.detach().float()
    err = (got - expected).abs().max().item()
    ref = expected.abs().max().item()
    print(f"  merge check on {flat_name}: max|merged - (base + lambda*delta)| = "
          f"{err:.3e} against a weight of max|W| = {ref:.3e}")
    if err > 1e-2 * max(ref, 1e-6):
        raise ValueError(
            f"merged weights are not base + {scale} * delta on {flat_name} "
            f"(max error {err:.3e}). peft's scaling semantics are not what this "
            f"script assumes; do not trust any checkpoint it wrote.")

    print(f"Saving merged model to {output_dir}...")
    model.save_pretrained(output_dir)
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    tokenizer.save_pretrained(output_dir)
    (pathlib.Path(output_dir) / "merge_info.json").write_text(
        json.dumps({"base_model": base_model_path, "adapter": adapter_path,
                    "scale": scale}, indent=2) + "\n", encoding="utf-8")
    print("Done.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Merge a LoRA adapter checkpoint into its base model")
    parser.add_argument("base_model_path", help="HF ID or path of the base model the adapter was trained on")
    parser.add_argument("adapter_path", help="Path to the checkpoint-* directory containing the adapter")
    parser.add_argument("output_dir", help="Where to write the merged, vLLM-loadable model")
    parser.add_argument("--scale", type=float, default=1.0,
                        help="lambda in base + lambda*(repaired - base). 1.0 (default) is "
                             "the ordinary merge and leaves the adapter untouched.")

    args = parser.parse_args()
    merge_lora(args.base_model_path, args.adapter_path, args.output_dir, args.scale)
