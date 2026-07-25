"""Evaluation harness: scores a model (optionally + LoRA adapter) on
test_clean.jsonl and test_adversarial.jsonl for both tasks, and reports the
clean-vs-adversarial gap and (optionally) zero-shot-vs-fine-tuned deltas.

Runs on GPU (Colab) for real numbers, or CPU/MPS for a small --limit smoke test.
Uses plain transformers + peft (not unsloth) so it's portable to wherever you
want to score a saved adapter, independent of where it was trained.

Usage:
  python src/eval/evaluate.py \
    --data-root data/processed \
    --base-model unsloth/Qwen2-VL-2B-Instruct \
    --adapter outputs/qwen2vl-2b-docguard-lora \
    --out results/eval_results.json

  # zero-shot baseline (no --adapter):
  python src/eval/evaluate.py --data-root data/processed \
    --base-model unsloth/Qwen2-VL-2B-Instruct --out results/eval_baseline.json
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.metrics import forgery_accuracy, ocr_accuracy
from train.dataset_utils import load_jsonl, resolve_image

try:
    from transformers.utils import logging as hf_logging

    hf_logging.set_verbosity_error()
except ImportError:
    pass


def build_model(base_model, adapter_path, device, load_in_4bit=True):
    import torch
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

    dtype = torch.float16 if device != "cpu" else torch.float32

    # 4-bit by default: matches how the adapter was trained (QLoRA) and keeps both host RAM
    # and GPU memory low enough for free-tier Colab, which otherwise can OOM loading the full
    # fp16 checkpoint -- especially with training-session memory still resident.
    if load_in_4bit and device == "cuda":
        from transformers import BitsAndBytesConfig

        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            base_model, quantization_config=quant_config, device_map={"": 0}
        )
    else:
        model = Qwen2VLForConditionalGeneration.from_pretrained(base_model, torch_dtype=dtype)
        model = model.to(device)

    # Cap image resolution: several source photos are ~9MP (e.g. 2304x4096), and Qwen2-VL's
    # default max_pixels (~12.8MP) lets those through uncapped, producing thousands of visual
    # tokens and a single attention call that can request 6+ GiB on its own -- independent of
    # anything else resident on the GPU. 1024*28*28 (~0.8MP) keeps receipts legible while
    # keeping worst-case attention memory bounded on a T4.
    processor = AutoProcessor.from_pretrained(
        base_model, min_pixels=256 * 28 * 28, max_pixels=1024 * 28 * 28
    )

    if adapter_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_path)

    # Qwen2-VL's default generation_config sets max_length=32768 alongside our explicit
    # max_new_tokens, which triggers a redundant-config warning on every single generate()
    # call. Clear it so the two don't conflict (max_new_tokens alone is unambiguous).
    model.generation_config.max_length = None

    model.eval()
    return model, processor


def generate(model, processor, image, instruction, device, max_new_tokens=160):
    import torch

    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": instruction}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt").to(device)
    with torch.no_grad():
        out_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    gen_ids = out_ids[:, inputs["input_ids"].shape[1]:]
    return processor.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()


def run_split(model, processor, records, data_root, device, limit=None):
    if limit:
        records = records[:limit]
    ocr_preds, ocr_gold = [], []
    forg_preds, forg_gold = [], []
    per_example = []
    t0 = time.time()
    for i, r in enumerate(records):
        img = resolve_image(r, data_root)
        instruction = r["conversations"][0]["content"][1]["text"]
        gold_text = r["conversations"][1]["content"][0]["text"]
        pred = generate(model, processor, img, instruction, device)
        per_example.append({"id": r["id"], "task": r["task"], "pred": pred, "gold": gold_text})
        if r["task"] == "ocr":
            ocr_preds.append(pred)
            ocr_gold.append(gold_text)
        else:
            forg_preds.append(pred)
            forg_gold.append(r["meta"]["label"])
        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            print(f"  [{i + 1}/{len(records)}] {elapsed:.1f}s elapsed", flush=True)

    results = {}
    if ocr_preds:
        results["ocr"] = ocr_accuracy(ocr_preds, ocr_gold)
    if forg_preds:
        results["forgery"] = forgery_accuracy(forg_preds, forg_gold)
    results["n_examples"] = len(records)
    results["wall_time_sec"] = time.time() - t0
    return results, per_example


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data/processed")
    ap.add_argument("--base-model", default="unsloth/Qwen2-VL-2B-Instruct")
    ap.add_argument("--adapter", default=None, help="path to LoRA adapter; omit for zero-shot baseline")
    ap.add_argument("--device", default=None)
    ap.add_argument("--limit", type=int, default=None, help="cap examples per split, for a quick smoke test")
    ap.add_argument("--out", default="results/eval_results.json")
    ap.add_argument("--save-predictions", action="store_true")
    ap.add_argument("--no-4bit", action="store_true", help="load full fp16 instead of 4-bit (uses much more memory)")
    args = ap.parse_args()

    import torch

    device = args.device or ("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"device={device} base_model={args.base_model} adapter={args.adapter} load_in_4bit={not args.no_4bit}")

    model, processor = build_model(args.base_model, args.adapter, device, load_in_4bit=not args.no_4bit)

    splits = {
        "clean": load_jsonl(os.path.join(args.data_root, "test_clean.jsonl")),
        "adversarial": load_jsonl(os.path.join(args.data_root, "test_adversarial.jsonl")),
    }

    all_results = {"base_model": args.base_model, "adapter": args.adapter, "device": device}
    all_predictions = {}
    for split_name, records in splits.items():
        print(f"=== evaluating {split_name} ({len(records)} records) ===")
        results, per_example = run_split(model, processor, records, args.data_root, device, limit=args.limit)
        all_results[split_name] = results
        all_predictions[split_name] = per_example
        print(json.dumps(results, indent=2))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"wrote {args.out}")

    if args.save_predictions:
        pred_path = args.out.replace(".json", "_predictions.json")
        with open(pred_path, "w") as f:
            json.dump(all_predictions, f, indent=2)
        print(f"wrote {pred_path}")


if __name__ == "__main__":
    main()
