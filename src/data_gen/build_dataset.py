"""Builds the DocGuard-VLM dataset from CORD-v2 receipts:

  train.jsonl          - clean OCR samples + genuine/tampered forgery samples
  test_clean.jsonl      - held-out clean eval set (same distribution as train)
  test_adversarial.jsonl - same held-out images under noise/blur/jpeg/rotation/
                            downscale perturbations (single + stacked), to
                            separately measure robustness beyond clean validation.

Images are copied to disk under data/processed/images/<split>/ so the produced
JSONL files only ever reference relative file paths (portable to Colab).
"""
import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import load_dataset
from tqdm import tqdm

from data_gen.cord_utils import extract_line_boxes
from data_gen.perturb import apply_random_perturbation, apply_stacked_perturbations
from data_gen.prompts import forgery_sample, ocr_sample, to_conversation
from data_gen.tamper import apply_random_tamper

RNG_SEED = 42


def extract_fields(gt_parse: dict) -> dict:
    items = []
    menu = gt_parse.get("menu") or []
    if isinstance(menu, dict):  # CORD gives a bare dict (not a list) for single-item receipts
        menu = [menu]
    for m in menu:
        if isinstance(m, dict):
            items.append(
                {
                    "name": m.get("nm", "unknown"),
                    "qty": m.get("cnt", "1"),
                    "price": m.get("price", "unknown"),
                }
            )
    sub_total = gt_parse.get("sub_total") or {}
    if isinstance(sub_total, list):  # occasionally a list of dicts too
        sub_total = sub_total[0] if sub_total else {}
    total = gt_parse.get("total") or {}
    if isinstance(total, list):
        total = total[0] if total else {}
    return {
        "items": items,
        "subtotal": sub_total.get("subtotal_price", "unknown"),
        "total": total.get("total_price", "unknown"),
    }


# Distinct per-split seed offsets. build_split reseeds at the top of every call, and
# test_clean/test_adversarial both draw 60-of-100 from equally-sized CORD splits -- with a
# shared seed, `random.shuffle(list(range(100)))` produces the *same* permutation both times,
# making label/tamper-type/instruction-phrasing identical position-by-position between the two
# splits (only the underlying photo differs). That defeats the point of a separate adversarial
# eval set. Offsetting the seed per split breaks that correlation.
SPLIT_SEED_OFFSETS = {"train": 0, "test_clean": 1000, "test_adversarial": 2000}


def build_split(hf_split_name, out_tag, n_images, images_dir, base_dir, donor_pool_size=40, adversarial=False, seed=RNG_SEED):
    split_offset = SPLIT_SEED_OFFSETS.get(out_tag, 0)
    random.seed(seed + split_offset)
    ds = load_dataset("naver-clova-ix/cord-v2", split=hf_split_name)
    n_images = min(n_images, len(ds))
    indices = list(range(len(ds)))
    random.shuffle(indices)
    indices = indices[:n_images]

    os.makedirs(images_dir, exist_ok=True)

    # donor pool for splice tampering: a handful of other images in this split
    donor_indices = random.sample([i for i in range(len(ds)) if i not in indices], min(donor_pool_size, len(ds) - n_images))
    donor_images = [ds[i]["image"].convert("RGB") for i in donor_indices]

    records = []
    for idx in tqdm(indices, desc=f"building {out_tag}"):
        ex = ds[idx]
        img = ex["image"].convert("RGB")
        gt_full = json.loads(ex["ground_truth"])
        gt_parse = gt_full.get("gt_parse", {})
        content_boxes = extract_line_boxes(gt_full)
        image_id = f"{out_tag}_{idx}"

        # --- OCR sample (always on a clean-content image; adversarial split
        # additionally perturbs pixels to test OCR robustness) ---
        ocr_img = img
        ocr_suffix = ""
        if adversarial:
            ocr_img, pert_name = apply_random_perturbation(img, seed=idx + split_offset)
            ocr_suffix = f"_adv-{pert_name}"
        ocr_path = os.path.join(images_dir, f"{image_id}_ocr{ocr_suffix}.png")
        ocr_img.save(ocr_path)
        fields = extract_fields(gt_parse)
        rec = ocr_sample(image_id, {"items": fields["items"], "total": fields["total"], "subtotal": fields["subtotal"]})
        rec["response"] = json.dumps(fields, ensure_ascii=False)
        records.append(to_conversation(rec, os.path.relpath(ocr_path, start=base_dir)))

        # --- Forgery sample: ~50/50 genuine vs tampered ---
        is_tampered = (idx % 2 == 0)
        if is_tampered:
            donor_pool = donor_images if donor_images else None
            result = apply_random_tamper(
                img, donor_pool=donor_pool, content_boxes=content_boxes, seed=idx + split_offset + 10_000
            )
            forg_img, bbox, ttype = result.image, result.bbox, result.tamper_type
        else:
            forg_img, bbox, ttype = img, None, None

        forg_suffix = ""
        if adversarial:
            forg_img, pert_names = apply_stacked_perturbations(forg_img, n=2, seed=idx + split_offset + 20_000)
            forg_suffix = "_adv-" + "-".join(pert_names)
        forg_path = os.path.join(images_dir, f"{image_id}_forgery{forg_suffix}.png")
        forg_img.save(forg_path)

        frec = forgery_sample(image_id, is_tampered, image_size=img.size, bbox=bbox, tamper_type=ttype)
        records.append(to_conversation(frec, os.path.relpath(forg_path, start=base_dir)))

    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="data/processed")
    ap.add_argument("--n-train", type=int, default=350)
    ap.add_argument("--n-test-clean", type=int, default=60)
    ap.add_argument("--n-test-adv", type=int, default=60)
    args = ap.parse_args()

    base = os.path.abspath(args.out_dir)
    splits = {
        "train": ("train", args.n_train, False),
        "test_clean": ("validation", args.n_test_clean, False),
        "test_adversarial": ("test", args.n_test_adv, True),
    }

    for tag, (hf_split, n, adversarial) in splits.items():
        images_dir = os.path.join(base, "images", tag)
        records = build_split(hf_split, tag, n, images_dir, base, adversarial=adversarial)
        out_path = os.path.join(base, f"{tag}.jsonl")
        with open(out_path, "w") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        n_ocr = sum(1 for r in records if r["task"] == "ocr")
        n_forg = sum(1 for r in records if r["task"] == "forgery")
        n_tampered = sum(1 for r in records if r["task"] == "forgery" and r["meta"]["label"] == "tampered")
        print(f"{tag}: {len(records)} records ({n_ocr} ocr, {n_forg} forgery [{n_tampered} tampered / {n_forg - n_tampered} authentic]) -> {out_path}")


if __name__ == "__main__":
    main()
