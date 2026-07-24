# DocGuard-VLM

LoRA fine-tuning of a small open-weight vision-language model (**Qwen2-VL-2B-Instruct**) for two
document-intelligence tasks on a single shared adapter:

1. **Structured field extraction (OCR)** — read line items, subtotal, and total off a receipt and
   return them as JSON.
2. **Forgery detection** — decide whether a document image is authentic or has been digitally
   tampered with (copy-move, splicing, or field/value editing), and explain what looks altered
   and roughly where.

Both tasks are trained together via multi-task instruction tuning, because a real
identity/document-verification intake pipeline needs both capabilities on the same image, not two
separate models.

Every model is scored on **two** held-out sets: a clean test set, and an adversarial test set where
the same images go through noise, blur, JPEG re-compression, rotation/skew, and downscaling —
because a model that only works on centered, clean scans doesn't generalize to real-world photo
uploads. The eval harness reports zero-shot baseline vs. fine-tuned, and clean vs. adversarial, so
the robustness gap is visible rather than hidden inside one aggregate accuracy number.

## Why this dataset, not a real ID/document forgery dataset

Real forged government ID scans aren't available to train on for obvious legal/ethical reasons.
Following the standard approach in the document-forgery-detection literature (e.g. DocTamper),
this project starts from a public, permissively licensed receipt dataset —
[CORD-v2](https://huggingface.co/datasets/naver-clova-ix/cord-v2) (Consolidated Receipt Dataset,
800 train / 100 val / 100 test images with word-level OCR ground truth) — and synthetically
tampers a subset of it. Receipts are a reasonable stand-in: they're real photographed documents
with printed text fields, line items, and totals, i.e. structurally similar to the kind of document
an identity/finance pipeline has to read and verify.

## Synthetic tampering pipeline (`src/data_gen/`)

`tamper.py` implements three tamper types, applied to a random **text line** from CORD's own
word-level bounding boxes (not a random patch of the image — an earlier version of this picked a
uniformly random region and would occasionally "tamper" blank background instead of the document,
which is a trivial and unrealistic training/eval signal; `cord_utils.py` recovers per-line boxes
from CORD's raw `valid_line` annotations to fix that):

- **copy_move** — a text-line region is duplicated and pasted elsewhere on the same document.
- **splice** — a region from a *different* document (donor) is pasted in.
- **patch_overlay** — a field is blanked to the local background color and overwritten with
  different digits/text, simulating an edited total/date/ID number.

`perturb.py` implements the separate adversarial/robustness perturbations used only for the
`test_adversarial` split: JPEG re-compression, Gaussian noise, Gaussian blur, rotation/skew, and
downscale-upscale (single and stacked).

`build_dataset.py` assembles:

| split | images | OCR samples | forgery samples (tampered / authentic) |
|---|---|---|---|
| `train` | 350 | 350 | 350 (~50/50) |
| `test_clean` | 60 | 60 | 60 (~50/50) |
| `test_adversarial` | 60 (perturbed) | 60 | 60 (~50/50) |

Output: `data/processed/{train,test_clean,test_adversarial}.jsonl`, one instruction/response
conversation per line, images under `data/processed/images/`.

## Fine-tuning (`notebooks/train_colab.ipynb`)

QLoRA fine-tune of `unsloth/Qwen2-VL-2B-Instruct` via [unsloth](https://github.com/unslothai/unsloth)
on a free Colab **T4** GPU — 4-bit base weights, rank-16 LoRA over vision + language + attention +
MLP layers. Runs in a few hours on a dataset this size. See the notebook for the exact
`SFTConfig`/`SFTTrainer` setup.

## Evaluation (`src/eval/`)

```
python src/eval/evaluate.py \
  --data-root data/processed \
  --base-model unsloth/Qwen2-VL-2B-Instruct \
  --adapter outputs/qwen2vl-2b-docguard-lora \
  --out results/eval_finetuned.json

# zero-shot baseline for comparison (omit --adapter):
python src/eval/evaluate.py --data-root data/processed \
  --base-model unsloth/Qwen2-VL-2B-Instruct --out results/eval_baseline.json
```

Metrics (`src/eval/metrics.py`, unit-testable without a GPU):

- **OCR**: JSON-validity rate (did the model emit parseable structured output at all — itself a
  real pipeline requirement) and total-amount exact-match rate.
- **Forgery**: accuracy, precision/recall/F1 on the "tampered" class, and an "unknown" rate for
  responses that don't commit to a verdict. Verdict parsing is negation-aware — naive substring
  matching on "tamper" misfires on phrases like *"no signs of tampering"*; see `parse_verdict`.

## Live demo (`src/serve/app.py`)

A small Gradio app that reuses the exact model-loading/generation code from the eval harness, so
the demo and the scored numbers can't silently diverge:

```
pip install gradio
python src/serve/app.py --adapter outputs/qwen2vl-2b-docguard-lora
```

Upload a document image, pick OCR or forgery-detection (or write a custom instruction), see the
model's output. Omit `--adapter` to demo the zero-shot base model for comparison.

## Results

_Fill in after running `evaluate.py` on both the zero-shot baseline and the fine-tuned adapter:_

| model | split | OCR JSON-valid | OCR total-match | Forgery accuracy | Forgery F1 (tampered) |
|---|---|---|---|---|---|
| Qwen2-VL-2B zero-shot | clean | | | | |
| Qwen2-VL-2B zero-shot | adversarial | | | | |
| Qwen2-VL-2B + LoRA (ours) | clean | | | | |
| Qwen2-VL-2B + LoRA (ours) | adversarial | | | | |

## Limitations

- Tampering is synthetic, not real forged documents — it teaches the *classes* of manipulation
  (copy-move, splicing, field edits) but the model's performance on real forged IDs is unverified.
- CORD is receipts, not identity documents; layout/field types differ from IDs/passports.
- Small dataset by LLM standards (470 unique source images) — sized to fit a one-day, one-GPU
  training budget, not to be state-of-the-art.
- `patch_overlay`'s fake replacement text is drawn from a small fixed vocabulary, so the model may
  partly learn to recognize that specific pattern rather than field-tampering in general.

## Repo layout

```
src/data_gen/   synthetic tampering, adversarial perturbations, dataset builder
src/train/      dataset loading helpers shared by the Colab notebook and eval
src/eval/       metrics + evaluation harness
src/serve/      Gradio live demo (reuses eval's model-loading/generation code)
notebooks/      Colab fine-tuning notebook
data/processed/ generated JSONL + images (not committed; regenerate with build_dataset.py)
results/        eval_*.json outputs
```

## Reproduce from scratch

```
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt   # data-gen deps only; see requirements.txt for GPU deps
PYTHONPATH=src python src/data_gen/build_dataset.py --out-dir data/processed \
  --n-train 350 --n-test-clean 60 --n-test-adv 60
# then run notebooks/train_colab.ipynb on Colab (T4 GPU), and
# src/eval/evaluate.py to score baseline vs. fine-tuned.
```
