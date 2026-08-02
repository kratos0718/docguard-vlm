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

Qwen2-VL-2B-Instruct, zero-shot vs. LoRA fine-tuned (rank 16, 2 epochs, 350 train examples),
scored on 120 held-out examples per split (60 OCR + 60 forgery):

| model | split | OCR JSON-valid | OCR total-match | Forgery accuracy | Forgery F1 (tampered) | Forgery recall (tampered) |
|---|---|---|---|---|---|---|
| Qwen2-VL-2B zero-shot | clean | 25.0% | 33.3% | 49.2% | 0.21 | 11.8% |
| Qwen2-VL-2B zero-shot | adversarial | 13.3% | 18.3% | 43.9% | 0.16 | 8.8% |
| Qwen2-VL-2B + LoRA (ours) | clean | 91.7% | 90.0% | 55.0% | 0.60 | 57.1% |
| Qwen2-VL-2B + LoRA (ours) | adversarial | 88.3% | 78.3% | 55.0% | 0.60 | 57.1% |

**OCR field extraction** is the clean win: the zero-shot model rarely even emits valid JSON in this
schema (25%/13%), while the fine-tuned adapter hits 91.7%/88.3% JSON-validity and 90.0%/78.3%
total-amount match — a ~3-4x improvement that holds up (with the expected degradation) under
adversarial perturbation.

**Forgery detection** improves substantially too — F1 roughly triples (0.21→0.60, 0.16→0.60) and
recall on the "tampered" class goes from essentially not-detecting (11.8%/8.8%) to actually
flagging most tampered documents (57.1%/57.1%). The zero-shot model's near-perfect precision with
terrible recall is a classic "always guess authentic" failure mode, not real forgery detection.

**A caveat worth stating plainly, not glossing over:** the fine-tuned model's forgery confusion
matrix is *bit-for-bit identical* between the clean and adversarial splits (tp=20, fp=12, fn=15,
tn=13 in both) — despite the two splits using genuinely different, adversarially-perturbed images.
The zero-shot baseline's confusion matrix, by contrast, *does* shift between splits (tp 4→3, tn
25→22), as you'd expect from a model actually responding to pixel differences. This is independent
evidence for a real bug I found and fixed in the dataset builder: `build_split()` reseeded Python's
RNG to the same value at the top of every split, and since `test_clean`/`test_adversarial` draw
60-of-100 from equally-sized CORD splits, that produced the *identical* shuffle permutation both
times — making tampered/authentic labels, tamper type, and instruction phrasing identical
position-by-position between the two splits (see `SPLIT_SEED_OFFSETS` in `build_dataset.py` for the
fix). The results above were measured against the *pre-fix* dataset, so the forgery-task numbers
likely overstate robustness: the fine-tuned model may be partly keying off the label-correlated
template patterns baked into the (pre-fix) data rather than purely visual evidence. The seeding fix
is in the codebase now; a full retrain on the corrected dataset would be needed to get a clean
answer on how much of the forgery-detection gain is genuine visual robustness vs. this artifact.

> 📝 **Write-up:** [The bug that was quietly inflating my own results](WRITEUP.md) — a
> post-mortem on how a dataset-seeding bug produced a bit-for-bit identical confusion matrix
> across two different test splits, how the zero-shot baseline made it diagnosable, and why the
> provisional number is published rather than quietly retrained away.

## Limitations

- Tampering is synthetic, not real forged documents — it teaches the *classes* of manipulation
  (copy-move, splicing, field edits) but the model's performance on real forged IDs is unverified.
- CORD is receipts, not identity documents; layout/field types differ from IDs/passports.
- Small dataset by LLM standards (470 unique source images) — sized to fit a one-day, one-GPU
  training budget, not to be state-of-the-art.
- `patch_overlay`'s fake replacement text is drawn from a small fixed vocabulary, so the model may
  partly learn to recognize that specific pattern rather than field-tampering in general.
- **Clean/adversarial forgery-detection comparison is confounded by a since-fixed seeding bug** —
  see the Results section above. The OCR numbers aren't affected (they don't depend on the
  tampered/authentic label pattern), but the forgery robustness claim should be treated as
  provisional until re-measured on a retrain against the corrected dataset.

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
