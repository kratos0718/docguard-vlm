# The bug that was quietly inflating my own results

*A post-mortem on a dataset-seeding bug in DocGuard-VLM, and why I published the worse number.*

---

I fine-tuned **Qwen2-VL-2B** with LoRA to do two things on the same receipt image: extract fields
as JSON, and decide whether the document had been digitally tampered with.

Because a model that only works on clean, centred scans is useless for real photo uploads, I
scored everything on **two** held-out sets — a clean one, and an adversarial one where the same
images go through noise, blur, JPEG re-compression, rotation and downscaling.

The results looked good:

| model | split | OCR JSON-valid | OCR total-match | Forgery F1 |
|---|---|---|---|---|
| zero-shot | clean | 25.0% | 33.3% | 0.21 |
| zero-shot | adversarial | 13.3% | 18.3% | 0.16 |
| **+ LoRA** | clean | **91.7%** | **90.0%** | **0.60** |
| **+ LoRA** | adversarial | **88.3%** | **78.3%** | **0.60** |

OCR validity up ~3.7×. Forgery F1 roughly tripled. And robustness held: forgery F1 was **0.60 on
both splits**.

That last part is what bothered me.

## The number that was too good

Degrade an image with noise, blur and JPEG artefacts and a vision model's behaviour should change
*somewhere*. Mine didn't move at all.

So I looked past the summary metric at the raw confusion matrices:

```
fine-tuned, clean split:        tp=20  fp=12  fn=15  tn=13
fine-tuned, adversarial split:  tp=20  fp=12  fn=15  tn=13
```

**Bit-for-bit identical.** Not "similar" — the same four integers, on two sets of genuinely
different images.

The control was already sitting in my results. The zero-shot baseline's matrix *did* shift
between splits (`tp 4→3`, `tn 25→22`) — exactly what you expect from a model actually responding
to pixels. Same evaluation code, same images, different behaviour.

That ruled out the eval harness and pointed at the data.

## Root cause

In my dataset builder, `build_split()` reseeded Python's RNG to the same value at the top of
**every** split.

`test_clean` and `test_adversarial` each draw 60 examples from equally-sized CORD splits. Same
seed + same population size + same draw count = **the identical shuffle permutation, both times.**

So position-by-position across the two splits, the examples had:
- the same authentic/tampered labels
- the same tamper type
- the same instruction phrasing

The images differed. Everything else was a carbon copy.

That matters because it means the two splits weren't independent measurements. Worse, it means the
model could partly be keying off **label-correlated template patterns baked into the data** rather
than looking at the document at all — and my "robustness" number would look perfect either way.

The fix was small — per-split seed offsets, so each split gets a genuinely different permutation:

```python
SPLIT_SEED_OFFSETS = {...}   # build_dataset.py
```

## What I did about the numbers

The results above were measured against the **pre-fix** dataset. I had two options.

I could quietly retrain, publish whatever came out, and never mention it. Nobody would have known —
the bug is invisible unless you go looking at raw confusion matrices for two splits you'd normally
never compare directly.

Instead the README now says, in the Results section and again under Limitations:

> The clean/adversarial forgery-detection comparison is confounded by a since-fixed seeding bug.
> The forgery robustness claim should be treated as **provisional** until re-measured on a retrain
> against the corrected dataset.

The OCR numbers are unaffected — they don't depend on the tampered/authentic label pattern at all —
so those stand. The forgery robustness claim doesn't, and I say so.

## Three things I took from this

**1. A metric that doesn't move is a bug report.** I was looking at F1 = 0.60 on both splits and
reading it as "robust". Stability that clean, on inputs that different, isn't a result — it's a
symptom. The summary number hid it; the confusion matrix showed it in one glance.

**2. Always keep a control you didn't tune.** The zero-shot baseline is what made this
diagnosable. Without a second model running through the same pipeline, I'd have had one weird
number and no way to tell whether the cause was the data, the eval code or the model.

**3. Seeds are experimental infrastructure, not boilerplate.** `random.seed(42)` is muscle memory,
and the whole point is reproducibility. But *the same* seed across splits that are supposed to be
independent silently couples them. Reproducible and correct are different properties.

## Why publish the worse number

Because the alternative is a portfolio of results nobody can check, and because the interesting
part of this project was never the F1 score — it was catching a confounder in my own work.

Anyone can fine-tune a 2B model with LoRA. Being able to say *"this number is not trustworthy, and
here is precisely why"* is the harder and more useful skill.

---

**Code, dataset builder and eval harness:**
[github.com/kratos0718/docguard-vlm](https://github.com/kratos0718/docguard-vlm)
