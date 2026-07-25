# DocGuard-VLM — The Complete Learning Guide (A to Z)

This document exists so you can defend every line of `docguard-vlm` and every word of the resume
bullet in an interview, cold, without needing to reread the README. It assumes **zero** prior
knowledge of deep learning terminology and builds up from there to the exact code you shipped.

Read it top to bottom once. After that, use the table of contents as a reference.

---

## Table of contents

- [Part 0 — How this document is organized](#part-0)
- [Part 1 — Foundations: the vocabulary](#part-1)
  - [1.1 What "machine learning" actually means](#p1-1)
  - [1.2 Neural networks in one page](#p1-2)
  - [1.3 What a Large Language Model (LLM) is](#p1-3)
  - [1.4 Tokens and tokenization](#p1-4)
  - [1.5 The Transformer and attention](#p1-5)
  - [1.6 What a Vision-Language Model (VLM) is](#p1-6)
  - [1.7 Pretraining vs. fine-tuning](#p1-7)
  - [1.8 What "2B parameters" means for memory](#p1-8)
- [Part 2 — The fine-tuning toolkit](#part-2)
  - [2.1 Full fine-tuning vs. PEFT](#p2-1)
  - [2.2 LoRA, the actual mechanism](#p2-2)
  - [2.3 QLoRA: LoRA + quantization](#p2-3)
  - [2.4 What unsloth actually does](#p2-4)
  - [2.5 The Hugging Face ecosystem, library by library](#p2-5)
  - [2.6 SFTTrainer / SFTConfig, every hyperparameter explained](#p2-6)
- [Part 3 — The project, file by file](#part-3)
- [Part 4 — The bugs we hit, and what each one taught us](#part-4)
- [Part 5 — Reading the results](#part-5)
- [Part 6 — Resume and interview mapping](#part-6)
- [Glossary, A–Z](#glossary)

---

<a id="part-0"></a>
## Part 0 — How this document is organized

Part 1 gives you the vocabulary a total beginner needs. Part 2 covers the specific fine-tuning
techniques (LoRA, QLoRA, unsloth) at the level of "what problem does this solve and how." Part 3
walks the actual repo, file by file. Part 4 is the debugging log turned into lessons — this is
genuinely some of the best interview material in the whole project, because it's proof of process,
not just a result. Part 5 explains what the numbers in the results table mean and how to talk about
them honestly. Part 6 maps every word of the resume bullet and every JD requirement back to
something real you did.

---

<a id="part-1"></a>
## Part 1 — Foundations: the vocabulary

<a id="p1-1"></a>
### 1.1 What "machine learning" actually means

At the core, machine learning is: **a function with adjustable knobs, tuned automatically using
examples.**

- The "function" is the model. It takes an input (an image, some text) and produces an output (a
  prediction).
- The "knobs" are called **parameters** (or **weights**) — plain numbers. A model with "2 billion
  parameters" just has 2 billion adjustable numbers inside it.
- "Tuned automatically" means: you don't hand-set these numbers. You show the model examples of
  correct input→output pairs, measure how wrong its current guesses are, and nudge every number a
  tiny bit in the direction that would have made it less wrong. Repeat this millions of times.

Three pieces make this loop work:

1. **A loss function** — a single number that says "how wrong was this prediction." Lower is
   better.
2. **Gradient descent** — the algorithm that figures out, for each of the billions of parameters,
   which direction (up or down) and by how much to nudge it to reduce the loss. This uses calculus
   (the *gradient* — literally the slope of the loss with respect to each parameter).
3. **An optimizer** — the specific bookkeeping strategy for applying those nudges. We used
   **AdamW**, which doesn't just apply the raw gradient each step — it keeps a running average of
   recent gradients (momentum) and adapts the step size per-parameter, which trains faster and more
   stably than plain gradient descent.

That's it. Everything else in this document is refinements on "adjustable numbers, tuned by
gradient descent, to reduce a loss."

<a id="p1-2"></a>
### 1.2 Neural networks in one page

A neural network is a specific *shape* for that function: a stack of simple layers, each doing:

```
output = activation(W · input + b)
```

- `W` (a matrix) and `b` (a vector) are parameters — the "knobs."
- `W · input` is just weighted sums: every output number is some combination of every input number,
  scaled by learned weights.
- `activation` is a small nonlinear function (like ReLU: `max(0, x)`) applied elementwise. Without
  this, stacking layers would collapse into one big linear function — nonlinearity is what lets
  the network represent complex patterns instead of just straight lines.

Stack many of these layers and you get a "deep" neural network — hence "deep learning." Two passes
happen during training:

- **Forward pass**: input flows through the layers, produces a prediction.
- **Backward pass** (**backpropagation**): starting from the loss, the chain rule of calculus is
  used to compute how much each parameter, all the way back through every layer, contributed to
  the error — this is the "gradient" that gradient descent uses.

You don't need to hand-derive any of this — PyTorch (the library everything here is built on)
computes backpropagation automatically (`autograd`). You just define the forward pass; PyTorch
figures out the backward pass for you.

<a id="p1-3"></a>
### 1.3 What a Large Language Model (LLM) is

An LLM is a neural network trained on one deceptively simple task: **predict the next token, given
everything before it.** Trained on a big enough chunk of the internet, with a big enough network,
this simple objective produces a model that has implicitly learned grammar, facts, reasoning
patterns, and coding ability — because predicting the next word well requires modeling all of that.

A base LLM (just next-token prediction) isn't naturally good at *following instructions* — it just
continues text plausibly. **Instruction tuning** (fine-tuning on examples of `instruction → good
response` pairs, formatted as a conversation) teaches the model to behave like an assistant instead
of an autocomplete engine. `Qwen2-VL-2B-**Instruct**` — the "Instruct" in the model name means
this step has already been done by the model's creators, before we ever touched it.

<a id="p1-4"></a>
### 1.4 Tokens and tokenization

Neural networks operate on numbers, not text. **Tokenization** is the process of chopping text into
small pieces (**tokens** — often subwords, e.g. "tampering" might become `tamper` + `ing`), and
mapping each token to an integer ID. Each ID then looks up a vector of numbers (an **embedding**) —
a point in a high-dimensional space where, after training, similar-meaning tokens end up near each
other. The model's actual input is a sequence of these embedding vectors.

`max_new_tokens=160` in our eval code means: "generate up to 160 of these tokens as the response,
then stop." It's a length budget, not a word-count budget — 160 tokens is usually 100-130 words of
English.

<a id="p1-5"></a>
### 1.5 The Transformer and attention

The **Transformer** is the specific neural network architecture behind essentially every modern LLM
and VLM, including Qwen2-VL. Its key idea is **self-attention**.

For each token in a sequence, self-attention asks: *"which other tokens in this sequence are
relevant to understanding me, and how much?"* Mechanically, every token produces three vectors from
its embedding via learned weight matrices:

- **Query (Q)** — "what am I looking for?"
- **Key (K)** — "what do I contain, that others might look for?"
- **Value (V)** — "what information do I actually offer, if selected?"

The attention score between token *i* and token *j* is `Q_i · K_j` (a dot product — high when the
vectors point in a similar direction). These scores get normalized with softmax into weights that
sum to 1, and the output for token *i* is a weighted sum of all tokens' Value vectors, weighted by
those scores. In formula form (this is the exact operation you saw crash in the traceback —
`scaled_dot_product_attention`):

```
Attention(Q, K, V) = softmax(Q · Kᵀ / √d_k) · V
```

"Multi-head attention" just runs several of these attention computations in parallel (with
different learned Q/K/V projections each), so different heads can specialize in different kinds of
relationships (e.g. one head tracking grammatical structure, another tracking topical relevance).

**Why this matters for a bug you personally hit:** the `Q · Kᵀ` step produces a matrix of size
`(sequence_length × sequence_length)` — its memory cost grows **quadratically** with sequence
length. A document image that gets tokenized into 12,000 visual tokens (because it was 9 megapixels
and uncapped) needs an attention matrix with ~144 million entries *per head*, in a single forward
pass. That's the actual mechanism behind the `CUDA out of memory: Tried to allocate 6.45 GiB` error
you hit — not a vague "ran out of memory," but this specific formula, at this specific step,
scaling quadratically. Capping `max_pixels` in the image processor is directly capping sequence
length to keep this bounded.

<a id="p1-6"></a>
### 1.6 What a Vision-Language Model (VLM) is

A VLM extends an LLM so it can also "read" images, by turning an image into something that looks
like a sequence of tokens the transformer can process alongside text:

1. The image is cut into small square **patches** (Qwen2-VL uses patches on the order of 14×14
   pixels).
2. Each patch is run through a **vision encoder** — a smaller neural network (itself built from
   transformer blocks, this time attending over patches instead of words) that turns each patch
   into an embedding vector, the same kind of object a text token embedding is.
3. These "visual tokens" are projected into the *same* embedding space the language side uses, and
   fed into the shared transformer alongside the text tokens from the prompt. From that point on,
   the model doesn't really distinguish "this came from a pixel patch" vs. "this came from a word"
   — it's all just vectors in a sequence, and self-attention lets text tokens attend to visual
   tokens and vice versa (this is literally how the model can look at a receipt image and answer a
   question about the total in text).

Qwen2-VL specifically uses **dynamic resolution**: instead of always resizing every image to one
fixed size, it lets image size determine token count (more pixels → more patches → more visual
tokens), within `min_pixels`/`max_pixels` bounds you can configure — exactly the knob we used in
`evaluate.py` to fix the OOM.

<a id="p1-7"></a>
### 1.7 Pretraining vs. fine-tuning

- **Pretraining**: training a model from (near-)scratch on a massive, general dataset (a large
  fraction of the internet, for an LLM). This is where the model learns language and vision broadly.
  It costs millions of dollars in compute for a model like Qwen2-VL. We did not do this — nobody
  fine-tuning on a laptop or free Colab does.
- **Fine-tuning**: taking an *already pretrained* model and training it further on a smaller,
  task-specific dataset, so it specializes without forgetting everything it already knows. This is
  what we did: we started from `Qwen2-VL-2B-Instruct` (already knows language, already knows how to
  look at images and describe them, already knows how to follow instructions) and fine-tuned it on
  700 of our own examples to specialize it for two specific tasks: structured receipt
  field-extraction and forgery verdicts in our exact format.

<a id="p1-8"></a>
### 1.8 What "2B parameters" means for memory

"2B" = 2,237,936,128 parameters, per our own training log. Each parameter is a number that has to
be stored in memory. The storage cost depends on the numeric precision:

| precision | bytes/parameter | 2.24B params ≈ |
|---|---|---|
| fp32 (32-bit float) | 4 | ~9.0 GB |
| fp16/bf16 (16-bit float) | 2 | ~4.5 GB |
| int8 (8-bit integer) | 1 | ~2.2 GB |
| 4-bit (e.g. NF4) | 0.5 | ~1.1 GB |

That's *just the weights, sitting still*. Training needs much more: gradients (same size as the
weights again), plus optimizer state — AdamW keeps two extra running-average values *per
parameter* (in fp32, for numerical stability), which is another ~8 bytes/parameter. Add it up for
naive full fine-tuning in fp16: weights (4.5GB) + gradients (4.5GB) + Adam state (~18GB) ≈ **27GB**,
comfortably blowing past a free Colab T4's 15GB of VRAM. This is the concrete reason Part 2 exists
— LoRA and 4-bit quantization aren't optional tricks, they're what make this project possible on
free hardware at all.

---

<a id="part-2"></a>
## Part 2 — The fine-tuning toolkit

<a id="p2-1"></a>
### 2.1 Full fine-tuning vs. Parameter-Efficient Fine-Tuning (PEFT)

**Full fine-tuning** updates every one of the 2.24B parameters. As shown above, that's memory you
don't have on free hardware, and it also risks **catastrophic forgetting** — aggressively updating
every weight on a small 700-example dataset can degrade everything the model already knew.

**PEFT** (Parameter-Efficient Fine-Tuning) is the family of techniques that freeze the original
weights entirely and add a small number of *new* trainable parameters instead. LoRA (below) is the
specific PEFT technique we used — it's in your resume's skills list and it's the literal `peft`
Python package imported throughout this codebase.

<a id="p2-2"></a>
### 2.2 LoRA, the actual mechanism

LoRA = **Low-Rank Adaptation**. Here's the actual math, not just the marketing description.

Take any weight matrix `W` inside the model that full fine-tuning would normally update directly —
say a matrix of shape `(d_out, d_in)` inside an attention layer. Full fine-tuning would learn a
full-size update `ΔW` (same shape as `W`) and use `W + ΔW`.

LoRA's insight: instead of learning `ΔW` directly (which has `d_out × d_in` numbers — huge), factor
it as the product of two much smaller matrices:

```
ΔW = B · A
```

where `A` has shape `(r, d_in)` and `B` has shape `(d_out, r)`, and `r` (the **rank**) is a small
number — in our config, `r = 16`. Both `d_in` and `d_out` in Qwen2-VL's attention/MLP layers are in
the thousands, so a rank-16 factorization has *vastly* fewer numbers than the full matrix: instead
of `d_out × d_in` trainable values, you have `r × (d_in + d_out)` — for a 1536×1536 matrix, that's
16×3072 ≈ 49K parameters instead of 2.36M. This is the entire reason your training log shows
**"Trainable parameters = 28,950,528 of 2,237,936,128 (1.29% trained)"** — 98.71% of the model
stayed completely frozen, and only these small A/B adapter pairs (inserted into the attention,
MLP, and vision layers, per `finetune_attention_modules=True` etc. in the config) were ever updated.

**Why does throwing away most of the expressiveness (using only rank 16 instead of full rank) still
work?** The empirical finding behind the LoRA paper (Hu et al., 2021) is that the *update* a model
needs for a specific downstream task tends to have low "intrinsic rank" — i.e. it lies mostly in a
small subspace of all possible directions, even though the full weight matrix doesn't. You don't
need to re-derive language and vision from scratch; you need a small, targeted nudge, and that nudge
compresses well.

`lora_alpha=16` is a scaling factor: the actual update applied is `(alpha / r) · B·A`, not just
`B·A`. With `alpha = r = 16`, the scale factor is 1 — a common, simple default. Raising `alpha`
relative to `r` effectively increases how strongly the adapter's learned update is weighted
relative to the frozen base weights.

At inference time, `W + (alpha/r)·B·A` is used as the effective weight — either computed on the fly
(what we do, keeping `W` frozen and `A`/`B` as a separate small adapter you can load/unload), or
"merged" into a new single matrix if you want to ship one final model with no PEFT machinery at
inference time.

<a id="p2-3"></a>
### 2.3 QLoRA: LoRA + quantization

**QLoRA** combines LoRA with **quantization** of the frozen base weights: instead of storing the
2.24B frozen parameters in fp16 (2 bytes each, ~4.5GB), they're stored in **4-bit** precision
(~1.1GB) using a scheme called **NF4** (4-bit NormalFloat) — specifically designed to represent
weights that are roughly normally-distributed (which trained neural network weights typically are)
more accurately than a naive uniform 4-bit split would. **Double quantization** further compresses
the small quantization-constant metadata itself, for a bit more savings.

Critically: only the *frozen* base weights are quantized. The LoRA adapter matrices (`A`, `B`) that
are actually being trained stay in higher precision (fp16), because gradients need enough numerical
precision to update meaningfully. This is exactly the `BitsAndBytesConfig` in `evaluate.py`:

```python
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,   # compute (matmuls) happen in fp16
    bnb_4bit_use_double_quant=True,         # compress the quantization constants too
    bnb_4bit_quant_type="nf4",              # the NF4 scheme described above
)
```

This is why both training and (after we fixed it) evaluation fit comfortably on a free T4's 15GB:
frozen weights at ~1.1GB instead of ~4.5GB, plus small LoRA adapters, plus activations — versus the
~27GB full fine-tuning would have needed.

<a id="p2-4"></a>
### 2.4 What unsloth actually does

**unsloth** is a library that reimplements the performance-critical inner loops of popular model
architectures (Qwen2-VL among them) as custom, fused GPU kernels (written in Triton, a
GPU-programming language), and monkey-patches them into Hugging Face's standard model classes at
load time. That's literally what the log line **"Unsloth: Will patch your computer to enable 2x
faster free finetuning"** means — it's rewriting parts of the model's forward/backward pass with
faster, leaner implementations before training starts.

Concretely, this buys you:
- Faster attention computation (a more memory-efficient path than the default `transformers`
  implementation).
- Lower peak memory during training, partly via smarter gradient checkpointing (trading a bit of
  recomputation for a lot of memory savings — instead of storing every intermediate activation for
  the backward pass, it recomputes some of them on the fly).
- The same mathematical result as standard LoRA/QLoRA training — unsloth changes *how fast and
  memory-efficiently* the computation happens, not *what* is being computed.

This is also the direct explanation for something you noticed empirically: **training** (via
unsloth) never hit the CUDA OOM that **evaluation** (via plain `transformers`) did on the exact same
large images — unsloth's patched attention path handles that more gracefully than the default one
`evaluate.py` deliberately uses (portability was the tradeoff — see Part 3).

<a id="p2-5"></a>
### 2.5 The Hugging Face ecosystem, library by library

| library | what it actually does here |
|---|---|
| `transformers` | Defines model architectures (`Qwen2VLForConditionalGeneration`), loads pretrained weights from the Hugging Face Hub, provides the `AutoProcessor` that turns images+text into model inputs. |
| `datasets` | Streams/loads datasets from the Hugging Face Hub — this is how `naver-clova-ix/cord-v2` gets pulled down in `build_dataset.py`. |
| `peft` | Implements LoRA (and other PEFT methods) — `PeftModel.from_pretrained(model, adapter_path)` in `evaluate.py` is this library attaching your trained adapter onto the frozen base model. |
| `trl` | "Transformer Reinforcement Learning" — despite the name, also provides `SFTTrainer`/`SFTConfig`, the standard supervised fine-tuning training loop used in the Colab notebook. |
| `accelerate` | Handles device placement (CPU/GPU/multi-GPU) and mixed-precision plumbing underneath `trl`/`transformers`, mostly invisibly. |
| `bitsandbytes` | The actual CUDA kernels implementing 4-bit/8-bit quantization (`BitsAndBytesConfig`) and 8-bit optimizers (`adamw_8bit`). |
| `unsloth` | Sits on top of all of the above, patching in faster kernels (2.4, above). |

<a id="p2-6"></a>
### 2.6 SFTTrainer / SFTConfig, every hyperparameter explained

From `notebooks/train_colab.ipynb`:

```python
SFTConfig(
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    warmup_ratio=0.05,
    num_train_epochs=2,
    learning_rate=2e-4,
    logging_steps=10,
    save_strategy="epoch",
    optim="adamw_8bit",
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    seed=3407,
    ...
)
```

- **`per_device_train_batch_size=2`** — process 2 training examples at once per forward/backward
  pass. Kept small because each example (an image + text) uses meaningful GPU memory, and we're
  already tight on a T4.
- **`gradient_accumulation_steps=4`** — instead of updating the weights after every batch of 2,
  accumulate (sum) the gradients over 4 consecutive batches before applying one weight update. This
  *simulates* a larger effective batch size (2 × 4 = 8) without needing the memory to hold 8
  examples' activations simultaneously. Larger effective batch sizes generally give more stable
  gradient estimates.
- **`num_train_epochs=2`** — one **epoch** = one full pass over all 700 training examples. We did
  two full passes.
- **`learning_rate=2e-4`** — the step size for gradient descent. `2e-4` (0.0002) is a fairly large
  learning rate by full-fine-tuning standards — but appropriate for LoRA, since we're only updating
  ~29M small adapter parameters, not the whole 2.24B-parameter model, so larger, faster-moving steps
  are safe.
- **`warmup_ratio=0.05`** — over the first 5% of training steps, linearly ramp the learning rate up
  from 0 to the target `2e-4`, instead of starting at full speed immediately. This avoids
  destabilizing the freshly-initialized LoRA adapter weights with large updates before the
  optimizer's internal statistics have "warmed up."
- **`lr_scheduler_type="cosine"`** — after warmup, the learning rate follows a cosine curve down
  toward 0 by the end of training, rather than staying constant. This is standard practice — large
  steps early (fast progress), small steps late (fine convergence).
- **`optim="adamw_8bit"`** — the AdamW optimizer (Part 1.1's "optimizer," specifically the variant
  that adds weight decay cleanly), but with its internal per-parameter state (the two running
  averages mentioned in 1.8) stored in 8-bit instead of 32-bit — another memory-saving trick, this
  time from `bitsandbytes`, layered on top of everything else.
- **`weight_decay=0.01`** — a regularization term: each step, weights are nudged slightly toward
  zero (in addition to the gradient-based update), which discourages any single weight from growing
  too large and helps generalization (reduces overfitting to the 700 training examples).
- **`save_strategy="epoch"`** — write a checkpoint to disk after each full epoch (this is what let
  training resume from a saved checkpoint after a Colab disconnect, once we wired `OUTPUT_DIR` to
  Google Drive).
- **`seed=3407`** — fixes the random number generator's starting state, so re-running training with
  identical code and data reproduces (nearly) identical results — important for being able to say
  "this result is reproducible," not a fluke of randomness.

The training log line **"Total steps = 176"** comes from: 700 examples ÷ effective batch size 8 =
87.5 → 88 steps per epoch (rounded), × 2 epochs = 176.

---

<a id="part-3"></a>
## Part 3 — The project, file by file

### `src/data_gen/tamper.py` — the synthetic forgery generator

Three functions, each producing a `TamperResult(image, bbox, tamper_type)`:

- **`copy_move(image, content_boxes)`** — crops a region (biased toward a real text line, see
  `_content_region` below), optionally flips it horizontally, and pastes it elsewhere on the *same*
  image. This mimics a real forgery technique: duplicating a genuine element (e.g. a stamp, a
  digit) to a different spot on the same document.
- **`splice(image, donor_image, content_boxes)`** — crops a same-shaped region from a *different*
  document (the "donor") and pastes it into the target image at a real-content location. Mimics
  combining content from two different physical documents.
- **`patch_overlay(image, content_boxes)`** — picks a text-line region, fills it with the local
  background color (sampled via the median pixel value of that region, so the patch blends in), and
  draws different fake text over it (from a small fixed vocabulary of plausible values). Mimics
  editing a specific field, like a total amount.

`_content_region(w, h, content_boxes, pad)` is the key helper that fixed a real bug: it picks a
region *anchored to one of the real OCR word/line boxes* (passed in from `cord_utils.py`) instead of
a uniformly random rectangle. Early in the project, uniformly random regions occasionally landed
tampering on blank background instead of the document — visually obvious and not representative of
a real forgery, which would target actual document content.

### `src/data_gen/cord_utils.py` — recovering real text-line boxes

CORD's raw `ground_truth` field includes a `valid_line` list: every OCR'd word has a `quad` (four
corner coordinates, since real photographed text can be slightly skewed, not a perfect rectangle)
and a `row_id` grouping words on the same printed line. `extract_line_boxes()` converts each word's
quad into an axis-aligned bounding box, and unions all boxes sharing a `row_id` into one line-level
box. These line boxes are what `tamper.py` anchors tampering to.

### `src/data_gen/perturb.py` — the adversarial/robustness perturbations

Five independent perturbation functions, used **only** in the `test_adversarial` split, never
during training:

- `jpeg_recompress` — re-encodes the image as low-quality JPEG (quality 20–45), simulating
  repeated compression from messaging apps / uploads.
- `gaussian_noise` — adds random per-pixel noise, simulating a poor camera sensor / low light.
- `gaussian_blur` — simulates a slightly out-of-focus photo.
- `rotate_skew` — rotates ±8°, simulating a document photographed at an angle.
- `downscale_upscale` — shrinks then re-enlarges the image, simulating a low-resolution capture.

`apply_stacked_perturbations(image, n=2)` composes two of these at once, for a harder adversarial
tier (used on the forgery-task images in `test_adversarial`).

**Why this exists at all**: the JD explicitly asks for "evaluation harnesses against noisy,
adversarial, and real-world inputs" — a model that only works on clean, centered scans isn't useful
for a real identity-verification intake pipeline where users photograph documents with phone
cameras under imperfect conditions.

### `src/data_gen/prompts.py` — instruction/response templates

Defines the instruction wording (several variants, chosen randomly, for both the OCR task and the
forgery task) and how ground-truth responses get formatted — e.g. `forgery_sample()` builds a
response like `"Tampered. In the top-left region of the document, a region appears to have been
pasted in from a different source document (splicing)."`, combining the tamper's location (computed
geometrically from the bounding box) and a fixed description per tamper type
(`TAMPER_TYPE_DESC`). `to_conversation()` wraps a sample into the chat-message format
(`{"role": "user"/"assistant", "content": [...]}`) that both unsloth's trainer and `evaluate.py`
expect.

### `src/data_gen/build_dataset.py` — assembling everything

For each selected CORD image: builds one OCR sample (extracting line items/subtotal/total from
CORD's `gt_parse` into a JSON string as the target response) and one forgery sample (roughly
50/50 tampered/authentic, via `is_tampered = idx % 2 == 0`). For the adversarial split, both the OCR
and forgery images additionally get run through `perturb.py`. Writes everything to
`{train,test_clean,test_adversarial}.jsonl` plus the actual image files under `data/processed/images/`.

`SPLIT_SEED_OFFSETS` (added after the seed-reuse bug, see Part 4) ensures each split's random draw
(which CORD images get selected, which tamper type gets applied, which instruction phrasing gets
used) is independent across splits, instead of accidentally identical.

### `notebooks/train_colab.ipynb` — the actual training run

Cell order: clone repo + rebuild dataset → mount Google Drive (checkpoint persistence) → install
unsloth/trl/peft (unpinned, after the version-pin bug) → config (`DATA_ROOT`, `OUTPUT_DIR`,
`LORA_R`, etc.) → load base model in 4-bit + attach LoRA via `FastVisionModel.get_peft_model()` →
convert the JSONL into unsloth's expected chat-message format → build `SFTTrainer` (with
resume-from-checkpoint logic) → `trainer.train()` → save adapter → smoke test → pointer to the eval
harness.

### `src/eval/metrics.py` — scoring, no GPU required

`ocr_accuracy()`: tries to `json.loads()` the model's response (recording whether it even produced
valid JSON — itself a real pipeline requirement, not just a nice-to-have) and compares the `total`
field's digits against ground truth.

`forgery_accuracy()`: computes a confusion matrix (`tp`/`fp`/`fn`/`tn` for the "tampered" class) and
derives accuracy/precision/recall/F1 from it, using `parse_verdict()` to map free-text model output
to `{authentic, tampered, unknown}`.

`parse_verdict()` is worth understanding in detail because a real bug lived here: naive substring
matching for `"tamper"` misfires on `"no signs of **tamper**ing"` — the word "tampering" *contains*
"tamper" as a substring. The fix checks **negation patterns first** (`_NEGATED_TAMPER_RE`, matching
things like `"no ... tamper"`, `"not ... tamper"`, `"n't ... tamper"` within a short window of
words), only falling through to the positive `_TAMPER_RE` match if no negation was found — and does
this on the first sentence before falling back to scanning the whole response, since verdicts are
usually stated up front.

### `src/eval/evaluate.py` — scoring a model end to end

Deliberately uses plain `transformers` + `peft` (not unsloth) so it's portable — you can score a
saved adapter anywhere, independent of the training environment. `build_model()` loads the base
model in 4-bit (matching how it was trained, and keeping memory low — see Part 4's RAM-OOM bug),
optionally attaches a LoRA adapter via `PeftModel.from_pretrained()`, and caps image resolution via
the processor's `min_pixels`/`max_pixels` (see Part 4's CUDA-OOM bug). `generate()` builds the chat
prompt, runs the model, and decodes only the newly-generated tokens (slicing off the input prompt
from the output). `run_split()` loops over a JSONL file, calls `generate()` per example, and routes
predictions to the right metric function based on `task`.

### `src/serve/app.py` — the live demo

A thin Gradio UI wrapped around `build_model()`/`generate()` from `evaluate.py` — deliberately
reusing that exact code, not reimplementing it, so the demo can never silently diverge from the
scored numbers.

---

<a id="part-4"></a>
## Part 4 — The bugs we hit, and what each one taught us

This is real debugging history, not a sanitized retelling — worth having ready as interview
material, because it's evidence of process.

**1. Colab runtime not actually GPU-attached.** `Runtime → Change runtime type` sets the *intended*
hardware, but the runtime still has to actually connect to a GPU instance — `!nvidia-smi` is the
ground-truth check. Lesson: verify the resource you think you have before spending time on the next
step.

**2. `unsloth`/`peft` version drift → `NameError: VARIANT_KWARG_KEYS`.** Pinning `trl==0.12.1` while
leaving `peft`/`transformers`/`unsloth` unpinned let pip install a newer `peft` that introduced a
"variant" concept in its LoRA layer forward pass; unsloth's auto-generated compiled kernel
(`unsloth_compiled_cache/Linear_peft_forward.py`, regenerated fresh on each run to fuse
operations for speed) referenced a symbol from that newer `peft` API without importing it correctly
— a real upstream bug, not a config mistake on our end. Fix: clear the stale generated cache, let
pip resolve `unsloth`/`unsloth_zoo`/`peft`/`trl` together without an outdated pin forcing a
mismatch. Lesson: in a fast-moving ecosystem, pinning *one* library while leaving its close
dependents unpinned is often worse than pinning none of them.

**3. Colab free-tier disconnects wiping the whole VM.** `Runtime → Restart session` clears only the
Python kernel's memory (variables, loaded model) — files on disk survive. But an idle-timeout
disconnect or reclaimed session can reset the *entire VM*, wiping cloned repos and installed
packages too, not just kernel state. Fix: mount Google Drive and save checkpoints there (`OUTPUT_DIR`
on Drive, plus `resume_from_checkpoint` logic checking for existing `checkpoint-*` folders), so a
disconnect costs re-running setup cells, not the actual training progress.

**4. `CUDA out of memory: Tried to allocate 6.45 GiB`, on a single attention call.** Traced via the
full traceback to `scaled_dot_product_attention` inside the vision encoder. Some CORD source photos
are ~9 megapixels (2304×4096); Qwen2-VL's default `max_pixels` (~12.8MP) let those through
uncapped, producing thousands of visual tokens for a single image — and per Part 1.5, attention
memory scales *quadratically* with token count. Fix: explicitly cap `min_pixels`/`max_pixels` on
the `AutoProcessor`. Lesson: "out of memory" errors are worth reading the *specific operation* in
the traceback, not just treating as generically "needs more GPU" — the fix here was about sequence
length, not raw VRAM.

**5. Host RAM crash ("session crashed after using all available RAM"), distinct from GPU VRAM.**
`evaluate.py` originally loaded the full fp16 checkpoint (~4.5GB) independently of how the adapter
was trained (4-bit QLoRA), on top of whatever the training session had already left resident — Colab
free tier's system RAM is a separate, often tighter, budget than GPU VRAM. Fix: load the eval model
in 4-bit too, by default, matching the training configuration.

**6. Log spam from a redundant generation-config warning, severe enough to truncate output.** Every
single `model.generate()` call printed a "both max_new_tokens and max_length are set" warning,
because Qwen2-VL's default `generation_config` ships with `max_length=32768` alongside our explicit
`max_new_tokens=160` — harmless, but printed hundreds of times across a long eval run, enough to
blow past a chat message's character limit. Fix: explicitly clear `model.generation_config.max_length
= None` after loading, removing the redundant setting instead of just suppressing its symptom.

**7. The big one — a seed-reuse bug silently correlating the clean and adversarial test splits.**
`build_split()` reset Python's global RNG to the *same* seed at the top of every call (train,
test_clean, test_adversarial). Since `test_clean` (CORD's "validation" split) and `test_adversarial`
(CORD's "test" split) both have exactly 100 images, `random.shuffle(list(range(100)))` — run right
after the same reset seed — produced the **identical permutation** both times. That meant, position
by position, the two splits shared the same tampered/authentic label, the same tamper type, and the
same instruction phrasing — only the underlying photographed document differed. This was **caught**,
not assumed, via a concrete piece of evidence: the fine-tuned model's forgery confusion matrix was
*bit-for-bit identical* between clean and adversarial (`tp=20, fp=12, fn=15, tn=13`, exactly, in
both), while the zero-shot baseline's confusion matrix genuinely differed between the two splits
(`tp` 4→3, `tn` 25→22) — exactly what you'd expect from a model actually responding to different
pixels. That asymmetry is what triggered tracing the bug back to `build_dataset.py:59`, confirming
it in code, and fixing it (`SPLIT_SEED_OFFSETS`, offsetting the seed per split so the permutations
diverge) rather than either ignoring the anomaly or panicking and discarding otherwise-valid results.
This is genuinely one of the strongest things to talk about in an interview: it demonstrates reading
metrics skeptically, forming a hypothesis, verifying it in the actual source code, and documenting
the fix and its limits honestly rather than quietly patching numbers.

---

<a id="part-5"></a>
## Part 5 — Reading the results

The full numbers live in `results/eval_results.json` (fine-tuned) and `results/eval_baseline.json`
(zero-shot), and the table in the README. Here's what each metric means:

- **`json_valid_rate`** — fraction of responses that could be parsed as valid JSON at all. This
  matters independent of *correctness*: a real pipeline needs a machine-parseable response, not just
  a roughly-right one a human could interpret.
- **`total_match_rate`** — fraction where the extracted `total` field's digits exactly matched
  ground truth (after stripping formatting like commas/currency symbols).
- **`accuracy`** (forgery) — fraction of forgery verdicts that matched ground truth (tampered vs.
  authentic).
- **`precision_tampered`** — of everything the model *called* tampered, what fraction actually was.
  High precision + low recall (the baseline's pattern: 100% precision, 11.8% recall) means the model
  is playing it safe, rarely committing to "tampered" — so on the rare occasions it does, it's
  usually right, but it's missing most real tampering.
- **`recall_tampered`** — of everything that *was actually* tampered, what fraction the model caught.
  This is usually the more important number for a fraud-detection use case — missing real forgeries
  (false negatives) is typically costlier than a false alarm.
- **`f1_tampered`** — the harmonic mean of precision and recall, a single number balancing both.
  Rewards models that don't just optimize one at the expense of the other.
- **`unknown_rate`** — fraction of responses `parse_verdict()` couldn't confidently classify as
  either verdict (excluded from the confusion matrix, tracked separately, since silently guessing
  would inflate or deflate accuracy dishonestly).

**The headline story**: fine-tuning took OCR from barely-functional (25%/13% valid JSON) to strong
(91.7%/88.3%) — because the zero-shot base model simply doesn't know your target JSON schema until
taught. Forgery detection improved substantially too (F1 roughly tripled, recall went from
essentially "never says tampered" to catching the majority of real cases) — but Part 4's bug #7
means the *clean-vs-adversarial robustness* comparison specifically (not the fine-tuned-vs-baseline
comparison, which remains valid) needs a retrain on the corrected dataset to fully trust. Say this
plainly if asked — it's a stronger answer than claiming clean robustness you haven't actually
verified.

---

<a id="part-6"></a>
## Part 6 — Resume and interview mapping

### The resume bullet, word by word

> **Fine-tuned Qwen2-VL-2B via QLoRA** — Part 2.2/2.3: LoRA rank-16 adapters over a 4-bit-quantized
> frozen base, trained via `SFTTrainer`.
>
> **for dual-task document intelligence — structured field extraction and digital-tampering
> detection** — one shared adapter, two tasks, distinguished only by the instruction text at
> inference time (Part 3, `prompts.py`).
>
> **on a synthetic-forgery pipeline built from scratch over real receipt photos** — `tamper.py` +
> `cord_utils.py`: copy-move/splice/patch-overlay, anchored to real OCR word boxes, over CORD-v2.
>
> **improved OCR field-extraction validity from 25%→92% and total-amount accuracy from 33%→90% over
> zero-shot baseline** — directly from `results/eval_baseline.json` vs. `results/eval_results.json`,
> clean split.
>
> **and tripled forgery-detection F1 (0.21→0.60)** — same source, forgery task, clean split.
>
> **Built a 5-perturbation adversarial evaluation harness** — `perturb.py`: JPEG recompression,
> Gaussian noise, Gaussian blur, rotation/skew, downscale-upscale.
>
> **to measure robustness beyond clean validation** — directly echoes the JD's own phrase ("Design
> evaluation benchmarks beyond clean validation datasets").
>
> **diagnosed and fixed a dataset-seeding bug that was silently confounding the clean-vs-adversarial
> forgery comparison, then documented the finding rather than reporting inflated numbers** — Part
> 4, bug #7, in full.

### JD requirement → what you actually did

| JD line | what maps to it |
|---|---|
| "Fine-tune open-weight LLMs and VLMs using SFT / PEFT / LoRA" | Exactly this project, literally. |
| "Adapt multimodal models for Document Understanding, OCR Enhancement" | The OCR field-extraction task. |
| "Adapt multimodal models for ... Identity Verification, Image Forgery Detection" | The forgery-detection task, verbatim topic match. |
| "Design evaluation benchmarks beyond clean validation datasets" | `test_adversarial.jsonl` + `perturb.py`. |
| "Build evaluation harnesses against noisy, adversarial, and real-world inputs" | Same, 5 perturbation types, single vs. stacked. |
| "Investigate model failures, develop hypotheses, and validate improvements through experimentation" | Bug #7, precisely — noticed an anomaly, hypothesized a cause, verified it in code, fixed and documented it. |
| "Explore synthetic data generation" | The entire `tamper.py`/`build_dataset.py` pipeline — a JD bonus point, and the core of this project. |
| "Hugging Face ecosystem" (bonus) | `transformers`, `datasets`, `peft`, `trl`, `bitsandbytes` — used directly, not just referenced. |
| "Comfortable training, debugging, and evaluating models — not just consuming AI APIs" | The entire Part 4 debugging log is the proof, not just the claim. |

### Likely interview questions, and where the honest answer lives

- *"Walk me through your project."* → the spoken pitch already drafted in this conversation, backed
  by Parts 3 and 5 here if they want depth.
- *"Why rank 16 for LoRA?"* → Part 2.2 — a common default balancing adapter expressiveness against
  trainable-parameter count; you could mention you didn't sweep this given the one-day/single-GPU
  budget, and that it's a natural next experiment.
- *"What would you do differently?"* → retrain on the corrected (seed-fixed) dataset to get a clean
  read on true clean-vs-adversarial forgery robustness (Part 4, bug #7); try real ID-document data
  if it were legally available; sweep LoRA rank/alpha.
- *"Why 4-bit quantization?"* → Part 2.3, plus the very concrete Part 4 story of the RAM crash it
  fixed.
- *"What's the hardest bug you hit?"* → bug #7, told as a story: identical confusion matrix → why
  would that be identical → traced to a shared RNG seed → fixed and documented, not just fixed
  quietly.

---

<a id="glossary"></a>
## Glossary, A–Z

**Accuracy** — fraction of predictions that exactly matched ground truth.

**AdamW** — the optimizer used for training; adapts per-parameter step size using running gradient
averages, plus a clean weight-decay term (Part 1.1, Part 2.6).

**Attention** — the core Transformer operation; lets each token weigh how relevant every other
token is to it (Part 1.5).

**Backpropagation** — the algorithm computing how much each parameter contributed to the loss, via
the calculus chain rule, so gradient descent knows which way to nudge it (Part 1.2).

**Batch size** — how many training examples are processed together in one forward/backward pass
(Part 2.6).

**Bounding box** — the rectangular coordinates (x0, y0, x1, y1) marking where something is in an
image; used here for OCR word/line locations and tamper regions.

**Checkpoint** — a saved snapshot of a model's weights at some point during training, so training
can resume from there instead of from scratch.

**Confusion matrix** — the 2×2 breakdown (true positive / false positive / false negative / true
negative) underlying precision, recall, and F1 (Part 5).

**Embedding** — a vector of numbers representing a token (or image patch) in a learned space where
similar things end up near each other (Part 1.4).

**Epoch** — one full pass over the entire training dataset (Part 2.6).

**Fine-tuning** — further training an already-pretrained model on a smaller, specific dataset
(Part 1.7).

**F1 score** — the harmonic mean of precision and recall; one number balancing both (Part 5).

**Forward pass** — running input through the model to produce a prediction (Part 1.2).

**Gradient descent** — the algorithm that nudges parameters in the direction that reduces loss
(Part 1.1).

**Gradient accumulation** — summing gradients over several small batches before applying one weight
update, simulating a larger batch size without the memory cost (Part 2.6).

**Instruction tuning** — fine-tuning a base LLM on instruction→response examples so it behaves like
an assistant, not just an autocomplete engine (Part 1.3).

**JSONL** — "JSON Lines," a file format where each line is one independent JSON object; used for
all the training/eval data in this project.

**LLM (Large Language Model)** — a neural network trained to predict the next token in text, at
large scale (Part 1.3).

**Loss function** — the single number quantifying how wrong a prediction was; what gradient descent
minimizes (Part 1.1).

**LoRA (Low-Rank Adaptation)** — a PEFT technique that freezes original weights and learns a
low-rank update via two small matrices instead (Part 2.2).

**Neural network** — a stack of simple weighted-sum-plus-nonlinearity layers, whose weights are
tuned by gradient descent (Part 1.2).

**NF4 (4-bit NormalFloat)** — the specific 4-bit quantization scheme used in QLoRA, tuned for
normally-distributed weight values (Part 2.3).

**Parameter** — a single adjustable number inside a model; "2B parameters" = 2 billion of them
(Part 1.1, 1.8).

**PEFT (Parameter-Efficient Fine-Tuning)** — fine-tuning approaches that update only a small number
of new parameters instead of the whole model (Part 2.1).

**Precision** — of everything predicted positive, what fraction actually was (Part 5).

**Pretraining** — training a model from scratch on a massive general dataset (Part 1.7).

**QLoRA** — LoRA combined with 4-bit quantization of the frozen base weights (Part 2.3).

**Quantization** — reducing the numeric precision used to store weights (e.g. 16-bit → 4-bit) to
save memory (Part 1.8, 2.3).

**Rank (of a LoRA adapter)** — the inner dimension `r` of the two small factor matrices; controls
how many trainable parameters the adapter has (Part 2.2).

**Recall** — of everything that actually was positive, what fraction got correctly predicted
(Part 5).

**Self-attention** — see Attention.

**SFTTrainer / SFTConfig** — the `trl` library's training loop and configuration object for
supervised fine-tuning on chat-formatted data (Part 2.6).

**Token** — a small piece of text (or, for images, a patch) that the model processes as one unit in
a sequence (Part 1.4, 1.6).

**Tokenization** — the process of chopping input into tokens and mapping them to IDs (Part 1.4).

**Transformer** — the neural network architecture built on self-attention, underlying all modern
LLMs/VLMs (Part 1.5).

**unsloth** — a library providing faster, more memory-efficient GPU kernels for training popular
model architectures, patched in at load time (Part 2.4).

**VLM (Vision-Language Model)** — an LLM extended to also process images, by turning image patches
into tokens in the same shared representation space as text (Part 1.6).

**Warmup (learning rate)** — gradually increasing the learning rate from 0 at the start of training,
before switching to the main schedule (Part 2.6).

**Weight decay** — a regularization term that slightly shrinks weights each training step, reducing
overfitting (Part 2.6).

**Zero-shot** — evaluating a model on a task without any task-specific fine-tuning; the baseline
this project's fine-tuned model is compared against throughout.
