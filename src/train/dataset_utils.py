"""Shared helpers for loading DocGuard-VLM JSONL records into the chat/conversation
format expected by unsloth's FastVisionModel / Qwen2-VL processor, both for
training (train.py, Colab notebook) and evaluation (eval/evaluate.py).
"""
import json
import os

from PIL import Image


def load_jsonl(path):
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def resolve_image(record, data_root):
    return Image.open(os.path.join(data_root, record["image"])).convert("RGB")


def to_unsloth_sample(record, data_root):
    """unsloth's vision SFT trainer expects each sample as
    {"messages": [...]} with image objects inlined into the user turn."""
    img = resolve_image(record, data_root)
    convo = record["conversations"]
    messages = []
    for turn in convo:
        content = []
        for c in turn["content"]:
            if c["type"] == "image":
                content.append({"type": "image", "image": img})
            else:
                content.append({"type": "text", "text": c["text"]})
        messages.append({"role": turn["role"], "content": content})
    return {"messages": messages}


def system_prompt():
    return (
        "You are a document intelligence assistant for an identity-verification "
        "pipeline. You either extract structured fields from receipts/documents, "
        "or assess documents for signs of digital tampering. Be precise and concise."
    )
