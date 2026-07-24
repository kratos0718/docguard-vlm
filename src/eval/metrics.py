"""Scoring functions for the two tasks. Kept dependency-free (no model imports)
so they can be unit-tested without a GPU."""
import json
import re


_NEGATED_TAMPER_RE = re.compile(
    r"(\bno\b|\bnot\b|n't\b|\bwithout\b|\bfree of\b|\bzero\b|\bnone\b)[^.\n]{0,25}"
    r"\b(tamper\w*|forg\w*|alter\w*|manipulat\w*|splic\w*|edit\w*)\b"
)
_TAMPER_RE = re.compile(r"\b(tamper\w*|forg\w*|alter\w*|manipulat\w*|splic\w*)\b")
_AUTHENTIC_RE = re.compile(r"\b(authentic|genuine|unaltered|original)\b")


def parse_verdict(text: str) -> str:
    """Maps free-text model output to {authentic, tampered, unknown}.

    Naive substring checks misfire on phrasing like "no signs of tampering"
    (contains "tamper" but means authentic), so negation is checked first,
    then explicit positive cues, in priority order.
    """
    t = text.lower()
    first_sentence = re.split(r"[.\n]", t, maxsplit=1)[0]

    for chunk in (first_sentence, t):
        if _NEGATED_TAMPER_RE.search(chunk):
            return "authentic"
        if _AUTHENTIC_RE.search(chunk):
            return "authentic"
        if _TAMPER_RE.search(chunk):
            return "tampered"
    return "unknown"


def forgery_accuracy(predictions, gold_labels):
    """predictions/gold_labels: list of raw strings ('tampered'/'authentic') and
    model output text respectively -> returns accuracy, precision/recall/F1 for
    the 'tampered' class, and the confusion breakdown."""
    tp = fp = fn = tn = unknown = 0
    for pred_text, gold in zip(predictions, gold_labels):
        pred = parse_verdict(pred_text)
        if pred == "unknown":
            unknown += 1
            continue
        if gold == "tampered" and pred == "tampered":
            tp += 1
        elif gold == "tampered" and pred == "authentic":
            fn += 1
        elif gold == "authentic" and pred == "tampered":
            fp += 1
        elif gold == "authentic" and pred == "authentic":
            tn += 1

    total = tp + fp + fn + tn
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "n": len(predictions),
        "accuracy": accuracy,
        "precision_tampered": precision,
        "recall_tampered": recall,
        "f1_tampered": f1,
        "unknown_rate": unknown / len(predictions) if predictions else 0.0,
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
    }


_NUM_RE = re.compile(r"[0-9]+(?:[.,][0-9]+)*")


def _normalize_number(s):
    if s is None:
        return None
    digits = re.sub(r"[^0-9]", "", str(s))
    return digits if digits else None


def try_parse_json(text: str):
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def ocr_field_match(pred_text: str, gold_response: str):
    """Loose field-level match on 'total' (JSON-parse pred; fall back to regex
    over raw text if the model didn't emit valid JSON -- this is itself a metric
    worth reporting, since a real pipeline needs a parseable output)."""
    gold = try_parse_json(gold_response) or {}
    gold_total = _normalize_number(gold.get("total"))

    pred = try_parse_json(pred_text)
    json_valid = pred is not None
    if pred is not None:
        pred_total = _normalize_number(pred.get("total"))
    else:
        nums = _NUM_RE.findall(pred_text)
        pred_total = _normalize_number(nums[-1]) if nums else None

    total_match = (gold_total is not None and pred_total is not None and gold_total == pred_total)
    return {"json_valid": json_valid, "total_match": total_match}


def ocr_accuracy(predictions, gold_responses):
    results = [ocr_field_match(p, g) for p, g in zip(predictions, gold_responses)]
    n = len(results) or 1
    return {
        "n": len(results),
        "json_valid_rate": sum(r["json_valid"] for r in results) / n,
        "total_match_rate": sum(r["total_match"] for r in results) / n,
    }
