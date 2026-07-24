"""Parses CORD's raw `ground_truth` (not just the gt_parse summary) to recover
word/line-level bounding boxes, so synthetic tampering can be placed on actual
document content instead of blank background."""


def _quad_to_bbox(quad):
    xs = [quad["x1"], quad["x2"], quad["x3"], quad["x4"]]
    ys = [quad["y1"], quad["y2"], quad["y3"], quad["y4"]]
    return min(xs), min(ys), max(xs), max(ys)


def extract_line_boxes(ground_truth: dict):
    """Returns a list of (x0, y0, x1, y1) axis-aligned boxes, one per text row,
    by unioning all word quads that share a row_id."""
    lines = {}
    for entry in ground_truth.get("valid_line", []) or []:
        for word in entry.get("words", []) or []:
            quad = word.get("quad")
            if not quad:
                continue
            row_id = word.get("row_id")
            x0, y0, x1, y1 = _quad_to_bbox(quad)
            if row_id not in lines:
                lines[row_id] = [x0, y0, x1, y1]
            else:
                b = lines[row_id]
                b[0], b[1] = min(b[0], x0), min(b[1], y0)
                b[2], b[3] = max(b[2], x1), max(b[3], y1)
    return [tuple(b) for b in lines.values() if b[2] > b[0] and b[3] > b[1]]
