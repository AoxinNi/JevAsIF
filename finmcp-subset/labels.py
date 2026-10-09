"""Review labels for D-group units: stored in review_labels.json, mirrored into review.csv."""
import csv
import json
import os
import tempfile
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
LABELS = HERE / "review_labels.json"
REVIEW_CSV = HERE / "review.csv"

SEMANTIC = {"yes": "是：要理解客户意思才能选", "no_rule": "否：按数字或规则就能算出", "no_other": "否：其他原因"}
REFERENCE = {"unique": "合理且唯一", "multi": "合理但不唯一", "arbitrary": "随意", "bad": "不合理"}
CSV_SEMANTIC = {"yes": "Y", "no_rule": "N（数字/规则可算）", "no_other": "N"}
CSV_REFERENCE = {"unique": "Y", "multi": "合理但不唯一", "arbitrary": "随意", "bad": "N"}
CSV_COLUMNS = ("真需要语义判断(Y/N)", "标准答案合理且唯一(Y/N/随意)", "备注")


def load() -> dict:
    if LABELS.exists():
        return json.loads(LABELS.read_text(encoding="utf-8")).get("labels", {})
    return {}


def _atomic_write(path: Path, text: str, encoding="utf-8"):
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".tmp")
    with os.fdopen(fd, "w", encoding=encoding, newline="") as f:
        f.write(text)
    os.replace(tmp, path)


def save(labels: dict):
    _atomic_write(LABELS, json.dumps({"version": 1, "labels": labels}, ensure_ascii=False, indent=1))
    sync_csv(labels)


def update(labels: dict, unit_id: str, label: dict | None) -> dict:
    if label is None:
        labels.pop(unit_id, None)
    else:
        labels[unit_id] = {**label, "updated": datetime.now().isoformat(timespec="seconds")}
    save(labels)
    return labels


def cells(label: dict | None) -> list:
    if not label:
        return ["", "", ""]
    note = "；".join([*label.get("tags", []), *([label["note"]] if label.get("note") else [])])
    return [CSV_SEMANTIC.get(label.get("semantic"), ""), CSV_REFERENCE.get(label.get("reference"), ""), note]


def sync_csv(labels: dict):
    """Fill the three judgment columns of review.csv from the labels."""
    if not REVIEW_CSV.exists():
        return
    with REVIEW_CSV.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.reader(f))
    header, body = rows[0], rows[1:]
    idx = [header.index(c) for c in CSV_COLUMNS]
    for row in body:
        for i, value in zip(idx, cells(labels.get(row[0]))):
            row[i] = value
    import io
    buf = io.StringIO()
    csv.writer(buf).writerows([header, *body])
    _atomic_write(REVIEW_CSV, "﻿" + buf.getvalue())
