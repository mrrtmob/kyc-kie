"""Shared pieces that need no PyTorch: the label set, the character vocabulary, BIO
conversion, Viterbi decoding and JSONL reading."""
import json
import unicodedata

import numpy as np

# what the tagger labels (spans in the text)
ENTITIES = ["DOC_NO", "PASSPORT_NO", "NAME_KM", "NAME_EN", "SURNAME", "GIVEN", "SEX", "DOB", "NATIONALITY",
            "POB", "ADDRESS", "HEIGHT", "ISSUE", "EXPIRY", "VISA_TYPE", "MARKS"]
LABELS = ["O"] + [f"{p}-{e}" for e in ENTITIES for p in ("B", "I")]
LABEL_ID = {l: i for i, l in enumerate(LABELS)}

# the fields of the final JSON, in output order
FIELDS = ["document_type", "document_number", "passport_number", "full_name_km", "full_name_en", "surname",
          "given_names", "sex", "date_of_birth", "nationality", "place_of_birth", "address", "height_cm",
          "distinguishing_marks",
          "issue_date", "expiry_date", "visa_type"]

# what the model's second output says about the whole text
DOC_CLASSES = ["other", "id_card", "passport", "visa"]

# ---- character vocabulary: fixed, so there is nothing to build or to save -------------
PAD = 0
_BUCKETS = ["<pad>", "<khmer>", "<latin>", "<digit>", "<cjk>", "<thai>", "<letter>", "<other>"]
_CHARS = (["\n"] + [chr(c) for c in range(32, 127)] + [chr(c) for c in range(0x1780, 0x17FA)]
          + list("«»—–•·…“”‘’°№×"))
VOCAB = _BUCKETS + _CHARS
CHAR_ID = {c: i for i, c in enumerate(VOCAB)}
_B = {b: i for i, b in enumerate(_BUCKETS)}


def char_id(ch):
    """One id per character. Characters outside the vocabulary go to a bucket for their
    script, so a Chinese or Thai name is 'some CJK' / 'some Thai' and not a hole."""
    i = CHAR_ID.get(ch)
    if i is not None:
        return i
    o = ord(ch)
    if 0x1780 <= o <= 0x17FF or 0x19E0 <= o <= 0x19FF:
        return _B["<khmer>"]
    if 0x2E80 <= o <= 0x9FFF or 0xAC00 <= o <= 0xD7AF or 0xF900 <= o <= 0xFAFF or 0xFF00 <= o <= 0xFFEF:
        return _B["<cjk>"]
    if 0x0E00 <= o <= 0x0E7F:
        return _B["<thai>"]
    if ch in "\t\r ​‌‍　":
        return CHAR_ID[" "]
    cat = unicodedata.category(ch)
    if cat == "Nd":
        return _B["<digit>"]
    if cat[0] == "L":
        base = unicodedata.normalize("NFD", ch)[0]  # É -> E, ữ -> u
        if base in CHAR_ID and base != ch:
            return CHAR_ID[base]
        return _B["<latin>"] if o < 0x250 else _B["<letter>"]
    if cat[0] == "M":
        return _B["<letter>"]
    return _B["<other>"]


def encode(text):
    return [char_id(c) for c in text]


# ---- spans <-> BIO ---------------------------------------------------------------
def spans_to_tags(n, spans):
    """spans: [{start, end, label}] (end exclusive, character offsets) -> n label ids."""
    tags = [0] * n
    for s in sorted(spans, key=lambda s: s["start"]):
        a, b = max(0, s["start"]), min(n, s["end"])
        if b <= a or s["label"] not in ENTITIES:
            continue
        tags[a] = LABEL_ID["B-" + s["label"]]
        for i in range(a + 1, b):
            tags[i] = LABEL_ID["I-" + s["label"]]
    return tags


def tags_to_spans(tags):
    spans, start, ent = [], None, None
    for i, t in enumerate(list(tags) + [0]):
        lab = LABELS[t] if i < len(tags) else "O"
        if start is not None and (lab == "O" or lab[0] == "B" or lab[2:] != ent):
            spans.append({"start": start, "end": i, "label": ent})
            start = ent = None
        if lab != "O" and start is None:  # a stray I- opens a span too
            start, ent = i, lab[2:]
    return spans


def allowed_transitions():
    """allowed[i, j]: label i may be followed by j (I-X only after B-X or I-X)."""
    n = len(LABELS)
    ok = np.ones((n, n), dtype=bool)
    for j, lj in enumerate(LABELS):
        if lj[0] == "I":
            for i, li in enumerate(LABELS):
                ok[i, j] = li != "O" and li[2:] == lj[2:]
    start_ok = np.array([l[0] != "I" for l in LABELS])
    return ok, start_ok


def viterbi(em, trans, start, end):
    """Best label sequence. em: (T, L) emissions; trans[i, j]: score of i -> j."""
    T = em.shape[0]
    score = start + em[0]
    back = np.zeros((T, em.shape[1]), dtype=np.int32)
    for t in range(1, T):
        cand = score[:, None] + trans
        back[t] = cand.argmax(0)
        score = cand.max(0) + em[t]
    best = int((score + end).argmax())
    path = [best]
    for t in range(T - 1, 0, -1):
        best = int(back[t, best])
        path.append(best)
    return path[::-1]


# ---- JSONL ---------------------------------------------------------------------
def resolve_spans(row):
    """Spans of a JSONL row as {start, end, label}. A span may give "text" instead of
    offsets: it is then the first occurrence of that text not already used."""
    text, out, used = row["text"], [], []
    for s in row.get("spans", []):
        if "start" in s and "end" in s:
            out.append({"start": int(s["start"]), "end": int(s["end"]), "label": s["label"]})
            continue
        pos, frm = -1, 0
        while True:
            pos = text.find(s["text"], frm)
            if pos < 0 or not any(a < pos + len(s["text"]) and pos < b for a, b in used):
                break
            frm = pos + 1
        if pos < 0:
            raise ValueError(f"row {row.get('id')}: span text {s['text']!r} is not in the text")
        used.append((pos, pos + len(s["text"])))
        out.append({"start": pos, "end": pos + len(s["text"]), "label": s["label"]})
    for s in out:
        if s["label"] not in ENTITIES:
            raise ValueError(f"row {row.get('id')}: unknown label {s['label']!r}; use one of {ENTITIES}")
    return out


def read_jsonl(path, limit=None):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                r["spans"] = resolve_spans(r)
                rows.append(r)
                if limit and len(rows) >= limit:
                    break
    return rows


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
