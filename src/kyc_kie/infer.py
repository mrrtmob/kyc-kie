"""Text in, JSON out: the ONNX runner, the Extractor and the command line.

    python -m kyc_kie scan.txt               # a text file
    python -m kyc_kie --text "..."           # text on the command line
    cat scan.txt | python -m kyc_kie         # text on standard input
    python -m kyc_kie scan.txt --debug       # also print the spans and where each value came from

Dates come out as YYYY-MM-DD, sex as M / F, height in centimetres; a field that is not
found is null. The model is the one inside the package; --model names another model.onnx.
"""
import argparse
import json
import os
import sys

import numpy as np

from .common import DOC_CLASSES, LABELS, VOCAB, encode, tags_to_spans, viterbi
from .rules import assemble


DEFAULT_MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model", "model.onnx")


class OnnxRunner:
    """model.onnx plus model.onnx.json (the CRF's transition scores and the label list)."""

    def __init__(self, path=None, threads=None):
        import onnxruntime as ort
        path = str(path or DEFAULT_MODEL)
        if not os.path.exists(path) or not os.path.exists(path + ".json"):
            raise FileNotFoundError(f"No model at {path} (model.onnx and model.onnx.json are needed).")
        opts = ort.SessionOptions()
        if threads:  # CPU threads for one document; None leaves it to onnxruntime
            opts.intra_op_num_threads = opts.inter_op_num_threads = int(threads)
        self.session = ort.InferenceSession(path, sess_options=opts, providers=["CPUExecutionProvider"])
        with open(path + ".json", encoding="utf-8") as f:
            meta = json.load(f)
        if meta["labels"] != LABELS or meta["vocab_size"] != len(VOCAB):
            raise SystemExit(f"{path} was exported with another label set or vocabulary than this code has.")
        self.trans, self.start, self.end = (np.array(meta[k], dtype=np.float32) for k in ("transitions", "start", "end"))

    def scores(self, ids):
        em, cls = self.session.run(None, {"ids": np.array([ids], dtype=np.int64)})
        return em[0], cls[0]


def tag(text, runner):
    """(spans [{start, end, label, text}], document class) for one text."""
    if not text.strip():
        return [], "other"
    em, cls = runner.scores(encode(text))
    spans = tags_to_spans(viterbi(em, runner.trans, runner.start, runner.end))
    for s in spans:
        s["text"] = text[s["start"]:s["end"]]
    return spans, DOC_CLASSES[int(cls.argmax())]


def run(text, runner, today=None):
    """(fields, sources, spans) for one raw OCR text. runner: anything with scores(ids) and
    the CRF's trans / start / end (OnnxRunner here)."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    spans, kind = tag(text, runner)
    fields, sources = assemble(text, spans, kind, today)
    return fields, sources, spans


class Extractor:
    """Loads the model once; call it for every document.

        ex = Extractor()                 # the model inside the package
        fields = ex(ocr_text)            # {"document_type": "id_card", "full_name_km": ..., ...}
        fields, sources, spans = ex.details(ocr_text)
    """

    def __init__(self, model=None, threads=None):
        self.runner = OnnxRunner(model, threads)

    def __call__(self, text, today=None):
        return run(text, self.runner, today)[0]

    extract = __call__

    def details(self, text, today=None):
        """(fields, where each value came from, the spans the tagger marked)."""
        return run(text, self.runner, today)


_default = None


def extract(text, today=None):
    """The JSON fields of one raw OCR text, with the model inside the package."""
    global _default
    if _default is None:
        _default = Extractor()
    return _default(text, today)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file", nargs="?", help="text file with the raw OCR text (default: standard input)")
    ap.add_argument("--text", help="the raw OCR text itself")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--debug", action="store_true")
    a = ap.parse_args()
    if a.text is not None:
        text = a.text.replace("\\n", "\n")
    elif a.file:
        with open(a.file, encoding="utf-8") as f:
            text = f.read()
    else:
        text = sys.stdin.read()
    fields, sources, spans = Extractor(a.model).details(text)
    if a.debug:
        for s in spans:
            print(f"  {s['label']:<12} {s['text']!r}", file=sys.stderr)
        for k, v in sources.items():
            print(f"  {k:<16} from {v}", file=sys.stderr)
    print(json.dumps(fields, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
