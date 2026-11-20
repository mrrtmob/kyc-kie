"""kyc-kie: raw OCR text of an ID card, passport or visa -> structured fields.

    from kyc_kie import extract
    fields = extract(ocr_text)

    from kyc_kie import Extractor
    ex = Extractor()                              # or Extractor("path/to/model.onnx")
    fields, sources, spans = ex.details(ocr_text)
"""
from .common import DOC_CLASSES, ENTITIES, FIELDS
from .infer import Extractor, OnnxRunner, extract
from .rules import assemble, find_dates, parse_date, parse_mrz

__version__ = "0.1.0"
__all__ = ["extract", "Extractor", "OnnxRunner", "FIELDS", "ENTITIES", "DOC_CLASSES", "assemble", "parse_mrz",
           "parse_date", "find_dates"]
