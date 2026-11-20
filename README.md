# kyc-kie: KYC text to JSON

Turns the raw text an OCR engine reads from an identity document into one JSON object.
Built for Khmer national ID cards, passports (Cambodian and foreign) and Cambodian visas,
with fields in Khmer, English or both.

This is a Python library: the model (about 1.9 MB, inside the package) and the code around
it.

A small character-level tagger (BiGRU-CRF, 472,488 parameters) marks the spans; plain code
reads the fixed-pattern fields and assembles the JSON. Nothing is generated: every value is
either a piece of the input text or parsed from one.

## Install

From this folder, into the environment of the project that will use it (a path or a git
URL of this folder works the same way):

```bash
pip install .
```

The only dependencies are `numpy` and
`onnxruntime`; Python 3.9 or newer.

## Use

```python
from kyc_kie import extract

fields = extract(ocr_text)        # a dict; see Output below
print(fields["full_name_km"], fields["date_of_birth"])
```

The first call loads the model (a few milliseconds per document after that). To keep the
extractor yourself, use another model file, or see how each value was found:

```python
from kyc_kie import Extractor

ex = Extractor()                                   # or Extractor("path/to/model.onnx")
fields = ex(ocr_text)
fields, sources, spans = ex.details(ocr_text)      # sources: {"sex": "mrz", "address": "model", ...}
```

The rules are usable on their own: `parse_mrz(text)`, `parse_date("០៥.០៣.១៩៩០")`,
`find_dates(text)`.

From the command line:

```bash
kyc-kie examples/1_khmer_id.txt --debug
```

(`python -m kyc_kie` is the same; text can also come from `--text "..."` or standard input.)

## Output

```json
{
  "document_type": "id_card",          // id_card, passport, visa or null
  "document_number": "010234567",      // the ID, passport or visa number; a Khmer ID's bracket is kept: "010234567 (01)"
  "passport_number": null,             // passports: same as document_number; visas: the passport the visa is in
  "full_name_km": "សុខ សុភា",
  "full_name_en": "SOK SOPHEA",
  "surname": "SOK",
  "given_names": "SOPHEA",
  "sex": "F",                          // M or F
  "date_of_birth": "1990-03-05",       // dates are YYYY-MM-DD
  "nationality": "KHM",
  "place_of_birth": "ឃុំព្រែកអញ្ចាញ ស្រុកមុខកំពូល កណ្តាល",
  "address": "ផ្ទះ១២ ផ្លូវ២៧១ ភូមិ៣ សង្កាត់ទឹកថ្លា ខណ្ឌសែនសុខ ភ្នំពេញ",
  "height_cm": 158,
  "distinguishing_marks": "ប្រជ្រុយចំ ០.៥ស.ម លើចិញ្ចើមឆ្វេង",   // ភិនភាគ
  "issue_date": "2020-06-12",
  "expiry_date": "2030-06-11",
  "visa_type": null,
  "mrz": "IDKHM0102345670<<<<<<<<<<<<<<<\n9003059F3006117KHM<<<<<<<<<<<4\nSOK<<SOPHEA<<<<<<<<<<<<<<<<<<<",
  "mrz_valid": true                    // the check digits fit; null when there are no machine-readable lines
}
```

A field that is not found is `null`. Free-text fields (names, places, address, nationality,
distinguishing marks, visa type) are copied from the text as the OCR engine read them: the model finds them, it
does not correct them.


## How a value is chosen

| Field | Order of trust |
|---|---|
| sex, date of birth, expiry date | machine-readable lines with right check digits, then a label on the same line (regex), then the tagger's span, then machine-readable lines with wrong check digits |
| issue date, height | label on the same line, then the tagger's span |
| document number | machine-readable lines with a right check digit, then a line that is only 9 digits (Khmer ID), then the tagger's span |
| names, nationality | the tagger's span, then the machine-readable lines |
| Khmer name, birth place, address, distinguishing marks, visa type | the tagger's span |
| mrz, mrz_valid | rules only |
| document type | first letter of the machine-readable lines when their check digits are right, then the model's class |

Dates are read in these forms: `០៥.០៣.១៩៩០`, `05/03/1990`, `05-03-1990`, `1990.03.05`,
`05 MAR 1990`, `05-Mar-1990`, `05 MAR / MARS 90`, `05 3月/MAR 1990`, `5 March 1990`,
`ថ្ងៃទី០៥ ខែមីនា ឆ្នាំ១៩៩០`. `O` and `l` between digits are read as `0` and `1`.


## Architecture

The work is split in two because the fields are of two kinds. Numbers, dates, sex and the
machine-readable lines have a fixed shape, and some carry check digits, so code can read
them and prove them right. Names, addresses and birth places have no shape; finding them
needs a model that has seen many documents.

```
                         raw OCR text (any length)
                                   │
              ┌────────────────────┴─────────────────────┐
              ▼                                          ▼
   1. TAGGER (model, 472,488 parameters)        2. RULES (plain code)
   one id per character (240-entry table)       machine-readable lines: found, repaired
   embedding, 64 numbers per character            (O/0, I/1 by column), check digits tested
   2-layer bidirectional GRU, 128 per way       dates: ten written forms -> YYYY-MM-DD
   linear layer: 33 label scores per char       label on the same line: birth date, issue,
   CRF: best legal label sequence (Viterbi)       expiry, sex, height; 9-digit Khmer ID line
   + document class from the averaged states
              │                                          │
              ▼                                          ▼
   spans: NAME_KM, ADDRESS, DOB, DOC_NO ...     values with proof (check digit) or an anchor
   class: id_card / passport / visa / other
              └────────────────────┬─────────────────────┘
                                   ▼
        3. ASSEMBLY: per field, take the most trusted source that has a value
                                   ▼
                                 JSON
```

**1. The tagger.** The text is read one character at a time, never split into words: OCR
errors break word lists, and Khmer has no spaces between words. Each character becomes an
id from a fixed table (ASCII, the Khmer block, a few signs; Chinese, Thai and other
scripts each share one id, and accented Latin letters use their base letter). The GRU
reads the whole text in both directions, so the label of a character can depend on a label
word several lines away, which is what makes "label on one line, value on the next" and
shuffled lines workable. The output for every character is one of 33 labels: `O` (not a
field) or `B-`/`I-` (begins / continues) for each of the 16 span types. The CRF on top
scores whole label sequences instead of single characters and forbids impossible ones
(an `I-ADDRESS` right after `O`), so spans come out whole. A second small output, from
the average and the maximum of the GRU's states, says what kind of document the text is.

| Part | Parameters |
|---|---|
| character embeddings (240 × 64) | 15,360 |
| bidirectional GRU, 2 layers, hidden 128 | 445,440 |
| linear layer to 33 labels | 8,481 |
| document class (4 classes) | 2,052 |
| CRF transitions, start, end | 1,155 |
| total | 472,488 |

**2. The rules** (`kyc_kie/rules.py`) never guess. Machine-readable lines are found even when
the OCR engine puts them on one text line or glues them together; characters are repaired
by column (a letter column cannot hold `0`, a digit column cannot hold `O`) and the check
digits decide whether the result is trusted. Dates are parsed from ten written forms in
Khmer and Latin digits. A value right after its label on the same line is taken by regex.

**3. Assembly** picks, per field, the first source that has a value in the order of trust
shown in the next section, normalises it (dates, sex, height, numbers) and reports where
it came from.

**Training data.** The tagger learns from 50,000 synthetic documents: made-up people,
real layouts (Khmer ID front, back and both sides; Khmer and foreign passports; visa
sticker and e-visa; two-language cards; free "label: value" documents in Khmer, English
or both; and non-ID documents), then OCR noise: look-alike characters, lost and stray
characters, lost Khmer marks, merged, swapped and lost lines, stray lines. Labels move
with the characters, so they stay exact. Place names are real (Cambodia's provinces, districts,
communes and villages); people and numbers are made up.

**On a device.** The model is one ONNX file of about 1.9 MB (float32) plus a small JSON
with the CRF scores. Decoding and the rules are plain code with no dependencies beyond
numpy, so they port directly to Kotlin, Swift or JavaScript.


## Results

Measured on text the model never saw in training.

**Speed** (text in, JSON out; one Apple M1 Pro core, ONNX; the OCR step is not included): about
88 documents per second, 11 ms per document on average, 18 ms at the 95th percentile. The
model loads in about 4 ms.

**Accuracy on 3,000 synthetic documents** (share of fields right): 99.6% on clean text, 98.1%
with light OCR noise, 94.5% with heavy noise.

| Field | Accuracy | Field | Accuracy |
|---|---|---|---|
| `distinguishing_marks` | 99.9% | `place_of_birth` | 98.8% |
| `sex` | 99.7% | `document_type` | 98.6% |
| `full_name_km` | 99.7% | `height_cm` | 97.8% |
| `visa_type` | 99.6% | `date_of_birth` | 97.3% |
| `address` | 99.3% | `full_name_en` | 97.2% |
| `passport_number` | 96.9% | `expiry_date` | 96.3% |
| `nationality` | 96.1% | `surname` | 95.3% |
| `given_names` | 95.0% | `document_number` | 94.8% |
| `issue_date` | 89.1% | | |

**On 89 real photographed documents** read by an OCR engine, the fields with a fixed shape come
out right 81% to 88% of the time (birth date 86.7%, expiry date 88.0%, sex 88.0%, height 85.5%,
issue date 83.9%, document number 80.7%). Free-text Khmer fields are limited by the OCR text
itself: the right Khmer name is in the OCR output letter for letter only 23% of the time, so
exact match cannot go higher than that, while the extracted address, birth place and
distinguishing marks are about 85% similar, character by character, to the true text.

## Limits

- Values are copied from the OCR text. An OCR mistake inside a name or an address stays in
  the output; only dates, sex, height and numbers are normalised, and only numbers and dates
  in machine-readable lines can be proven by check digits.
- A digit damaged by OCR in a date or number with no machine-readable line to compare cannot
  be detected.
- Built for Cambodian ID cards, passports and Cambodian visas. Other countries' ID cards are
  not covered; foreign passports are covered through their English labels and machine-readable lines.
- `document_type` is the least reliable field on very short or partial texts.

## Files

| File | What it is |
|---|---|
| `src/kyc_kie/model/model.onnx`, `model.onnx.json` | the tagger, and its CRF scores and label list |
| `src/kyc_kie/infer.py` | `extract`, `Extractor`, the ONNX runner, the command line |
| `src/kyc_kie/common.py` | labels, character table, BIO conversion, Viterbi decoding |
| `src/kyc_kie/rules.py` | machine-readable lines, dates, label regexes, JSON assembly |
| `examples/` | four noisy texts of made-up people |


## Licence

MIT. See `LICENSE`.
