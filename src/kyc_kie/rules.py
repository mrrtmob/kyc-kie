"""The plain-code half of the pipeline: machine-readable zone (with check digits), dates,
label-anchored regexes, and the assembly of the final JSON from the tagger's spans.
No PyTorch here."""
import datetime
import re
import unicodedata

from .common import FIELDS

KH2LAT = str.maketrans("០១២៣៤៥៦៧៨៩", "0123456789")
KH_MONTHS = ["មករា", "កុម្ភៈ", "មីនា", "មេសា", "ឧសភា", "មិថុនា", "កក្កដា", "សីហា", "កញ្ញា", "តុលា", "វិច្ឆិកា", "ធ្នូ"]
_MONTHS = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6, "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
    "JANUARY": 1, "FEBRUARY": 2, "MARCH": 3, "APRIL": 4, "JUNE": 6, "JULY": 7, "AUGUST": 8, "SEPTEMBER": 9, "SEPT": 9,
    "OCTOBER": 10, "NOVEMBER": 11, "DECEMBER": 12,
    "JANV": 1, "FEV": 2, "FEVR": 2, "MARS": 3, "AVR": 4, "MAI": 5, "JUIN": 6, "JUIL": 7, "AOUT": 8, "DEZ": 12, "OKT": 10,
}


def _strip_accents(s):
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


# ======================= machine-readable zone =======================
_TO_DIGIT = str.maketrans("OQDIL|ZSBG", "0001112586")
_TO_ALPHA = str.maketrans("0125868", "OIZSBGB")


def check_digit(s):
    total = 0
    for i, c in enumerate(s):
        v = int(c) if c.isdigit() else (ord(c) - 55 if "A" <= c <= "Z" else 0)
        total += v * (7, 3, 1)[i % 3]
    return str(total % 10)


def _mrz_clean(tok):
    t = tok.upper().replace("«", "<<").replace("‹", "<").replace("〈", "<")
    return re.sub(r"[^A-Z0-9<]", "", t)


def _looks_mrz(t):
    return 24 <= len(t) <= 48 and (t.count("<") >= 3 or (len(t) >= 28 and sum(c.isdigit() for c in t) >= 10))


def mrz_tokens(text):
    """Machine-readable lines in reading order. An OCR engine may put the two or three
    lines on one text line with spaces between them, or break one line with a space."""
    out = []
    for line in text.split("\n"):
        toks = []
        for t in (_mrz_clean(x) for x in line.split()):
            if len(t) > 48 and t.count("<") >= 6:  # lines glued together: cut at the standard widths
                n, w = min(((3, 30), (2, 30), (2, 44), (2, 36)), key=lambda nw: abs(len(t) - nw[0] * nw[1]))
                toks += [t[i * w:(i + 1) * w if i < n - 1 else None] for i in range(n)]
            else:
                toks.append(t)
        toks = [t for t in toks if _looks_mrz(t)]
        if not toks:
            whole = _mrz_clean(line)
            if _looks_mrz(whole) and line.count("<") >= 3:
                toks = [whole]
        out.extend(toks)
    return out


def _yymmdd(s, birth, today):
    try:
        yy, mm, dd = int(s[0:2]), int(s[2:4]), int(s[4:6])
        if birth:
            year = 2000 + yy if 2000 + yy <= today.year else 1900 + yy
        else:
            year = 2000 + yy if yy < 80 else 1900 + yy
        return datetime.date(year, mm, dd).isoformat()
    except ValueError:
        return None


def _date_window(tok, i, today):
    """birth(6) check sex expiry(6) check starting at i, both check digits right."""
    w = tok[i:i + 15]
    if len(w) < 15:
        return None
    b, e = w[0:7].translate(_TO_DIGIT), w[8:15].translate(_TO_DIGIT)
    sex = w[7]
    if not (b.isdigit() and e.isdigit() and sex in "MFX<"):
        return None
    if check_digit(b[:6]) != b[6] or check_digit(e[:6]) != e[6]:
        return None
    dob, exp = _yymmdd(b, True, today), _yymmdd(e, False, today)
    if not dob or not exp:
        return None
    return {"dob": dob, "expiry": exp, "sex": sex if sex in "MF" else None}


def _number(field9, check, digits_only=False):
    """(number, check digit right?) for a 9-character document number field."""
    cands = [field9.translate(_TO_DIGIT)] if digits_only else [field9, field9.replace("O", "0"), field9.replace("I", "1").replace("O", "0")]
    chk = check.translate(_TO_DIGIT)
    for c in cands:
        if len(c) == 9 and check_digit(c) == chk:
            return c.rstrip("<"), True
    return cands[0].rstrip("<"), False


def _names(s):
    s = re.split(r"<{3,}", s.translate(_TO_ALPHA).lstrip("<"))[0].rstrip("<")  # what follows the filler is noise
    if not s:
        return None, None
    parts = s.split("<<", 1)
    sur = parts[0].replace("<", " ").strip() or None
    giv = (parts[1].replace("<", " ").strip() or None) if len(parts) > 1 else None
    return sur, giv


def parse_mrz(text, today=None):
    """None when the text has no machine-readable lines. Otherwise what they say, with
    number_ok / dates_ok telling whether the check digits were right."""
    today = today or datetime.date.today()
    toks = mrz_tokens(text)
    if not toks:
        return None
    r = {"lines": toks, "type": None, "number": None, "number_ok": False, "nationality": None, "dob": None,
         "expiry": None, "sex": None, "dates_ok": False, "surname": None, "given": None}
    data, off = None, None
    for k, t in enumerate(toks):  # the line with the dates: the usual offsets first, then anywhere
        for i in [13, 0] + [j for j in range(len(t) - 14) if j not in (13, 0)]:
            w = _date_window(t, i, today)
            if w:
                data, off = k, i
                r.update(w, dates_ok=True)
                break
        if data is not None:
            break
    two_line = None
    if data is not None:
        t = toks[data]
        if off >= 8:  # passport / visa: number(9) check nationality(3) dates
            two_line = True
            r["nationality"] = t[off - 3:off].translate(_TO_ALPHA).replace("<", "") or None
            r["number"], r["number_ok"] = _number(t[max(0, off - 13):off - 4].ljust(9, "<")[:9], t[off - 4])
        else:  # card: dates first, nationality after; the number is on the line above
            two_line = False
            r["nationality"] = t[off + 15:off + 18].translate(_TO_ALPHA).replace("<", "") or None
    else:  # no valid check digits anywhere: read fixed columns, trusted last
        long = [k for k, t in enumerate(toks) if len(t) >= 34]
        if long:
            two_line, data = True, long[-1]
            t = toks[data].ljust(28, "<")
            r["number"], r["number_ok"] = _number(t[0:9], t[9])  # the number has its own check digit
            r["number"] = r["number"] or None
            r["nationality"] = t[10:13].translate(_TO_ALPHA).replace("<", "") or None
            r["dob"], r["expiry"] = _yymmdd(t[13:19].translate(_TO_DIGIT), True, today), _yymmdd(t[21:27].translate(_TO_DIGIT), False, today)
            r["sex"] = t[20] if t[20] in "MF" else None
        elif len(toks) >= 2:
            two_line = False
            cand = [k for k, t in enumerate(toks) if sum(c.isdigit() for c in t[:15].translate(_TO_DIGIT)) >= 11 and not t.startswith(("ID", "I<", "AC", "C<"))]
            if cand:
                data = cand[0]
                t = toks[data].ljust(18, "<")
                r["dob"], r["expiry"] = _yymmdd(t[0:6].translate(_TO_DIGIT), True, today), _yymmdd(t[8:14].translate(_TO_DIGIT), False, today)
                r["sex"] = t[7] if t[7] in "MF" else None
                r["nationality"] = t[15:18].translate(_TO_ALPHA).replace("<", "") or None
    others = [k for k in range(len(toks)) if k != data]
    if two_line is False:
        first = [k for k in others if re.match(r"[IAC1][A-Z<0-9][A-Z<0-9]{3}", toks[k]) and sum(c.isdigit() for c in toks[k]) >= 6]
        if first:
            t = toks[first[0]]
            r["type"] = "id_card"
            cambodia = t[2:5].translate(_TO_ALPHA) == "KHM"
            r["number"], r["number_ok"] = _number(t[5:14].ljust(9, "<"), t[14:15] or "?", digits_only=cambodia)
            others = [k for k in others if k != first[0]]
        name = [k for k in others if "<<" in toks[k].rstrip("<") or sum(c.isalpha() for c in toks[k]) >= 6]
        if name:
            r["surname"], r["given"] = _names(toks[name[-1]])
    else:
        name = [k for k in others if sum(c.isalpha() for c in toks[k]) > sum(c.isdigit() for c in toks[k]) + 4]
        if name:
            t = toks[name[0]]
            if t[0] in "PV" and len(t) >= 30:
                r["type"] = "passport" if t[0] == "P" else "visa"
                t = t[5:]
            elif re.match(r"[A-Z<]{2}[A-Z]{3}", t) and len(t) >= 40:
                t = t[5:]
            r["surname"], r["given"] = _names(t)
    return r


# ======================= dates =======================
_N = "[0-9]"
_L = "A-Za-zÀ-ÿ"
_DATE_RES = [
    ("ymd", re.compile(rf"(?<![0-9])({_N}{{4}})\s?[./\-年]\s?({_N}{{1,2}})\s?[./\-月]\s?({_N}{{1,2}})(?![0-9])")),
    ("dmy", re.compile(rf"(?<![0-9])({_N}{{1,2}})\s?[./\-]\s?({_N}{{1,2}})\s?[./\-]\s?({_N}{{4}})(?![0-9])")),
    ("dMy", re.compile(rf"(?<![0-9])({_N}{{1,2}})\s?[-/. ]?\s?(?:{_N}{{1,2}}月\s?/\s?)?([{_L}]{{3,9}})\.?(?:\s?/\s?[{_L}]{{3,9}}\.?)?\s?[-/., ]?\s?({_N}{{4}}|{_N}{{2}})(?![0-9])")),
    ("kh", re.compile(rf"({_N}{{1,2}})\s?(?:ខែ\s?)?({'|'.join(KH_MONTHS)})\s?(?:ឆ្នាំ\s?)?({_N}{{4}})")),
]
_FIX = re.compile(r"(?<=[0-9])[OoIl|](?=[0-9./\-])|(?<=[0-9./\- ])[OoIl|](?=[0-9])")


def _numeric(text):
    """The text with Khmer digits as 0-9 and O / l between digits read as 0 / 1. Same length."""
    t = text.translate(KH2LAT)
    for _ in range(2):
        t = _FIX.sub(lambda m: "0" if m.group() in "Oo" else "1", t)
    return t


def _iso(y, m, d, today):
    try:
        y, m, d = int(y), int(m), int(d)
        if y < 100:
            y += 1900 if y > today.year % 100 + 15 else 2000
        if not 1900 <= y <= 2100:
            return None
        return datetime.date(y, m, d).isoformat()
    except ValueError:
        return None


def find_dates(text, today=None):
    """[(start, end, 'YYYY-MM-DD')] for every date in the text, left to right."""
    today = today or datetime.date.today()
    t, found = _numeric(text), []
    for kind, rx in _DATE_RES:
        for m in rx.finditer(t):
            if any(a < m.end() and m.start() < b for a, b, _ in found):
                continue
            a, b, c = m.groups()
            if kind == "ymd":
                iso = _iso(a, b, c, today)
            elif kind == "dmy":
                iso = _iso(c, b, a, today)
            elif kind == "kh":
                iso = _iso(c, KH_MONTHS.index(b) + 1, a, today)
            else:
                mon = _strip_accents(b).upper()
                mm = _MONTHS.get(mon) or _MONTHS.get(mon[:4]) or _MONTHS.get(mon[:3])
                iso = _iso(c, mm, a, today) if mm else None
            if iso:
                found.append((m.start(), m.end(), iso))
    return sorted(found)


def parse_date(s, today=None):
    d = find_dates(s, today)
    if d:
        return d[0][2]
    digits = re.sub(r"[^0-9]", "", _numeric(s).translate(_TO_DIGIT) if not re.search(r"[A-Za-z]{3}", s) else "")
    if len(digits) == 8:  # separators lost
        today = today or datetime.date.today()
        return _iso(digits[4:], digits[2:4], digits[:2], today) or _iso(digits[:4], digits[4:6], digits[6:], today)
    return None


# ======================= label-anchored rules =======================
_DOB_L = ["ថ្ងៃខែឆ្នាំកំណើត", "ថ្ងៃកំណើត", "date of birth", "birth date", "dob", "date de naissance", "出生日期", "ngày sinh"]
_ISSUE_L = ["សុពលភាព", "ថ្ងៃចេញ", "កាលបរិច្ឆេទចេញ", "ថ្ងៃផ្តល់", "ចេញថ្ងៃទី", "date of issue", "issue date", "issued on",
            "签发日期", "ngày cấp", "date de délivrance"]
_EXPIRY_L = ["ថ្ងៃផុតកំណត់", "សុពលភាពដល់", "ផុតសុពលភាព", "ដល់ថ្ងៃ", "date of expiry", "expiry date", "valid until", "expires",
             "date of expiration", "有效期至", "có giá trị đến", "date d'expiration"]
_ROLE_RX = [(role, re.compile("|".join(re.escape(x) for x in sorted(ls, key=len, reverse=True)), re.I))
            for role, ls in (("date_of_birth", _DOB_L), ("issue_date", _ISSUE_L), ("expiry_date", _EXPIRY_L))]
_SEX_RX = re.compile(r"(?i:ភេទ|sex|gender|性别)[^\n]{0,12}?(ប្រុស|ស្រី|(?i:\bfemale\b|\bmale\b)|\b[MF]\b|男|女)")
_HEIGHT_RX = re.compile(r"(?i:កំពស់|កម្ពស់|height)[^\n0-9០-៩]{0,12}([0-9០-៩]{1,3}(?:[.,][0-9០-៩]{1,2})?)")
_KHID_RX = re.compile(r"^[ \t]*([0-9០-៩]{9}(?:[ \t]*\([ \t]*[0-9០-៩]{1,2}[ \t]*\))?)[ \t]*$", re.M)
_SUFFIX_RX = re.compile(r"^(.*?[A-Z0-9])\s*\(\s*([0-9]{1,2})\s*\)?\s*$")


def norm_sex(s):
    if not s:
        return None
    u = s.strip().upper()
    if "ស្រ" in s or "女" in s or "ី" in s or re.search(r"\bF\b|FEM|NỮ", u):
        return "F"
    if "ប" in s or "ុ" in s or "男" in s or re.search(r"\bM\b|MAL|NAM\b", u):
        return "M"
    if u[:1] in ("F", "M"):
        return u[0]
    return None


def norm_height(s):
    m = re.search(r"[0-9]{1,3}(?:[.,][0-9]{1,2})?", (s or "").translate(KH2LAT))
    if not m:
        return None
    v = float(m.group().replace(",", "."))
    if v < 3:
        v *= 100
    v = int(round(v))
    return v if 50 <= v <= 250 else None


def norm_number(s):
    """Letters and digits only. A Khmer ID's issue count in brackets is kept: '010234567 (០១)' -> '010234567 (01)'."""
    s = (s or "").translate(KH2LAT).upper().strip()
    m = _SUFFIX_RX.match(s)
    n = re.sub(r"[^A-Z0-9]", "", m.group(1) if m else s)
    if m and n:
        return f"{n} ({m.group(2)})"
    return n or None


def _with_suffix(number, *seen):
    """The machine-readable lines have the bare number; add the bracket the card prints after it."""
    for s in seen:
        if number and s and " (" in s and s.split(" (")[0] == number:
            return s
    return number


def anchored(text, today=None):
    """Fields a label on the same line gives away: {field: value}."""
    out = {}
    start = 0
    for line in text.split("\n"):
        dates = find_dates(line, today)
        labels = [(m.start(), m.end(), role) for role, rx in _ROLE_RX for m in rx.finditer(line)]
        for i, (a, b, iso) in enumerate(dates):
            prev_end = dates[i - 1][1] if i else 0
            near = [l for l in labels if prev_end <= l[0] and l[1] <= a]
            if near:
                role = max(near, key=lambda l: (l[1], l[1] - l[0]))[2]
                out.setdefault(role, iso)
        start += len(line) + 1
    m = _SEX_RX.search(text)
    if m and norm_sex(m.group(1)):
        out["sex"] = norm_sex(m.group(1))
    m = _HEIGHT_RX.search(text)
    if m and norm_height(m.group(1)):
        out["height_cm"] = norm_height(m.group(1))
    return out


_VISA_KW = ["visa", "ទិដ្ឋាការ"]
_PASS_KW = ["passport", "លិខិតឆ្លងដែន", "passeport", "护照", "hộ chiếu"]
_ID_KW = ["អត្តសញ្ញាណប័ណ្ណ", "identity card", "គោត្តនាមនិងនាម", "id card", "ភិនភាគ", "idkhm", "身份证"]


def detect_type(text, mrz, labels, model_type=None):
    """Machine-readable line with right check digits, then the model's class, then keywords."""
    if mrz and mrz["type"] and mrz["dates_ok"]:
        return mrz["type"]
    if model_type is not None:
        return None if model_type == "other" else model_type
    if mrz and mrz["type"]:
        return mrz["type"]
    low = text.lower()
    if any(k in low for k in _VISA_KW):
        return "visa"
    if any(k in low for k in _ID_KW):
        return "id_card"
    if any(k in low for k in _PASS_KW):
        return "passport"
    if "VISA_TYPE" in labels:
        return "visa"
    if labels & {"SURNAME", "GIVEN", "PASSPORT_NO"}:
        return "passport"
    if labels & {"NAME_KM", "DOC_NO", "ADDRESS", "POB"}:
        return "id_card"
    return None


def clean(s, upper=False):
    s = re.sub(r"\s+", " ", s).strip(" \t:;,.៖-/|")
    return (s.upper() if upper else s) or None


def assemble(text, spans, doc_type=None, today=None):
    """Final JSON from the raw text, the tagger's spans [{start, end, label}] and the
    model's document class (id_card / passport / visa / other; None: use keywords).
    Returns (fields, where each value came from: mrz / rule / model)."""
    by = {}
    for s in sorted(spans, key=lambda s: s["start"]):
        by.setdefault(s["label"], []).append(text[s["start"]:s["end"]])
    mrz = parse_mrz(text, today)
    rule = anchored(text, today)
    f, src = dict.fromkeys(FIELDS), {}

    def put(name, *choices):
        for value, where in choices:
            if value not in (None, ""):
                f[name], src[name] = value, where
                return

    def first(label, fn):
        for t in by.get(label, []):
            v = fn(t)
            if v is not None:
                return v
        return None

    def joined(label, upper=False):
        return clean(" ".join(by[label]), upper) if label in by else None

    ok = bool(mrz and mrz["dates_ok"])
    put("document_type", (detect_type(text, mrz, set(by), doc_type), "model" if doc_type else "rule"))
    kind = f["document_type"]
    put("sex", (mrz["sex"] if ok else None, "mrz"), (rule.get("sex"), "rule"), (first("SEX", norm_sex), "model"),
        (mrz["sex"] if mrz else None, "mrz, check digit wrong"))
    put("date_of_birth", (mrz["dob"] if ok else None, "mrz"), (rule.get("date_of_birth"), "rule"),
        (first("DOB", lambda t: parse_date(t, today)), "model"), (mrz["dob"] if mrz else None, "mrz, check digit wrong"))
    put("expiry_date", (mrz["expiry"] if ok else None, "mrz"), (rule.get("expiry_date"), "rule"),
        (first("EXPIRY", lambda t: parse_date(t, today)), "model"), (mrz["expiry"] if mrz else None, "mrz, check digit wrong"))
    put("issue_date", (rule.get("issue_date"), "rule"), (first("ISSUE", lambda t: parse_date(t, today)), "model"))
    put("height_cm", (rule.get("height_cm"), "rule"), (first("HEIGHT", norm_height), "model"))

    mnum, mok = (mrz["number"], mrz["number_ok"]) if mrz else (None, False)
    doc_no, pass_no = first("DOC_NO", norm_number), first("PASSPORT_NO", norm_number)
    if kind == "visa":
        put("document_number", (doc_no, "model"))
        put("passport_number", (pass_no, "model"), (mnum, "mrz" if mok else "mrz, check digit wrong"))
    elif kind == "passport":
        put("document_number", (mnum if mok else None, "mrz"), (pass_no, "model"), (doc_no, "model"), (mnum, "mrz, check digit wrong"))
        put("passport_number", (f["document_number"], src.get("document_number")))
    else:
        km = _KHID_RX.search(text) if kind == "id_card" else None
        printed = norm_number(km.group(1)) if km else None
        put("document_number", (_with_suffix(mnum, printed, doc_no) if mok else None, "mrz"), (printed, "rule"),
            (doc_no, "model"), (mnum, "mrz, check digit wrong"))
        put("passport_number", (pass_no, "model"))

    put("full_name_km", (joined("NAME_KM"), "model"))
    put("surname", (joined("SURNAME", True), "model"), (mrz["surname"] if mrz else None, "mrz"))
    put("given_names", (joined("GIVEN", True), "model"), (mrz["given"] if mrz else None, "mrz"))
    both = " ".join(x for x in (f["surname"], f["given_names"]) if x) or None
    put("full_name_en", (joined("NAME_EN", True), "model"), (both, "surname + given names"))
    put("nationality", (joined("NATIONALITY"), "model"), (mrz["nationality"] if mrz else None, "mrz"))
    put("place_of_birth", (joined("POB"), "model"))
    put("address", (joined("ADDRESS"), "model"))
    put("visa_type", (joined("VISA_TYPE"), "model"))
    put("distinguishing_marks", (joined("MARKS"), "model"))
    # the machine-readable lines as read (cleaned to A-Z, 0-9 and <), and whether their check digits fit
    f["mrz"] = "\n".join(mrz["lines"]) if mrz else None
    f["mrz_valid"] = (mrz["dates_ok"] and (mrz["number_ok"] or mrz["number"] is None)) if mrz else None
    if mrz:
        src["mrz"] = src["mrz_valid"] = "rule"
    return f, src
