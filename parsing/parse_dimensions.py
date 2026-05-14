import re
from typing import List, Dict, Any, Optional

_ALLOWED_ATTRS = {"height", "width", "length", "depth", "thickness", "diameter", "weight"}


def parse_normalized_dimensions(normalized: str) -> Optional[Dict[str, Any]]:
    text = (normalized or "").strip()
    if not text:
        return None

    text = text.replace(";", "\n")
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    parts: List[Dict[str, Any]] = []
    any_measurements = False

    for line in lines:
        line_part, line_core = "", line
        
        m_prefix = re.match(r"^([^:]+):\s*(.*)$", line)
        if m_prefix:
            line_part = m_prefix.group(1).strip()
            line_core = m_prefix.group(2).strip()

        measurements = _parse_line_simple(line_core)
        if not measurements:
            return None
        
        any_measurements = True
        parts.append({"part": line_part, "measurements": measurements})

    if not any_measurements:
        return None

    return {"parts": parts}


def _parse_line_simple(line: str) -> Optional[List[Dict[str, Any]]]:
    # normalize separators and whitespace just in case
    s = re.sub(r"[×X]", "x", line)
    s = re.sub(r"\s+", " ", s).strip()

    segments = [seg.strip() for seg in s.split("x") if seg.strip()]
    if not segments:
        return None

    measurements: List[Dict[str, Any]] = []
    for seg in segments:
        m = _parse_segment(seg)
        if m is None:
            return None
        if m.get('value') != 0.0:
            measurements.append(m)

    if not measurements:
        return None

    return measurements


_SEG_RE = re.compile(
    r"""
    ^\s*
    (?:(?P<prefix>max\.?|min\.?)\s+)?
    (?P<label>height|width|length|depth|thickness|diameter|weight)
    \s+
    (?P<val>\d+(?:\.\d+)?)
    \s*
    (?P<unit>cm|mm|g|kg|in)
    \s*$
    """,
    re.IGNORECASE | re.VERBOSE,
)


def _parse_segment(seg: str) -> Optional[Dict[str, Any]]:
    m = _SEG_RE.match(seg)
    if not m:
        return None

    label = m.group("label").lower()
    val_str = m.group("val")
    unit = m.group("unit").lower()
    prefix = m.group("prefix")
    note = ""

    try:
        val = float(val_str)
    except ValueError:
        return None

    if prefix:
        p = prefix.lower()
        if p.startswith("max"):
            note = "max"
        elif p.startswith("min"):
            note = "min"

    if label in {"height", "width", "length", "depth", "thickness", "diameter"}:
        if unit == "cm":
            value = val
        elif unit == "mm":
            value = val / 10.0
        else:
            return None

    elif label == "weight":
        if unit == "g":
            value = val
        elif unit == "kg":
            value = val * 1000.0
        else:
            return None
    else:
        return None

    return {"attribute": label, "value": value, "note": note}