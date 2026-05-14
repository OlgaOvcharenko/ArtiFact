import re
import unicodedata
from typing import Optional

# Pre-compiled Setup Regexes
RE_ERA_BCE = re.compile(r"\b(?:B\.?\s*C\.?|B\.?\s*C\.?\.|BCE)\b", re.I)
RE_ERA_CE  = re.compile(r"\b(?:(?:A\.?\s*D\.?)|(?:C\.?\s*E\.?))\b", re.I)

RE_START_UNCERTAINTY = re.compile(r"^\s*(?:possibly|probably|maybe|c\.?|circa|about)\s+", re.I)
RE_UNCERTAINTY = re.compile(r"\b(?:possibly|probably|maybe|circa|about)\b", re.I)

ORD_MAP = {
    "first": "1st", "second": "2nd", "third": "3rd", "fourth": "4th", "fifth": "5th",
    "sixth": "6th", "seventh": "7th", "eighth": "8th", "ninth": "9th", "tenth": "10th",
    "eleventh": "11th", "twelfth": "12th", "thirteenth": "13th", "fourteenth": "14th",
    "fifteenth": "15th", "sixteenth": "16th", "seventeenth": "17th", "eighteenth": "18th",
    "nineteenth": "19th", "twentieth": "20th", "twenty-first": "21st", "twentyfirst": "21st",
    "last": "last"
}

RE_ORD_WORDS = re.compile(
    r"\b(" + "|".join(ORD_MAP.keys()) + r")\s+(century|millennium|quarter|half)\b", 
    re.I
)

PATTERN_YEAR = r"\d{1,4}(?:-\d{1,4})?" 
PATTERN_CENTURY = r"\d{1,2}(?:st|nd|rd|th)(?:-\d{1,2}(?:st|nd|rd|th))?\s+century"
PATTERN_PARTIAL = (
    r"(?:1st|2nd|3rd|4th|last)\s+quarter\s+of\s+the\s+" + PATTERN_CENTURY + "|"
    r"(?:1st|2nd)\s+half\s+of\s+the\s+" + PATTERN_CENTURY + "|"
    r"(?:early|late|mid|middle)\s+(?:of\s+the\s+)?(" + PATTERN_CENTURY +")"
)
PATTERN_MODIFIED_RANGE = (
    r"(?:early|late|mid|middle)?\s*\d{1,2}(?:st|nd|rd|th)\s*-\s*(?:early|late|mid|middle)?\s*\d{1,2}(?:st|nd|rd|th)\s+century"
)
PATTERN_CROSS_ERA = r"\d+\s*(?:BCE|CE)\s*-\s*\d+\s*(?:BCE|CE)"

RE_STRICT_VALIDATOR = re.compile(
    rf"^(?:ca\.\s)?(?:(?:before|after)\s+)?(?:{PATTERN_YEAR}|{PATTERN_PARTIAL}|{PATTERN_CENTURY}|{PATTERN_MODIFIED_RANGE}|{PATTERN_CROSS_ERA})(?:\s(?:BCE|CE))?$",
    re.I
)

RE_MULTI_SPACE = re.compile(r"\s+")
RE_RANGE_BEFORE_AFTER = re.compile(r"(\d{4})\s*-\s*in\s+or\s+(?:before|after)\s+(\d{4})(?:-\d{2}-\d{2})?", flags=re.I)
RE_START_BEFORE_AFTER = re.compile(r"^in\s+or\s+(?:before|after)\s+(\d{4})\s*-\s*", flags=re.I)
RE_END_BEFORE_AFTER = re.compile(r"-\s*in\s+or\s+(?:before|after)\s+(\d{4})", flags=re.I)
RE_CENTURIES = re.compile(r"\bcenturies\b", flags=re.I)
RE_DASHES = re.compile(r"[‐‒–—―]+")
RE_SPACED_DASH = re.compile(r"\s*-\s*")
RE_SLASH = re.compile(r"/")
RE_CENTURY_RANGE = re.compile(r"((?:early|late|mid|middle)?\s*\d+(?:st|nd|rd|th))\s+century\s*-\s*((?:early|late|mid|middle)?\s*\d+(?:st|nd|rd|th))", flags=re.I)
RE_ERA_DOT = re.compile(r"(BCE|CE)\s*\.\s*$", flags=re.I)
RE_Q_MARK_SPACE = re.compile(r"\s*\(\?\)$")
RE_QUARTER_HALF = re.compile(r"\b(quarter|half)\s+(?!(?:of\s+the\b))(\d{1,2}(?:st|nd|rd|th))", flags=re.I)
RE_QUARTER_HALF_OF = re.compile(r"\b(quarter|half)\s+of\s+(\d{1,2}(?:st|nd|rd|th))", flags=re.I)
RE_0S_DECADE = re.compile(r"\b(\d{3})0s\b")
RE_BCE_CHECK = re.compile(r"\b(BCE)\b", flags=re.I)
RE_CENTURY_YEAR_RANGE1 = re.compile(r"(?<!st|nd|rd|th)(?<!\d)\b(\d{2})(\d{2})-(\d{2})\b")
RE_CENTURY_YEAR_RANGE2 = re.compile(r"(?<!st|nd|rd|th)(?<!\d)\b(\d{3})(\d)-(\d)\b")
RE_MULTI_CA = re.compile(r"(?:ca\.?\s*)+", flags=re.I)
RE_CA_NO_SPACE = re.compile(r"ca\.(\d)")
RE_CA_RANGE1 = re.compile(r"^ca\.\s*(\d+)\s*-\s*(?:ca\.|c\.)\s*(\d+)", flags=re.I)
RE_CA_RANGE2 = re.compile(r"^ca\.\s*(\d+)-ca\.\s*(\d+)", flags=re.I)
RE_CA_RANGE3 = re.compile(r"(\d+)\s*-\s*(?:ca\.|c\.)\s*(\d+)", flags=re.I)
RE_STANDALONE_C = re.compile(r"\bc\.\s+(\d)", flags=re.I)
RE_MID_DASH = re.compile(r"\bmid-", flags=re.I)
RE_TRAILING_DOT = re.compile(r"\.\s*$")

def normalize_date(raw: str) -> Optional[str]:
    if not isinstance(raw, str) or not raw.strip():
        return None

    s = unicodedata.normalize("NFKC", raw).strip()
    s = RE_MULTI_SPACE.sub(" ", s)
    
    s = RE_RANGE_BEFORE_AFTER.sub(r"\1-\2", s)
    
    s = RE_START_BEFORE_AFTER.sub(r"\1-", s)
    
    s = RE_END_BEFORE_AFTER.sub(r"-\1", s)
    
    if RE_START_UNCERTAINTY.match(s):
        s = RE_START_UNCERTAINTY.sub("ca. ", s)
        
    s = RE_CENTURIES.sub("century", s)

    s = RE_DASHES.sub("-", s)
    s = RE_SPACED_DASH.sub("-", s)
    s = RE_SLASH.sub("-", s) 

    s = RE_CENTURY_RANGE.sub(r"\1-\2", s) 

    s = RE_ERA_BCE.sub("BCE", s)
    s = RE_ERA_CE.sub("CE", s)
    s = RE_ERA_DOT.sub(r"\1", s)

    s = RE_Q_MARK_SPACE.sub("?", s)
    s = RE_UNCERTAINTY.sub("ca.", s)
    if s.endswith("?"):
        s = "ca. " + s.rstrip("?")

    def replace_ord(m):
        word = m.group(1).lower()
        unit = m.group(2)
        val = ORD_MAP.get(word, word)
        return f"{val} {unit}"
        
    s = RE_ORD_WORDS.sub(replace_ord, s)

    s = RE_QUARTER_HALF.sub(r"\1 of the \2", s)
    s = RE_QUARTER_HALF_OF.sub(r"\1 of the \2", s)

    s = RE_0S_DECADE.sub(r"\g<1>0-\g<1>9", s)
    
    if not RE_BCE_CHECK.search(s):
        s = RE_CENTURY_YEAR_RANGE1.sub(r"\1\2-\1\3", s)
        s = RE_CENTURY_YEAR_RANGE2.sub(r"\1\2-\1\3", s)

    s = RE_MULTI_CA.sub("ca. ", s)
    s = RE_CA_NO_SPACE.sub(r"ca. \1", s)    

    s = RE_CA_RANGE1.sub(r"ca. \1-ca. \2", s)
    s = RE_CA_RANGE2.sub(r"ca. \1-\2", s)
    
    s = RE_CA_RANGE3.sub(r"\1-\2", s)

    s = RE_STANDALONE_C.sub(r"ca. \1", s)
    
    s = RE_MID_DASH.sub("mid ", s)

    s = s.strip()
    s = RE_TRAILING_DOT.sub("", s)
    
    if RE_STRICT_VALIDATOR.match(s):
        return s
    
    return None