import re
from typing import Any, Optional, List

def html_to_text(s: Optional[str]) -> Optional[str]:
    """strip html tags and convert simple tags to linebreaks."""
    if s is None:
        return None
    s = re.sub(r"<br\s*/?>", "\n", s, flags=re.I)
    s = re.sub(r"</p\s*>", "\n\n", s, flags=re.I)
    s = re.sub(r"<.*?>", "", s)
    return re.sub(r"\n{3,}", "\n\n", s).strip()

def list_to_str(x: Any) -> Optional[str]:
    if x is None:
        return None
    if isinstance(x, list):
        return ", ".join(str(v) for v in x if v is not None)
    return str(x)

def slugify(name: str) -> str:
    if not name:
        return "unknown"
    s = re.sub(r"\s+", "_", name.strip())
    s = re.sub(r"[^\w\-\.]", "", s)
    return s.lower()[:120]

def resolve_language(val: Any) -> Any:
    """
    If val is a dict with 'English' or 'Dutch' keys, return the English value if present,
    else the Dutch value.
    """
    if isinstance(val, dict):
        def has_content(x):
            return x is not None and len(x) > 0

        if has_content(val.get("English")):
            return val.get("English")
        if has_content(val.get("Dutch")):
            return val.get("Dutch")
        if has_content(val.get("und")):
            return val.get("und")

    return val
