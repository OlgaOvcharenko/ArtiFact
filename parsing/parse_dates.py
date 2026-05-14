import re
from typing import Optional, Dict, Any

def _get_century_bounds(century: int, era: str) -> tuple[int, int]:
    """
    Returns the absolute start and end years of a century.
    - CE: 5th C -> 401 to 500
    - BCE: 5th C -> 500 to 401
    """
    if era == "BCE":
        start = century * 100
        end = (century - 1) * 100 + 1
    else:
        start = (century - 1) * 100 + 1
        end = start + 99
    return start, end

def _parse_ordinal_word(word: str) -> int:
    """
    Maps words like '1st', 'second', 'last' to integers.
    """
    word = word.lower()
    if "last" in word:
        return -1
    
    digit_match = re.search(r"(\d+)", word)
    if digit_match:
        return int(digit_match.group(1))
    
    mapping = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10}
    return mapping.get(word, 1)

def _calculate_partial(segment_str: str, unit: str, century: int, era: str) -> tuple[int, int]:
    """
    Calculates exact years for quarters/halves.
    Example: "1st quarter of the 5th century BCE" -> (-500, -476)
    """
    c_start, c_end = _get_century_bounds(century, era)
    
    segment = _parse_ordinal_word(segment_str)
    
    span = 25 if "quarter" in unit.lower() else 50
    
    if segment == -1:
        segment = 4 if span == 25 else 2

    offset_chunks = segment - 1 
    total_year_offset = offset_chunks * span

    if era == "BCE":
        p_start = c_start - total_year_offset
        p_end = p_start - (span - 1)
    else:
        p_start = c_start + total_year_offset
        p_end = p_start + (span - 1)

    return p_start, p_end

def _calculate_section(mod: str, century: int, era: str) -> tuple[int, int]:
    """
    Calculates years for early/mid/late.
    Approximation:
    - Early: 0-33
    - Mid: 34-66
    - Late: 67-99
    """
    c_start, c_end = _get_century_bounds(century, era)

    
    mod = mod.lower()
    if "early" in mod:
        offset_start, offset_end = 0, 32
    elif "mid" in mod:
        offset_start, offset_end = 33, 65
    elif "late" in mod:
        offset_start, offset_end = 66, 99
    else:
        offset_start, offset_end = 0, 99
        
    if era == "BCE":
        p_start = c_start - offset_start
        p_end = c_start - offset_end
    else:

        p_start = c_start + offset_start
        p_end = c_start + offset_end
        
    return p_start, p_end

def _get_millennium_bounds(millennium: int, era: str) -> tuple[int, int]:
    """
    Returns the absolute (positive) start and end years of a millennium.
    - CE: 3rd mill -> 2001 to 3000
    - BCE: 3rd mill -> 3000 to 2001
    """
    if era == "BCE":
        start = millennium * 1000
        end = (millennium - 1) * 1000 + 1
    else:
        start = (millennium - 1) * 1000 + 1
        end = millennium * 1000
    return start, end

def _calculate_millennium_section(mod: str, millennium: int, era: str) -> tuple[int, int]:
    """
    Calculates years for early/mid/late within a millennium.
    Approximation (thirds):
    - Early: first 333 years
    - Mid: middle 334 years
    - Late: last 333 years
    """
    c_start, c_end = _get_millennium_bounds(millennium, era)
    
    mod = mod.lower()
    if "early" in mod:
        offset_start, offset_end = 0, 332
    elif "mid" in mod:
        offset_start, offset_end = 333, 666
    elif "late" in mod:
        offset_start, offset_end = 667, 999
    else:
        offset_start, offset_end = 0, 999
        
    if era == "BCE":
        p_start = c_start - offset_start
        p_end = c_start - offset_end
    else:
        p_start = c_start + offset_start
        p_end = c_start + offset_end
        
    return p_start, p_end

def parse_normalized_date(normalized_date: str) -> Optional[Dict[str, Any]]:
    """
    Parses a normalized date string into structured start/end integers.
    BCE years are negative.
    """
    if not normalized_date:
        return None
    
    m = re.match(r"^(?:(ca\.|before|after)\s)?(.+?)(?:\s(BCE|CE|B\.?C\.?|A\.?D\.?))?$", normalized_date, re.I)
    if not m:
        return None

    prefix = (m.group(1) or "").lower().strip()
    is_circa = "ca." in prefix
    
    core = m.group(2).strip()
    m_paren = re.search(r"^(.*?)\s*\([^)]*\)$", core)
    if m_paren:
        core = m_paren.group(1).strip()
        
    era_raw = (m.group(3) or "CE").upper()
    era = "BCE" if "B" in era_raw else "CE"

    start_year: int = 0
    end_year: int = 0

    # 1. YYYY-MM-DD or YYYY-MM
    m_iso = re.match(r"^(\d{4})[-\u2013]\d{2}(?:[-\u2013]\d{2})?$", core)
    
    # 2. Ordinal Quarter/Half Century
    m_partial = re.match(
        r"^(first|second|third|fourth|last|\d+(?:st|nd|rd|th))\s+(quarter|half)\s+of\s+the\s+(\d+)(?:st|nd|rd|th)\s+century", 
        core, 
        re.I
    )
    
    # 3. Early/Mid/Late Century
    m_modified = re.match(
        r"^(early|mid|middle|late)\s+(?:of\s+the\s+)?(\d+)(?:st|nd|rd|th)\s+century",
        core,
        re.I
    )

    # 4. Cross-Era Century Range (must evaluate BEFORE Same Era)
    m_cross_era_cent = re.match(
        r"^(?:(early|mid|middle|late)\s+)?(?:of\s+the\s+)?(\d+)(?:st|nd|rd|th)\s+century\s+([A-Z\.]+)\s*[-\u2013/]\s*(?:(early|mid|middle|late)\s+)?(?:of\s+the\s+)?(\d+)(?:st|nd|rd|th)\s+century\s+([A-Z\.]+)", 
        normalized_date,
        re.I
    )
    
    # 4.5 Century Range (Same Era)
    m_cent_range = re.match(
        r"^(?:(early|mid|middle|late)\s+)?(\d+)(?:st|nd|rd|th)\s*[-\u2013]\s*(?:(early|mid|middle|late)\s+)?(\d+)(?:st|nd|rd|th)\s+century", 
        core, 
        re.I
    )

    # 5. Simple Century
    m_century = re.match(r"^(\d+)(?:st|nd|rd|th)\s+century", core, re.I)
    
    # 5.1 Simple Millennium
    m_millennium = re.match(r"^(\d+)(?:st|nd|rd|th)\s+millennium", core, re.I)

    # 5.2 Modified Millennium
    m_mod_millennium = re.match(
        r"^(early|mid|middle|late)\s+(?:of\s+the\s+)?(\d+)(?:st|nd|rd|th)\s+millennium",
        core,
        re.I
    )

    # 5.3 Millennium Range
    m_millennium_range = re.match(
        r"^(?:(early|mid|middle|late)\s+)?(\d+)(?:st|nd|rd|th)\s*[-\u2013]\s*(?:(early|mid|middle|late)\s+)?(\d+)(?:st|nd|rd|th)\s+millennium", 
        core, 
        re.I
    )
    
    # 5.4 Decades
    m_decade = re.match(r"^(\d{3})0s(?:(?:\s|[-\u2013/])(\d{3})0s)?$", core, re.I)

    # 6. Year Range (with optional abbreviation handling)
    m_year_range = re.match(r"^(\d+)\s*[-\u2013/]\s*(\d{1,4})$", core)

    # 7. Single Year
    m_year = re.match(r"^(\d+)$", core)
    
    # 8. Cross-era ranges
    m_cross_era_full = re.match(
        r"^(?:ca\.\s)?(\d+)\s+([A-Z\.]+)\s*[-\u2013/]\s*(\d+)\s+([A-Z\.]+)$",
        normalized_date,
        re.I
    )

    if m_iso:
        y = int(m_iso.group(1))
        start_year = end_year = y

    elif m_partial:
        seg, unit, cent = m_partial.groups()
        s, e = _calculate_partial(seg, unit, int(cent), era)
        start_year, end_year = s, e
        
    elif m_millennium_range:
        mod1, m1, mod2, m2 = m_millennium_range.groups()
        s1, e1 = _calculate_millennium_section(mod1 or "", int(m1), era)
        s2, e2 = _calculate_millennium_section(mod2 or "", int(m2), era)
        start_year, end_year = s1, e2

    elif m_cross_era_cent:
        mod1, c1, era1, mod2, c2, era2 = m_cross_era_cent.groups()
        c1, c2 = int(c1), int(c2)
        c1_era = "BCE" if "B" in era1.upper() else "CE"
        c2_era = "BCE" if "B" in era2.upper() else "CE"
        
        s1, e1 = _calculate_section(mod1 or "", c1, c1_era)
        s2, e2 = _calculate_section(mod2 or "", c2, c2_era)
        
        return {
            "date_begin": s1,
            "date_end": e2,
            "date_begin_bce": c1_era == "BCE",
            "date_end_bce": c2_era == "BCE",
            "circa": is_circa,
            "notes": normalized_date
        }

    elif m_cent_range:
        mod1, c1, mod2, c2 = m_cent_range.groups()
        c1, c2 = int(c1), int(c2)
        s1, e1 = _calculate_section(mod1 or "", c1, era)
        s2, e2 = _calculate_section(mod2 or "", c2, era)
        start_year, end_year = s1, e2

    elif m_mod_millennium:
        mod, mill = m_mod_millennium.groups()
        s, e = _calculate_millennium_section(mod, int(mill), era)
        start_year, end_year = s, e

    elif m_millennium:
        mill = int(m_millennium.group(1))
        start_year, end_year = _get_millennium_bounds(mill, era)

    elif m_modified:
        mod, cent = m_modified.groups()
        s, e = _calculate_section(mod, int(cent), era)
        start_year, end_year = s, e
        
    elif m_decade:
        y1_str, y2_str = m_decade.groups()
        y1 = int(y1_str + "0")
        if y2_str:
            y2 = int(y2_str + "9")
        else:
            y2 = y1 + 9
        start_year, end_year = y1, y2

    elif m_cross_era_full:
        y1, era1, y2, era2 = m_cross_era_full.groups()
        
        y1_int, y2_int = int(y1), int(y2)
        b_bce = "B" in era1.upper()
        e_bce = "B" in era2.upper()
        
        # Canonical order
        if b_bce and e_bce:
            if y1_int < y2_int: y1_int, y2_int = y2_int, y1_int
        elif not b_bce and not e_bce:
            if y1_int > y2_int: y1_int, y2_int = y2_int, y1_int
            
        return {
            "date_begin": y1_int,
            "date_end": y2_int,
            "date_begin_bce": b_bce,
            "date_end_bce": e_bce,
            "circa": is_circa,
            "notes": normalized_date
        }

    elif m_century:
        c = int(m_century.group(1))
        start_year, end_year = _get_century_bounds(c, era)

    elif m_year_range:
        y1_str, y2_str = m_year_range.groups()
        y1 = int(y1_str)

        if len(y2_str) < len(y1_str):
            prefix = y1_str[:-len(y2_str)]
            y2_expanded = int(prefix + y2_str)
            
            if era == "BCE":
                if y2_expanded < y1:
                    y2 = y2_expanded
                else:
                    y2 = int(y2_str)
            else:
                if y2_expanded > y1:
                    y2 = y2_expanded
                else:
                    if len(prefix) > 0:
                        try:
                            rollover_y2 = int(str(int(prefix) + 1) + y2_str)
                            if rollover_y2 > y1:
                                y2 = rollover_y2
                            else:
                                y2 = int(y2_str)
                        except ValueError:
                            y2 = int(y2_str)
                    else:
                        y2 = int(y2_str)
        else:
            y2 = int(y2_str)
        start_year, end_year = y1, y2

    elif m_year:
        y = int(m_year.group(1))
        start_year = end_year = y

    else:
        return None 
    
    date_begin_bce = (era == "BCE")
    date_end_bce = (era == "BCE")

    if era == "BCE":
        if start_year < end_year:
            start_year, end_year = end_year, start_year
    else:
        if start_year > end_year:
            start_year, end_year = end_year, start_year

    return {
        "date_begin": start_year,
        "date_end": end_year,
        "date_begin_bce": date_begin_bce,
        "date_end_bce": date_end_bce,
        "circa": is_circa,
        "notes": normalized_date
    }