import re
from typing import Optional, Dict, List, Any


# =============================================================================
# REGEX BUILDING BLOCKS - Iteratively generated with Gemini
# =============================================================================

# Number patterns
NUM = r'(\d+(?:\.\d+)?)'              # Integer or decimal: 24.7
FRAC = r'(\d+/\d+)'                    # Simple fraction: 3/4
NUM_FRAC = r'((?:\d+\s+)?\d+/\d+|\d+(?:\.\d+)?)'    # Mixed, simple fraction, or number
NUM_FRAC_EXT = r'((?:\d+[-\s]+)?\d+/\d+|\d+(?:\.\d+)?)'  # Also handles hyphenated: 10-1/2

# Dimension separators
SEP = r'\s*[×x]\s*'                   # × or x with optional whitespace

# Unit patterns
UNIT_CM = r'cm\.?'
UNIT_IN = r'in\.?'

# Flexible labels (non-capturing groups so index extraction is safe)
# Catches: H., height:, h:, Max. H., etc.
PREFIX_OPT = r'(?:(?:[Mm]ax|[Mm]in)\.?\s+)?'
LBL_H = PREFIX_OPT + r'(?:[Hh][\.:]*|[Hh]eight[\.:]*)'
LBL_W = PREFIX_OPT + r'(?:[Ww][\.:]*|[Ww]idth[\.:]*)'
LBL_D = PREFIX_OPT + r'(?:[Dd][\.:]*|[Dd]epth[\.:]*)'
LBL_L = PREFIX_OPT + r'(?:[Ll][\.:]*|[Ll]ength[\.:]*)'
LBL_DIAM = PREFIX_OPT + r'(?:[Dd]iam[\.:]*|[Dd]iameter[\.:]*)'


def parse_num(s: str) -> float:
    """Parse a number string (integer, decimal, or mixed fraction) to float."""
    s = s.strip()
    if '/' in s:
        parts = s.split()
        if len(parts) == 2:  # Mixed fraction: "9 3/4"
            whole = float(parts[0])
            num, denom = parts[1].split('/')
            return whole + float(num) / float(denom)
        else:  # Simple fraction: "3/4"
            num, denom = s.split('/')
            return float(num) / float(denom)
    return float(s)


def build_2d_pattern(label: Optional[str] = None) -> str:
    """Build pattern for 2D dimensions: H × W cm (H × W in.)"""
    prefix = f'{label}:\\s*' if label else ''
    return (
        prefix +
        NUM + SEP + NUM + r'\s*' + UNIT_CM + r'\s*'
        r'\(' + NUM_FRAC + SEP + NUM_FRAC + r'\s*' + UNIT_IN + r'\.\)'
    )


def build_3d_pattern(label: Optional[str] = None) -> str:
    """Build pattern for 3D dimensions: H × W × D cm (H × W × D in.)"""
    prefix = f'{label}:\\s*' if label else ''
    return (
        prefix +
        NUM + SEP + NUM + SEP + NUM + r'\s*' + UNIT_CM + r'\s*'
        r'\(' + NUM_FRAC + SEP + NUM_FRAC + SEP + NUM_FRAC + r'\s*' + UNIT_IN + r'\.\)'
    )


def build_single_pattern(label: str) -> str:
    """Build pattern for single dimension: h.: N cm (N in.)"""
    # If the label is one of the macros, don't append punctuation 
    # (since the macro already handles it). Otherwise, assume old behavior.
    if label.startswith('(?:'):
        return (
            label + r'\s*' +
            NUM + r'\s*' + UNIT_CM + r'\s*'
            r'\(' + NUM_FRAC + r'\s*' + UNIT_IN + r'\.\)'
        )
    else:
        return (
            label + r'\.?:\s*' +
            NUM + r'\s*' + UNIT_CM + r'\s*'
            r'\(' + NUM_FRAC + r'\s*' + UNIT_IN + r'\.\)'
        )


# =============================================================================
# RESULT BUILDERS
# =============================================================================

def make_measurement(attribute: str, value: float, note: str = "") -> Dict:
    """Create a single measurement dict."""
    return {
        "attribute": attribute,
        "value": value,
        "note": note
    }

def make_part(measurements: List[Dict], part_name: Optional[str] = None) -> Dict:
    """Create a part dict with measurements."""
    return {
        "part": part_name if part_name else None,
        "measurements": measurements
    }

def make_2d_result(h_cm: float, w_cm: float, label: Optional[str] = None) -> Dict:
    """Create part dict for 2D dimensions."""
    measurements = [
        make_measurement("height", round(h_cm, 2)),
        make_measurement("width", round(w_cm, 2))
    ]
    return make_part(measurements, label)


def make_3d_result(h_cm: float, w_cm: float, d_cm: float, label: Optional[str] = None) -> Dict:
    """Create part dict for 3D dimensions."""
    measurements = [
        make_measurement("height", round(h_cm, 2)),
        make_measurement("width", round(w_cm, 2)),
        make_measurement("depth", round(d_cm, 2))
    ]
    return make_part(measurements, label)


def make_single_result(value_cm: float, dimension_type: str, label: Optional[str] = None) -> Dict:
    """Create part dict for single dimension (height, diameter, etc.)."""
    measurements = [
        make_measurement(dimension_type, round(value_cm, 2))
    ]
    return make_part(measurements, label)


# =============================================================================
# PATTERNS PARSER
# =============================================================================

class PatternParser:
    """
    Pattern-based parser for dimension strings.
    Attempts to match against known templates in order of frequency.
    """
    
    def __init__(self):
        self.patterns = self._build_patterns()
        self.parse_stats = {'matched': 0, 'unmatched': 0, 'by_pattern': {}}
    
    def _build_patterns(self) -> List[tuple]:
        """Build list of (name, regex, parser_func) tuples."""
        patterns = []
        
        # Pattern 1: Simple 2D (30.6%)
        # 24.7 × 35.9 cm (9 3/4 × 14 3/16 in.)
        p1 = r'^' + build_2d_pattern() + r'$'
        patterns.append(('simple_2d', re.compile(p1, re.IGNORECASE), self._parse_simple_2d))
        
        # Pattern 2: image + sheet (5.4%)
        p2 = r'^' + build_2d_pattern('image') + r';\s*' + build_2d_pattern('sheet') + r'$'
        patterns.append(('image_sheet', re.compile(p2, re.IGNORECASE), self._parse_image_sheet))
        
        # Pattern 5: image + plate + sheet (3.5%)
        p5 = r'^' + build_2d_pattern('image') + r';\s*' + build_2d_pattern('plate') + r';\s*' + build_2d_pattern('sheet') + r'$'
        patterns.append(('image_plate_sheet', re.compile(p5, re.IGNORECASE), self._parse_image_plate_sheet))
        
        # Pattern 6: Simple 3D (3.3%)
        p6 = r'^' + build_3d_pattern() + r'$'
        patterns.append(('simple_3d', re.compile(p6, re.IGNORECASE), self._parse_simple_3d))
        
        # Pattern 7: plate + sheet (3.2%)
        p7 = r'^' + build_2d_pattern('plate') + r';\s*' + build_2d_pattern('sheet') + r'$'
        patterns.append(('plate_sheet', re.compile(p7, re.IGNORECASE), self._parse_plate_sheet))
        
        # Pattern 8: image/plate + sheet (2.3%)
        p8 = r'^image/plate:\s*' + NUM + SEP + NUM + r'\s*' + UNIT_CM + r'\s*\(' + NUM_FRAC + SEP + NUM_FRAC + r'\s*' + UNIT_IN + r'\.\);\s*' + build_2d_pattern('sheet') + r'$'
        patterns.append(('image_plate_combined_sheet', re.compile(p8, re.IGNORECASE), self._parse_image_plate_combined_sheet))
        
        # Pattern 9: h.: single height (2.1%)
        p9 = r'^' + build_single_pattern('h') + r'$'
        patterns.append(('height_only', re.compile(p9, re.IGNORECASE), self._parse_height_only))
        
        # Pattern 10: diam.: single diameter (2.1%)
        p10 = r'^' + build_single_pattern('diam') + r'$'
        patterns.append(('diameter_only', re.compile(p10, re.IGNORECASE), self._parse_diameter_only))
        
        # Pattern 11: 2D + diam (1.6%)
        p11 = r'^' + build_2d_pattern() + r';\s*' + build_single_pattern('diam') + r'$'
        patterns.append(('2d_plus_diam', re.compile(p11, re.IGNORECASE), self._parse_2d_plus_diam))
        
        # Pattern 16: image/paper + mount (0.7%)
        p16 = r'^image/paper:\s*' + NUM + SEP + NUM + r'\s*' + UNIT_CM + r'\s*\(' + NUM_FRAC + SEP + NUM_FRAC + r'\s*' + UNIT_IN + r'\.\);\s*' + build_2d_pattern('mount') + r'$'
        patterns.append(('image_paper_mount', re.compile(p16, re.IGNORECASE), self._parse_image_paper_mount))
        
        # Pattern 18: sheet only (0.7%)
        p18 = r'^' + build_2d_pattern('sheet') + r'$'
        patterns.append(('sheet_only', re.compile(p18, re.IGNORECASE), self._parse_sheet_only))
        
        # Pattern 19: image/paper + album page (0.6%)
        p19 = r'^image/paper:\s*' + NUM + SEP + NUM + r'\s*' + UNIT_CM + r'\s*\(' + NUM_FRAC + SEP + NUM_FRAC + r'\s*' + UNIT_IN + r'\.\);\s*album page:\s*' + NUM + SEP + NUM + r'\s*' + UNIT_CM + r'\s*\(' + NUM_FRAC + SEP + NUM_FRAC + r'\s*' + UNIT_IN + r'\.\)$'
        patterns.append(('image_paper_album', re.compile(p19, re.IGNORECASE), self._parse_image_paper_album))
        
        # Pattern 20: folded sheet (0.5%)
        p20 = r'^folded sheet:\s*' + NUM + SEP + NUM + r'\s*' + UNIT_CM + r'\s*\(' + NUM_FRAC + SEP + NUM_FRAC + r'\s*' + UNIT_IN + r'\.\)$'
        patterns.append(('folded_sheet', re.compile(p20, re.IGNORECASE), self._parse_folded_sheet))
        
        # =====================================================================
        # RIJKS PATTERNS (explicit labels: height, width, diameter, etc.)
        # Format: "height 130 mm x width 80 mm"
        # =====================================================================
        
        # Rijks Unit patterns (mm or cm)
        RIJKS_UNIT = r'(?:mm|cm)'
        
        # Helper to convert mm to cm
        def to_cm(val: float, unit: str) -> float:
            return val / 10.0 if unit.lower() == 'mm' else val
        self._to_cm = to_cm
        
        # Rijks Pattern 1: height X unit x width Y unit (79.2%)
        p_rijks_2d = r'^height\s+' + NUM + r'\s*(' + RIJKS_UNIT + r')\s*x\s*width\s+' + NUM + r'\s*(' + RIJKS_UNIT + r')$'
        patterns.append(('rijks_2d', re.compile(p_rijks_2d, re.IGNORECASE), self._parse_rijks_2d))
        
        # Rijks Pattern 2: diameter X unit x weight Y (coins - 3.2%)
        p_rijks_diam_weight = r'^diameter\s+' + NUM + r'\s*(' + RIJKS_UNIT + r')\s*x\s*weight\s+' + NUM + r'(?:\s*g)?$'
        patterns.append(('rijks_diam_weight', re.compile(p_rijks_diam_weight, re.IGNORECASE), self._parse_rijks_diam_weight))
        
        # Rijks Pattern 3: height X unit x width Y unit x depth Z unit (1.5%)
        p_rijks_3d = r'^height\s+' + NUM + r'\s*(' + RIJKS_UNIT + r')\s*x\s*width\s+' + NUM + r'\s*(' + RIJKS_UNIT + r')\s*x\s*depth\s+' + NUM + r'\s*(' + RIJKS_UNIT + r')$'
        patterns.append(('rijks_3d', re.compile(p_rijks_3d, re.IGNORECASE), self._parse_rijks_3d))
        
        # Rijks Pattern 4: height X unit x diameter Y unit (0.9%)
        p_rijks_h_diam = r'^height\s+' + NUM + r'\s*(' + RIJKS_UNIT + r')\s*x\s*diameter\s+' + NUM + r'\s*(' + RIJKS_UNIT + r')$'
        patterns.append(('rijks_h_diam', re.compile(p_rijks_h_diam, re.IGNORECASE), self._parse_rijks_h_diam))
        
        # Rijks Pattern 5: diameter X unit x height Y unit (swapped order) (0.7%)
        p_rijks_diam_h = r'^diameter\s+' + NUM + r'\s*(' + RIJKS_UNIT + r')\s*x\s*height\s+' + NUM + r'\s*(' + RIJKS_UNIT + r')$'
        patterns.append(('rijks_diam_h', re.compile(p_rijks_diam_h, re.IGNORECASE), self._parse_rijks_diam_h))
        
        # Rijks Pattern 6: length X unit x width Y unit (0.7%)
        p_rijks_lw = r'^length\s+' + NUM + r'\s*(' + RIJKS_UNIT + r')\s*x\s*width\s+' + NUM + r'\s*(' + RIJKS_UNIT + r')$'
        patterns.append(('rijks_lw', re.compile(p_rijks_lw, re.IGNORECASE), self._parse_rijks_lw))
        
        # Rijks Pattern 7: height X unit x width Y unit x thickness Z unit (0.7%)
        p_rijks_3d_thick = r'^height\s+' + NUM + r'\s*(' + RIJKS_UNIT + r')\s*x\s*width\s+' + NUM + r'\s*(' + RIJKS_UNIT + r')\s*x\s*thickness\s+' + NUM + r'\s*(' + RIJKS_UNIT + r')$'
        patterns.append(('rijks_3d_thick', re.compile(p_rijks_3d_thick, re.IGNORECASE), self._parse_rijks_3d_thick))
        
        # Rijks Pattern 8: diameter X unit (single, 0.4%)
        p_rijks_diam_only = r'^diameter\s+' + NUM + r'\s*(' + RIJKS_UNIT + r')$'
        patterns.append(('rijks_diam_only', re.compile(p_rijks_diam_only, re.IGNORECASE), self._parse_rijks_diam_only))
        
        # Rijks Pattern 9: height X unit (single, 0.1%)
        p_rijks_h_only = r'^height\s+' + NUM + r'\s*(' + RIJKS_UNIT + r')$'
        patterns.append(('rijks_h_only', re.compile(p_rijks_h_only, re.IGNORECASE), self._parse_rijks_h_only))
        
        # Rijks Pattern 10: width X unit x height Y unit (swapped, 0.4%)
        p_rijks_wh = r'^width\s+' + NUM + r'\s*(' + RIJKS_UNIT + r')\s*x\s*height\s+' + NUM + r'\s*(' + RIJKS_UNIT + r')$'
        patterns.append(('rijks_wh', re.compile(p_rijks_wh, re.IGNORECASE), self._parse_rijks_wh))
        
        # Rijks Pattern 11: diameter X unit x height Y unit x weight Z (coins)
        p_rijks_diam_h_weight = r'^diameter\s+' + NUM + r'\s*(' + RIJKS_UNIT + r')\s*x\s*height\s+' + NUM + r'\s*(' + RIJKS_UNIT + r')\s*x\s*weight\s+' + NUM + r'(?:\s*g)?$'
        patterns.append(('rijks_diam_h_weight', re.compile(p_rijks_diam_h_weight, re.IGNORECASE), self._parse_rijks_diam_h_weight))
        
        # Rijks Pattern 12: height X unit, width Y unit (comma separator)
        p_rijks_2d_comma = r'^height\s+' + NUM + r'\s*(' + RIJKS_UNIT + r'),\s*width\s+' + NUM + r'\s*(' + RIJKS_UNIT + r')$'
        patterns.append(('rijks_2d_comma', re.compile(p_rijks_2d_comma, re.IGNORECASE), self._parse_rijks_2d))
        
        # Rijks Pattern 13: height X unit x width Y unit x weight Z (2D + weight)
        p_rijks_2d_weight = r'^height\s+' + NUM + r'\s*(' + RIJKS_UNIT + r')\s*x\s*width\s+' + NUM + r'\s*(' + RIJKS_UNIT + r')\s*x\s*weight\s+' + NUM + r'(?:\s*g)?$'
        patterns.append(('rijks_2d_weight', re.compile(p_rijks_2d_weight, re.IGNORECASE), self._parse_rijks_2d_weight))
        
        # Rijks Pattern 14: height X unit x width Y unit x depth Z unit x weight W (3D + weight)
        p_rijks_3d_weight = r'^height\s+' + NUM + r'\s*(' + RIJKS_UNIT + r')\s*x\s*width\s+' + NUM + r'\s*(' + RIJKS_UNIT + r')\s*x\s*depth\s+' + NUM + r'\s*(' + RIJKS_UNIT + r')\s*x\s*weight\s+' + NUM + r'(?:\s*g)?$'
        patterns.append(('rijks_3d_weight', re.compile(p_rijks_3d_weight, re.IGNORECASE), self._parse_rijks_3d_weight))
        
        # Rijks Pattern 14b: length X unit x width Y unit x weight Z (length+width+weight)
        p_rijks_lw_weight = r'^length\s+' + NUM + r'\s*(' + RIJKS_UNIT + r')\s*x\s*width\s+' + NUM + r'\s*(' + RIJKS_UNIT + r')\s*x\s*weight\s+' + NUM + r'(?:\s*g)?$'
        patterns.append(('rijks_lw_weight', re.compile(p_rijks_lw_weight, re.IGNORECASE), self._parse_rijks_lw_weight))
        
        # Rijks Pattern 15: length X unit (single)
        p_rijks_l_only = r'^length\s+' + NUM + r'\s*(' + RIJKS_UNIT + r')$'
        patterns.append(('rijks_l_only', re.compile(p_rijks_l_only, re.IGNORECASE), self._parse_rijks_l_only))
        
        # Rijks Pattern 16: depth X cm, height Y cm x width Z cm
        p_rijks_depth_2d = r'^depth\s+' + NUM + r'\s*(' + RIJKS_UNIT + r'),\s*height\s+' + NUM + r'\s*(' + RIJKS_UNIT + r')\s*x\s*width\s+' + NUM + r'\s*(' + RIJKS_UNIT + r')$'
        patterns.append(('rijks_depth_2d', re.compile(p_rijks_depth_2d, re.IGNORECASE), self._parse_rijks_depth_2d))
        
        # Rijks Pattern 17: length X unit x width Y unit x thickness Z unit (x weight W optional)
        p_rijks_lwt = r'^length\s+' + NUM + r'\s*(' + RIJKS_UNIT + r')\s*x\s*width\s+' + NUM + r'\s*(' + RIJKS_UNIT + r')\s*x\s*thickness\s+' + NUM + r'\s*(' + RIJKS_UNIT + r')(?:\s*x\s*weight\s+' + NUM + r'(?:\s*g)?)?$'
        patterns.append(('rijks_lwt', re.compile(p_rijks_lwt, re.IGNORECASE), self._parse_rijks_lwt))
        
        # Rijks Pattern 18: width X unit x height Y unit x thickness Z unit (x weight W optional)
        p_rijks_wht = r'^width\s+' + NUM + r'\s*(' + RIJKS_UNIT + r')\s*x\s*height\s+' + NUM + r'\s*(' + RIJKS_UNIT + r')\s*x\s*thickness\s+' + NUM + r'\s*(' + RIJKS_UNIT + r')(?:\s*x\s*weight\s+' + NUM + r'(?:\s*g)?)?$'
        patterns.append(('rijks_wht', re.compile(p_rijks_wht, re.IGNORECASE), self._parse_rijks_wht))
        
        # =====================================================================
        # MET PATTERNS (imperial-first: H x W in. (H x W cm))
        # Format: "47 1/2 × 7 3/8 in. (120.7 × 18.7 cm)"
        # =====================================================================
        
        # MET separator (both x and ×)
        MET_SEP = r'\s*[×x]\s*'
        
        # Helper to parse hyphenated fractions
        def parse_met_num(s: str) -> float:
            s = s.strip()
            if '/' in s:
                # Could be "10-1/2" or "10 1/2" or "1/2"
                s = s.replace('-', ' ')  # Normalize
                parts = s.split()
                if len(parts) == 2:
                    whole = float(parts[0])
                    num, denom = parts[1].split('/')
                    return whole + float(num) / float(denom)
                else:
                    num, denom = s.split('/')
                    return float(num) / float(denom)
            return float(s)
        self._parse_met_num = parse_met_num
        
        # MET Pattern 1: sheet: H x W in. (H x W cm) (7.99% + 6.96% = 14.95%)
        p_met_sheet = r'^sheet:\s*' + NUM_FRAC_EXT + MET_SEP + NUM_FRAC_EXT + r'\s*in\.?\s*\(\s*' + NUM + MET_SEP + NUM + r'\s*cm\.?\s*\)$'
        patterns.append(('met_sheet', re.compile(p_met_sheet, re.IGNORECASE), self._parse_met_sheet))
        
        # MET Pattern 2: Simple H x W in. (H x W cm) (5.75%)
        p_met_simple_2d = r'^' + NUM_FRAC_EXT + MET_SEP + NUM_FRAC_EXT + r'\s*in\.?\s*\(\s*' + NUM + MET_SEP + NUM + r'\s*cm\.?\s*\)$'
        patterns.append(('met_simple_2d', re.compile(p_met_simple_2d, re.IGNORECASE), self._parse_met_simple_2d))
        
        # MET Pattern 3: h. H in. (H cm) (2.51%)
        p_met_h_only = r'^' + LBL_H + r'\s*' + NUM_FRAC_EXT + r'\s*in\.?\s*\(\s*' + NUM + r'\s*cm\.?\s*\)$'
        patterns.append(('met_h_only', re.compile(p_met_h_only, re.IGNORECASE), self._parse_met_h_only))
        
        # MET Pattern 4: overall: H x W in. (H x W cm) (1.9%)
        p_met_overall = r'^overall:\s*' + NUM_FRAC_EXT + MET_SEP + NUM_FRAC_EXT + r'\s*in\.?\s*\(\s*' + NUM + MET_SEP + NUM + r'\s*cm\.?\s*\)$'
        patterns.append(('met_overall', re.compile(p_met_overall, re.IGNORECASE), self._parse_met_overall))
        
        # MET Pattern 5: height: H in. (H cm) (1.41%)
        p_met_height = r'^' + LBL_H + r'\s*' + NUM_FRAC_EXT + r'\s*in\.?\s*\(\s*' + NUM + r'\s*cm\.?\s*\)$'
        patterns.append(('met_height', re.compile(p_met_height, re.IGNORECASE), self._parse_met_height))
        
        # MET Pattern 6: diameter: D in. (D cm) (0.82%)
        p_met_diameter = r'^' + LBL_DIAM + r'\s*' + NUM_FRAC_EXT + r'\s*in\.?\s*\(\s*' + NUM + r'\s*cm\.?\s*\)$'
        patterns.append(('met_diameter', re.compile(p_met_diameter, re.IGNORECASE), self._parse_met_diameter))
        
        # MET Pattern 7: diam. D in. (D cm) (0.82%)
        p_met_diam = r'^' + LBL_DIAM + r'\s*' + NUM_FRAC_EXT + r'\s*in\.?\s*\(\s*' + NUM + r'\s*cm\.?\s*\)$'
        patterns.append(('met_diam', re.compile(p_met_diam, re.IGNORECASE), self._parse_met_diam))
        
        # MET Pattern 8: sheet (trimmed): H x W in. (H x W cm) (1.05%)
        p_met_sheet_trimmed = r'^sheet\s*\(trimmed\):\s*' + NUM_FRAC_EXT + MET_SEP + NUM_FRAC_EXT + r'\s*in\.?\s*\(\s*' + NUM + MET_SEP + NUM + r'\s*cm\.?\s*\)$'
        patterns.append(('met_sheet_trimmed', re.compile(p_met_sheet_trimmed, re.IGNORECASE), self._parse_met_sheet_trimmed))
        
        # MET Pattern 9: H x W x D in. (H x W x D cm) - 3D
        p_met_simple_3d = r'^' + NUM_FRAC_EXT + MET_SEP + NUM_FRAC_EXT + MET_SEP + NUM_FRAC_EXT + r'\s*in\.?\s*\(\s*' + NUM + MET_SEP + NUM + MET_SEP + NUM + r'\s*cm\.?\s*\)$'
        patterns.append(('met_simple_3d', re.compile(p_met_simple_3d, re.IGNORECASE), self._parse_met_simple_3d))
        
        # MET Pattern 10: h. H in. (H cm); diam. D in. (D cm) (0.99%)
        p_met_h_diam = r'^' + LBL_H + r'\s*' + NUM_FRAC_EXT + r'\s*in\.?\s*\(\s*' + NUM + r'\s*cm\.?\s*\)(?:;|\s+)' + LBL_DIAM + r'\s*' + NUM_FRAC_EXT + r'\s*in\.?\s*\(\s*' + NUM + r'\s*cm\.?\s*\)$'
        patterns.append(('met_h_diam', re.compile(p_met_h_diam, re.IGNORECASE), self._parse_met_h_diam))
        
        # MET Pattern 11: h. H in. (H cm); w. W in. (W cm) (0.84%)
        p_met_h_w = r'^' + LBL_H + r'\s*' + NUM_FRAC_EXT + r'\s*in\.?\s*\(\s*' + NUM + r'\s*cm\.?\s*\)(?:;|\s+)' + LBL_W + r'\s*' + NUM_FRAC_EXT + r'\s*in\.?\s*\(\s*' + NUM + r'\s*cm\.?\s*\)$'
        patterns.append(('met_h_w', re.compile(p_met_h_w, re.IGNORECASE), self._parse_met_h_w))
        
        # MET Pattern 12: Single number in. (cm) (0.70%)
        p_met_single = r'^' + NUM_FRAC_EXT + r'\s*in\.?\s*\(\s*' + NUM + r'\s*cm\.?\s*\)$'
        patterns.append(('met_single', re.compile(p_met_single, re.IGNORECASE), self._parse_met_single))
        
        # =====================================================================
        # MET PATTERNS - EXTENDED (for top 10 unparsed)
        # =====================================================================
        
        # MET Pattern 13: plate: H × W in. (cm) sheet: H × W in. (cm) (0.78%)
        p_met_plate_sheet = r'^plate:\s*' + NUM_FRAC_EXT + MET_SEP + NUM_FRAC_EXT + r'\s*in\.?\s*\(\s*' + NUM + MET_SEP + NUM + r'\s*cm\.?\s*\)\s*sheet:\s*' + NUM_FRAC_EXT + MET_SEP + NUM_FRAC_EXT + r'\s*in\.?\s*\(\s*' + NUM + MET_SEP + NUM + r'\s*cm\.?\s*\)$'
        patterns.append(('met_plate_sheet', re.compile(p_met_plate_sheet, re.IGNORECASE), self._parse_met_plate_sheet))
        
        # MET Pattern 14: sheet: ... plate: ... (reversed order) (0.66%)
        p_met_sheet_plate = r'^sheet:\s*' + NUM_FRAC_EXT + MET_SEP + NUM_FRAC_EXT + r'\s*in\.?\s*\(\s*' + NUM + MET_SEP + NUM + r'\s*cm\.?\s*\)\s*plate:\s*' + NUM_FRAC_EXT + MET_SEP + NUM_FRAC_EXT + r'\s*in\.?\s*\(\s*' + NUM + MET_SEP + NUM + r'\s*cm\.?\s*\)$'
        patterns.append(('met_sheet_plate', re.compile(p_met_sheet_plate, re.IGNORECASE), self._parse_met_sheet_plate))
        
        # MET Pattern 15: l. X x w. Y inches (X x Y cm) (0.58% + 0.53%)
        p_met_lw_inches = r'^' + LBL_L + r'\s*' + NUM_FRAC_EXT + r'\s*x\s*' + LBL_W + r'\s*' + NUM_FRAC_EXT + r'\s*inches\s*\(\s*' + NUM + r'\s*x\s*' + NUM + r'\s*cm\.?\s*\)$'
        patterns.append(('met_lw_inches', re.compile(p_met_lw_inches, re.IGNORECASE), self._parse_met_lw))
        
        # MET Pattern 16: overall: 3D (0.52% + 0.49%)
        p_met_overall_3d = r'^overall:\s*' + NUM_FRAC_EXT + MET_SEP + NUM_FRAC_EXT + MET_SEP + NUM_FRAC_EXT + r'\s*in\.?\s*\(\s*' + NUM + MET_SEP + NUM + MET_SEP + NUM + r'\s*cm\.?\s*\)$'
        patterns.append(('met_overall_3d', re.compile(p_met_overall_3d, re.IGNORECASE), self._parse_met_overall_3d))
        
        # MET Pattern 17: h. X in. (cm) w. Y in. (cm) - space separator instead of semicolon (0.52%)
        p_met_h_w_space = r'^' + LBL_H + r'\s*' + NUM_FRAC_EXT + r'\s*in\.?\s*\(\s*' + NUM + r'\s*cm\.?\s*\)\s+' + LBL_W + r'\s*' + NUM_FRAC_EXT + r'\s*in\.?\s*\(\s*' + NUM + r'\s*cm\.?\s*\)$'
        patterns.append(('met_h_w_space', re.compile(p_met_h_w_space, re.IGNORECASE), self._parse_met_h_w))
        
        # MET Pattern 17b: multi-line 3D variant (h. X in. (cm) w. Y in. (cm) d. Z in. (cm))
        p_met_h_w_d_space = r'^' + LBL_H + r'\s*' + NUM_FRAC_EXT + r'\s*in\.?\s*\(\s*' + NUM + r'\s*cm\.?\s*\)\s+' + \
                            LBL_W + r'\s*' + NUM_FRAC_EXT + r'\s*in\.?\s*\(\s*' + NUM + r'\s*cm\.?\s*\)\s+' + \
                            LBL_D + r'\s*' + NUM_FRAC_EXT + r'\s*in\.?\s*\(\s*' + NUM + r'\s*cm\.?\s*\)$'
        patterns.append(('met_h_w_d_space', re.compile(p_met_h_w_d_space, re.IGNORECASE), self._parse_met_h_w_d_space))
        
        # MET Pattern 18: image: ... sheet: ... (0.51%)
        p_met_image_sheet = r'^image:\s*' + NUM_FRAC_EXT + MET_SEP + NUM_FRAC_EXT + r'\s*in\.?\s*\(\s*' + NUM + MET_SEP + NUM + r'\s*cm\.?\s*\)\s*sheet:\s*' + NUM_FRAC_EXT + MET_SEP + NUM_FRAC_EXT + r'\s*in\.?\s*\(\s*' + NUM + MET_SEP + NUM + r'\s*cm\.?\s*\)$'
        patterns.append(('met_image_sheet', re.compile(p_met_image_sheet, re.IGNORECASE), self._parse_met_image_sheet))
        
        # MET Pattern 19: l. X x w. Y inches X x Y cm - missing parentheses (0.50%)
        p_met_lw_noparen = r'^' + LBL_L + r'\s*' + NUM_FRAC_EXT + r'\s*x\s*' + LBL_W + r'\s*' + NUM_FRAC_EXT + r'\s*inches\s+' + NUM + r'\s*x\s*' + NUM + r'\s*cm\.?$'
        patterns.append(('met_lw_noparen', re.compile(p_met_lw_noparen, re.IGNORECASE), self._parse_met_lw))
        
        # MET Pattern 20: l. X in. (cm) - length only (0.48%)
        p_met_l_only = r'^' + LBL_L + r'\s*' + NUM_FRAC_EXT + r'\s*in\.?\s*\(\s*' + NUM + r'\s*cm\.?\s*\)$'
        patterns.append(('met_l_only', re.compile(p_met_l_only, re.IGNORECASE), self._parse_met_l_only))
        
        # MET Pattern 21: H. X × W. Y × D. Z cm (A × B × C in.) - metric first, inline labels
        p_met_inline_metric_3d = r'^' + LBL_H + r'\s*' + NUM + MET_SEP + LBL_W + r'\s*' + NUM + MET_SEP + LBL_D + r'\s*' + NUM + r'\s*cm\.?\s*\(\s*' + NUM_FRAC_EXT + MET_SEP + NUM_FRAC_EXT + MET_SEP + NUM_FRAC_EXT + r'\s*in\.?\s*\)$'
        patterns.append(('met_inline_metric_3d', re.compile(p_met_inline_metric_3d, re.IGNORECASE), self._parse_met_inline_metric_3d))
        
        return patterns
    
    # =========================================================================
    # Parser methods for each pattern
    # =========================================================================
    
    def _wrap_parts(self, parts: List[Dict]) -> Dict:
        """Wrap list of parts directly into the final structure."""
        return {"parts": parts}
    
    def _parse_simple_2d(self, match) -> Dict:
        h, w = float(match.group(1)), float(match.group(2))
        return self._wrap_parts([make_2d_result(h, w)])
    
    def _parse_image_sheet(self, match) -> Dict:
        return self._wrap_parts([
            make_2d_result(float(match.group(1)), float(match.group(2)), 'image'),
            make_2d_result(float(match.group(5)), float(match.group(6)), 'sheet'),
        ])
    
    def _parse_image_plate_sheet(self, match) -> Dict:
        return self._wrap_parts([
            make_2d_result(float(match.group(1)), float(match.group(2)), 'image'),
            make_2d_result(float(match.group(5)), float(match.group(6)), 'plate'),
            make_2d_result(float(match.group(9)), float(match.group(10)), 'sheet'),
        ])
    
    def _parse_simple_3d(self, match) -> Dict:
        h, w, d = float(match.group(1)), float(match.group(2)), float(match.group(3))
        return self._wrap_parts([make_3d_result(h, w, d)])
    
    def _parse_plate_sheet(self, match) -> Dict:
        return self._wrap_parts([
            make_2d_result(float(match.group(1)), float(match.group(2)), 'plate'),
            make_2d_result(float(match.group(5)), float(match.group(6)), 'sheet'),
        ])
    
    def _parse_image_plate_combined_sheet(self, match) -> Dict:
        return self._wrap_parts([
            make_2d_result(float(match.group(1)), float(match.group(2)), 'image/plate'),
            make_2d_result(float(match.group(5)), float(match.group(6)), 'sheet'),
        ])
    
    def _parse_height_only(self, match) -> Dict:
        return self._wrap_parts([make_single_result(float(match.group(1)), 'height')])
    
    def _parse_diameter_only(self, match) -> Dict:
        return self._wrap_parts([make_single_result(float(match.group(1)), 'diameter')])
    
    def _parse_2d_plus_diam(self, match) -> Dict:
        return self._wrap_parts([
            make_2d_result(float(match.group(1)), float(match.group(2))),
            make_single_result(float(match.group(5)), 'diameter'),
        ])
    
    def _parse_image_paper_mount(self, match) -> Dict:
        return self._wrap_parts([
            make_2d_result(float(match.group(1)), float(match.group(2)), 'image/paper'),
            make_2d_result(float(match.group(5)), float(match.group(6)), 'mount'),
        ])
    
    def _parse_sheet_only(self, match) -> Dict:
        return self._wrap_parts([make_2d_result(float(match.group(1)), float(match.group(2)), 'sheet')])
    
    def _parse_image_paper_album(self, match) -> Dict:
        return self._wrap_parts([
            make_2d_result(float(match.group(1)), float(match.group(2)), 'image/paper'),
            make_2d_result(float(match.group(5)), float(match.group(6)), 'album page'),
        ])
    
    def _parse_folded_sheet(self, match) -> Dict:
        return self._wrap_parts([make_2d_result(float(match.group(1)), float(match.group(2)), 'folded sheet')])
    
    # =========================================================================
    # Rijks Parser methods
    # =========================================================================
    
    def _parse_rijks_2d(self, match) -> Dict:
        h = self._to_cm(float(match.group(1)), match.group(2))
        w = self._to_cm(float(match.group(3)), match.group(4))
        measurements = [
            make_measurement('height', round(h, 2)),
            make_measurement('width', round(w, 2))
        ]
        if len(match.groups()) > 4 and match.group(5):
            measurements.append(make_measurement('weight', float(match.group(5))))
        
        # Filter out measurements with value 0.0
        valid_measurements = [m for m in measurements if m.get('value') != 0.0]
        
        # If no valid measurements remain, return an empty parts list or handle as an error
        if not valid_measurements:
            return self._wrap_parts([]) # Or raise an error, depending on desired behavior
            
        return self._wrap_parts([make_part(valid_measurements)])
    
    def _parse_rijks_diam(self, match) -> Dict:
        d = self._to_cm(float(match.group(1)), match.group(2))
        measurements = [make_measurement('diameter', round(d, 2))]
        if len(match.groups()) > 2 and match.group(3):
            measurements.append(make_measurement('weight', float(match.group(3))))
        return self._wrap_parts([make_part(measurements)])
    
    def _parse_rijks_3d(self, match) -> Dict:
        h = self._to_cm(float(match.group(1)), match.group(2))
        w = self._to_cm(float(match.group(3)), match.group(4))
        d = self._to_cm(float(match.group(5)), match.group(6))
        measurements = [
            make_measurement('height', round(h, 2)),
            make_measurement('width', round(w, 2)),
            make_measurement('depth', round(d, 2))
        ]
        if len(match.groups()) > 6 and match.group(7):
            measurements.append(make_measurement('weight', float(match.group(7))))
        return self._wrap_parts([make_part(measurements)])
    
    def _parse_rijks_h_diam(self, match) -> Dict:
        h = self._to_cm(float(match.group(1)), match.group(2))
        d = self._to_cm(float(match.group(3)), match.group(4))
        return self._wrap_parts([make_part([
            make_measurement('height', round(h, 2)),
            make_measurement('diameter', round(d, 2))
        ])])
    
    def _parse_rijks_diam_h(self, match) -> Dict:
        d = self._to_cm(float(match.group(1)), match.group(2))
        h = self._to_cm(float(match.group(3)), match.group(4))
        measurements = [
            make_measurement('diameter', round(d, 2)),
            make_measurement('height', round(h, 2))
        ]
        if len(match.groups()) > 4 and match.group(5):
            measurements.append(make_measurement('weight', float(match.group(5))))
        return self._wrap_parts([make_part(measurements)])
    
    def _parse_rijks_lw(self, match) -> Dict:
        l = self._to_cm(float(match.group(1)), match.group(2))
        w = self._to_cm(float(match.group(3)), match.group(4))
        return self._wrap_parts([make_part([
            make_measurement('length', round(l, 2)),
            make_measurement('width', round(w, 2))
        ])])
    
    def _parse_rijks_3d_thick(self, match) -> Dict:
        h = self._to_cm(float(match.group(1)), match.group(2))
        w = self._to_cm(float(match.group(3)), match.group(4))
        t = self._to_cm(float(match.group(5)), match.group(6))
        return self._wrap_parts([make_part([
            make_measurement('height', round(h, 2)),
            make_measurement('width', round(w, 2)),
            make_measurement('thickness', round(t, 2))
        ])])
    
    def _parse_rijks_diam_only(self, match) -> Dict:
        d = self._to_cm(float(match.group(1)), match.group(2))
        return self._wrap_parts([make_single_result(d, 'diameter')])
    
    def _parse_rijks_h_only(self, match) -> Dict:
        h = self._to_cm(float(match.group(1)), match.group(2))
        return self._wrap_parts([make_single_result(h, 'height')])
    
    def _parse_rijks_wh(self, match) -> Dict:
        w = self._to_cm(float(match.group(1)), match.group(2))
        h = self._to_cm(float(match.group(3)), match.group(4))
        return self._wrap_parts([make_part([
            make_measurement('width', round(w, 2)),
            make_measurement('height', round(h, 2))
        ])])
    
    def _parse_rijks_l_only(self, match) -> Dict:
        l = self._to_cm(float(match.group(1)), match.group(2))
        return self._wrap_parts([make_single_result(l, 'length')])
    
    # =========================================================================
    # Rijks Weight-inclusive Parser methods
    # =========================================================================
    
    def _parse_rijks_diam_weight(self, match) -> Dict:
        """diameter X unit x weight Y g"""
        d = self._to_cm(float(match.group(1)), match.group(2))
        w_g = float(match.group(3))
        return self._wrap_parts([make_part([
            make_measurement('diameter', round(d, 2)),
            make_measurement('weight', round(w_g, 2))
        ])])
    
    def _parse_rijks_diam_h_weight(self, match) -> Dict:
        """diameter X unit x height Y unit x weight Z g"""
        d = self._to_cm(float(match.group(1)), match.group(2))
        h = self._to_cm(float(match.group(3)), match.group(4))
        w_g = float(match.group(5))
        return self._wrap_parts([make_part([
            make_measurement('diameter', round(d, 2)),
            make_measurement('height', round(h, 2)),
            make_measurement('weight', round(w_g, 2))
        ])])
    
    def _parse_rijks_2d_weight(self, match) -> Dict:
        """height X unit x width Y unit x weight Z g"""
        h = self._to_cm(float(match.group(1)), match.group(2))
        w = self._to_cm(float(match.group(3)), match.group(4))
        w_g = float(match.group(5))
        return self._wrap_parts([make_part([
            make_measurement('height', round(h, 2)),
            make_measurement('width', round(w, 2)),
            make_measurement('weight', round(w_g, 2))
        ])])
    
    def _parse_rijks_3d_weight(self, match) -> Dict:
        """height X unit x width Y unit x depth Z unit x weight W g"""
        h = self._to_cm(float(match.group(1)), match.group(2))
        w = self._to_cm(float(match.group(3)), match.group(4))
        d = self._to_cm(float(match.group(5)), match.group(6))
        w_g = float(match.group(7))
        return self._wrap_parts([make_part([
            make_measurement('height', round(h, 2)),
            make_measurement('width', round(w, 2)),
            make_measurement('depth', round(d, 2)),
            make_measurement('weight', round(w_g, 2))
        ])])
    
    def _parse_rijks_lw_weight(self, match) -> Dict:
        """length X unit x width Y unit x weight Z g"""
        l = self._to_cm(float(match.group(1)), match.group(2))
        w = self._to_cm(float(match.group(3)), match.group(4))
        w_g = float(match.group(5))
        return self._wrap_parts([make_part([
            make_measurement('length', round(l, 2)),
            make_measurement('width', round(w, 2)),
            make_measurement('weight', round(w_g, 2))
        ])])
    
    def _parse_rijks_depth_2d(self, match) -> Dict:
        d = self._to_cm(float(match.group(1)), match.group(2))
        h = self._to_cm(float(match.group(3)), match.group(4))
        w = self._to_cm(float(match.group(5)), match.group(6))
        return self._wrap_parts([make_part([
            make_measurement('depth', round(d, 2)),
            make_measurement('height', round(h, 2)),
            make_measurement('width', round(w, 2))
        ])])
    
    def _parse_rijks_lwt(self, match) -> Dict:
        l = self._to_cm(float(match.group(1)), match.group(2))
        w = self._to_cm(float(match.group(3)), match.group(4))
        t = self._to_cm(float(match.group(5)), match.group(6))
        measurements = [
            make_measurement('length', round(l, 2)),
            make_measurement('width', round(w, 2)),
            make_measurement('thickness', round(t, 2))
        ]
        if len(match.groups()) > 6 and match.group(7):
            measurements.append(make_measurement('weight', float(match.group(7))))
        return self._wrap_parts([make_part(measurements)])
    
    def _parse_rijks_wht(self, match) -> Dict:
        w = self._to_cm(float(match.group(1)), match.group(2))
        h = self._to_cm(float(match.group(3)), match.group(4))
        t = self._to_cm(float(match.group(5)), match.group(6))
        measurements = [
            make_measurement('width', round(w, 2)),
            make_measurement('height', round(h, 2)),
            make_measurement('thickness', round(t, 2))
        ]
        if len(match.groups()) > 6 and match.group(7):
            measurements.append(make_measurement('weight', float(match.group(7))))
        return self._wrap_parts([make_part(measurements)])
    
    # =========================================================================
    # MET Parser methods (use cm values from parentheses)
    # =========================================================================
    
    def _parse_met_sheet(self, match) -> Dict:
        # Groups: 1=in_h, 2=in_w, 3=cm_h, 4=cm_w
        h_cm = float(match.group(3))
        w_cm = float(match.group(4))
        return self._wrap_parts([make_2d_result(h_cm, w_cm, 'sheet')])
    
    def _parse_met_simple_2d(self, match) -> Dict:
        h_cm = float(match.group(3))
        w_cm = float(match.group(4))
        return self._wrap_parts([make_2d_result(h_cm, w_cm)])
    
    def _parse_met_h_only(self, match) -> Dict:
        # Groups: 1=in_h, 2=cm_h
        h_cm = float(match.group(2))
        return self._wrap_parts([make_single_result(h_cm, 'height')])
    
    def _parse_met_overall(self, match) -> Dict:
        h_cm = float(match.group(3))
        w_cm = float(match.group(4))
        return self._wrap_parts([make_2d_result(h_cm, w_cm, 'overall')])
    
    def _parse_met_height(self, match) -> Dict:
        h_cm = float(match.group(2))
        return self._wrap_parts([make_single_result(h_cm, 'height')])
    
    def _parse_met_diameter(self, match) -> Dict:
        d_cm = float(match.group(2))
        return self._wrap_parts([make_single_result(d_cm, 'diameter')])
    
    def _parse_met_diam(self, match) -> Dict:
        d_cm = float(match.group(2))
        return self._wrap_parts([make_single_result(d_cm, 'diameter')])
    
    def _parse_met_sheet_trimmed(self, match) -> Dict:
        h_cm = float(match.group(3))
        w_cm = float(match.group(4))
        return self._wrap_parts([make_2d_result(h_cm, w_cm, 'sheet (trimmed)')])
    
    def _parse_met_simple_3d(self, match) -> Dict:
        # Groups: 1,2,3=in h/w/d, 4,5,6=cm h/w/d
        h_cm = float(match.group(4))
        w_cm = float(match.group(5))
        d_cm = float(match.group(6))
        return self._wrap_parts([make_3d_result(h_cm, w_cm, d_cm)])
    
    def _parse_met_h_diam(self, match) -> Dict:
        # Groups: 1=in_h, 2=cm_h, 3=in_diam, 4=cm_diam
        h_cm = float(match.group(2))
        d_cm = float(match.group(4))
        return self._wrap_parts([make_part([
            make_measurement('height', round(h_cm, 2)),
            make_measurement('diameter', round(d_cm, 2))
        ])])
    
    def _parse_met_h_w(self, match) -> Dict:
        # Groups: 1=in_h, 2=cm_h, 3=in_w, 4=cm_w
        h_cm = float(match.group(2))
        w_cm = float(match.group(4))
        return self._wrap_parts([make_2d_result(h_cm, w_cm)])
    
    def _parse_met_single(self, match) -> Dict:
        # Groups: 1=in_val, 2=cm_val
        val_cm = float(match.group(2))
        return self._wrap_parts([make_single_result(val_cm, 'height')])
    
    def _parse_met_h_w_d_space(self, match) -> Dict:
        # Groups: 1=in_h, 2=cm_h, 3=in_w, 4=cm_w, 5=in_d, 6=cm_d
        h_cm = float(match.group(2))
        w_cm = float(match.group(4))
        d_cm = float(match.group(6))
        return self._wrap_parts([make_3d_result(h_cm, w_cm, d_cm)])
        
    def _parse_met_inline_metric_3d(self, match) -> Dict:
        # Groups 1=cm_h, 2=cm_w, 3=cm_d, 4=in_h, 5=in_w, 6=in_d
        h_cm = float(match.group(1))
        w_cm = float(match.group(2))
        d_cm = float(match.group(3))
        return self._wrap_parts([make_3d_result(h_cm, w_cm, d_cm)])

    # Extended MET parsers
    
    def _parse_met_plate_sheet(self, match) -> Dict:
        # plate: groups 1-4, sheet: groups 5-8 (in h/w, cm h/w)
        plate_h = float(match.group(3))
        plate_w = float(match.group(4))
        sheet_h = float(match.group(7))
        sheet_w = float(match.group(8))
        return self._wrap_parts([
            make_2d_result(plate_h, plate_w, 'plate'),
            make_2d_result(sheet_h, sheet_w, 'sheet'),
        ])
    
    def _parse_met_sheet_plate(self, match) -> Dict:
        # sheet first, then plate
        sheet_h = float(match.group(3))
        sheet_w = float(match.group(4))
        plate_h = float(match.group(7))
        plate_w = float(match.group(8))
        return self._wrap_parts([
            make_2d_result(sheet_h, sheet_w, 'sheet'),
            make_2d_result(plate_h, plate_w, 'plate'),
        ])
    
    def _parse_met_lw(self, match) -> Dict:
        # l. X x w. Y - groups 1/2 = in l/w, 3/4 = cm l/w
        l_cm = float(match.group(3))
        w_cm = float(match.group(4))
        return self._wrap_parts([make_part([
            make_measurement('length', round(l_cm, 2)),
            make_measurement('width', round(w_cm, 2))
        ])])
    
    def _parse_met_overall_3d(self, match) -> Dict:
        # overall: 3D - groups 1-3 = in, 4-6 = cm
        h_cm = float(match.group(4))
        w_cm = float(match.group(5))
        d_cm = float(match.group(6))
        return self._wrap_parts([make_3d_result(h_cm, w_cm, d_cm, 'overall')])
    
    def _parse_met_image_sheet(self, match) -> Dict:
        # image: groups 1-4, sheet: groups 5-8
        image_h = float(match.group(3))
        image_w = float(match.group(4))
        sheet_h = float(match.group(7))
        sheet_w = float(match.group(8))
        return self._wrap_parts([
            make_2d_result(image_h, image_w, 'image'),
            make_2d_result(sheet_h, sheet_w, 'sheet'),
        ])
    
    def _parse_met_l_only(self, match) -> Dict:
        # l. X in. (cm) - groups: 1=in, 2=cm
        l_cm = float(match.group(2))
        return self._wrap_parts([make_single_result(l_cm, 'length')])
    
    # =========================================================================
    # Main parse method
    # =========================================================================
    
    def parse(self, s: str) -> Optional[Dict]:
        """
        Parse a dimension string using pattern matching.
        Returns dimension dict with structure {"parts": [...]}, or None if no pattern matches.
        """
        if not s or not isinstance(s, str):
            return None
        
        s = s.strip()
        
        for name, pattern, parser_func in self.patterns:
            match = pattern.match(s)
            if match:
                self.parse_stats['matched'] += 1
                self.parse_stats['by_pattern'][name] = self.parse_stats['by_pattern'].get(name, 0) + 1
                return parser_func(match)
        
        self.parse_stats['unmatched'] += 1
        return None
    
    def get_stats(self) -> Dict:
        """Return parsing statistics."""
        total = self.parse_stats['matched'] + self.parse_stats['unmatched']
        return {
            'total': total,
            'matched': self.parse_stats['matched'],
            'unmatched': self.parse_stats['unmatched'],
            'match_rate': round(100 * self.parse_stats['matched'] / total, 2) if total > 0 else 0,
            'by_pattern': self.parse_stats['by_pattern'],
        }
