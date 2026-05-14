import re
import sys
import os
from typing import Optional, Dict, List, Any

# TO DO: change
try:
    from dimensions_parsing.pattern_parser import PatternParser
except ImportError:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from dimensions_parsing.pattern_parser import PatternParser

_PARSER = PatternParser()

def _format_measurement(m: Dict[str, Any]) -> str:
    """Format a single measurement dict to 'attribute/value/unit' string."""
    attr = m['attribute']
    val = m['value']
    unit = "cm" 
    if attr == 'weight':
        unit = "g"
        
    return f"{attr} {val} {unit}"

def _format_pattern_result(data: Dict[str, Any]) -> str:
    """
    Convert PatternParser JSON output back to the normalized string format:
    "part: height X cm x width Y cm ; part2: ..."
    """
    parts_out = []
    
    for part in data.get('parts', []):
        measurements = part.get('measurements', [])
        if not measurements:
            continue
            
        m_strs = [_format_measurement(m) for m in measurements]
        segment = " x ".join(m_strs)
        
        part_name = part.get('part')
        if part_name:
            segment = f"{part_name}: {segment}"
            
        parts_out.append(segment)
        
    return " ; ".join(parts_out)

def normalize_dimensions(raw: str) -> Optional[str]:
    """Main entry point for dimension normalization using PatternParser."""
    if not raw:
        return None
        
    result = _PARSER.parse(raw)
    
    if result:
        return _format_pattern_result(result)
        
    return None
