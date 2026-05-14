import re
import json
import os
import nltk
from nltk.corpus import stopwords as nltk_stopwords
from typing import List, Dict, Set, Optional, Union

EXTRA_STOPWORDS = {"from", "with", "on", "and", "in", "to", "by", "of", "over", "applied"}
COLORS = {"black", "white", "red", "blue", "green", "yellow", "brown", "orange", "purple", "pink", "gray", "grey", "cream", "buff", "tan", "beige", "dark", "light", "pale"}


term_dictionary: Dict[str, str] = {}
sorted_keys: List[str] = []

term_patterns: Dict[str, re.Pattern] = {}

llm_dictionary: Dict[str, Dict[str, List[str]]] = {}
base_materials: Set[str] = set()
base_techniques: Set[str] = set()
stop_pattern = None

def ensure_resources():
    """Ensure NLTK stopwords and dictionaries are loaded."""
    global stop_pattern, term_dictionary, sorted_keys, term_patterns

    #add stopwords
    if stop_pattern is None:
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords', quiet=True)
            
        stop_idx = set(nltk_stopwords.words('english'))
        stop_idx.update(EXTRA_STOPWORDS)
        stop_idx.update(COLORS)
        
        sorted_stops = sorted(list(stop_idx), key=len, reverse=True)
        pattern_str = r'\b(' + '|'.join(map(re.escape, sorted_stops)) + r')\b'
        stop_pattern = re.compile(pattern_str, re.IGNORECASE)

    if not term_dictionary:
        dict_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "medium_parsing", "term_classification.json")
        if not os.path.exists(dict_path):
            dict_path = os.path.join(os.path.dirname(__file__), "term_classification.json")
            
        if os.path.exists(dict_path):
            try:
                with open(dict_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                term_dictionary = {}
                for term, val in data.items():
                    norm_term = term.lower().replace('-', ' ')
                    if isinstance(val, dict):
                        term_dictionary[norm_term] = val
                    else:
                        term_dictionary[norm_term] = {"type": val}
                
                sorted_keys = sorted(term_dictionary.keys(), key=len, reverse=True)
                
                for key in sorted_keys:
                    pattern = r'\b' + re.escape(key) + r'\b'
                    term_patterns[key] = re.compile(pattern, re.IGNORECASE)
                    
            except Exception as e:
                print(f"Error loading {dict_path}: {e}")


def parse_medium(text: Union[str, List[str]]) -> Optional[Dict[str, List[str]]]:
    ensure_resources()
    
    if not text:
        return None

    if isinstance(text, list):
        full_text = " ".join(text)
    else:
        full_text = str(text)

    working_text = full_text.lower()
    
    noise_phrases = ["laid down", "placed down", "pasted down", "put down", "glued down", "mounted down", "setting down", "set down", "faced down", "face down"]
    for phrase in noise_phrases:
        working_text = working_text.replace(phrase, " ")
        
    parenthetical_texts = re.findall(r'\((.*?)\)', working_text)
    
    working_text = re.sub(r'\(.*?\)', '', working_text)
    working_text = working_text.replace('-', ' ')
    
    extracted_materials = set()
    extracted_techniques = set()

    def extract_from_text(text_to_scan, allow_materials=True):
        mats = set()
        techs = set()
        current_text = text_to_scan
        for key in sorted_keys:
            compiled_pattern = term_patterns.get(key)
            if compiled_pattern is None: continue
            if compiled_pattern.search(current_text):
                info = term_dictionary[key]
                term_type = info.get("type", "unknown")
                resolved_term = info.get("alias_of", key)
                
                if term_type == "material" and allow_materials:
                    mats.add(resolved_term)
                elif term_type == "technique":
                    techs.add(resolved_term)
                elif term_type == "ambiguous":
                    if allow_materials: mats.add(resolved_term)
                    techs.add(resolved_term)
                
                current_text = compiled_pattern.sub(' ', current_text)
        return mats, techs

    for p_text in parenthetical_texts:
        p_text = p_text.replace('-', ' ')
        _, p_techs = extract_from_text(p_text, allow_materials=False)
        extracted_techniques.update(p_techs)
        
    main_mats, main_techs = extract_from_text(working_text, allow_materials=True)
    extracted_materials.update(main_mats)
    extracted_techniques.update(main_techs)

    result = {
        "materials": sorted(list(extracted_materials)),
        "techniques": sorted(list(extracted_techniques))
    }

    if not result["materials"] and not result["techniques"]:
        return None
        
    return result