import re
from typing import List, Dict, Optional, Tuple

class ArtistParser:
    def __init__(self):
        self.year_pattern = re.compile(r'\b(1[0-9]|20)\d{2}\b')
        self.century_pattern = re.compile(r'\b\d{1,2}(th|st|nd|rd)\s+century\b', re.IGNORECASE)
        self.decade_pattern = re.compile(r'\b(1[0-9]|20)\d{2}s\b')
        
        # Pattern A: Newline Format
        self.newline_pattern = re.compile(
            r'^(?P<name>.+?)\n(?P<date_str>.*)$',
            re.MULTILINE
        )

        # Pattern B: Inline Format
        self.inline_pattern = re.compile(
            r'^(?:(?P<role>attributed to|workshop of|after|follower of|circle of|manner of|published by|designed by|printed by|engraved by)\s+)?(?P<name>.+?)\s*\((?P<meta>(?P<nat>[^)\n]+?),\s*(?P<date_str>.*?))\)$',
            re.IGNORECASE
        )

        self.non_artist_keywords = {
            'china', 'japan', 'egypt', 'france', 'italy', 'belgium', 'peru', 'mexico',
            'netherlands', 'spain', 'england', 'united states', 'america', 'germany',
            'switzerland', 'greece', 'turkey', 'india', 'thailand', 'vietnam', 'korea',
            'indonesia', 'russia', 'austria', 'chicago', 'london', 'paris', 'rome',
            'chinese', 'japanese', 'egyptian', 'french', 'italian', 'belgian', 'dutch',
            'spanish', 'british', 'american', 'german', 'swiss', 'greek', 'mexican',
            'canadian', 'european', 'asian', 'african', 'pre-columbian', 'ancient',
            'moche', 'inca', 'aztec', 'maya', 'roman', 'greek', 'islamic', 'byzantine',
            'han-chinese', 'chimu', 'chimú', 'nasca', 'paracas', 'valdivia', 'vicus',
            'unknown maker', 'unknown artist', 'anonymous', 'artist unknown', 'unknown',
            'unknown culture', 'culture unknown', 'sculptor unknown', 'probably', 'possibly'
        }
        self.geographical_terms = {
            'north', 'south', 'east', 'west', 'coast', 'valley', 'river', 'lake', 
            'region', 'mountain', 'states', 'central', 'middle', 'upper', 'lower',
            'province', 'city', 'country', 'island', 'islands', 'peninsula'
        }
        
        self.complex_indicators = {
            'by', 'of', 'after', 'follower', 'circle', 'manner', 'workshop', 'studio', 'school'
        }
        self.multi_artist_indicators = {'and', 'with', ';'}

    def has_dates(self, text: str) -> bool:
        if not isinstance(text, str): return False
        lower = text.lower()
        return bool(self.year_pattern.search(lower) or 
                    self.century_pattern.search(lower) or 
                    self.decade_pattern.search(lower))

    def is_confirmed_non_artist(self, text: str) -> bool:
        if self.has_dates(text):
            return False
            
        parts = [p.strip().lower() for p in re.split(r'[\n,]', text) if p.strip()]
        if not parts:
            return True
            
        for p in parts:
            if p in self.non_artist_keywords:
                continue
            if p in self.geographical_terms:
                continue
            words = p.split()
            if all(w in self.non_artist_keywords or w in self.geographical_terms for w in words):
                continue
                
            return False
            
        return True

    def _parse_date_str(self, date_str: str) -> Tuple[Optional[int], Optional[int]]:
        s = date_str.lower().strip()
        
        s = re.sub(r'\(\?\)', '', s)
        s = re.sub(r'\?', '', s)
        s = re.sub(r'\b(c\.|ca\.|circa|about|approximately|fl\.)\.?\s*', '', s)
        s = s.strip()
        
        range_match = re.search(r'^(\d{4})\s*[-–]\s*(\d{4})$', s)
        if range_match:
            return int(range_match.group(1)), int(range_match.group(2))
            
        century_match = re.search(r'^(\d{1,2})(?:st|nd|rd|th)\s+century$', s)
        if century_match:
            cent = int(century_match.group(1))
            start_year = (cent - 1) * 100
            end_year = (cent * 100) - 1
            return start_year, end_year
            
        if 'founded' in s:
            year_match = re.search(r'(\d{4})', s)
            if year_match:
                return int(year_match.group(1)), None
                
        if 'active' in s:
            decade_match = re.search(r'(\d{4})s', s)
            if decade_match:
                start = int(decade_match.group(1))
                return start, start + 9
            range_match = re.search(r'(\d{4})\s*[-–]\s*(\d{4})', s)
            if range_match:
                return int(range_match.group(1)), int(range_match.group(2))
                
        return None, None

    def _expand_two_digit_year(self, date_str: str) -> str:
        pattern = r'(\d{4})\s*[-–]\s*(\d{2})(?!\d)'
        
        def replacer(m):
            start_year = m.group(1)
            end_suffix = m.group(2)
            # Take century from start year
            century = start_year[:2]
            expanded_end = century + end_suffix
            return f"{start_year}-{expanded_end}"
        
        return re.sub(pattern, replacer, date_str)

    def _handle_slash_dates(self, date_str: str) -> str:
        pattern = r'(\d{4})/\d+\s*[-–]\s*(\d{4})'
        match = re.search(pattern, date_str)
        if match:
            return f"{match.group(1)}-{match.group(2)}"
        return date_str

    def _parse_single_line(self, line: str) -> Optional[Dict]:
        line = line.strip()
        if not line:
            return None
            
        processed = self._expand_two_digit_year(line)
        processed = self._handle_slash_dates(processed)
        
        match_b = self.inline_pattern.match(processed)
        if match_b:
            date_str = match_b.group('date_str')
            date_str = self._expand_two_digit_year(date_str)
            date_str = self._handle_slash_dates(date_str)
            begin, end = self._parse_date_str(date_str)
            
            if begin is not None or end is not None:
                role = match_b.group('role')
                if role:
                    role = role.strip().lower()
                else:
                    role = 'primary'
                    
                return {
                    'artist_name': match_b.group('name').strip(),
                    'artist_role': role,
                    'artist_nationality': match_b.group('nat').strip(),
                    'artist_date_begin': begin,
                    'artist_date_end': end
                }
        
        return None

    def _is_valid_parse(self, parsed: Dict) -> bool:
        name = (parsed.get('artist_name') or '').lower()
        nat = (parsed.get('artist_nationality') or '').lower()
        role = (parsed.get('artist_role') or 'primary').lower()
        
        if role != 'primary':
            return False
            
        if any(f" {k} " in f" {nat} " for k in self.complex_indicators):
            return False
        if '(' in nat or ')' in nat:
            return False
            
        if any(f" {k} " in f" {name} " for k in self.complex_indicators):
            return False
        if '(' in name or ')' in name:
            return False
        if self.year_pattern.search(name):
            return False

        if len(name) > 100 or len(nat) > 50:
            return False
            
        return True

    def parse_all(self, text: str) -> list:
        if not isinstance(text, str):
            return []
        
        text = text.strip()
        if not text:
            return []
            
        if any(f" {m} " in f" {text.lower()} " for m in self.multi_artist_indicators):
            return []
        
        results = []
        lines = text.split('\n')
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            if not line:
                i += 1
                continue
            
            if i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                combined = f"{line}\n{next_line}"
                
                processed_next = self._expand_two_digit_year(next_line)
                processed_next = self._handle_slash_dates(processed_next)
                processed_combined = f"{line}\n{processed_next}"
                
                match_a = self.newline_pattern.match(processed_combined)
                if match_a:
                    meta = match_a.group('date_str').strip()
                    name = match_a.group('name').strip()
                    
                    if ',' in meta:
                        nat, date_part = meta.rsplit(',', 1)
                    else:
                        nat = None
                        date_part = meta
                    
                    date_part = self._expand_two_digit_year(date_part)
                    date_part = self._handle_slash_dates(date_part)
                    
                    begin, end = self._parse_date_str(date_part)
                    
                    if begin is not None or end is not None:
                        candidate = {
                            'artist_name': name,
                            'artist_role': 'primary',
                            'artist_nationality': nat.strip() if nat else None,
                            'artist_date_begin': begin,
                            'artist_date_end': end
                        }
                        if self._is_valid_parse(candidate):
                            results.append(candidate)
                            i += 2
                            continue
            
            parsed = self._parse_single_line(line)
            if parsed and self._is_valid_parse(parsed):
                results.append(parsed)
            else:
                return []
            
            i += 1
            
        if results:
            return results
            
        if self.is_confirmed_non_artist(text):
            return None
            
        return []

def parse_artist_info_string(text: str) -> List[Dict]:
    parser = ArtistParser()
    return parser.parse_all(text)
