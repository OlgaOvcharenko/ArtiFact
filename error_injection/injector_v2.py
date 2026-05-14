import json
import os
import random
import re
import ast
import pandas as pd
import numpy as np

class ErrorInjectorV2:
    def __init__(self, knowledge_dir):
        self.knowledge_dir = knowledge_dir
        
        print(f"Loading knowledge base from {knowledge_dir}:")
        self.timelines = self._load_json('timelines.json')
        self.regions = self._load_json('regions.json')
        self.artist_clusters = self._load_json('artist_clusters.json')
        self.material_cooccurrence = self._load_json('material_cooccurrence.json')
        self.technique_cooccurrence = self._load_json('technique_cooccurrence.json')
        self.cross_correlations = self._load_json('cross_correlations.json')
        self.culture_adjacency = self._load_json('culture_adjacency.json')
        self.culture_to_continent = self._load_json('culture_to_continent.json')
        self.place_knowledge = self._load_json('place_knowledge.json')
        
        self.artist_to_cluster = {}
        self._build_artist_index()
        
        self.embeddings = None
        self.emb_ids = None
        self.id_to_emb_idx = {}
        self._load_embeddings()
        
        self.ref_data = {}
        
        print("Knowledge base loaded.")

    def _load_json(self, filename):
        path = os.path.join(self.knowledge_dir, filename)
        if os.path.exists(path):
            with open(path, 'r') as f:
                return json.load(f)
        return {}

    def _build_artist_index(self):
        for key, artists in self.artist_clusters.items():
            for art in artists:
                self.artist_to_cluster[art] = key

    def _load_embeddings(self):
        emb_dir = os.path.join(os.path.dirname(self.knowledge_dir), 'embeddings')
        paths = [
            (os.path.join(emb_dir, 'clip_embeddings.npy'), os.path.join(emb_dir, 'object_ids.npy')),
            (os.path.join(emb_dir, 'clip_embeddings_partial.npy'), os.path.join(emb_dir, 'object_ids_partial.npy'))
        ]
        for emb_path, id_path in paths:
            if os.path.exists(emb_path) and os.path.exists(id_path):
                try:
                    self.embeddings = np.load(emb_path, mmap_mode='r')
                    self.emb_ids = np.load(id_path, allow_pickle=True)
                    for idx, oid in enumerate(self.emb_ids):
                        self.id_to_emb_idx[str(oid)] = idx
                    print(f"Loaded {len(self.embeddings)} CLIP embeddings (Memory Mapped).")
                    break
                except Exception as e:
                    print(f"Embedding load failed: {e}")


    def precompute_lookups(self, df):
        """Builds fast lookup indices for the reference dataframe."""
        print(f"Pre-indexing {len(df):,} rows:")
        
        self.ref_data['obj'] = df['object_name'].fillna('').astype(str).str.strip().str.lower()
        self.ref_data['mat'] = df['materials'].fillna('').astype(str).str.strip().str.lower()
        self.ref_data['tec'] = df['techniques'].fillna('').astype(str).str.strip().str.lower()
        self.ref_data['nat'] = df['artist_nationality'].fillna('').astype(str).str.strip().str.lower()
        self.ref_data['art'] = df['artist_name'].fillna('').astype(str).str.strip().str.lower()
        self.ref_data['date'] = pd.to_numeric(df['date_begin'], errors='coerce')
        self.ref_data['class'] = df['classification'].fillna('').astype(str).str.strip().str.lower() if 'classification' in df.columns else pd.Series(['']*len(df))
        self.ref_data['img'] = df['image_url']
        self.ref_data['ids'] = df['object_ID'].astype(str)
        self.ref_data['dims'] = df['dimensions_json'] if 'dimensions_json' in df.columns else pd.Series(['']*len(df))
        
        self.ref_data['idx_by_art'] = df.groupby(self.ref_data['art']).indices
        self.ref_data['idx_by_class'] = df.groupby(self.ref_data['class']).indices
        self.ref_data['idx_by_obj'] = df.groupby(self.ref_data['obj']).indices

        heist_cols = ['artist_name', 'artist_role', 'artist_date_begin', 'artist_date_end', 'artist_nationality', 'culture', 'location']
        if 'country' in df.columns:
            heist_cols.append('country')
        existing_cols = [c for c in heist_cols if c in df.columns]
        self.ref_data['heist'] = df[existing_cols].copy()
        
        print("Indexing complete.")

    def parse_list(self, val):
        if pd.isna(val) or val == '[]' or val == '' or val == 'nan' or val is None:
            return []
        if isinstance(val, list): return val
        val_str = str(val).strip()
        try:
            parsed = ast.literal_eval(val_str)
            if isinstance(parsed, list):
                return [str(x).strip() for x in parsed if x]
            return [str(parsed).strip()]
        except:
            if ',' in val_str:
                return [x.strip() for x in val_str.split(',') if x.strip()]
            return [val_str]

    def _get_year(self, row, col):
        try:
            val = row.get(col)
            if pd.isna(val) or val == '' or val is None: return None
            return int(float(val))
        except:
            return None

    def _propagate_text(self, row, replacements):
        for col in ['title', 'description']:
            val = str(row.get(col) or '')
            if not val or val == 'nan': continue
            for old, new in replacements:
                if pd.isna(old) or not old: continue
                s_old, s_new = str(old).strip(), str(new).strip()
                if s_old.lower() in ['nan', 'none', '', 'null']: continue
                
                pattern = r'\b' + re.escape(s_old) + r'\b'
                val = re.sub(pattern, s_new, val, flags=re.IGNORECASE)
            row[col] = val
        return row

    def _is_too_similar(self, a, b):
        a, b = str(a).lower().strip(), str(b).lower().strip()
        if a == b: return True
        if a in b or b in a: return True
        a_words = set(a.split())
        b_words = set(b.split())
        if not a_words or not b_words: return False
        overlap = a_words.intersection(b_words)
        if len(overlap) / max(len(a_words), len(b_words)) > 0.7:
            return True
        return False

    def _is_ambiguous(self, text):
        if not text or pd.isna(text): return False
        t = str(text).lower()
        connectors = [' or ', ' and ', '?', 'possibly', 'probably', '/', ' maybe']
        return any(c in t for c in connectors)

    def _is_anonymous(self, name):
        if not name or pd.isna(name): return True
        n = str(name).lower().strip()
        unknown_tokens = ['unknown', 'anonymous', 'unidentified', 'nan', 'attributed to', 'style of', 'manner of']
        return any(x in n for x in unknown_tokens)

    def inject_date_error(self, row):
        if not self._is_anonymous(row.get('artist_name')):
            return None
        start = self._get_year(row, 'date_begin')
        end = self._get_year(row, 'date_end')
        start_bce = bool(row.get('date_begin_bce', False))
        end_bce = bool(row.get('date_end_bce', False))
        if start is None: return None
        shift = random.choice([-300, -200, -100, 100, 200, 300])
        def to_astro(year, is_bce):
            if year is None: return None
            return year if not is_bce else -(year - 1)
        def from_astro(astro):
            if astro is None: return None, None
            if astro > 0: return astro, False
            return -astro + 1, True
        astro_start = to_astro(start, start_bce)
        astro_end = to_astro(end, end_bce)
        new_astro_start = astro_start + shift
        new_astro_end = (astro_end + shift) if astro_end is not None else None
        new_start_abs, new_start_bce = from_astro(new_astro_start)
        new_end_abs, new_end_bce = from_astro(new_astro_end)
        if new_start_abs > 2025 and not new_start_bce: return None
        if new_end_abs and new_end_abs > 2025 and not new_end_bce: return None
        corrupted = row.copy()
        corrupted['date_begin'] = new_start_abs
        corrupted['date_begin_bce'] = new_start_bce
        meta_new = {'start': new_start_abs, 'start_bce': new_start_bce}
        if new_end_abs is not None:
            corrupted['date_end'] = new_end_abs
            corrupted['date_end_bce'] = new_end_bce
            meta_new['end'] = new_end_abs
            meta_new['end_bce'] = new_end_bce
        else:
            meta_new['end'] = None
            meta_new['end_bce'] = None
        return corrupted, {
            'error_type': 'date_error', 
            'subtype': 'century_shift', 
            'original': {'start': start, 'start_bce': start_bce, 'end': end, 'end_bce': end_bce}, 
            'new': meta_new
        }

    # dimensions
    def inject_dimension_error(self, row):
        dims_json = row.get('dimensions_json')
        if pd.isna(dims_json) or dims_json is None: return None
        try:
            dims = json.loads(dims_json) if isinstance(dims_json, str) else dims_json
            if 'parts' not in dims or not dims['parts']: return None
            error_subtype = random.choice(['scale_error', 'aspect_swap'])
            if error_subtype == 'scale_error':
                factor = random.choice([0.1, 10])
                for part in dims['parts']:
                    for m in part.get('measurements', []):
                        if 'value' in m and isinstance(m['value'], (int, float)):
                            m['value'] = round(m['value'] * factor, 2)
            elif error_subtype == 'aspect_swap':
                if len(dims['parts']) != 1: return None
                part = dims['parts'][0]
                ms = part.get('measurements', [])
                if not (2 <= len(ms) <= 3): return None
                h_idx, w_idx = -1, -1
                for i, m in enumerate(ms):
                    if m.get('attribute') == 'height': h_idx = i
                    elif m.get('attribute') == 'width': w_idx = i
                if h_idx != -1 and w_idx != -1:
                    ms[h_idx]['value'], ms[w_idx]['value'] = ms[w_idx]['value'], ms[h_idx]['value']
                else: return None
            corrupted = row.copy()
            corrupted['dimensions_json'] = json.dumps(dims) if isinstance(dims_json, str) else dims
            return corrupted, {'error_type': 'dimension_error', 'subtype': error_subtype, 'original': dims_json, 'new': corrupted['dimensions_json']}
        except: return None

    # materials
    def inject_material_error(self, row, subtype='material_interchange'):
        mats = self.parse_list(row.get('materials'))
        if not mats: return None
        obj = str(row.get('object_name') or '').lower().strip()
        year = self._get_year(row, 'date_end') or 2025
        error_subtype = subtype.split('_')[-1]
        chosen = None
        if error_subtype == 'anachronism':
            pool = set()
            for m in mats: 
                pool.update(self.material_cooccurrence.get(m, {}).keys())
            candidates = list(pool)
            random.shuffle(candidates)
            for cand in candidates:
                if cand in mats or any(self._is_too_similar(cand, m) for m in mats): continue
                if self._is_ambiguous(cand): continue
                time_data = self.timelines['materials'].get(cand)
                if not time_data or time_data['start'] <= (year + 50): continue
                if obj and obj in self.cross_correlations.get('material_object', {}).get(cand, {}):
                    chosen = cand; break
        if not chosen or error_subtype == 'interchange':
            error_subtype = 'interchange'
            techs = self.parse_list(row.get('techniques'))
            candidates = []
            mat_obj = self.cross_correlations.get('material_object', {})
            for cand, objs in mat_obj.items():
                if cand in mats or any(self._is_too_similar(cand, m) for m in mats): continue
                if self._is_ambiguous(cand): continue
                obj_score = objs.get(obj, 0)
                tech_score = 0
                if techs:
                    cand_techs = self.cross_correlations.get('material_technique', {}).get(cand, {})
                    tech_score = sum(cand_techs.get(t, 0) for t in techs)
                if obj_score > 5 or tech_score > 5:
                    candidates.append((cand, obj_score + tech_score))
            if candidates:
                candidates.sort(key=lambda x: x[1], reverse=True)
                chosen = random.choice([x[0] for x in candidates[:3]])
        if not chosen: return None
        new_mats = mats.copy()
        if len(new_mats) > 0:
            new_mats[random.randrange(len(new_mats))] = chosen
        else:
            new_mats = [chosen]
            
        corrupted = row.copy()
        corrupted['materials'] = json.dumps(new_mats) if isinstance(row.get('materials'), str) else new_mats
        return corrupted, {'error_type': 'material_error', 'subtype': subtype, 'original': mats, 'new': new_mats, 'added': chosen}

    # technique
    def inject_technique_error(self, row, subtype='technique_interchange'):
        techs = self.parse_list(row.get('techniques'))
        if not techs: return None
        obj = str(row.get('object_name') or '').lower().strip()
        year = self._get_year(row, 'date_end') or 2025
        mats = self.parse_list(row.get('materials'))
        error_subtype = subtype.split('_')[-1]
        chosen = None
        if error_subtype == 'anachronism':
            pool = set()
            for t in techs: pool.update(self.technique_cooccurrence.get(t, {}).keys())
            candidates = list(pool)
            random.shuffle(candidates)
            for cand in candidates:
                if cand in techs or any(self._is_too_similar(cand, t) for t in techs): continue
                if self._is_ambiguous(cand): continue
                time_data = self.timelines.get('techniques', {}).get(cand)
                if not time_data or time_data['start'] <= (year + 50): continue
                if obj and obj in self.cross_correlations.get('technique_object', {}).get(cand, {}):
                    chosen = cand; break
        if not chosen or error_subtype == 'interchange':
            error_subtype = 'interchange'
            candidates = []
            tec_obj = self.cross_correlations.get('technique_object', {})
            forbidden_pairs = [{'photography', 'albumen silver print'}, {'pen', 'writing'}]
            
            for cand, objs in tec_obj.items():
                if cand in techs or any(self._is_too_similar(cand, t) for t in techs): continue
                if self._is_ambiguous(cand): continue
                
                is_forbidden = False
                for t in techs:
                    if set([cand, t]) in forbidden_pairs:
                        is_forbidden = True; break
                if is_forbidden: continue
                
                obj_score = objs.get(obj, 0)
                mat_score = 0
                if mats:
                    cand_mats = self.cross_correlations.get('material_technique', {})
                    for m in mats:
                        if cand in cand_mats.get(m, {}): mat_score += cand_mats[m][cand]
                if obj_score > 5 or mat_score > 5:
                    candidates.append((cand, obj_score + mat_score))
            if candidates:
                candidates.sort(key=lambda x: x[1], reverse=True)
                chosen = random.choice([x[0] for x in candidates[:3]])
        if not chosen: return None
        new_techs = techs.copy()
        if len(new_techs) > 0:
            new_techs[random.randrange(len(new_techs))] = chosen
        else:
            new_techs = [chosen]
            
        corrupted = row.copy()
        corrupted['techniques'] = json.dumps(new_techs) if isinstance(row.get('techniques'), str) else new_techs
        return corrupted, {'error_type': 'technique_error', 'subtype': subtype, 'original': techs, 'new': new_techs, 'added_or_swapped': chosen}

    # artist
    def inject_artist_error(self, row, reference_df=None, subtype='artist_tier_1_easy'):
        orig_artist = str(row.get('artist_name') or '').strip().lower()
        if self._is_anonymous(orig_artist): return None
        r_obj = str(row.get('object_name') or '').strip().lower()
        r_tec = str(row.get('techniques') or '').strip().lower()
        r_nat = str(row.get('artist_nationality') or '').strip().lower()
        try: r_date = float(row.get('date_begin'))
        except: r_date = None
        rd = self.ref_data
        valid_idx = np.where((rd['art'] != orig_artist) & (~rd['art'].apply(self._is_anonymous)))[0]
        candidates_idx = []
        if subtype == 'artist_tier_4_hardest' and r_obj and r_tec and r_tec != 'nan' and r_nat and r_date is not None:
            mask = (rd['obj'] == r_obj) & (rd['tec'] == r_tec) & (rd['nat'] == r_nat) & (np.abs(rd['date'] - r_date) <= 50)
            candidates_idx = np.intersect1d(valid_idx, np.where(mask)[0])
        elif subtype == 'artist_tier_3_hard' and r_obj and r_tec and r_tec != 'nan' and r_nat and r_date is not None:
            mask = (rd['obj'] == r_obj) & (rd['tec'] == r_tec) & (rd['nat'] != r_nat) & (rd['nat'] != '') & (np.abs(rd['date'] - r_date) <= 50)
            candidates_idx = np.intersect1d(valid_idx, np.where(mask)[0])
        elif subtype == 'artist_tier_2_medium' and r_obj and r_nat and r_date is not None:
            mask = (rd['obj'] == r_obj) & (np.abs(rd['date'] - r_date) <= 50)
            candidates_idx = np.intersect1d(valid_idx, np.where(mask)[0])
        elif subtype == 'artist_tier_1_easy':
            candidates_idx = valid_idx
        if len(candidates_idx) == 0: return None
        chosen_idx = random.choice(candidates_idx)
        heist_target = self.ref_data['heist'].iloc[chosen_idx]
        corrupted = row.copy()
        for f in ['artist_name', 'artist_role', 'artist_date_begin', 'artist_date_end', 'artist_nationality', 'culture', 'location']:
            corrupted[f] = heist_target.get(f)
        
        
        corrupted = self._propagate_text(corrupted, [(str(row.get('artist_name')), str(corrupted['artist_name'])), (str(row.get('artist_nationality')), str(corrupted['artist_nationality']))])
        
        return row.copy(), {'error_type': 'artist_error', 'subtype': subtype, 'original': row.get('artist_name'), 'new': corrupted}

    def inject_culture_error(self, row, subtype='culture_tight_swap'):
        if not self._is_anonymous(row.get('artist_name')): return None
        cult = row.get('culture')
        if not cult or pd.isna(cult): return None
        if self._is_ambiguous(cult): return None
        cult_clean = str(cult).lower().strip()
        orig_words = set(re.findall(r'\w+', cult_clean))
        chosen_fake = None
        if subtype == 'culture_tight_swap':
            candidates = self.culture_adjacency.get(cult_clean, [])
            safe = [c for c in candidates if not orig_words.intersection(set(re.findall(r'\w+', c.lower()))) and not self._is_ambiguous(c)]
            if safe: chosen_fake = random.choice(safe)
        elif subtype == 'culture_continent_swap':
            cont = self.culture_to_continent.get(cult_clean)
            if cont:
                candidates = [c for c, ct in self.culture_to_continent.items() if ct == cont and c != cult_clean]
                safe = [c for c in candidates if not orig_words.intersection(set(re.findall(r'\w+', c.lower()))) and not self._is_ambiguous(c)]
                if safe: chosen_fake = random.choice(safe)
        if not chosen_fake: return None
        corrupted = row.copy()
        corrupted['culture'] = chosen_fake
        corrupted = self._propagate_text(corrupted, [(cult, chosen_fake)])
        return corrupted, {'error_type': 'culture_error', 'subtype': subtype, 'original': cult, 'new': chosen_fake}

    def inject_place_error(self, row, subtype='country_level_swap'):
        location = row.get('location')
        if not location or pd.isna(location) or str(location).lower() == 'unknown': return None
        if self._is_ambiguous(location): return None
        
        location_clean = str(location).lower().strip()
        
        city_to_country = self.place_knowledge.get('city_to_country', {})
        country_to_cities = self.place_knowledge.get('country_to_cities', {})
        country_adj = self.place_knowledge.get('country_neighbors', {})
        city_adj = self.place_knowledge.get('city_neighbors', {})
        
        is_country = location_clean in country_to_cities or location_clean in country_adj
        is_city = location_clean in city_to_country or location_clean in city_adj
        
        chosen_fake = None
        corrupted = row.copy()
        meta = {'error_type': 'place_error', 'subtype': subtype}
        
        if subtype == 'country_level_swap' and is_country:
            adjacents = country_adj.get(location_clean, [])
            safe = [c for c in adjacents if not self._is_too_similar(c, location_clean) and not self._is_ambiguous(c)]
            if safe: chosen_fake = random.choice(safe)
            
        elif subtype == 'city_level_swap' and is_city:
            adjacents = city_adj.get(location_clean, [])
            safe = [c for c in adjacents if not self._is_too_similar(c, location_clean) and not self._is_ambiguous(c)]
            if safe: chosen_fake = random.choice(safe)
            
        if not chosen_fake: return None
        
        corrupted['location'] = chosen_fake
        corrupted = self._propagate_text(corrupted, [(str(location), chosen_fake)])
        
        meta['original'] = location
        meta['new'] = chosen_fake
        return corrupted, meta

    def execute_swap(self, row_a, row_b, subtype):
        if subtype.startswith('artist'): error_type = 'artist'
        elif subtype.startswith('image') or subtype == 'embedding_swap': error_type = 'image'
        elif subtype.startswith('culture'): error_type = 'culture'
        elif subtype.startswith('country') or subtype.startswith('city'): error_type = 'place'
        else: return None, None
        corrupted_a = row_a.copy()
        corrupted_b = row_b.copy()
        meta_a = {'error_type': f'{error_type}_error', 'subtype': subtype}
        meta_b = {'error_type': f'{error_type}_error', 'subtype': subtype}
        if error_type == 'artist':
            cols_to_swap = ['artist_name', 'artist_role', 'artist_date_begin', 'artist_date_end', 'artist_nationality', 'culture', 'location']
            for c in cols_to_swap:
                corrupted_a[c] = row_b.get(c)
                corrupted_b[c] = row_a.get(c)
            corrupted_a = self._propagate_text(corrupted_a, [(str(row_a.get('artist_name')), str(row_b.get('artist_name'))), (str(row_a.get('artist_nationality')), str(row_b.get('artist_nationality')))])
            corrupted_b = self._propagate_text(corrupted_b, [(str(row_b.get('artist_name')), str(row_a.get('artist_name'))), (str(row_b.get('artist_nationality')), str(row_a.get('artist_nationality')))])
            meta_a['original'], meta_a['new'] = row_a.get('artist_name'), row_b.get('artist_name')
            meta_b['original'], meta_b['new'] = row_b.get('artist_name'), row_a.get('artist_name')
        elif error_type == 'image':
            corrupted_a['image_url'], corrupted_b['image_url'] = row_b.get('image_url'), row_a.get('image_url')
            meta_a['original'], meta_a['new'] = row_a.get('image_url'), row_b.get('image_url')
            meta_a['new_id'] = row_b.get('object_ID')
            meta_b['original'], meta_b['new'] = row_b.get('image_url'), row_a.get('image_url')
            meta_b['new_id'] = row_a.get('object_ID')
        elif error_type == 'culture' or error_type == 'place':
            c_a, c_b = row_a.get('culture'), row_b.get('culture')
            l_a, l_b = row_a.get('location'), row_b.get('location')
            
            corrupted_a['culture'], corrupted_a['location'] = c_b, l_b
            corrupted_b['culture'], corrupted_b['location'] = c_a, l_a
            
            if error_type == 'culture':
                meta_a['original'], meta_a['new'] = c_a, c_b
                meta_b['original'], meta_b['new'] = c_b, c_a
            else:
                meta_a['original'], meta_a['new'] = l_a, l_b
                meta_b['original'], meta_b['new'] = l_b, l_a
        return (corrupted_a, meta_a), (corrupted_b, meta_b)

    def inject_image_error(self, row, reference_df=None, subtype='image_tier_1_easy'):
        orig_img = row.get('image_url')
        if not orig_img or pd.isna(orig_img): return None
        row_obj = str(row.get('object_name') or '').strip().lower()
        row_artist = str(row.get('artist_name') or '').strip().lower()
        row_class = str(row.get('classification') or '').strip().lower()
        rd = self.ref_data
        chosen_img = None
        chosen_idx = None
        if subtype == 'embedding_swap' and self.embeddings is not None:
            oid = str(row.get('object_ID'))
            if oid in self.id_to_emb_idx:
                idx = self.id_to_emb_idx[oid]
                scores = np.dot(self.embeddings, self.embeddings[idx])
                
                k = min(201, len(scores))
                top_k_unsorted = np.argpartition(scores, -k)[-k:]
                top_k = top_k_unsorted[np.argsort(scores[top_k_unsorted])]

                top_k = top_k[:-1]
                
                if not hasattr(self, '_id_to_pool_idx'):
                    self._id_to_pool_idx = {str(val): i for i, val in enumerate(rd['ids'])}
                for n_idx in reversed(top_k):
                    if scores[n_idx] < 0.75: break
                    neighbor_oid = str(self.emb_ids[n_idx])
                    pool_idx = self._id_to_pool_idx.get(neighbor_oid)
                    if pool_idx is None: continue
                    if rd['art'].iloc[pool_idx] != row_artist:
                        chosen_img = rd['img'].iloc[pool_idx]
                        chosen_idx = pool_idx
                        break
        elif subtype == 'image_tier_1_easy':
            mask = (rd['class'] != row_class) & (rd['class'] != '') if row_class else np.array([False]*len(rd['img']))
            diff_indices = np.where(mask)[0]
            if len(diff_indices) == 0:
                mask = (rd['obj'] != row_obj) & (rd['obj'] != '')
                diff_indices = np.where(mask)[0]
            if len(diff_indices) > 0:
                chosen_idx = random.choice(diff_indices)
                chosen_img = rd['img'].iloc[chosen_idx]
        elif subtype == 'image_tier_2_medium':
            idx_list = rd['idx_by_obj'].get(row_obj, [])
            if len(idx_list) > 1:
                chosen_idx = random.choice(idx_list)
                chosen_img = rd['img'].iloc[chosen_idx]
        elif subtype == 'image_tier_3_hard':
            idx_list = rd['idx_by_art'].get(row_artist, [])
            if len(idx_list) > 1 and not self._is_anonymous(row_artist):
                valid_peers = [i for i in idx_list if rd['img'].iloc[i] != orig_img]
                if valid_peers:
                    chosen_idx = random.choice(valid_peers)
                    chosen_img = rd['img'].iloc[chosen_idx]
        if not chosen_img: return None
        return row.copy(), {'error_type': 'image_error', 'subtype': subtype, 'original': orig_img, 'new': chosen_img, 'new_id': rd['ids'].iloc[chosen_idx] if chosen_idx is not None else None}
