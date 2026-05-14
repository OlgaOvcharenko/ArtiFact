import pandas as pd
import json
import os
import glob
import ast
import numpy as np
from collections import defaultdict
from itertools import combinations

BASE_DIR = '/Users/lulo/Documents/DEEM/Thesis/artwork-dataset'
INPUT_DIR = os.path.join(BASE_DIR, '6_unified')
OUT_DIR = os.path.join(BASE_DIR, 'error_injection/knowledge_v4')
MIN_FREQ = 10

os.makedirs(OUT_DIR, exist_ok=True)


def parse_json_list(val):
    """Parse a JSON list string like '["oil", "canvas"]' into a Python list. For materials and techqniques"""
    if pd.isna(val) or not str(val).strip():
        return []
    try:
        parsed = json.loads(val)
        if isinstance(parsed, list):
            return [str(x).strip().lower() for x in parsed if str(x).strip()]
    except Exception:
        try:
            parsed = ast.literal_eval(str(val))
            if isinstance(parsed, list):
                return [str(x).strip().lower() for x in parsed if str(x).strip()]
        except Exception:
            pass
    s = str(val).strip().lower()
    return [s] if s else []


def load_all_data():
    files = sorted(glob.glob(os.path.join(INPUT_DIR, '*.csv')))
    files = [f for f in files if not os.path.basename(f).startswith(('_', 'overview'))]
    
    basename_map = {}
    for f in files:
        base = os.path.basename(f)
        parts = base.rsplit('_sample_', 1)
        prefix = parts[0] if len(parts) > 1 else base
        if prefix not in basename_map:
            basename_map[prefix] = f
        else:
            if os.path.getsize(f) > os.path.getsize(basename_map[prefix]):
                basename_map[prefix] = f
    
    deduped_files = sorted(basename_map.values())
    print(f"Loading {len(deduped_files)} files from {INPUT_DIR}/")
    
    dfs = []
    for f in deduped_files:
        df = pd.read_csv(f, low_memory=False)
        if 'country' not in df.columns and 'location' in df.columns:
            df['country'] = df['location']
        print(f"  {os.path.basename(f)}: {len(df):,} rows")
        dfs.append(df)
    
    combined = pd.concat(dfs, ignore_index=True)
    print(f"Total: {len(combined):,} rows\n")
    return combined


def build_timelines(df):
    """Date ranges for all materials, techniques, and object_names."""
    print("Building timelines:")
    
    valid = df.dropna(subset=['date_begin', 'date_end']).copy()
    valid['date_begin'] = pd.to_numeric(valid['date_begin'], errors='coerce')
    valid['date_end'] = pd.to_numeric(valid['date_end'], errors='coerce')
    
    if 'date_begin_bce' in valid.columns:
        valid['date_begin'] = valid.apply(lambda r: r['date_begin'] * -1 if str(r.get('date_begin_bce')).lower() == 'true' else r['date_begin'], axis=1)
    if 'date_end_bce' in valid.columns:
        valid['date_end'] = valid.apply(lambda r: r['date_end'] * -1 if str(r.get('date_end_bce')).lower() == 'true' else r['date_end'], axis=1)

    valid = valid.dropna(subset=['date_begin', 'date_end'])
    valid['avg_year'] = (valid['date_begin'] + valid['date_end']) / 2
    
    timelines = {'materials': {}, 'techniques': {}, 'object_name': {}}
    
    # Materials
    mat_years = defaultdict(list)
    for _, row in valid.iterrows():
        for m in parse_json_list(row.get('materials')):
            mat_years[m].append(row['avg_year'])
    
    for m, years in mat_years.items():
        if len(years) < MIN_FREQ: continue
        timelines['materials'][m] = {
            'start': int(np.percentile(years, 1)),
            'end': int(np.percentile(years, 99)),
            'count': len(years)
        }
    
    # Techniques
    tech_years = defaultdict(list)
    for _, row in valid.iterrows():
        for t in parse_json_list(row.get('techniques')):
            tech_years[t].append(row['avg_year'])
    
    for t, years in tech_years.items():
        if len(years) < MIN_FREQ: continue
        timelines['techniques'][t] = {
            'start': int(np.percentile(years, 1)),
            'end': int(np.percentile(years, 99)),
            'count': len(years)
        }
    
    # Object Names
    obj_years = defaultdict(list)
    for _, row in valid.iterrows():
        obj = row.get('object_name')
        if pd.notna(obj):
            obj_years[str(obj).strip().lower()].append(row['avg_year'])
    
    for o, years in obj_years.items():
        if len(years) < MIN_FREQ: continue
        timelines['object_name'][o] = {
            'start': int(np.percentile(years, 1)),
            'end': int(np.percentile(years, 99)),
            'count': len(years)
        }
    
    save_json(timelines, 'timelines.json')
    print(f"  Materials: {len(timelines['materials'])} | "
          f"Techniques: {len(timelines['techniques'])} | "
          f"Objects: {len(timelines['object_name'])}")


def build_regions(df):
    """Geographic distribution of materials/techniques/objects by country."""
    print("Building regions:")
    
    valid = df.dropna(subset=['country'])
    regions = {'materials': {}, 'techniques': {}, 'object_name': {}}
    
    # Materials
    mat_geo = defaultdict(lambda: defaultdict(int))
    for _, row in valid.iterrows():
        country = str(row.get('country') or '').strip().lower()
        if not country or country == 'nan': continue
        for m in parse_json_list(row.get('materials')):
            mat_geo[m][country] += 1
    
    for m, countries in mat_geo.items():
        total = sum(countries.values())
        if total < MIN_FREQ: continue
        sorted_c = sorted(countries.items(), key=lambda x: -x[1])
        kept = []
        cum = 0
        for c, cnt in sorted_c:
            pct = round(cnt / total * 100, 1)
            kept.append({'country': c, 'percent': pct})
            cum += pct
            if cum > 90: break
        regions['materials'][m] = kept
    
    # Techniques
    tech_geo = defaultdict(lambda: defaultdict(int))
    for _, row in valid.iterrows():
        country = str(row.get('country') or '').strip().lower()
        for t in parse_json_list(row.get('techniques')):
            tech_geo[t][country] += 1
    
    for t, countries in tech_geo.items():
        total = sum(countries.values())
        if total < MIN_FREQ: continue
        sorted_c = sorted(countries.items(), key=lambda x: -x[1])
        kept = []
        cum = 0
        for c, cnt in sorted_c:
            pct = round(cnt / total * 100, 1)
            kept.append({'country': c, 'percent': pct})
            cum += pct
            if cum > 90: break
        regions['techniques'][t] = kept
    
    # Object names
    obj_geo = defaultdict(lambda: defaultdict(int))
    for _, row in valid.iterrows():
        country = str(row.get('country') or '').strip().lower()
        obj = row.get('object_name')
        if pd.notna(obj):
            obj_geo[str(obj).strip().lower()][country] += 1
    
    for o, countries in obj_geo.items():
        total = sum(countries.values())
        if total < MIN_FREQ: continue
        sorted_c = sorted(countries.items(), key=lambda x: -x[1])
        kept = []
        cum = 0
        for c, cnt in sorted_c:
            pct = round(cnt / total * 100, 1)
            kept.append({'country': c, 'percent': pct})
            cum += pct
            if cum > 90: break
        regions['object_name'][o] = kept
    
    save_json(regions, 'regions.json')
    print(f"  Materials: {len(regions['materials'])} | "
          f"Techniques: {len(regions['techniques'])} | "
          f"Objects: {len(regions['object_name'])}")


def build_cooccurrence(df):
    """Material and technique co-occurrence matrices. Like self co-occurent"""
    print("Building co-occurrence:")
    
    # Material
    mat_cooc = defaultdict(lambda: defaultdict(int))
    mat_total = defaultdict(int)
    
    for _, row in df.iterrows():
        mats = parse_json_list(row.get('materials'))
        for m in mats:
            mat_total[m] += 1
        for a, b in combinations(sorted(set(mats)), 2):
            mat_cooc[a][b] += 1
            mat_cooc[b][a] += 1
    
    mat_knowledge = {}
    for m, neighbors in mat_cooc.items():
        if mat_total[m] < MIN_FREQ: continue
        mat_knowledge[m] = {}
        for n, cnt in neighbors.items():
            if cnt < 2: continue  # skip very rare pairs
            mat_knowledge[m][n] = {
                'count': cnt,
                'probability': round(cnt / mat_total[m], 4)
            }
    
    save_json(mat_knowledge, 'material_cooccurrence.json')
    print(f"  Material pairs: {sum(len(v) for v in mat_knowledge.values())}")
    
    # Technique 
    tech_cooc = defaultdict(lambda: defaultdict(int))
    tech_total = defaultdict(int)
    
    for _, row in df.iterrows():
        techs = parse_json_list(row.get('techniques'))
        for t in techs:
            tech_total[t] += 1
        for a, b in combinations(sorted(set(techs)), 2):
            tech_cooc[a][b] += 1
            tech_cooc[b][a] += 1
    
    tech_knowledge = {}
    for t, neighbors in tech_cooc.items():
        if tech_total[t] < MIN_FREQ: continue
        tech_knowledge[t] = {}
        for n, cnt in neighbors.items():
            if cnt < 2: continue
            tech_knowledge[t][n] = {
                'count': cnt,
                'probability': round(cnt / tech_total[t], 4)
            }
    
    save_json(tech_knowledge, 'technique_cooccurrence.json')
    print(f"  Technique pairs: {sum(len(v) for v in tech_knowledge.values())}")


def build_cross_correlations(df):
    """Cross-field co-occurrence matrices."""
    print("Building cross-correlations:")
    
    mat_tech = defaultdict(lambda: defaultdict(int))
    mat_obj  = defaultdict(lambda: defaultdict(int))
    tech_obj = defaultdict(lambda: defaultdict(int))
    
    for _, row in df.iterrows():
        mats = parse_json_list(row.get('materials'))
        techs = parse_json_list(row.get('techniques'))
        obj = str(row.get('object_name') or '').strip().lower()
        
        for m in mats:
            for t in techs:
                mat_tech[m][t] += 1
            if obj:
                mat_obj[m][obj] += 1
        for t in techs:
            if obj:
                tech_obj[t][obj] += 1
    
    def filter_and_sort(d):
        return dict(sorted({k: v for k, v in d.items() if v >= 2}.items(), key=lambda item: -item[1]))

    cross = {
        'material_technique': {m: filter_and_sort(ts) for m, ts in mat_tech.items() if sum(ts.values()) >= MIN_FREQ},
        'material_object':    {m: filter_and_sort(os_) for m, os_ in mat_obj.items() if sum(os_.values()) >= MIN_FREQ},
        'technique_object':   {t: filter_and_sort(os_) for t, os_ in tech_obj.items() if sum(os_.values()) >= MIN_FREQ},
    }
    
    save_json(cross, 'cross_correlations.json')
    print(f"MatxTech: {len(cross['material_technique'])} | "
          f"MatxObj: {len(cross['material_object'])} | "
          f"TechxObj: {len(cross['technique_object'])}")


def build_artists(df):
    """Artist profiles and peer clusters for swapping."""
    print("Building artist profiles and clusters:")
    
    artists = defaultdict(lambda: {
        'count': 0, 'dates': [], 'end_dates': [],
        'nationalities': defaultdict(int), 'roles': defaultdict(int),
        'objects': defaultdict(int), 'techniques': defaultdict(int)
    })
    
    valid = df.dropna(subset=['artist_name'])
    for _, row in valid.iterrows():
        name = str(row.get('artist_name')).strip().lower()
        if not name or name == 'nan':
            continue
            
        anon_variants = ['unknown', 'anonymous', 'unidentified', 'attributed to', 'workshop of', 'circle of', 'school of', 'style of', 'follower of']
        if any(v in name for v in anon_variants):
            continue
        
        a = artists[name]
        a['count'] += 1
        
        try:
            d = float(row.get('artist_date_begin', np.nan))
            if not np.isnan(d): a['dates'].append(d)
        except: pass
        
        try:
            d = float(row.get('artist_date_end', np.nan))
            if not np.isnan(d): a['end_dates'].append(d)
        except: pass
        
        nat = row.get('artist_nationality')
        if pd.notna(nat): a['nationalities'][str(nat).strip().lower()] += 1
        
        role = row.get('artist_role')
        if pd.notna(role): a['roles'][str(role).strip().lower()] += 1
        
        obj = row.get('object_name')
        if pd.notna(obj): a['objects'][str(obj).strip().lower()] += 1
        
        for t in parse_json_list(row.get('techniques')):
            a['techniques'][t] += 1
    
    profiles = {}
    cluster_data = []
    
    for name, data in artists.items():
        if data['count'] < 2: continue 
        
        avg_start = int(sum(data['dates']) / len(data['dates'])) if data['dates'] else None
        avg_end = int(sum(data['end_dates']) / len(data['end_dates'])) if data['end_dates'] else None
        
        primary_nat = max(data['nationalities'].items(), key=lambda x: x[1])[0] if data['nationalities'] else None
        primary_role = max(data['roles'].items(), key=lambda x: x[1])[0] if data['roles'] else None
        primary_obj = max(data['objects'].items(), key=lambda x: x[1])[0] if data['objects'] else None
        
        profiles[name] = {
            'artist_name': name,
            'artist_role': primary_role,
            'artist_nationality': primary_nat,
            'artist_date_begin': avg_start,
            'artist_date_end': avg_end,
            'count': data['count'],
            'primary_object': primary_obj,
        }
        
        if avg_start is not None:
            cluster_data.append({
                'name': name,
                'era': (avg_start // 50) * 50,
                'nationality': primary_nat or 'unknown',
                'primary_object': primary_obj or 'unknown',
            })
    
    save_json(profiles, 'artist_profiles.json')
    print(f"  Profiles: {len(profiles):,}")
    
    clusters = defaultdict(list)
    for art in cluster_data:
        key = f"{art['era']}_{art['primary_object']}"
        clusters[key].append(art['name'])
    
    final_clusters = {k: sorted(v) for k, v in clusters.items() if len(v) > 1}
    
    save_json(final_clusters, 'artist_clusters.json')
    print(f"Clusters: {len(final_clusters):,} ({sum(len(v) for v in final_clusters.values()):,} artists)")

def build_culture_adjacency(df):
    print("Building culture adjacency:")
    
    valid = df.dropna(subset=['culture']).copy()
    valid['culture'] = (valid['culture']).astype(str).str.strip().str.lower()
    
    valid['date_begin'] = pd.to_numeric(valid['date_begin'], errors='coerce')
    valid['era'] = (valid['date_begin'] // 100) * 100
    
    cult_total_counts = valid['culture'].value_counts().to_dict()
    cult_country_counts = defaultdict(lambda: defaultdict(int))
    cult_material_counts = defaultdict(lambda: defaultdict(int))
    cult_era_counts = defaultdict(lambda: defaultdict(int))
    
    # hardcoded high frequencies
    country_to_continent = {
        'netherlands': 'europe', 'france': 'europe', 'italy': 'europe', 
        'germany': 'europe', 'united kingdom': 'europe', 'belgium': 'europe',
        'spain': 'europe', 'austria': 'europe', 'czech republic': 'europe',
        'greece': 'europe', 'cyprus': 'europe', 'switzerland': 'europe',
        'egypt': 'africa', 'china': 'asia', 'japan': 'asia', 'iran': 'asia',
        'india': 'asia', 'korea': 'asia', 'vietnam': 'asia', 'thailand': 'asia',
        'pakistan': 'asia', 'iraq': 'asia', 'syria': 'asia', 'turkey': 'asia',
        'peru': 'americas', 'mexico': 'americas', 'united states': 'americas',
        'canada': 'americas', 'bolivia': 'americas', 'chile': 'americas', 'ecuador': 'americas',
        'colombia': 'americas', 'guatemala': 'americas'
    }

    # harcoded fallback for ancient cultures that lack geographic metadata
    CULTURE_TO_CONTINENT_FALLBACK = {
        'greek': 'europe',
        'roman': 'europe',
        'etruscan': 'europe',
        'cypriot': 'europe',
        'byzantine': 'europe',
        'egyptian': 'africa',
        'chinese': 'asia',
        'japanese': 'asia',
        'indian': 'asia',
        'sasanian': 'asia',
        'iranian': 'asia',
        'mesopotamian': 'asia',
        'sumerian': 'asia',
        'assyrian': 'asia',
        'babylonian': 'asia',
        'moche': 'americas',
        'inca': 'americas',
        'maya': 'americas',
        'aztec': 'americas'
    }

    for _, row in valid.iterrows():
        cult = row['culture']
        if cult == 'nan' or not cult: continue
        # Skip ambiguous cultures like "british or french"
        if ' or ' in cult: continue
        
        country = str(row.get('country') or '').strip().lower()
        if country and country != 'nan':
            cult_country_counts[cult][country] += 1
            
        for m in parse_json_list(row.get('materials')):
            cult_material_counts[cult][m] += 1
            
        era = row['era']
        if pd.notna(era):
            cult_era_counts[cult][int(era)] += 1
            
    significant_countries = {}
    significant_continents = {}
    significant_materials = {}
    significant_eras = {}
    
    THRESHOLD = 0.05
    
    cult_freq = valid['culture'].value_counts()
    freq_cultures = sorted(set(c for c in cult_freq[cult_freq >= MIN_FREQ].index if ' or ' not in c))
    
    for cult in freq_cultures:
        total = cult_total_counts.get(cult, 0)
        if total == 0: continue
        
        sig_cos = set()
        sig_conts = set()
        for co, count in cult_country_counts[cult].items():
            if count >= 2 and (count / total) >= THRESHOLD:
                sig_cos.add(co)
                cont = country_to_continent.get(co, 'unknown')
                if cont != 'unknown':
                    sig_conts.add(cont)
        
        if not sig_conts:
            for prefix, cont in CULTURE_TO_CONTINENT_FALLBACK.items():
                if cult.startswith(prefix):
                    sig_conts.add(cont)
                    break

        significant_countries[cult] = sig_cos
        significant_continents[cult] = sig_conts
        
        sig_mats = set()
        for m, count in cult_material_counts[cult].items():
            if count >= 2 and (count / total) >= THRESHOLD:
                sig_mats.add(m)
        significant_materials[cult] = sig_mats
        
        sig_eras = set()
        for era, count in cult_era_counts[cult].items():
            if count >= 2 and (count / total) >= THRESHOLD:
                sig_eras.add(era)
        significant_eras[cult] = sig_eras

    adjacency = defaultdict(list)
    
    for i, c1 in enumerate(freq_cultures):
        for c2 in freq_cultures[i+1:]:
            is_adjacent = False
            
            # country
            shared_countries = significant_countries.get(c1, set()) & significant_countries.get(c2, set())
            if shared_countries:
                is_adjacent = True
            else:
                # continent, time, and material overlap
                shared_conts = significant_continents.get(c1, set()) & significant_continents.get(c2, set())
                shared_eras = significant_eras.get(c1, set()) & significant_eras.get(c2, set())
                
                m1 = significant_materials.get(c1, set())
                m2 = significant_materials.get(c2, set())
                high_mat_overlap = False
                if m1 and m2:
                    jaccard = len(m1 & m2) / len(m1 | m2)
                    if jaccard > 0.50:
                        high_mat_overlap = True
                
                if shared_conts and shared_eras and high_mat_overlap:
                    is_adjacent = True
                    
            if is_adjacent:
                adjacency[c1].append(c2)
                adjacency[c2].append(c1)
    
    save_json(dict(adjacency), 'culture_adjacency.json')
    
    culture_to_cont = {}
    for cult in freq_cultures:
        conts = list(significant_continents.get(cult, []))
        if conts:
            culture_to_cont[cult] = conts[0] # Pick primary continent
            
    save_json(culture_to_cont, 'culture_to_continent.json')
    
    print(f"Cultures with adjacency: {len(adjacency):,} | Continents mapped: {len(culture_to_cont):,}")

def build_place_knowledge(df):
    """Data-driven city→country mapping and categorized neighbor graphs (Country vs City)."""
    print("Building place knowledge (using clean legacy dictionary):")

    # Load Legacy Knowledge
    legacy_path = os.path.join(BASE_DIR, 'error_injection/knowledge/place_knowledge.json')
    if not os.path.exists(legacy_path):
        print(f"  Warning: Legacy dictionary not found at {legacy_path}")
        return

    with open(legacy_path, 'r') as f:
        legacy_data = json.load(f)

    city_to_country = legacy_data.get('city_to_country', {})
    country_to_cities = legacy_data.get('country_to_cities', {})
    country_neighbors = legacy_data.get('country_neighbors', {})

    # Build clean city_neighbors strictly from same-country siblings
    city_neighbors = defaultdict(list)
    for country, cities in country_to_cities.items():
        city_list = list(cities)
        for i, c1 in enumerate(city_list):
            for c2 in city_list[i+1:]:
                city_neighbors[c1].append(c2)
                city_neighbors[c2].append(c1)

    place_knowledge = {
        'city_to_country': city_to_country,
        'country_to_cities': country_to_cities,
        'country_neighbors': country_neighbors,
        'city_neighbors': dict(city_neighbors)
    }

    save_json(place_knowledge, 'place_knowledge.json')
    print(f"Countries: {len(country_to_cities):,} | Cities: {len(city_to_country):,}")
    print(f"Country neighbors: {len(country_neighbors):,} | City neighbors: {len(city_neighbors):,}")

def build_nationality_to_country(df):
    """Derive a nationality→country mapping by co-occurring artist_nationality
    with the country field. For each nationality, the most frequent associated
    country wins."""
    print("Building nationality→country mapping:")

    valid = df.dropna(subset=['artist_nationality', 'country'])

    counts = defaultdict(lambda: defaultdict(int))
    for _, row in valid.iterrows():
        nat = str(row['artist_nationality']).strip().lower()
        cou = str(row['country']).strip().lower()
        if nat and nat != 'nan' and cou and cou != 'nan':
            counts[nat][cou] += 1

    nat_to_country = {}
    for nat, countries in counts.items():
        best_country, best_count = max(countries.items(), key=lambda x: x[1])
        if best_count >= 2:
            nat_to_country[nat] = best_country

    save_json(nat_to_country, 'nationality_to_country.json')
    print(f"Nationalities mapped: {len(nat_to_country):,}")

def save_json(data, filename):
    path = os.path.join(OUT_DIR, filename)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def main():
    print("=" * 60)
    print("WORLD KNOWLEDGE RECONSTRUCTION")
    print(f"Source: {INPUT_DIR}")
    print(f"Output: {OUT_DIR}")
    print("=" * 60 + "\n")
    
    df = load_all_data()
    
    build_timelines(df)
    build_regions(df)
    build_cooccurrence(df)
    build_cross_correlations(df)
    build_artists(df)
    build_culture_adjacency(df)
    build_place_knowledge(df)
    build_nationality_to_country(df)
    
    print("\n" + "=" * 60)
    print("All knowledge files saved to:", OUT_DIR)
    print("=" * 60)


if __name__ == "__main__":
    main()
