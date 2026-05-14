import requests, re, html, json, os, hashlib
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple
from rijks_net import get_session

EN_LANG_AAT = "http://vocab.getty.edu/aat/300388277"
NL_LANG_AAT = "http://vocab.getty.edu/aat/300388256"

DESC_AAT = {"http://vocab.getty.edu/aat/300048722"}
DIM_AAT  = {"http://vocab.getty.edu/aat/300435430"}
MED_AAT  = {"http://vocab.getty.edu/aat/300435429"}
CRED_AAT = {"http://vocab.getty.edu/aat/300026687"}
PROV_AAT = {"http://vocab.getty.edu/aat/300444174"}
LABEL_AAT = {"http://vocab.getty.edu/aat/300417268"}
INSCRIPTION_AAT = {"http://vocab.getty.edu/aat/300435414"}
REJECTED_AAT = {"http://vocab.getty.edu/aat/300404908"}
TECHNIQUE_IGNORE = {
    "manufacturer", "maker", "print maker", "die cutter", "jeweler", 
    "publisher", "engraver", "etcher", "lithographer", "artist", 
    "creator", "designer", "draftsman", "painter", "sculptor"
}


UNIT_LABELS = {
    "http://vocab.getty.edu/aat/300379098": "cm",
    "http://vocab.getty.edu/aat/300379100": "mm",
    "http://vocab.getty.edu/aat/300379097": "m",
}

YEAR_SPAN_RE = re.compile(r'(?i)(?:c(?:a|irca)\.?\s*)?(\d{3,4})(?:\s*[-–]\s*(\d{3,4}))?')

_BR_RE  = re.compile(r'(?is)<\s*br\s*/?\s*>')
_TAG_RE = re.compile(r'(?is)<[^>]+>')

LD_CACHE_DIR = os.environ.get("RIJKS_LD_CACHE", "../notebooks/cache/rijks-jsonld")
os.makedirs(LD_CACHE_DIR, exist_ok=True)

def _ld_cache_path(uri: str) -> str:
    h = hashlib.md5(uri.encode("utf-8")).hexdigest()
    return os.path.join(LD_CACHE_DIR, f"{h}.json")

LD_HEADERS = {"Accept": "application/ld+json, application/json;q=0.9"}

from extraction.core.network import SessionManager
LD_SESSION_MGR = SessionManager(rotate_every=25)

def fetch_ld(uri: str, timeout: int = 30, force: bool = False) -> Optional[Dict[str, Any]]:
    cache_file = _ld_cache_path(uri)
    if not force and os.path.exists(cache_file):
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass

    try:
        r = LD_SESSION_MGR.get(uri, headers=LD_HEADERS, timeout=timeout, allow_redirects=True)
        r.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"[error] Failed fetch_ld request for {uri}: {e}")
        return None

    ctype = (r.headers.get("Content-Type") or "").lower()
    if "json" not in ctype:
        print(f"[warning] Non-JSON response for {uri} (Content-Type: {ctype})")
        return None

    try:
        data = r.json()
    except ValueError:
        try:
            data = json.loads(r.content.decode(r.encoding or "utf-8", errors="ignore"))
        except Exception as e:
            print(f"[error] Failed to parse JSON for {uri}: {e}")
            if os.path.exists(cache_file):
                os.remove(cache_file)
            return None

    if not isinstance(data, dict):
        print(f"[warning] Expected dict for {uri}, got {type(data)}")
        if os.path.exists(cache_file):
            os.remove(cache_file)
        return None

    try:
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
    except Exception:
        pass

    return data

def as_list(x):
    if x is None:
        return []
    return x if isinstance(x, list) else [x]

def classified_ids(node) -> set:
    out = set()
    if not isinstance(node, dict):
        return out
    for c in as_list(node.get("classified_as")):
        if isinstance(c, dict) and c.get("id"): out.add(c["id"])
        elif isinstance(c, str): out.add(c)
    return out

def strip_html(text: str, collapse=True) -> str:
    if not isinstance(text, str):
        return ""
    t = _BR_RE.sub(" ", text)
    t = _TAG_RE.sub("", t)
    t = html.unescape(t).replace("\xa0", " ")
    if collapse:
        t = "\n".join(" ".join(line.split()) for line in t.splitlines()).strip()
    return t

def lang_tags(node) -> List[str]:
    tags = []
    if not isinstance(node, dict):
        return ["und"]
    for lang in as_list(node.get("language")):
        if isinstance(lang, dict):
            lid = (lang.get("id") or "").strip()
            if lid == EN_LANG_AAT: tags.append("English")
            elif lid == NL_LANG_AAT: tags.append("Dutch")
            else: tags.append("und")
        elif isinstance(lang, str):
            if lang == EN_LANG_AAT: tags.append("English")
            elif lang == NL_LANG_AAT: tags.append("Dutch")
            else: tags.append("und")
    return tags or ["und"]

def lang_code(node) -> Optional[str]:
    if not isinstance(node, dict):
        return None
    lids = [ (l.get("id") or "") for l in as_list(node.get("language")) if isinstance(l, dict) ]
    if EN_LANG_AAT in lids: return "English"
    if NL_LANG_AAT in lids: return "Dutch"
    return None

def labels_by_lang(name_nodes):
    out = {"English": [], "Dutch": []}
    for n in name_nodes or []:
        if not isinstance(n, dict) or n.get("type") != "Name":
            continue
        raw = n.get("content") or ""
        if isinstance(raw, list):
            raw = " ".join(str(x) for x in raw)
        txt = str(raw).strip()
        if not txt:
            continue
        for tag in lang_tags(n):
            if tag in ("English", "Dutch") and txt not in out[tag]:
                out[tag].append(txt)
    return {k: v for k, v in out.items() if v}

def ensure_multilang_labels(x) -> Dict[str, List[str]]:
    if isinstance(x, str):
        return {"und": [x]}
    if isinstance(x, dict):
        return {k: [v for v in (vals or []) if isinstance(v, str) and v.strip()]
                for k, vals in x.items() if (vals or [])}
    return {}

def merge_multilang_dicts(dicts):
    merged  = {}
    for d in dicts or []:
        if not isinstance(d, dict):
            continue
        for lang, vals in d.items():
            if not vals:
                continue
            merged.setdefault(lang, [])
            for v in vals:
                if v not in merged[lang]:
                    merged[lang].append(v)
    return merged


@lru_cache(maxsize=10000)
def read_rijksmuseum_place( uri: str):
    data = fetch_ld(uri)
    if not isinstance(data, dict):
        return None
    return labels_by_lang(data.get("identified_by", []))

@lru_cache(maxsize=10000)
def read_rijksmuseum_concept(uri: str):
    if not uri:
        return None
    try:
        data = fetch_ld(uri)
    except requests.exceptions.RequestException as e:
        print(e)
        return None
    if not isinstance(data, dict):
        return None
    return labels_by_lang(data.get("identified_by"))

@lru_cache(maxsize=10000)
def read_rijksmuseum_person(uri: str, depth: int = 1):
    data = fetch_ld(uri)
    if not isinstance(data, dict):
        return None

    result = {
        "id": uri,
        "names_by_lang": labels_by_lang(data.get("identified_by")),
        "birth": {"date": None, "place_by_lang": None},
        "death": {"date": None, "place_by_lang": None},
        "activities_by_lang": {},
        "equivalent": [e.get("id") for e in (data.get("equivalent") or []) if isinstance(e, dict) and e.get("id")]
    }

    born = data.get("born")
    if isinstance(born, dict):
        ts = born.get("timespan") or {}
        result["birth"]["date"] = ts.get("begin_of_the_begin") or ts.get("end_of_the_end")
        if depth > 0:
            for pl in (born.get("took_place_at") or []):
                if isinstance(pl, dict) and pl.get("id"):
                    labels = read_rijksmuseum_place(pl["id"])
                    if labels:
                        result["birth"]["place_by_lang"] = labels
                        break

    died = data.get("died")
    if isinstance(died, dict):
        ts = died.get("timespan") or {}
        result["death"]["date"] = ts.get("end_of_the_end") or ts.get("begin_of_the_begin")
        if depth > 0:
            for pl in (died.get("took_place_at") or []):
                if isinstance(pl, dict) and pl.get("id"):
                    labels = read_rijksmuseum_place(pl["id"])
                    if labels:
                        result["death"]["place_by_lang"] = labels
                        break

    act_labels: Dict[str, set] = {}
    for act in as_list(data.get("carried_out")):
        if not isinstance(act, dict): continue
        for cl in as_list(act.get("classified_as")):
            if not isinstance(cl, dict): continue
            uri2 = cl.get("id")
            if not uri2:
                continue
            if depth > 0:
                labs = read_rijksmuseum_concept(uri2) or {}
                for lang, texts in labs.items():
                    act_labels.setdefault(lang, set()).update(texts)
    result["activities_by_lang"] = {k: sorted(v) for k, v in act_labels.items()}
    return result

def pick_identifier(cho_ld, cho_uri=None):
    if cho_uri:
        # Guarantee 100% uniqueness to prevent cross-row overwrites during normalization
        return str(cho_uri).split('/')[-1]
        
    for idb in as_list(cho_ld.get("identified_by")):
        if not isinstance(idb, dict):
            continue
        if idb.get("type") == "Identifier" and idb.get("content"):
            content = idb["content"]
            if isinstance(content, list):
                content = content[0]
            return str(content)
    return None

def pick_titles_by_lang(cho_ld):
    return labels_by_lang([n for n in as_list(cho_ld.get("identified_by")) if isinstance(n, dict) and n.get("type") == "Name"])

def pick_titles(cho_ld):
    return pick_titles_by_lang(cho_ld)

def _aa_is_rejected(aa):
    return bool(classified_ids(aa) & REJECTED_AAT)

def pick_artist_info(cho_ld):
    def add_from_cob(cob, bag):
        for a in as_list(cob):
            if isinstance(a, dict) and a.get("id") and a.get("type") in ("Person", "Actor"):
                bag.append((a["id"], None))

    def add_from_assignment(aa, bag):
        if not isinstance(aa, dict) or aa.get("assigned_property") != "carried_out_by":
            return
        if _aa_is_rejected(aa):
            return
        for a in as_list(aa.get("assigned")):
            if not isinstance(a, dict):
                continue
            if a.get("id") and a.get("type") in ("Person", "Actor"):
                bag.append((a["id"], None))
            elif a.get("type") == "Production":
                add_from_cob(a.get("carried_out_by"), bag)
            elif a.get("type") == "Group":  # workshop of [artist]
                formed = a.get("formed_by") or {}
                for infl in as_list(formed.get("influenced_by")):
                    if infl.get("id") and infl.get("type") == "Person":
                        bag.append((infl["id"], "workshop_of"))

    pb = cho_ld.get("produced_by") or {}
    candidates: List[Tuple[str, Optional[str]]] = []
    if isinstance(pb, dict):
        add_from_cob(pb.get("carried_out_by"), candidates)
        for aa in as_list(pb.get("assigned_by")):
            add_from_assignment(aa, candidates)
        for part in as_list(pb.get("part")):
            if isinstance(part, dict):
                add_from_cob(part.get("carried_out_by"), candidates)
                for aa in as_list(part.get("assigned_by")):
                    add_from_assignment(aa, candidates)

    seen, order = {}, []
    for pid, qual in candidates:
        if pid not in seen:
            seen[pid] = qual
            order.append(pid)

    people = []
    for pid in order:
        info = read_rijksmuseum_person(pid, depth=1)
        if info:
            if seen[pid]:
                info = dict(info); info["qualifier"] = seen[pid]
            people.append(info)
    return people[0] if len(people) == 1 else people

def _looks_like_provenance_text(txt: str) -> bool:
    t = txt.strip()
    if t.startswith("…") or t.startswith("..."): return True
    if "{" in t and "}" in t: return True
    if t.count(";") >= 2: return True
    return False

def pick_descriptions(cho_ld):
    out: Dict[str, str] = {}

    for so in as_list(cho_ld.get("subject_of")):
        for lo, langs in walk_lo_with_lang(so, so.get("language") if isinstance(so, dict) else None):
            if classified_ids(lo) & DESC_AAT:
                raw_val = lo.get("content") or ""
                if isinstance(raw_val, list):
                    raw_val = " ".join(str(x) for x in raw_val)
                raw = str(raw_val).strip()
                if not raw:
                    continue
                txt = strip_html(raw)
                if not txt:
                    continue
                for tag in lang_tags({"language": langs}):
                    if tag not in out:
                        out[tag] = txt

    missing = {"English", "Dutch"} - set(out.keys())
    if missing:
        EXCLUDE = set().union(DIM_AAT, MED_AAT, DESC_AAT, CRED_AAT, PROV_AAT, INSCRIPTION_AAT)
        candidates = []
        for lo in as_list(cho_ld.get("referred_to_by")):
            if not isinstance(lo, dict):
                continue
            kinds = classified_ids(lo)
            if kinds & EXCLUDE:
                continue
            raw_val = lo.get("content") or ""
            if isinstance(raw_val, list):
                raw_val = " ".join(str(x) for x in raw_val)
            raw = str(raw_val).strip()
            if len(raw) < 40:
                continue
            txt = strip_html(raw)
            if not txt or _looks_like_provenance_text(txt):
                continue
            rank = 0 if (kinds & LABEL_AAT) else 1
            candidates.append((rank, lo, txt))

        candidates.sort(key=lambda x: (x[0], -len(x[2])))
        for _, lo, txt in candidates:
            lc = lang_code(lo)
            if lc in missing and lc not in out:
                out[lc] = txt
                missing.discard(lc)
            if not missing:
                break
    return out or None

def walk_lo_with_lang(node, inherited=None):
    if not isinstance(node, dict):
        return
    here_lang = (node.get("language") or inherited) if isinstance(node, dict) else inherited
    if isinstance(node, dict) and node.get("type") == "LinguisticObject":
        raw_val = node.get("content") or ""
        if isinstance(raw_val, list):
            raw_val = " ".join(str(x) for x in raw_val)
        if str(raw_val).strip():
            yield node, here_lang
    for ch in as_list(node.get("part") if isinstance(node, dict) else None):
        yield from walk_lo_with_lang(ch, here_lang)

def pick_type(cho_ld):
    for t in as_list(cho_ld.get("classified_as")):
        if not isinstance(t, dict):
            continue
        lab = t.get("_label") or t.get("label")
        if lab:
            return {"und": [lab]}
        tid = t.get("id")
        if tid:
            lab = read_rijksmuseum_concept(tid)
            if lab:
                return lab
    return None


def pick_subjects(cho_ld):
    pieces = []
    for s in as_list(cho_ld.get("shows")):
        if not isinstance(s, dict): continue
        for it in as_list(s.get("represents_instance_of_type")) + as_list(s.get("about")):
            if isinstance(it, dict) and it.get("id"):
                labs = read_rijksmuseum_concept(it["id"]) or {}
                pieces.append(labs)
            else:
                lab = (isinstance(it, dict) and (it.get("_label") or it.get("label"))) or (it if isinstance(it, str) else None)
                if lab:
                    pieces.append({"und": [lab]})
    return merge_multilang_dicts([ensure_multilang_labels(p) for p in pieces]) or None

def _resolve_place_node(pl):
    if isinstance(pl, str):
        return read_rijksmuseum_place(pl)
    if isinstance(pl, dict):
        lab = pl.get("_label") or pl.get("label")
        if lab:
            return {"und": [lab]}
        if pl.get("id"):
            return read_rijksmuseum_place(pl["id"])
    return None

def pick_place(cho_ld):
    pb = cho_ld.get("produced_by") or {}
    if isinstance(pb, dict):
        for part in as_list(pb.get("part")):
            if not isinstance(part, dict): continue
            for pl in as_list(part.get("took_place_at")):
                lab = _resolve_place_node(pl)
                if lab:
                    return lab
    for s in as_list(cho_ld.get("shows")):
        if not isinstance(s, dict): continue
        for pl in as_list(s.get("represents")):
            lab = _resolve_place_node(pl)
            if lab:
                return lab
    return None

def _looks_like_room_or_code(txt):
    t = txt.lower()
    return any(ch.isdigit() for ch in t) or "-" in t or t.startswith(("hg", "zaal", "room"))

def pick_current_location(cho_ld):
    loc = cho_ld.get("current_location") or {}
    if isinstance(loc, str):
        lab = _resolve_place_node(loc)
        if lab: return lab
    elif isinstance(loc, dict) and loc.get("id"):
        lab = _resolve_place_node(loc["id"])
        if lab: return lab

    name_candidates = []
    for idb in as_list(loc.get("identified_by")):
        for part in as_list(idb.get("part")):
            if part.get("type") == "Name" and part.get("content"):
                name_candidates.append(part)
        if idb.get("type") == "Name" and idb.get("content"):
            name_candidates.append(idb)

    for it in name_candidates:
        txt = (it.get("content") or "").strip()
        if txt and not _looks_like_room_or_code(txt):
            return {"und": [txt]}
    for it in name_candidates:
        txt = (it.get("content") or "").strip()
        if txt:
            return {"und": [txt]}
    return None

def pick_cho_uri(cho_ld):
    return cho_ld.get("id")

def pick_rights(cho_ld):
    for s in as_list(cho_ld.get("shows")):
        if not isinstance(s, dict): continue
        for r in as_list(s.get("subject_to")):
            if not isinstance(r, dict): continue
            for c in as_list(r.get("classified_as")):
                if not isinstance(c, dict): continue
                return c.get("_label") or c.get("label") or c.get("id")
    for so in as_list(cho_ld.get("subject_of")):
        if not isinstance(so, dict): continue
        for r in as_list(so.get("subject_to")):
            if not isinstance(r, dict): continue
            for c in as_list(r.get("classified_as")):
                if not isinstance(c, dict): continue
                return c.get("_label") or c.get("label") or c.get("id")
    return None

def _group_lang_text(items):
    out = {"English": [], "Dutch": []}
    for it in items or []:
        if not isinstance(it, dict):
            continue
        txt = (it.get("content") or "").strip()
        if not txt:
            continue
        lc = lang_code(it)
        if lc in out:
            out[lc].append(txt)
    return {k: v for k, v in out.items() if v}

def pick_inscriptions(cho_ld):
    los = []
    for lo in as_list(cho_ld.get("referred_to_by")):
        if classified_ids(lo) & INSCRIPTION_AAT and (lo.get("content") or "").strip():
            los.append(lo)
    return _group_lang_text(los) or None

def pick_set_memberships(cho_ld):
    ids = [s.get("id") for s in as_list(cho_ld.get("member_of")) if isinstance(s, dict) and s.get("id")]
    return {"ids": ids} if ids else None

def pick_iiif_info_from_digital(dig_ld):
    """
    From a DigitalObject JSON, return a IIIF info.json URL if possible.
    """
    for ap in (dig_ld.get("access_point") or []):
        ap_id = ap.get("id")
        if not ap_id:
            continue
        return ap_id
    return None

def pick_image_url(cho_ld):
    """
    From a HumanMadeObject JSON, follow:
      shows -> VisualItem -> digitally_shown_by -> DigitalObject
    and return the IIIF info.json URL.
    """
    # Step 1: VisualItem from the object
    shows = as_list(cho_ld.get("shows"))
    if not shows or not isinstance(shows[0], dict):
        return None
    vis_id = shows[0].get("id")
    if not vis_id:
        return None

    vis_ld = fetch_ld(vis_id)
    if not isinstance(vis_ld, dict):
        return None

    # Step 2: DigitalObject from the VisualItem
    d_shown = as_list(vis_ld.get("digitally_shown_by"))
    if not d_shown or not isinstance(d_shown[0], dict):
        return None
    dig_id = d_shown[0].get("id")
    if not dig_id:
        return None

    dig_ld = fetch_ld(dig_id)
    if not isinstance(dig_ld, dict):
        return None

    # Step 3: IIIF info.json from the DigitalObject
    return pick_iiif_info_from_digital(dig_ld)

def build_row(cho_uri: str) -> Dict[str, Any]:
    cho_ld = fetch_ld(cho_uri) if cho_uri else None
    if not isinstance(cho_ld, dict):
        cho_ld = {}

    artist_info = pick_artist_info(cho_ld) or {}
    artist_birth = artist_info.get("birth") if isinstance(artist_info, dict) else None
    artist_death = artist_info.get("death") if isinstance(artist_info, dict) else None

    return {
        # Identifiers
        "object_ID": pick_identifier(cho_ld, cho_uri=cho_uri),
        "title": pick_titles(cho_ld),
        "object_name": pick_type(cho_ld),
        "classification": None,
        "department": pick_set_memberships(cho_ld),

        # Dates
        "date": None,
        "date_begin": None,
        "date_end": None,

        # Medium / Size
        "medium": None,
        "material": None,
        "technique": None,
        "dimensions": None,

        # Location
        "culture": None,
        "country": pick_place(cho_ld),
        "region": None,
        "city": None,
        "period": None,
        "dynasty": None,
        "reign": None,

        # Artist
        "artist_name": (artist_info.get("names_by_lang") if isinstance(artist_info, dict) else None),
        "artist_role": (artist_info.get("activities_by_lang") if isinstance(artist_info, dict) else None),
        "artist_place_of_birth": (artist_birth and artist_birth.get("place_by_lang")),
        "artist_place_of_death": (artist_death and artist_death.get("place_by_lang")),
        "artist_date_begin": (artist_birth and artist_birth.get("date")),
        "artist_date_end": (artist_death and artist_death.get("date")),
        "artist_gender": None,
        "artist_equivalent": (artist_info.get("equivalent") if isinstance(artist_info, dict) else None),

        # Subject / Inscriptions
        "subjects": None,
        "inscriptions": pick_inscriptions(cho_ld),
    
        # Extra Info
        "description": pick_descriptions(cho_ld),
        "current_location": pick_current_location(cho_ld),
        "cho_uri": pick_cho_uri(cho_ld),
        "rights": pick_rights(cho_ld),
        
        "place": None, 

        # Image
        "image": None,
    }