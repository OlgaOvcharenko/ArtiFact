from typing import Any, Dict, List, Optional, Tuple
import xml.etree.ElementTree as ET

NS = {
    "oai": "http://www.openarchives.org/OAI/2.0/",
    "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "dc":  "http://purl.org/dc/elements/1.1/",
    "dct": "http://purl.org/dc/terms/",
    "dcterms": 'http://purl.org/dc/terms/',
    "edm": "http://www.europeana.eu/schemas/edm/",
    "ore": "http://www.openarchives.org/ore/terms/",
    "skos":"http://www.w3.org/2004/02/skos/core#",
    "rdfs":"http://www.w3.org/2000/01/rdf-schema#",
    "edmfp": "http://www.europeanafashion.eu/edmfp/",
}

def _is_empty(x):
    if x is None: return True
    if isinstance(x, (list, dict, set)) and not x: return True
    if isinstance(x, str) and not x.strip(): return True
    return False

def _lang_from_xml(elem) -> Optional[str]:
    lang = (elem.attrib.get('{http://www.w3.org/XML/1998/namespace}lang') or '').lower()
    if lang.startswith('en'): return 'English'
    if lang.startswith('nl'): return 'Dutch'
    return None

def _collect_literals(rdf: ET.Element, xpath: str) -> List[Tuple[str, Optional[str]]]:
    out = []
    for el in rdf.findall(xpath, NS):
        txt = (el.text or '').strip()
        if txt:
            out.append((txt, _lang_from_xml(el)))
    return out

def _first_resource_attr(nodes) -> Optional[str]:
    for n in nodes:
        res = n.attrib.get('{%s}resource' % NS['rdf'])
        if res:
            return res
    return None

def _about_index(rdf: ET.Element) -> Dict[str, ET.Element]:
    idx = {}
    rdfns = '{%s}about' % NS['rdf']
    for el in rdf.findall(".//*[@rdf:about]", NS):
        about = el.attrib.get(rdfns)
        if about:
            idx[about] = el
    return idx

def _literal_with_lang(el) -> Optional[Tuple[str, Optional[str]]]:
    if el is None:
        return None
    txt = (el.text or "").strip()
    if not txt:
        return None
    lang = (
        el.attrib.get('{http://www.w3.org/XML/1998/namespace}lang')
        or el.attrib.get('lang')
    )
    return (txt, lang)

def _label_children(el: ET.Element) -> List[Tuple[str, Optional[str]]]:
    out = []
    for xp in (".//skos:prefLabel", ".//rdfs:label", ".//dc:title", ".//skos:altLabel", ".//prefLabel"):
        for lab in el.findall(xp, NS):
            pair = _literal_with_lang(lab)
            if pair:
                out.append(pair)
        if out:
            break
    return out

def _collect_resolved(rdf: ET.Element, xpath: str) -> List[Tuple[str, Optional[str]]]:
    idx = _about_index(rdf)
    out: List[Tuple[str, Optional[str]]] = []
    res_attr = '{%s}resource' % NS['rdf']

    for node in rdf.findall(xpath, NS):
        lit = _literal_with_lang(node)
        if lit:
            out.append(lit)
            continue
        uri = node.attrib.get(res_attr)
        if uri and uri in idx:
            out.extend(_label_children(idx[uri]))
    return out

def fill_row_from_rdf(row: Dict[str, Any], rdf: ET.Element) -> Dict[str, Any]:
    if _is_empty(row.get("image")):
        aggr = rdf.find(".//ore:Aggregation", NS)
        image = None
        if aggr is not None:
            wr = aggr.find(".//edm:isShownBy/edm:WebResource", NS)
            if wr is not None:
                image = wr.attrib.get(f"{{{NS['rdf']}}}about")
        if not image:
            nodes = rdf.findall(".//edm:isShownBy", NS)
            img = _first_resource_attr(nodes)
            if not img:
                for wr in rdf.findall(".//edm:WebResource", NS):
                    about = wr.attrib.get('{%s}about' % NS['rdf'])
                    fmts  = [ (f.text or '').lower() for f in wr.findall(".//dc:format", NS) ]
                    if about and any(("image" in f or f.startswith('image/')) for f in fmts):
                        img = about; break
                image = img
        row["image"] = image

    if _is_empty(row.get("date")):
        for xp in (".//dcterms:created", ".//dcterms:date", ".//dc:date"):
            dates = _collect_literals(rdf, xp)
            if dates:
                row["date"] = dates[0][0]
                break

    if _is_empty(row.get("material")):
        meds = _collect_resolved(rdf, ".//dcterms:medium")
        if meds:
            ml = {"English": [], "Dutch": []}
            for txt, lg in meds:
                if lg and str(lg).startswith("en"):
                    ml["English"].append(txt)
                elif lg and str(lg).startswith("nl"):
                    ml["Dutch"].append(txt)
            ml = {k: list(dict.fromkeys(v)) for k, v in ml.items() if v}
            row["material"] = ml or row.get("material")

    if _is_empty(row.get("technique")):
        meds = _collect_resolved(rdf, ".//edmfp:technique")
        if meds:
            ml = {"English": [], "Dutch": []}
            for txt, lg in meds:
                if lg and str(lg).startswith("en"):
                    ml["English"].append(txt)
                elif lg and str(lg).startswith("nl"):
                    ml["Dutch"].append(txt)
            ml = {k: list(dict.fromkeys(v)) for k, v in ml.items() if v}
        row["technique"] = ml or row.get("technique")

    if _is_empty(row.get("dimensions")):
        exts = _collect_literals(rdf, ".//dcterms:extent")
        if exts:
            row["dimensions"] = exts[0][0]

    if _is_empty(row.get("country")):
        places = _collect_resolved(rdf, ".//dcterms:spatial") or _collect_resolved(rdf, ".//dc:coverage")
        if places:
            row["country"] = places[0][0]

    if _is_empty(row.get("subjects")):
        subs = []
        subs.extend(_collect_resolved(rdf, ".//edm:ProvidedCHO/dc:subject"))
        subs.extend(_collect_resolved(rdf, ".//edm:Aggregation/dc:subject"))
        if subs:
            ml: Dict[str, List[str]] = {}
            for txt, lg in subs:
                key = ("English" if (lg and str(lg).startswith("en"))
                    else "Dutch" if (lg and str(lg).startswith("nl"))
                    else "und")
                ml.setdefault(key, []).append(txt)
            row["subjects"] = {k: list(dict.fromkeys(v)) for k, v in ml.items()}

    if _is_empty(row.get("object_ID_oai")):
        ids = _collect_literals(rdf, ".//dc:identifier")
        if ids:
            row["Object ID"] = ids[0][0]

    return row