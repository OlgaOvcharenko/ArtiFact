dim_prompts = {{
    "arms_and_armor": """You are evaluating a collection of Arms and Armor. Given unstructured dimension text, output a JSON object. Output **only** valid JSON in exactly this structure:

    }}
    "scratchpad": "<analyze parts and measurements step-by-step>",
    "parts": [
        }}
        "part": "<name of part or object, if multiple are described>",
        "measurements": [
            }}
            "attribute": "<one of: height | width | depth | length | thickness | diameter | weight>",
            "value": <numeric value in centimeters or grams>,
            "note": "<any contextual note or label>"
            }}
        ]
        }}
    ]
    }}

    Normalization and parsing rules
    - Always return a "parts" list, even if there is only one part.
    - Leave "part" empty unless a clear name like "overall", "body", "cover", "frame", "sheath", "scabbard", "head", "blade", or an object/accession label (e.g., "13.112.14a") is present.
    - Abbreviations are measurement attributes, not part names:
        H. → height, W. → width, L. → length, D. → depth, Diam./Dia./Ø → diameter, Th./Thickness/T./Thk. → thickness, Wt. → weight.
    Distinguish depth vs. diameter: D. = depth, Diam./Dia./Ø = diameter.
    - Cal./Caliber indicates bore diameter. Map to attribute="diameter" and set note="caliber" unless the word "caliber" already appears in the note.
    - If multiple abbreviations (H., W., D., L., Diam.) occur in the same phrase for one item, treat them as measurements of a single part.
    - Keep any qualifiers in notes verbatim: "with scabbard", "without sheath", "of head", "of blade", "including nape defense", "excluding nape defense", "overall", "greatest", "smallest", "approx.", etc.
    - Unlabeled tuples of dimensions like "A × B [× C]" are mapped positionally to height, width, depth.
    - Units:
    • Prefer metric values in parentheses if present, use them directly (cm for size, g for weight).  
    • Otherwise convert: in → cm (×2.54), ft-in → cm (ft×30.48 + in×2.54), mm → cm (÷10), m → cm (×100).  
    • Weights: lb and oz → g (lb×453.59237 + oz×28.34952), oz → g (×28.34952), kg → g (×1000).  
    • Fractions like 2 1/8 in. → 2.125 in., handle unicode fraction glyphs and narrow spaces.  
    • Round converted cm and g to 1 decimal unless the source provides a more precise metric value.
    - Never invent numbers, never infer missing measures.
    - Ensure the output is valid JSON only — no commentary or markdown.

    Examples
    Input:
    "Menuki (a) L. 2 1/8 in. (5.4 cm); Wt. 0.2 oz. (5.7 g); menuki (b); L. 2 1/8 in. (5.4 cm); Wt. 0.3 oz. (8.5 g)"
    Output:
    }}
    "parts": [
        }}
        "part": "Menuki (a)",
        "measurements": [
            {{"attribute": "length", "value": 5.4, "note": ""}},
            {{"attribute": "weight", "value": 5.7, "note": ""}}
        ]
        {{,
        }}
        "part": "menuki (b)",
        "measurements": [
            {{"attribute": "length", "value": 5.4, "note": ""}},
            {{"attribute": "weight", "value": 8.5, "note": ""}}
        ]
        }}
    ]
    }}

    Input:
    "Helmet (a); H. including nape defense 20 3/4 in. (52.7 cm); H. excluding nape defense 11 1/4 in. (28.6 cm); W. 9 in. (22.9 cm); D. 10 1/4 in. (26 cm); Wt. 4 lb. 9 oz. (2069.5 g)"
    Output:
    }}
    "parts": [
        }}
        "part": "Helmet (a)",
        "measurements": [
            {{"attribute": "height", "value": 52.7, "note": "including nape defense"}},
            {{"attribute": "height", "value": 28.6, "note": "excluding nape defense"}},
            {{"attribute": "width", "value": 22.9, "note": ""}},
            {{"attribute": "depth", "value": 26.0, "note": ""}},
            {{"attribute": "weight", "value": 2069.5, "note": ""}}
        ]
        }}
    ]
    }}

    Now process this input:
    "}}dimensions}}"

    and return it in the form of:
    ```json
    ```
    """,

    "musical_instruments" : """You are evaluating a collection of Musical Instruments. Given unstructured dimension text, output a JSON object. Output **only** valid JSON in exactly this structure:

    }}
    "scratchpad": "<analyze parts and measurements step-by-step>",
    "parts": [
        }}
        "part": "<name of part or object, if multiple are described>",
        "measurements": [
            }}
            "attribute": "<one of: height | width | depth | length | thickness | diameter | weight>",
            "value": <numeric value in centimeters or grams>,
            "note": "<any contextual note or label>"
            }}
        ]
        }}
    ]
    }}

    Normalization and parsing rules
    - Always return a "parts" list, even if there is only one part.
    - Leave "part" empty unless a clear name or label is present (e.g., "overall", "bell", "mouthpiece", "reed", "bow", "beater", "stick", "frog", "case", "box", "stand", "head", "shell", "body", "back", "neck", an accession/object label like "13.112.14a").
    - Abbreviations and synonyms are measurement attributes, not part names:
        H./Height → height; W./Width/Great W. → width; D./Depth → depth; L./Len./Length → length;
        Diam./Dia./Ø/"Diameter" → diameter; Th./Thickness/T./Thk. → thickness; Wt./Weight → weight.
    - Instrument-specific phrases: map to a base attribute and keep specificity in notes.
    Examples: "string length", "scale length", "sounding length", "back length", "body length", "stop length" → attribute=length with note preserved;
    "diameter of bell"/"bell diameter" → attribute=diameter with note="bell";
    "bore"/"bore diameter"/"embouchure diameter" → attribute=diameter with note preserved.
    - Phrases "of <part>" (e.g., "L. of mouthpiece", "Diam. of bell") define measurements for that <part>; create/append that part.
    - If multiple abbreviations (H., W., D., L., Diam.) occur for one item, treat them as measurements of a single part (unless a new part label appears).
    - Unlabeled tuples "A × B [× C]" map positionally to height, width, depth for the current part.
    - Units:
    • Prefer metric values in parentheses if present; use them directly (cm for size, g for weight).
    • Otherwise convert: in → cm (×2.54); ft-in → cm (ft×30.48 + in×2.54); mm → cm (÷10); m → cm (×100).
    • Weights: lb and oz → g (lb×453.59237 + oz×28.34952); oz → g (×28.34952); kg → g (×1000).
    • Fractions (e.g., 2 1/8 in., 7-9/16 in.) must be converted to decimals before unit conversion.
    - A hyphen range like "35.7–35.9 cm" → record two measurements with the same attribute, values 35.7 and 35.9, notes "min" and "max" plus any part-specific text.
    - Preserve qualifiers verbatim in notes where relevant: "with case", "without mouthpiece", "including crook", "excluding lid", "lower bouts", "of back", "at greatest point", "overall", "approx.", etc.

    Examples
    Input:
    "L.:36.8 cm (14-1/2 in.) without key; Bell Diam.:15 cm (5-15/16 in.); L. of mouthpiece: 9.7 cm (3-13/16 in.)"
    Output:
    }}
    "parts": [
        }}
        "part": "",
        "measurements": [
            {{"attribute": "length", "value": 36.8, "note": "without key"}}
        ]
        {{,
        }}
        "part": "bell",
        "measurements": [
            {{"attribute": "diameter", "value": 15.0, "note": ""}}
        ]
        {{,
        }}
        "part": "mouthpiece",
        "measurements": [
            {{"attribute": "length", "value": 9.7, "note": ""}}
        ]
        }}
    ]
    }}

    Input:
    "Great W. 3.6 × L. 53.9 cm (1 7/16 in. × 21 1/4 in.)"
    Output:
    }}
    "parts": [
        }}
        "part": "",
        "measurements": [
            {{"attribute": "width", "value": 3.6, "note": "greatest"}},
            {{"attribute": "length", "value": 53.9, "note": ""}}
        ]
        }}
    ]
    }}

    Input:
    "L. ±30.2 cm (±11 7/8 in.)"
    Output:
    }}
    "parts": [
        }}
        "part": "",
        "measurements": [
            {{"attribute": "length", "value": 30.2, "note": "±"}}
        ]
        }}
    ]
    }}

    Now process this input:
    "}}dimensions}}"

    and return it in the form of:
    ```json
    ```
    """,

    "H_W_D": """You are evaluating a collection of artworks. Given unstructured dimension text, output a JSON object. Output ONLY valid JSON in exactly this structure:

    }}
    "scratchpad": "<analyze parts and measurements step-by-step>",
    "parts": [
        }}
        "part": "<name of part or object, if multiple are described>",
        "measurements": [
            }}
            "attribute": "<one of: height | width | depth | length | thickness | diameter | weight>",
            "value": <numeric value in centimeters or grams>,
            "note": "<any contextual note or label>"
            }}
        ]
        }}
    ]
    }}

    1) Within a clause, leave "part" empty unless a clear clause-level label appears (e.g., "Overall", "Image", "Sheet", "Frame", "Base", "Mount", "Foot", "Lip", "Figure", explicit object/folio/accession labels). Part labels are case-insensitive and not measurement attributes.
    2) H./Height → height; W./Width/Great W./Max. W. → width; D./Depth → depth; L./Len./Length → length; Th./Thickness/T./Thk. → thickness; Diam./Dia./Ø/Diameter → diameter; Wt./Weight → weight.
    3) Distinguish depth vs. diameter: "D." is depth; "Diam./Dia./Ø" is diameter.
    4) Preserve qualifiers verbatim in "note": e.g., "max.", "min.", "overall", "with stand", "with base", "framed", "including …", "excluding …".
    5) Phrases "of <feature>" (e.g., "Diam. (foot)", "Diam. of rim", "H. at neck", "L. of scarab"): keep the measurement on the current part and put the phrase in "note" (e.g., "foot", "of rim", "at neck", "of scarab"). Only create a new part when the noun denotes a true object-level subpart (e.g., "ring", "scarab", "image", "sheet", "frame", "mount", "panel", "folio").
    6) Unlabeled tuples "A x B [x C]" map positionally to height, width, depth for the current part. Two values → height, width. One value + explicit diameter context → diameter.
    7) If both an unlabeled inches tuple and a parenthetical centimeters tuple appear, the centimeters tuple is canonical for values and order. Do not swap height/width.
    8) Within a single named part, use this precedence to identify the plain object dimensions and avoid overwrite: plain (no qualifier) > "overall" > "image/sheet" > "with stand/base/halo/framed". Include all as separate measurements; do not replace higher-precedence values with lower-precedence variants.
    9) Prefer metric values (cm for size; g for weight) whenever they are provided in the text (often in parentheses). Use the given metric numbers verbatim; do not re-convert them from inches.
    10) When only non-metric units are present, convert as follows: in → cm (×2.54); ft-in → cm (ft×30.48 + in×2.54); mm → cm (÷10); m → cm (×100). Weights: lb & oz → g (lb×453.59237 + oz×28.34952); oz → g (×28.34952); kg → g (×1000).
    11) Fractions and Unicode fractions must be converted to decimals before unit conversion (e.g., "2 1/8 in." → 5.4 cm).
    12) Precision: if a metric value is present, keep its precision exactly as given. For converted values, round to one decimal place.


    Examples
    Input:
    "Figure: H. 36 3/8 in. (92.4 cm); W. 27 1/2 in. (69.9 cm); D. 19 5/8 in. (49.8 cm)
    Figure with base: H. 63 3/4 in. (161.9 cm); W. 38 3/4 in. (98.4 cm); D. 39 1/8 in. (99.4 cm)
    Figure with base and halo: H. 86 in. (218.4 cm)"
    Output:
    }}
    "parts": [
        }}
        "part": "Figure",
        "measurements": [
            {{"attribute": "height", "value": 92.4, "note": ""}},
            {{"attribute": "width", "value": 69.9, "note": ""}},
            {{"attribute": "depth", "value": 49.8, "note": ""}}
        ]
        {{,
        }}
        "part": "Figure with base",
        "measurements": [
            {{"attribute": "height", "value": 161.9, "note": ""}},
            {{"attribute": "width", "value": 98.4, "note": ""}},
            {{"attribute": "depth", "value": 99.4, "note": ""}}
        ]
        {{,
        }}
        "part": "Figure with base and halo",
        "measurements": [
            {{"attribute": "height", "value": 218.4, "note": ""}}
        ]
        }}
    ]
    }}

    Input:
    "Diam. of ring 2.3 cm (7/8 in.); L. of scarab 1.8 cm (11/16 in.); W. of scarab 1.2 cm (1/2 in.)"
    Output:
    }}
    "parts": [
        }}
        "part": "ring",
        "measurements": [
            {{"attribute": "diameter", "value": 2.3, "note": ""}}
        ]
        {{,
        }}
        "part": "scarab",
        "measurements": [
            {{"attribute": "length", "value": 1.8, "note": ""}},
            {{"attribute": "width", "value": 1.2, "note": ""}}
        ]
        }}
    ]
    }}

    Input:
    "H. from-to: 7 1/2 - 8 in. (19.1 - 20.3 cm)
    H. (neck and lip): 3 1/2 in. (8.9 cm)
    Diam. (base) 3 1/2 in. (8.9 cm)
    Diam. (lip) 2 1/2 in. (6.4 cm)"
    Output:
    }}
    "parts": [
        }}
        "part": "",
        "measurements": [
            {{"attribute": "height", "value": 20.3, "note": "max"}},
            {{"attribute": "height", "value": 19.1, "note": "min"}},
            {{"attribute": "height", "value": 8.9, "note": "neck and lip"}},
            {{"attribute": "diameter", "value": 8.9, "note": "base"}},
            {{"attribute": "diameter", "value": 6.4, "note": "lip"}}
        ]
        }}
    ]
    }}

    Now process this input:
    "}}dimensions}}"
    ```json
    ```
    """,

    "overall" : """You are evaluating a collection of artworks. Given a dimension in unstructured free text form, output a JSON object. You must output **only** valid JSON in the following structure:

    }}
    "scratchpad": "<analyze parts and measurements step-by-step>",
    "parts": [
        }}
        "part": "<name of part or object, if multiple are described>",
        "measurements": [
            }}
            "attribute": "<one of: height | width | depth | length | thickness | diameter | weight>",
            "value": <numeric value in centimeters or grams>,
            "note": "<any contextual note or label>"
            {{,
            ...
        ]
        {{,
        ...
    ]
    }}

    Rules:
    - Always return a "parts" list, even if there is only one part.
    - Each measurement must have:
    • "attribute": standardized to one of the seven valid attributes above.
    • "value": numeric, converted to centimeters (cm) or grams (g).
    • "note": any contextual information (e.g. "loom width", "approx.", "of bell", "without sheath").
    - If there are multiple abbreviations like "H.", "W.", "D.", or "L." within the same phrase, treat them as measurements of a single part, not as separate parts.
    - Leave "part" empty unless a clear name like "body", "cover", "frame", "sheath", "image", "sheet", "plate", "panel", "mount", etc. is present.
    - Abbreviations such as "H.", "W.", "D.", "L.", or "Diam." are measurement attributes, not part names.
    - If multiple unit systems are given (e.g. inches and cm), always use the metric values provided; otherwise convert to cm or g.
    - If only inches are given, convert to cm.

    Examples:
    Input:
    "Overall: 19 11/16 × 22 1/16 in. (50 × 56 cm)
    Sight: 15 3/8 × 17 11/16 in. (39 × 45 cm)
    Rabbet: 16 1/4 × 18 11/16 in. (41.3 × 47.4 cm)"

    Output:
    }}
    "parts": [
        }}
        "part": "Overall",
        "measurements": [
            {{"attribute": "height", "value": 50.0, "note": ""}},
            {{"attribute": "width", "value": 56.0, "note": ""}}
        ]
        {{,
        }}
        "part": "Sight",
        "measurements": [
            {{"attribute": "height", "value": 39.0, "note": ""}},
            {{"attribute": "width", "value": 45.0, "note": ""}}
        ]
        {{,
        }}
        "part": "Rabbet",
        "measurements": [
            {{"attribute": "height", "value": 41.3, "note": ""}},
            {{"attribute": "width", "value": 47.4, "note": ""}}
        ]
        }}
    ]
    }}

    Now process this input:
    "}}dimensions}}"

    and return it in the form of:
    ```json
    ```
    """,

    "bundle_dimensions": """You are evaluating dimensions for a bundle of departments (American Decorative Arts; Arts of Africa, Oceania, and the Americas; European Paintings; Greek and Roman Art; Modern Art). Given unstructured dimension text, output a JSON object. Output **only** valid JSON in exactly this structure:

    }}
    "scratchpad": "<analyze parts and measurements step-by-step>",
    "parts": [
        }}
        "part": "<name of part or object, if multiple are described>",
        "measurements": [
            }}
            "attribute": "<one of: height | width | depth | length | thickness | diameter | weight>",
            "value": <numeric value in centimeters or grams>,
            "note": "<any contextual note or label>"
            }}
        ]
        }}
    ]
    }}

    Normalization and parsing rules
    - Always return a "parts" list, even if there is only one part.
    - Default part label: use "overall" unless a clear name is present (e.g., "canvas", "panel", "board", "frame", "sight", "sheet", "plate", "image", "block", "base", "mount", "cover", "body", "neck", "rim", "foot", "handle", "bowl", explicit object/accession labels).
    - Abbreviations and synonyms (attributes, not part names):
    H./Height → height; W./Width/Great W./Max. W. → width; D./Depth → depth; L./Len./Length → length;
    Diam./Dia./Ø/"Diameter" → diameter; Th./Thickness → thickness; Wt./Weight → weight.
    Distinguish depth vs. diameter: D. = depth; Diam./Dia./Ø = diameter.
    - Part association:
    • Phrases like "of <part>", "(<part>)", or leading labels ("Sheet:", "Plate:", "Canvas:", "Base:") assign the following measurements to that part.
    • For "with frame"/"without frame"/"framed": keep part "overall" and place the phrase in the measurement "note".
    • For prints on paper: prefer parts "sheet", "plate", "image" (or "block" when stated).
    • For vessels/objects: keep qualifiers like "rim", "foot", "across handles", "at greatest width" in the "note".
    - Multiple attributes in one phrase (e.g., "H., W., D.") belong to the current part unless a new part label appears.
    - Unlabeled tuples "A × B [× C]" map positionally to height, width, depth for the current part.
    - Ranges and qualifiers:
    • Hyphen/en-dash ranges like "35.7–35.9 cm" → two measurements with the same attribute; use notes "min" and "max" plus any given qualifier.
    • "max.", "min.", "overall", "including …", "excluding …", "sight", "across handles", "at foot", "at rim", "at greatest" must be preserved verbatim in "note".
    - Units:
    • Prefer metric values in parentheses if present; use them directly (cm for size, g for weight).
    • Otherwise convert: in → cm (×2.54); ft-in → cm (ft×30.48 + in×2.54); mm → cm (÷10); m → cm (×100).
    • Weights: lb & oz → g (lb×453.59237 + oz×28.34952); oz → g (×28.34952); kg → g (×1000).
    • Convert mixed and unicode fractions to decimals before unit conversion (e.g., 2 1/8 in. → 2.125 in.).
    • Round converted cm and g to one decimal place unless the provided metric shows more precision; retain that precision.
    - Never invent numbers; never infer missing measures. Output must be valid JSON only — no commentary or markdown.

    Examples
    Input:
    "Canvas: 45 × 35 cm; with frame: 60 × 50 cm; Sight: 44 × 34 cm"
    Output:
    }}
    "parts": [
        }}
        "part": "canvas",
        "measurements": [
            {{"attribute": "height", "value": 45.0, "note": ""}},
            {{"attribute": "width", "value": 35.0, "note": ""}}
        ]
        {{,
        }}
        "part": "",
        "measurements": [
            {{"attribute": "height", "value": 60.0, "note": "with frame"}},
            {{"attribute": "width", "value": 50.0, "note": "with frame"}}
        ]
        {{,
        }}
        "part": "sight",
        "measurements": [
            {{"attribute": "height", "value": 44.0, "note": ""}},
            {{"attribute": "width", "value": 34.0, "note": ""}}
        ]
        }}
    ]
    }}

    Input:
    "Bowl: H. 8 cm; Diam. (rim) 18.5 cm; Diam. (foot) 6.2 cm"
    Output:
    }}
    "parts": [
        }}
        "part": "bowl",
        "measurements": [
            {{"attribute": "height", "value": 8.0, "note": ""}},
            {{"attribute": "diameter", "value": 18.5, "note": "rim"}},
            {{"attribute": "diameter", "value": 6.2, "note": "foot"}}
        ]
        }}
    ]
    }}


    Now process this input:
    "}}dimensions}}"

    Return only the JSON object.
    ```json
    ```
    """
}}