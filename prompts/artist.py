artist_prompts = {
    "artist": """Task: Extract entity details from art records. You must determine if the record refers to a specific **NAMED PERSON** or a general **CULTURAL/GEOGRAPHIC ORIGIN**.

   Input Data:
   - artist_information: "{artist_information}"
   - artist_titles: "{artist_titles}"

   Output Schema (JSON):
   {
   "scratchpad": str,
   "artist_name": str, 
   "artist_nationality": str, 
   "artist_date_begin": str, 
   "artist_date_end": str, 
   "artist_place_of_birth": str, 
   "artist_place_of_death": str, 
   "country": str, 
   "region": str, 
   "culture": str, 
   "city": str
   }

   ### CRITICAL RULES:

   **0. SCRATCHPAD (REASONING)**
   * ALWAYS start your JSON response with a "scratchpad" key where you write your step-by-step classification and extraction reasoning.

   **1. DETECTING "ARTIST" RECORDS (Scenario A)**
   Treat the record as an **ARTIST** (and fill `artist_...` fields) if the input contains:
   * A specific person's name (e.g., "Rembrandt Peale").
   * Multiple people (e.g., "Vincent Laforme and Francis J. Laforme").
   * **Attributions**: "Workshop of...", "Circle of...", "Imitator of...", "After a design by...". Capture this **entire phrase** as the `artist_name`.
   * **"Unknown [Nationality] Artist"**: If the text says "Unknown Artist" or "Unknown Italian Artist", treat this as a **Named Artist**. 
      * `artist_name`: "Unknown Artist" (or "Imitator of...", etc.)
      * `artist_nationality`: Extract the nationality (e.g., "Italian", "English").

   **2. DETECTING "CULTURAL" RECORDS (Scenario B)**
   Treat the record as **CULTURAL** (fill `country`/`region`/`culture`) ONLY if:
   * There is absolutely NO person, workshop, or "Unknown Artist" mentioned.
   * The text refers *only* to a culture, place, or style (e.g., "Nayarit", "Chinese", "Probably Shaanxi province").

   **3. FIELD CONSTRAINTS**
   * **IF ARTIST IS FILLED**: `country`, `region`, `culture`, `city` MUST BE EMPTY.
   * **IF CULTURAL IS FILLED**: All `artist_...` fields MUST BE EMPTY.
   * **Geography**: Do **NOT** infer modern countries for ancient regions. If the text says "Ionia", put "Ionia" in `region` and leave `country` EMPTY unless "Greece" is explicitly written.
   * **Dates**: Extract years only.
   * **Full Names**: If multiple artists are listed, include **all** names in `artist_name`.

   ### Examples:

   Input: artist_information: "Workshop of Jan Emens Mennicken (Flemish, 1568-1594)", artist_titles: "['Jan Emens Mennicken']"
   Output:
   ```json
   {
   "scratchpad": "Contains a specific person's name 'Jan Emens Mennicken' with an attribution 'Workshop of'. Treat as ARTIST.",
   "artist_name": "Workshop of Jan Emens Mennicken",
   "artist_nationality": "Flemish",
   "artist_date_begin": "1568",
   "artist_date_end": "1594",
   "artist_place_of_birth": "",
   "artist_place_of_death": "",
   "country": "",
   "region": "",
   "culture": "",
   "city": ""
   }

   Input: artist_information: "Pablo Picasso (Spanish, 1881-1973)", artist_titles: "['Pablo Picasso']"
   Output:
   ```json
   {
      "scratchpad": "Contains a specific person's name 'Pablo Picasso'. Treat as ARTIST.",
      "artist_name": "Pablo Picasso",
      "artist_nationality": "Spanish",
      "artist_date_begin": "1881",
      "artist_date_end": "1973",
      "artist_place_of_birth": "",
      "artist_place_of_death": "",
      "country": "",
      "region": "",
      "culture": "",
      "city": ""
   }


    Input: artist_information: "China, Qishan county, Shaanxi province", artist_titles: "[]"
    ```json
   {
      "scratchpad": "Does not contain a person or workshop. Only places 'China, Qishan county, Shaanxi province'. Treat as CULTURAL.",
      "artist_name": "",
      "artist_nationality": "",
      "artist_date_begin": "",
      "artist_date_end": "",
      "artist_place_of_birth": "",
      "artist_place_of_death": "",
      "country": "China",
      "region": "Shaanxi province",
      "culture": "",
      "city": "Qishan county"
   }

    Now process this input:
    artist_information: "{artist_information}"
    artist_titles: "{artist_titles}"

    Output:```json
    ```"""
}
