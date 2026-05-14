from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Any

@dataclass
class ArtworkRecord:
    # IDs
    object_ID: str
    title: Optional[str] = None
    object_name: Optional[str] = None
    classification: Optional[str] = None
    department: Optional[str] = None
    
    # Dates
    date: Optional[str] = None
    date_begin: Optional[str] = None
    date_end: Optional[str] = None
    
    # Medium / Dims
    medium: Optional[str] = None
    material: Optional[str] = None
    technique: Optional[str] = None
    dimensions: Optional[str] = None
    
    # Geography / Culture / Period
    culture: Optional[str] = None
    country: Optional[str] = None
    region: Optional[str] = None
    city: Optional[str] = None
    period: Optional[str] = None
    dynasty: Optional[str] = None
    reign: Optional[str] = None
    
    # Artist Information
    artist_name: Optional[str] = None
    artist_role: Optional[str] = None
    artist_nationality: Optional[str] = None
    artist_date_begin: Optional[str] = None
    artist_date_end: Optional[str] = None
    artist_display_bio: Optional[str] = None
    artist_information: Optional[str] = None
    artist_titles: Optional[str] = None
    
    # Tags / Content
    subjects: Optional[str] = None
    inscriptions: Optional[str] = None
    description: Optional[str] = None
    
    # Image / Links
    image_url: Optional[str] = None
    cho_uri: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
