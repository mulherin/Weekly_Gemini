# domain.py
# Light domain types to aid readability (no runtime dependency on other layers).
from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass
class DigestItems:
    items: List[Dict[str, Any]]
    signal_quality: Dict[str, Any]
    exposure_weights: Dict[str, float]

@dataclass
class ExecItem:
    main_point: str
    supporting_points: List[str]
    flow: List[str]
