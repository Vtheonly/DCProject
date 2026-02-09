"""
Circuit Domain Models - Industrial DC Microgrid Platform

Defines the core electrical components of the microgrid for cross-language
interoperability with MATLAB. These models are designed to be serialized
perfectly to/from MATLAB structs.
"""
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
import uuid

@dataclass
class Bus:
    """Electrical Bus (Node)."""
    id: int
    name: str = ""
    voltage_kv: float = 0.4  # Default 400V
    type: str = "PQ"  # PQ, PV, Slack
    x: float = 0.0  # Visualization coordinate
    y: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class Line:
    """Transmission Line / Cable."""
    id: int
    from_bus: int
    to_bus: int
    r_ohm: float
    x_ohm: float
    length_km: float = 0.1
    c_nf_per_km: float = 0.0
    status: int = 1  # 1=Closed, 0=Open

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class Generator:
    """Power Source (Solar, Wind, Grid Infeed)."""
    id: int
    bus_id: int
    p_mw: float
    q_mvar: float = 0.0
    p_max_mw: float = 0.0
    p_min_mw: float = 0.0
    status: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class Load:
    """Power Consumer."""
    id: int
    bus_id: int
    p_mw: float
    q_mvar: float = 0.0
    status: int = 1
    priority: int = 1  # 1=Critical, 2=Normal, 3=Sheddable

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class CircuitModel:
    """
    Complete Microgrid Circuit Model.
    
    Acts as the Single Source of Truth for:
    1. Python Simulation (GridEmulator)
    2. MATLAB Analysis (Power Flow)
    3. UI Visualization
    """
    name: str = "Microgrid_Default"
    base_mva: float = 1.0
    buses: List[Bus] = field(default_factory=list)
    lines: List[Line] = field(default_factory=list)
    generators: List[Generator] = field(default_factory=list)
    loads: List[Load] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert entire model to dictionary for serialization."""
        return {
            "name": self.name,
            "base_mva": self.base_mva,
            "limit_violations": [],  # Placeholder for analysis results
            "buses": [b.to_dict() for b in self.buses],
            "lines": [l.to_dict() for l in self.lines],
            "generators": [g.to_dict() for g in self.generators],
            "loads": [l.to_dict() for l in self.loads]
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CircuitModel':
        """Create model from dictionary (e.g. loaded from JSON/MAT)."""
        model = cls(
            name=data.get("name", "Imported_Grid"),
            base_mva=data.get("base_mva", 1.0)
        )
        
        # Helper to safely parse lists of dicts
        def parse_list(key, cls_type):
            items = []
            raw_list = data.get(key, [])
            # Handle MATLAB's squeezed arrays if necessary (often come as numpy arrays)
            if hasattr(raw_list, 'tolist'):
                raw_list = raw_list.tolist()
                
            for item in raw_list:
                # Filter out keys that don't belong to the dataclass (simpler forward compatibility)
                valid_keys = cls_type.__dataclass_fields__.keys()
                filtered_item = {k: v for k, v in item.items() if k in valid_keys}
                items.append(cls_type(**filtered_item))
            return items

        try:
            model.buses = parse_list("buses", Bus)
            model.lines = parse_list("lines", Line)
            model.generators = parse_list("generators", Generator)
            model.loads = parse_list("loads", Load)
        except Exception as e:
            # Fallback for manual/simpler struct structures if needed
            print(f"Warning during efficient parsing: {e}")
            
        return model

    def get_bus_by_id(self, bus_id: int) -> Optional[Bus]:
        for b in self.buses:
            if b.id == bus_id:
                return b
        return None
