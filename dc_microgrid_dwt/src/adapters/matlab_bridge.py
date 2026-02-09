"""
MATLAB Bridge Adapter - Industrial DC Microgrid Platform

Handles interoperability with MATLAB/Simulink ecosystem by reading and writing
.mat files compliant with scipy.io's format. This enables the Python-based
grid emulator to load physical models defined in MATLAB.
"""
import scipy.io as sio
import numpy as np
import logging
from typing import Dict, Any, Optional

from src.domain.circuit import CircuitModel, Bus, Line, Generator, Load

logger = logging.getLogger(__name__)

class MatlabBridge:
    """
    Adapter to bridge Python Circuit Models and MATLAB .mat files.
    """

    @staticmethod
    def load_model(filepath: str) -> Optional[CircuitModel]:
        """Load a CircuitModel from a .mat file."""
        try:
            mat_data = sio.loadmat(filepath, struct_as_record=False, squeeze_me=True)
            
            # Expecting a struct named 'microgrid' or 'circuit' in the MAT file
            if 'microgrid' in mat_data:
                root = mat_data['microgrid']
            elif 'circuit' in mat_data:
                root = mat_data['circuit']
            else:
                logger.error(f"No 'microgrid' or 'circuit' struct found in {filepath}")
                return None

            # Create empty model
            model = CircuitModel()
            
            # --- Parse Buses ---
            if hasattr(root, 'bus'):
                raw_buses = root.bus if isinstance(root.bus, np.ndarray) else [root.bus]
                for b in raw_buses:
                    # Handle both struct attributes (b.id) and dict-like access if mismatch
                    # Scipy loadmat with struct_as_record=False returns objects
                    try:
                        bus_obj = Bus(
                            id=int(getattr(b, 'id', 0)),
                            voltage_kv=float(getattr(b, 'V', 1.0)), # Often 'V' in MATLAB
                            name=str(getattr(b, 'name', f"Bus_{int(getattr(b, 'id', 0))}")),
                            type=str(getattr(b, 'type', 'PQ')),
                            x=float(getattr(b, 'x', 0.0)),
                            y=float(getattr(b, 'y', 0.0))
                        )
                        model.buses.append(bus_obj)
                    except Exception as e:
                        logger.warning(f"Failed to parse bus: {e}")

            # --- Parse Lines ---
            if hasattr(root, 'line'):
                raw_lines = root.line if isinstance(root.line, np.ndarray) else [root.line]
                for l in raw_lines:
                    try:
                        line_obj = Line(
                            id=int(getattr(l, 'id', 0)),
                            from_bus=int(getattr(l, 'from', 0)), # 'from' is reserved in Python, but attribute access works
                            to_bus=int(getattr(l, 'to', 0)),
                            r_ohm=float(getattr(l, 'R', 0.01)),
                            x_ohm=float(getattr(l, 'X', 0.01)),
                            status=int(getattr(l, 'status', 1))
                        )
                        model.lines.append(line_obj)
                    except Exception as e:
                        logger.warning(f"Failed to parse line: {e}")

            # --- Parse Generators ---
            if hasattr(root, 'gen'):
                raw_gens = root.gen if isinstance(root.gen, np.ndarray) else [root.gen]
                for g in raw_gens:
                    try:
                        gen_obj = Generator(
                            id=int(getattr(g, 'id', 0)),
                            bus_id=int(getattr(g, 'bus', 0)),
                            p_mw=float(getattr(g, 'P', 0.0)), # 'P' in MATLAB
                            p_max_mw=float(getattr(g, 'Pmax', 100.0)),
                            status=int(getattr(g, 'status', 1))
                        )
                        model.generators.append(gen_obj)
                    except Exception as e:
                        logger.warning(f"Failed to parse generator: {e}")

            # --- Parse Loads ---
            if hasattr(root, 'load'):
                raw_loads = root.load if isinstance(root.load, np.ndarray) else [root.load]
                for l in raw_loads:
                    try:
                        load_obj = Load(
                            id=int(getattr(l, 'id', 0)),
                            bus_id=int(getattr(l, 'bus', 0)),
                            p_mw=float(getattr(l, 'P', 0.0)),
                            status=int(getattr(l, 'status', 1))
                        )
                        model.loads.append(load_obj)
                    except Exception as e:
                        logger.warning(f"Failed to parse load: {e}")

            logger.info(f"Successfully loaded circuit from {filepath} with {len(model.buses)} buses")
            return model

        except Exception as e:
            logger.error(f"Failed to load MAT file: {e}")
            return None

    @staticmethod
    def save_model(model: CircuitModel, filepath: str) -> bool:
        """Save a CircuitModel to a .mat file."""
        try:
            # We need to construct a dictionary that scipy.io.savemat can convert to a struct
            # Structure: {'microgrid': {'bus': [...], 'line': [...]}}
            
            # Helper to convert list of dataclasses to "struct array" (list of dicts)
            def to_struct_list(objects):
                return [obj.to_dict() for obj in objects]
            
            # MATLAB conventions: Use 'V' for voltage, 'P' for power, etc.
            # We map our dataclass fields to standard MATLAB power system field names
            
            mat_buses = []
            for b in model.buses:
                mat_buses.append({
                    'id': b.id, 'V': b.voltage_kv, 'name': b.name, 'type': b.type,
                    'x': b.x, 'y': b.y
                })
                
            mat_lines = []
            for l in model.lines:
                mat_lines.append({
                    'id': l.id, 'from': l.from_bus, 'to': l.to_bus, 
                    'R': l.r_ohm, 'X': l.x_ohm, 'status': l.status
                })
                
            mat_gens = []
            for g in model.generators:
                mat_gens.append({
                    'id': g.id, 'bus': g.bus_id, 'P': g.p_mw, 'Pmax': g.p_max_mw,
                    'status': g.status
                })
                
            mat_loads = []
            for l in model.loads:
                mat_loads.append({
                    'id': l.id, 'bus': l.bus_id, 'P': l.p_mw, 
                    'status': l.status
                })

            data_dict = {
                'microgrid': {
                    'bus': mat_buses,
                    'line': mat_lines,
                    'gen': mat_gens,
                    'load': mat_loads,
                    'base_mva': model.base_mva
                }
            }
            
            sio.savemat(filepath, data_dict)
            logger.info(f"Successfully saved circuit to {filepath}")
            return True

        except Exception as e:
            logger.error(f"Failed to save MAT file: {e}")
            return False
