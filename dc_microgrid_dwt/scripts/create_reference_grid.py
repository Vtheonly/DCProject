import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from src.domain.circuit import CircuitModel, Bus, Line, Generator, Load
from src.adapters.matlab_bridge import MatlabBridge

def create_reference_grid():
    model = CircuitModel(name="Industrial_DC_Microgrid", base_mva=5.0)

    # --- BUSES ---
    # Bus 1: Main Source Bus (Slack)
    model.buses.append(Bus(id=1, name="Main_PCC_Bus", voltage_kv=0.4, type="Slack", x=0.5, y=0.9))
    # Bus 2: Solar Array Integration
    model.buses.append(Bus(id=2, name="Solar_Bus", voltage_kv=0.4, type="PV", x=0.2, y=0.7))
    # Bus 3: Battery Storage
    model.buses.append(Bus(id=3, name="Battery_Bus", voltage_kv=0.4, type="PV", x=0.8, y=0.7))
    # Bus 4: Heavy Load Center
    model.buses.append(Bus(id=4, name="Production_Line_A", voltage_kv=0.4, type="PQ", x=0.3, y=0.4))
    # Bus 5: Light Load Center
    model.buses.append(Bus(id=5, name="Production_Line_B", voltage_kv=0.4, type="PQ", x=0.7, y=0.4))
    # Bus 6: Auxiliary Loads
    model.buses.append(Bus(id=6, name="Lighting_HVAC", voltage_kv=0.4, type="PQ", x=0.5, y=0.2))

    # --- GENERATORS ---
    # Grid Infeed
    model.generators.append(Generator(id=1, bus_id=1, p_mw=2.0, p_max_mw=5.0, status=1))
    # Solar Array
    model.generators.append(Generator(id=2, bus_id=2, p_mw=0.8, p_max_mw=1.0, status=1))
    # Battery (Discharging)
    model.generators.append(Generator(id=3, bus_id=3, p_mw=0.5, p_max_mw=1.0, status=1))

    # --- LOADS ---
    model.loads.append(Load(id=1, bus_id=4, p_mw=1.2, priority=1)) # Critical
    model.loads.append(Load(id=2, bus_id=5, p_mw=0.8, priority=2))
    model.loads.append(Load(id=3, bus_id=6, p_mw=0.3, priority=3))

    # --- LINES (Cables) ---
    # Main ring or radial structure
    # PCC to Solar
    model.lines.append(Line(id=1, from_bus=1, to_bus=2, r_ohm=0.01, x_ohm=0.005, length_km=0.1))
    # PCC to Battery
    model.lines.append(Line(id=2, from_bus=1, to_bus=3, r_ohm=0.01, x_ohm=0.005, length_km=0.1))
    # Solar to Load A
    model.lines.append(Line(id=3, from_bus=2, to_bus=4, r_ohm=0.02, x_ohm=0.01, length_km=0.2))
    # Battery to Load B
    model.lines.append(Line(id=4, from_bus=3, to_bus=5, r_ohm=0.02, x_ohm=0.01, length_km=0.2))
    # Load A to Aux
    model.lines.append(Line(id=5, from_bus=4, to_bus=6, r_ohm=0.03, x_ohm=0.015, length_km=0.15))
    # Load B to Aux
    model.lines.append(Line(id=6, from_bus=5, to_bus=6, r_ohm=0.03, x_ohm=0.015, length_km=0.15))

    # Save
    filepath = os.path.abspath("reference_microgrid.mat")
    MatlabBridge.save_model(model, filepath)
    print(f"Generated reference model at: {filepath}")

if __name__ == "__main__":
    create_reference_grid()
