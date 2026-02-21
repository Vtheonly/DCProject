import sys
import os
import unittest
import shutil

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from src.domain.circuit import CircuitModel, Bus, Line, Generator, Load
from src.adapters.matlab_bridge import MatlabBridge

class TestMatlabBridge(unittest.TestCase):
    
    def setUp(self):
        self.filename = "test_circuit.mat"
        # Create a sample circuit
        self.model = CircuitModel(name="TestGrid", base_mva=10.0)
        
        self.model.buses.append(Bus(id=1, name="Bus1", voltage_kv=11.0, type="Slack"))
        self.model.buses.append(Bus(id=2, name="Bus2", voltage_kv=11.0, type="PQ"))
        
        self.model.lines.append(Line(id=1, from_bus=1, to_bus=2, r_ohm=0.5, x_ohm=1.2))
        
        self.model.generators.append(Generator(id=1, bus_id=1, p_mw=5.0))
        
        self.model.loads.append(Load(id=1, bus_id=2, p_mw=3.0))

    def tearDown(self):
        if os.path.exists(self.filename):
            os.remove(self.filename)

    def test_round_trip(self):
        # 1. Save
        success = MatlabBridge.save_model(self.model, self.filename)
        self.assertTrue(success, "Failed to save model")
        
        # 2. Load
        loaded_model = MatlabBridge.load_model(self.filename)
        self.assertIsNotNone(loaded_model, "Failed to load model")
        
        # 3. Verify
        self.assertEqual(len(loaded_model.buses), 2)
        self.assertEqual(len(loaded_model.lines), 1)
        self.assertEqual(len(loaded_model.generators), 1)
        self.assertEqual(len(loaded_model.loads), 1)
        
        # Check specific values
        b1 = loaded_model.get_bus_by_id(1)
        self.assertIsNotNone(b1)
        self.assertEqual(b1.name, "Bus1")
        self.assertAlmostEqual(b1.voltage_kv, 11.0)
        
        l1 = loaded_model.lines[0]
        self.assertEqual(l1.from_bus, 1)
        self.assertEqual(l1.to_bus, 2)
        self.assertAlmostEqual(l1.r_ohm, 0.5)

        print("✅ Round-trip test passed!")

if __name__ == "__main__":
    unittest.main()
