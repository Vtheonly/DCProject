import sys
import os
import time
import unittest
import threading

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from src.framework.bus import EventBus
from src.domain.events import FaultLocationEvent, DWTResultEvent
from src.adapters.grid_emulator import GridEmulator
from src.adapters.matlab_bridge import MatlabBridge
from src.agents.processing.fault_locator import PreciseFaultLocatorAgent

class TestModelDrivenWorkflow(unittest.TestCase):
    
    def setUp(self):
        self.bus = EventBus()
        self.emulator = GridEmulator(base_voltage=400.0)
        self.agent = PreciseFaultLocatorAgent("Locator", self.bus, {"emulator": self.emulator})
        
        # Ensure reference model exists
        self.ref_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../reference_microgrid.mat'))
        if not os.path.exists(self.ref_path):
            self.fail("Reference model not found. Run create_reference_grid.py first.")

    def tearDown(self):
        self.emulator.stop()
        self.agent.stop()

    def test_workflow(self):
        # 1. Start Emulator WITHOUT Model
        print("--- Testing Empty State ---")
        self.emulator.start()
        time.sleep(1)
        
        # Read sensor - should be 0.0 because no topology
        val = self.emulator.read()
        self.assertEqual(val, 0.0, "Emulator should output 0.0 without model")
        print("✅ Empty state confirmed.")
        
        # 2. Load Model
        print("--- Loading Reference Model ---")
        model = MatlabBridge.load_model(self.ref_path)
        self.assertIsNotNone(model)
        self.emulator.load_circuit(model)
        time.sleep(1)
        
        # Read sensor - should be ~400.0 (Base Voltage)
        val = self.emulator.read()
        self.assertGreater(val, 300.0, "Emulator should be active after loading model")
        print(f"✅ Model loaded. Voltage: {val:.2f}V")
        
        # 3. Inject Fault/Simulate DWT
        # Since we are testing logic, we will bypass the raw signal generation and 
        # simulate the DWT event causing a Fault Location
        print("--- Testing Fault Location Logic ---")
        self.agent.start()
        
        # Capture events
        received_locs = []
        def on_loc(evt): received_locs.append(evt)
        self.bus.subscribe(FaultLocationEvent, on_loc)
        
        # Simulate DWT Event (Far Fault)
        d1_coeffs = [0]*100
        d1_coeffs[50] = 50.0 # Peak
        evt = DWTResultEvent(
            timestamp=time.time(),
            coeffs=[[], [], [], [], d1_coeffs],
            energy_levels={"D1": 1.0, "D2": 1.0}, # Ratio 1 -> ~500m
            window_id=1
        )
        self.bus.publish(evt)
        time.sleep(0.5)
        
        self.assertTrue(len(received_locs) > 0, "Should have detected fault location")
        loc = received_locs[0]
        print(f"✅ Fault Located at {loc.distance_m:.2f}m in {loc.zone}")
        self.assertGreater(loc.distance_m, 400.0)

if __name__ == "__main__":
    unittest.main()
