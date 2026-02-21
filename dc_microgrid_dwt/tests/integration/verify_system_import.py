import sys
import unittest
from unittest.mock import MagicMock

# Mock Streamlit before importing system
sys.modules["streamlit"] = MagicMock()
sys.modules["streamlit"].session_state = {}

try:
    from src.ui.system import start_system, stop_system
    from src.agents.processing.dsp_runner import DSPRunnerAgent
    from src.adapters.high_speed_loop import HighSpeedDetectionLoop
    print("✅ All critical system modules imported successfully")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

class TestSystemIntegrity(unittest.TestCase):
    def test_instantiation(self):
        """Verify classes can be instantiated."""
        try:
            # Check HighSpeedLoop
            hsl = HighSpeedDetectionLoop(
                MagicMock(), MagicMock(), MagicMock(),
                sample_rate=1000, ui_throttle=10
            ) 
            self.assertIsNotNone(hsl)
            print("✅ HighSpeedDetectionLoop instantiated")
            
        except Exception as e:
            self.fail(f"Instantiation failed: {e}")

if __name__ == "__main__":
    unittest.main()
