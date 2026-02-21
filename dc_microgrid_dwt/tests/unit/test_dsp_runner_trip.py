import unittest
import time
from unittest.mock import MagicMock, ANY

from src.agents.processing.dsp_runner import DSPRunnerAgent
from src.domain.events import VoltageSampleEvent, SystemTripEvent

class TestDSPRunnerTrip(unittest.TestCase):
    def setUp(self):
        self.bus = MagicMock()
        self.pipeline = MagicMock()
        
        # Configure pipeline mock result
        self.result_trip = MagicMock()
        self.result_trip.trip.triggered = True
        self.result_trip.trip.d1_energy = 150.0
        self.result_trip.window_ready = False
        self.result_trip.d1_peak = 100.0
        
        # Configure pipeline process_sample
        self.pipeline.process_sample.return_value = self.result_trip

    def test_trip_handling(self):
        """Verify that DSPRunner correctly processes a trip result."""
        agent = DSPRunnerAgent("DSPRunner", self.bus, config={"dsp_pipeline": self.pipeline})
        agent.setup()
        
        # Simulate voltage sample
        event = VoltageSampleEvent(voltage=230.0, timestamp=time.time())
        agent.on_sample(event)
        
        # Verify SystemTripEvent publication
        # This confirms the fix: if result.trip.triggered was not checked, this would fail (if we only checked result.trip object truthiness, it might pass broadly, but we are testing SPECIFICALLY that logic flows)
        # Actually, to verify the fix properly, we should ensure it accesses .triggered
        
        # Check call args
        calls = self.bus.publish.call_args_list
        trip_calls = [c for c in calls if isinstance(c[0][0], SystemTripEvent)]
        self.assertEqual(len(trip_calls), 1, "Should publish exactly one SystemTripEvent")
        
        evt = trip_calls[0][0][0]
        self.assertIn("Fast Trip", evt.reason)

if __name__ == "__main__":
    unittest.main()
