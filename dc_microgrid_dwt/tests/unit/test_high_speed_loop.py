import unittest
import time
import threading
from unittest.mock import MagicMock, ANY
import queue

from src.adapters.high_speed_loop import HighSpeedDetectionLoop
from src.domain.events import SystemTripEvent, ProcessingResultEvent

class MockSensor:
    def read(self):
        return 230.0

class TestHighSpeedLoop(unittest.TestCase):
    def setUp(self):
        self.sensor = MockSensor()
        self.pipeline = MagicMock()
        self.bus = MagicMock()
        
        # Setup pipeline mock result
        self.result_normal = MagicMock()
        self.result_normal.trip.triggered = False
        self.result_normal.window_ready = False
        
        self.result_trip = MagicMock()
        self.result_trip.trip.triggered = True
        self.result_trip.trip.d1_energy = 150.0
        self.result_trip.window_ready = True
        self.result_trip.d1_peak = 100.0
        self.result_trip.energy_dict.return_value = {"D1": 150.0}

    def test_normal_operation(self):
        # Configure pipeline to return normal results
        self.pipeline.process_sample.return_value = self.result_normal
        
        loop = HighSpeedDetectionLoop(
            self.sensor, self.pipeline, self.bus,
            sample_rate=1000, ui_throttle=10
        )
        
        loop.start()
        time.sleep(0.1)
        loop.stop()
        
        # Verify calls
        self.assertTrue(self.pipeline.process_sample.call_count > 0)
        # Should NOT publish trip
        self.bus.publish.assert_not_called()

    def test_trip_detection(self):
        # Configure pipeline to return TRIP results
        self.pipeline.process_sample.return_value = self.result_trip
        
        loop = HighSpeedDetectionLoop(
            self.sensor, self.pipeline, self.bus,
            sample_rate=1000, ui_throttle=10
        )
        
        loop.start()
        time.sleep(0.1)
        loop.stop()
        
        # Verify trip publication
        # We expect multiple trips because we return TRIP every sample
        calls = self.bus.publish.call_args_list
        trip_calls = [c for c in calls if isinstance(c[0][0], SystemTripEvent)]
        self.assertTrue(len(trip_calls) > 0)
        
        # Verify throttled UI updates
        ui_calls = [c for c in calls if isinstance(c[0][0], ProcessingResultEvent)]
        self.assertTrue(len(ui_calls) > 0)
        # Should be roughly 1/10th of total samples

if __name__ == "__main__":
    unittest.main()
