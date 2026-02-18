import logging
import time
from src.framework.base_agent import BaseAgent
from src.domain.events import VoltageSampleEvent, ProcessingResultEvent, SystemTripEvent

class DSPRunnerAgent(BaseAgent):
    """
    High-Speed DSP Runner Agent.
    
    Subscribes to VoltageSampleEvent and processes each sample using the 
    C++ DSP pipeline. Runs in the background thread (EventBus dispatch),
    ensuring low latency fault detection independent of UI refresh rate.
    """
    
    def setup(self):
        self.pipeline = self.config.get('dsp_pipeline')
        if not self.pipeline:
            self.logger.warning("No DSP pipeline provided! DSP Runner inactive.")
            return
        
        self.subscribe(VoltageSampleEvent, self.on_sample)
        self.logger.info("DSP Runner initialized and attached to VoltageSampleEvent")

    def on_sample(self, event: VoltageSampleEvent):
        """Process a single voltage sample."""
        if not self.pipeline:
            return
            
        try:
            # High-speed C++ processing
            result = self.pipeline.process_sample(event.voltage)
            
            # 1. Fast Trip Logic
            if result.trip:
                trip_event = SystemTripEvent(
                    reason="Fast Trip (DSP Core)",
                    source=self.name,
                    timestamp=event.timestamp
                )
                self.logger.critical("FAST TRIP TRIGGERED BY DSP CORE")
                self.publish(trip_event)
            
            # 2. Window Completion Handling
            if result.window_ready:
                energy = result.energy_dict()
                
                # Create result event for UI/Logging
                res_event = ProcessingResultEvent(
                    d1_peak=0.0, # Peak detection logic inside DSP if needed
                    d1_energy=energy.get("D1", 0.0),
                    is_faulty=result.trip,
                    timestamp=event.timestamp
                )
                # Attach full energy spectrum for dashboard
                res_event.energy_levels = energy
                
                self.publish(res_event)
                
        except Exception as e:
            # Don't crash the sampling thread
            self.logger.error(f"DSP processing error: {e}")
