from src.framework.base_agent import BaseAgent
from src.domain.events import FaultDetectedEvent, SystemTripEvent

class ZetaLogicAgent(BaseAgent):
    def setup(self):
        self.subscribe(FaultDetectedEvent, self.on_fault)
        self.subscribe(SystemTripEvent, self.on_trip)

    def on_fault(self, event: FaultDetectedEvent):
        # Immediate reaction: Reduce PWM duty cycle
        self.logger.info("Reducing PWM duty cycle due to potential fault.")
        # hardware_pwm.set_duty(10) 

    def on_trip(self, event: SystemTripEvent):
        # Hard shutdown
        self.logger.info("Zeta Converter Shutdown.")
        # hardware_pwm.stop()
