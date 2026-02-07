import pywt
import numpy as np
from src.framework.base_agent import BaseAgent
from src.domain.events import WindowReadyEvent, DWTCoefficientsEvent

class DWTEngineAgent(BaseAgent):
    def setup(self):
        self.wavelet_name = self.config.get('wavelet', {}).get('family', 'db4')
        self.level = self.config.get('wavelet', {}).get('level', 4)
        self.mode = self.config.get('wavelet', {}).get('mode', 'symmetric')
        
        # Pre-check wavelet availability
        if self.wavelet_name not in pywt.wavelist():
            self.logger.error(f"Wavelet {self.wavelet_name} not available!")

        self.subscribe(WindowReadyEvent, self.on_window)

    def on_window(self, event: WindowReadyEvent):
        data = event.window_data
        
        try:
            # Perform DWT
            # wavedec returns [cA_n, cD_n, cD_n-1, ..., cD_1]
            coeffs = pywt.wavedec(data, self.wavelet_name, mode=self.mode, level=self.level)
            
            # Publish coefficients
            dwt_event = DWTCoefficientsEvent(
                coeffs=coeffs,
                wavelet=self.wavelet_name,
                window_id=event.window_id
            )
            self.publish(dwt_event)
            
        except Exception as e:
            self.logger.error(f"DWT Error: {e}")
