"""
DWT Engine Agent - Industrial DC Microgrid Platform

Enhanced DWT processing with runtime configuration, energy spectrum
calculation, and detailed coefficient output for visualization.
"""
import pywt
import numpy as np
import logging
from typing import Dict, Any, List

from src.framework.base_agent import BaseAgent
from src.domain.events import WindowReadyEvent, DWTCoefficientsEvent, DWTResultEvent

logger = logging.getLogger(__name__)


class DWTEngineAgent(BaseAgent):
    """
    Enhanced Discrete Wavelet Transform Engine.
    
    Features:
    - Runtime wavelet family and level configuration
    - Energy spectrum calculation per decomposition level
    - Publishes both DWTCoefficientsEvent and DWTResultEvent
    
    Subscribes: WindowReadyEvent
    Publishes: DWTCoefficientsEvent, DWTResultEvent
    """
    
    def setup(self):
        """Initialize DWT engine with configuration."""
        # Get settings from config
        # Handle both nested dict (original design) and flat string (system.py usage)
        wavelet_config = self.config.get('wavelet', {})
        
        if isinstance(wavelet_config, str):
            self.wavelet_name = wavelet_config
            # Look for level/mode in top-level config
            self.level = self.config.get('level', 4)
            self.mode = self.config.get('mode', 'symmetric')
        else:
            self.wavelet_name = wavelet_config.get('family', 'db4')
            self.level = wavelet_config.get('level', 4)
            self.mode = wavelet_config.get('mode', 'symmetric')
        
        # Validate wavelet
        if self.wavelet_name not in pywt.wavelist():
            self.logger.warning(f"Wavelet {self.wavelet_name} not available, using db4")
            self.wavelet_name = 'db4'
        
        # Subscribe to window events
        self.subscribe(WindowReadyEvent, self.on_window)
        
        self.logger.info(f"DWT Engine initialized: {self.wavelet_name}, level={self.level}")

    def update_settings(self, wavelet: str = None, level: int = None, mode: str = None):
        """
        Update DWT settings at runtime.
        
        Args:
            wavelet: New wavelet family (e.g., 'db4', 'haar', 'sym5')
            level: New decomposition level
            mode: New signal extension mode
        """
        if wavelet is not None:
            if wavelet in pywt.wavelist():
                self.wavelet_name = wavelet
                self.logger.info(f"Wavelet updated to: {wavelet}")
            else:
                self.logger.warning(f"Invalid wavelet: {wavelet}")
                
        if level is not None:
            if 1 <= level <= 10:
                self.level = level
                self.logger.info(f"Level updated to: {level}")
                
        if mode is not None:
            self.mode = mode

    def on_window(self, event: WindowReadyEvent):
        """Process a window of data through DWT."""
        data = event.window_data
        
        if data is None or len(data) == 0:
            return
            
        try:
            # Perform DWT decomposition
            # wavedec returns [cA_n, cD_n, cD_n-1, ..., cD_1]
            coeffs = pywt.wavedec(
                data, 
                self.wavelet_name, 
                mode=self.mode, 
                level=self.level
            )
            
            # Calculate energy spectrum
            energy_levels = self._calculate_energy_spectrum(coeffs)
            
            # Publish legacy event for backward compatibility
            legacy_event = DWTCoefficientsEvent(
                source=self.name,
                coeffs=coeffs,
                wavelet=self.wavelet_name,
                window_id=event.window_id
            )
            self.publish(legacy_event)
            
            # Publish enhanced result event
            result_event = DWTResultEvent(
                source=self.name,
                coeffs=[c.tolist() if hasattr(c, 'tolist') else c for c in coeffs],
                energy_levels=energy_levels,
                wavelet=self.wavelet_name,
                level=self.level,
                window_id=event.window_id
            )
            self.publish(result_event)
            
            # Log metrics
            self.log_metric("dwt_d1_energy", energy_levels.get("D1", 0))
            
        except Exception as e:
            self.logger.error(f"DWT processing error: {e}", exc_info=True)

    def _calculate_energy_spectrum(self, coeffs: List[np.ndarray]) -> Dict[str, float]:
        """
        Calculate energy at each decomposition level.
        
        Energy = sum of squared coefficients
        
        Args:
            coeffs: List of coefficient arrays [cA_n, cD_n, ..., cD_1]
            
        Returns:
            Dictionary mapping level names to energy values
        """
        energy = {}
        
        if not coeffs:
            return energy
        
        # Approximation coefficient (last level)
        energy[f"A{self.level}"] = float(np.sum(np.square(coeffs[0])))
        
        # Detail coefficients (D_n, D_n-1, ..., D_1)
        for i, c in enumerate(coeffs[1:], 1):
            level_num = self.level - i + 1
            energy[f"D{level_num}"] = float(np.sum(np.square(c)))
        
        return energy

    def get_available_wavelets(self) -> List[str]:
        """Return list of available wavelet families."""
        return pywt.wavelist()

    def get_settings(self) -> Dict[str, Any]:
        """Get current engine settings."""
        return {
            "wavelet": self.wavelet_name,
            "level": self.level,
            "mode": self.mode
        }
