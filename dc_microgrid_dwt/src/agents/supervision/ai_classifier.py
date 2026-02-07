"""
AI Classifier Agent - Industrial DC Microgrid Platform

Provides AI-assisted fault diagnosis using pattern recognition
on DWT energy levels. Outputs probable causes with confidence scores.
"""
import logging
from typing import Dict, List, Any

from src.framework.base_agent import BaseAgent
from src.domain.events import (
    DWTCoefficientsEvent, DWTResultEvent, ProcessingResultEvent,
    AIAnalysisEvent, FaultDetectedEvent
)
from src.domain.models import FaultType

logger = logging.getLogger(__name__)


class AIClassifierAgent(BaseAgent):
    """
    AI-Assisted Fault Diagnosis Agent.
    
    Uses rule-based classification on DWT energy patterns to
    diagnose fault types. Expandable to ML models.
    
    Features:
    - Pattern matching on D1-D4 energy levels
    - Multi-cause probability ranking
    - Confidence scoring
    - Historical trend analysis
    
    Subscribes: DWTResultEvent, DWTCoefficientsEvent, ProcessingResultEvent
    Publishes: AIAnalysisEvent
    """
    
    def setup(self):
        """Initialize classifier with thresholds."""
        self.subscribe(DWTResultEvent, self.on_dwt_result)
        self.subscribe(ProcessingResultEvent, self.on_processing_result)
        
        # Energy history for trend analysis
        self.energy_history = []
        self.max_history = 50
        
        # Classification thresholds (tuned for typical faults)
        self.thresholds = {
            "d1_spike": self.config.get("d1_spike_threshold", 100.0),
            "d2_spike": self.config.get("d2_spike_threshold", 50.0),
            "high_freq_ratio": self.config.get("high_freq_ratio", 0.6),
            "noise_variance": self.config.get("noise_variance", 25.0)
        }
        
        self.logger.info("AI Classifier initialized")

    def on_dwt_result(self, event: DWTResultEvent):
        """Analyze DWT results for fault diagnosis."""
        try:
            energy = event.energy_levels
            
            if not energy:
                return
                
            # Store for trend analysis
            self.energy_history.append(energy)
            if len(self.energy_history) > self.max_history:
                self.energy_history.pop(0)
            
            # Perform classification
            diagnosis = self._classify(energy)
            
            # Only publish if fault probability is significant
            if diagnosis["fault_probability"] > 0.1:
                ai_event = AIAnalysisEvent(
                    source=self.name,
                    fault_probability=diagnosis["fault_probability"],
                    diagnosis=diagnosis["diagnosis"],
                    confidence=diagnosis["confidence"],
                    probable_causes=diagnosis["probable_causes"]
                )
                self.publish(ai_event)
                
        except Exception as e:
            self.logger.error(f"Classification error: {e}")

    def on_processing_result(self, event: ProcessingResultEvent):
        """Analyze processing results for quick classification."""
        if event.is_faulty:
            # Use peak energy for quick classification
            energy = {"D1": event.d1_energy, "D1_peak": event.d1_peak}
            diagnosis = self._classify(energy)
            
            if diagnosis["fault_probability"] > 0.5:
                ai_event = AIAnalysisEvent(
                    source=self.name,
                    fault_probability=diagnosis["fault_probability"],
                    diagnosis=diagnosis["diagnosis"],
                    confidence=diagnosis["confidence"],
                    probable_causes=diagnosis["probable_causes"]
                )
                self.publish(ai_event)

    def _classify(self, energy: Dict[str, float]) -> Dict[str, Any]:
        """
        Classify fault type based on energy patterns.
        
        Uses rule-based logic (can be replaced with ML model).
        """
        probable_causes = []
        fault_probability = 0.0
        diagnosis = "Normal Operation"
        confidence = 0.0
        
        d1 = energy.get("D1", energy.get("D1_peak", 0))
        d2 = energy.get("D2", 0)
        d3 = energy.get("D3", 0)
        d4 = energy.get("D4", 0)
        a4 = energy.get("A4", 0)
        
        total_detail = d1 + d2 + d3 + d4 + 0.001
        high_freq_ratio = d1 / total_detail
        
        # Rule 1: High D1 spike → Line-to-Line fault
        if d1 > self.thresholds["d1_spike"]:
            prob = min(0.95, d1 / (self.thresholds["d1_spike"] * 2))
            probable_causes.append({
                "cause": "Line-to-Line Short Circuit",
                "probability": prob
            })
            fault_probability = max(fault_probability, prob)
        
        # Rule 2: High D1 + D2 with oscillation → Arc fault
        if d1 > self.thresholds["d1_spike"] * 0.5 and d2 > self.thresholds["d2_spike"]:
            if high_freq_ratio > 0.4:
                prob = min(0.85, (d1 + d2) / 200)
                probable_causes.append({
                    "cause": "Arc Fault",
                    "probability": prob
                })
                fault_probability = max(fault_probability, prob * 0.9)
        
        # Rule 3: Gradual A4 decline → Voltage drift/sag
        if len(self.energy_history) > 10:
            trend = self._calculate_trend("A4")
            if trend < -10:  # Declining trend
                prob = min(0.7, abs(trend) / 50)
                probable_causes.append({
                    "cause": "Voltage Sag/Drift",
                    "probability": prob
                })
                fault_probability = max(fault_probability, prob * 0.8)
        
        # Rule 4: High variance across all bands → Noise
        variance = self._calculate_variance([d1, d2, d3, d4])
        if variance > self.thresholds["noise_variance"]:
            prob = min(0.6, variance / 50)
            probable_causes.append({
                "cause": "High Noise / EMI",
                "probability": prob
            })
            fault_probability = max(fault_probability, prob * 0.7)
        
        # Rule 5: Sudden drop in all energy → Sensor failure
        if total_detail < 1.0 and len(self.energy_history) > 5:
            avg_past = sum(
                sum(h.values()) for h in self.energy_history[-5:]
            ) / 5
            if avg_past > 50:  # Was normal, now near zero
                probable_causes.append({
                    "cause": "Sensor Failure",
                    "probability": 0.8
                })
                fault_probability = max(fault_probability, 0.75)
        
        # Sort by probability
        probable_causes.sort(key=lambda x: x["probability"], reverse=True)
        
        # Determine diagnosis and confidence
        if probable_causes:
            diagnosis = probable_causes[0]["cause"]
            # Confidence based on probability spread
            if len(probable_causes) == 1:
                confidence = probable_causes[0]["probability"]
            else:
                # Higher confidence if top cause is clearly dominant
                confidence = probable_causes[0]["probability"] - probable_causes[1]["probability"] / 2
                confidence = max(0.1, min(1.0, confidence))
        else:
            diagnosis = "Normal Operation"
            confidence = 1.0 - fault_probability
        
        return {
            "fault_probability": fault_probability,
            "diagnosis": diagnosis,
            "confidence": confidence,
            "probable_causes": probable_causes
        }

    def _calculate_trend(self, key: str) -> float:
        """Calculate trend (slope) for a specific energy key."""
        if len(self.energy_history) < 5:
            return 0.0
            
        values = [h.get(key, 0) for h in self.energy_history[-10:]]
        if len(values) < 2:
            return 0.0
            
        # Simple linear regression slope
        n = len(values)
        x_mean = (n - 1) / 2
        y_mean = sum(values) / n
        
        numerator = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(values))
        denominator = sum((i - x_mean) ** 2 for i in range(n))
        
        if denominator == 0:
            return 0.0
            
        return numerator / denominator

    def _calculate_variance(self, values: List[float]) -> float:
        """Calculate variance of a list of values."""
        if not values:
            return 0.0
        mean = sum(values) / len(values)
        return sum((v - mean) ** 2 for v in values) / len(values)
