"""
Pages Package — DC Microgrid Fault Detection Platform

Exports all page render functions for use by the main app.py entry point.
"""
from src.ui.pages.dashboard import render_dashboard
from src.ui.pages.digital_twin import render_digital_twin
from src.ui.pages.wavelet_inspector import render_wavelet_inspector
from src.ui.pages.fault_analysis import render_fault_analysis
from src.ui.pages.circuit_designer import render_circuit_designer
from src.ui.pages.system_health import render_system_health
from src.ui.pages.reports import render_reports
from src.ui.pages.system_log import render_system_log

__all__ = [
    "render_dashboard",
    "render_digital_twin",
    "render_wavelet_inspector",
    "render_fault_analysis",
    "render_circuit_designer",
    "render_system_health",
    "render_reports",
    "render_system_log",
]
