"""
Pages Package — DC Microgrid Fault Detection Platform

Exports all page render functions for use by the main app.py entry point.
"""
from src.ui.views.dashboard import render_dashboard
from src.ui.views.digital_twin import render_digital_twin
from src.ui.views.wavelet_inspector import render_wavelet_inspector
from src.ui.views.fault_analysis import render_fault_analysis
from src.ui.views.circuit_designer import render_circuit_designer
from src.ui.views.system_health import render_system_health
from src.ui.views.reports import render_reports
from src.ui.views.system_log import render_system_log

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
