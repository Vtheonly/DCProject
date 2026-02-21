"""
DC Microgrid Fault Detection — Main Entry Point

This is a thin wrapper that initializes the session state, styles,
and renders the main sidebar and selected page. All logic is now
modularized in the `src/ui/` package.
"""
import os
import sys
import time
import streamlit as st

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.ui.state import init_session_state
from src.ui.styles import get_custom_css
from src.ui.sidebar import render_sidebar
from src.ui.system import process_events, update_component_history

# Import page renderers
from src.ui.views import (
    render_dashboard, render_digital_twin, render_wavelet_inspector,
    render_fault_analysis, render_circuit_designer, render_system_health,
    render_reports, render_system_log, render_unified_grid
)


def main():
    # 1. Page Config
    st.set_page_config(
        page_title="DC Microgrid Fault Detection",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # 2. Load Custom CSS
    st.markdown(get_custom_css(), unsafe_allow_html=True)

    # 3. Initialize Session State
    init_session_state()

    # 4. Process Background Events (Bridge -> UI State)
    process_events()
    update_component_history()

    # 5. Render Sidebar and get selected page
    page = render_sidebar()

    # 6. Routing & Rendering
    if page == "Dashboard":
        render_dashboard()
    elif page == "Unified Grid":
        render_unified_grid()
    elif page == "Digital Twin":
        render_digital_twin()
    elif page == "Wavelet Inspector":
        render_wavelet_inspector()
    elif page == "Fault Analysis":
        render_fault_analysis()
    elif page == "Circuit Designer":
        render_circuit_designer()
    elif page == "System Health":
        render_system_health()
    elif page == "Reports":
        render_reports()
    elif page == "System Log":
        render_system_log()
    else:
        render_dashboard()

    # 7. Auto-refresh for real-time feel
    # Rerun every 100ms if system is running to pull new events
    if st.session_state.system_running:
        # Smart throttle: fast refresh only when events are pending
        bridge = st.session_state.get("bridge_agent")
        if bridge and not bridge.get_queue().empty():
            time.sleep(0.05)  # Events pending — refresh quickly
        else:
            time.sleep(0.2)   # Idle — conserve CPU
        st.rerun()


if __name__ == "__main__":
    main()