"""
Sidebar Module — DC Microgrid Fault Detection Platform

Renders the sidebar with system controls, fault injection, and
navigation links.
"""
import streamlit as st
from src.ui.system import (
    start_system, stop_system, inject_fault, clear_fault,
    add_log, get_node_ids
)


def render_sidebar():
    """Render the complete sidebar with controls and navigation."""
    with st.sidebar:
        # --- Logo & Title ---
        st.markdown("""
        <div style="text-align: center; padding: 10px 0;">
            <h2 style="color: #e94560; margin: 0;">⚡ DC Microgrid</h2>
            <p style="color: #8892b0; font-size: 12px; margin: 0;">
                Wavelet Fault Detection Platform
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.divider()

        # --- System Controls ---
        st.subheader("🎮 System Control")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("▶ Start", key="btn_start", use_container_width=True):
                start_system()
        with col2:
            if st.button("⏹ Stop", key="btn_stop", use_container_width=True):
                stop_system()

        # Status indicator
        if st.session_state.system_running:
            st.success("🟢 System Running")
            if st.session_state.dsp_available:
                st.caption("⚡ C++ DSP Fast Path Active")
        else:
            st.info("⚪ System Stopped")

        st.divider()

        # --- Navigation ---
        st.subheader("📍 Navigation")
        pages = [
            "Dashboard", "Digital Twin", "Wavelet Inspector",
            "Fault Analysis", "Circuit Designer",
            "System Health", "Reports", "System Log"
        ]
        for page in pages:
            if st.button(
                page,
                key=f"nav_{page}",
                use_container_width=True,
                type="primary" if st.session_state.current_page == page else "secondary"
            ):
                st.session_state.current_page = page
                st.rerun()

        st.divider()

        # --- Fault Injection ---
        st.subheader("⚡ Fault Injection")

        if not st.session_state.system_running:
            st.caption("Start system first to inject faults.")
        else:
            fault_type = st.selectbox(
                "Fault Type",
                ["LINE_TO_LINE", "LINE_TO_GROUND", "ARC_FAULT",
                 "NOISE", "DRIFT", "SENSOR_FAILURE"],
                key="fault_type_select"
            )
            severity = st.slider("Severity", 0.1, 1.0, 0.7, 0.05, key="fault_severity")

            # Dynamic node list from topology
            node_ids = get_node_ids()
            location = st.selectbox("Location (Node)", node_ids, key="fault_location_select")

            distance = st.number_input(
                "Distance (m)", min_value=0.0, max_value=500.0,
                value=10.0, step=5.0, key="fault_distance_input"
            )

            col1, col2 = st.columns(2)
            with col1:
                if st.button("⚡ Inject", key="btn_inject", use_container_width=True):
                    props = {"distance": distance}
                    inject_fault(fault_type, severity, location, properties=props)
            with col2:
                if st.button("🔄 Clear", key="btn_clear", use_container_width=True):
                    clear_fault()

            if st.session_state.fault_active:
                st.error(f"🔴 Active: {st.session_state.fault_type}")

        # --- Quick Info ---
        st.divider()
        st.caption("v2.0 • C++ DSP Core • Hybrid Edge Architecture")
