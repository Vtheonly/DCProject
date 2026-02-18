"""
System Log Page — Real-time log viewer.

Shows timestamped system events, warnings, errors, and status messages.
"""
import streamlit as st


def render_system_log():
    """Render the system log page."""
    st.markdown("""
    <div class="page-header">
        <h2>📜 System Log</h2>
        <p>Real-time system events, warnings, and errors</p>
    </div>
    """, unsafe_allow_html=True)

    # Controls
    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("🗑️ Clear Log", key="btn_clear_log"):
            st.session_state.system_log = []
            st.rerun()

    # Filter
    log_filter = st.selectbox(
        "Filter", ["All", "INFO", "WARNING", "ERROR"],
        key="log_filter", label_visibility="collapsed"
    )

    # Display log entries
    entries = st.session_state.get("system_log", [])
    if not entries:
        st.info("No log entries yet. Start the system to see activity.")
        return

    # Apply filter
    if log_filter != "All":
        entries = [e for e in entries if f"[{log_filter}]" in e]

    # Show newest first
    for entry in reversed(entries[-100:]):
        css_class = "info"
        if "[WARNING]" in entry:
            css_class = "warning"
        elif "[ERROR]" in entry:
            css_class = "error"

        st.markdown(f'<div class="log-entry {css_class}">{entry}</div>',
                    unsafe_allow_html=True)
