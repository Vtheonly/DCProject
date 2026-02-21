"""
Reports Page — Generate and view fault analysis reports.

Shows fault reports, daily summaries, and benchmark results.
"""
import streamlit as st


def render_reports():
    """Render the reports page."""
    st.markdown("""
    <div class="page-header">
        <h2>📋 Reports</h2>
        <p>Fault analysis reports, daily summaries, and benchmark results</p>
    </div>
    """, unsafe_allow_html=True)

    # Summary metrics
    fault_count = len(st.session_state.get("fault_events", []))
    trip_count = 1 if st.session_state.get("trip_active") else 0

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Fault Events", fault_count)
    with col2:
        st.metric("Trip Events", trip_count)
    with col3:
        dsp = st.session_state.get("dsp_pipeline")
        samples = dsp.total_samples if dsp else 0
        st.metric("Samples Processed", f"{samples:,}")

    # Fault events table
    events = st.session_state.get("fault_events", [])
    if events:
        st.markdown("#### 🔍 Fault Event History")
        import time
        import pandas as pd

        rows = []
        for evt in events:
            rows.append({
                "Time": time.strftime("%H:%M:%S", time.localtime(evt.get("time", 0))),
                "Type": evt.get("type", "—"),
                "Zone": evt.get("zone", "—"),
                "Distance (m)": f"{evt.get('distance', 0):.1f}",
            })
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("No fault events recorded — system has been running clean.")

    # DSP Performance
    if st.session_state.get("dsp_available") and st.session_state.get("dsp_pipeline"):
        dsp = st.session_state.dsp_pipeline
        st.markdown("#### ⚡ DSP Performance")
        st.markdown(f"""
        | Metric | Value |
        |--------|-------|
        | Total Samples | {dsp.total_samples:,} |
        | Total Trips | {dsp.total_trips} |
        | Avg Processing Time | {dsp.avg_processing_us:.1f}μs |
        | Architecture | C++ DSP (native) |
        """)
