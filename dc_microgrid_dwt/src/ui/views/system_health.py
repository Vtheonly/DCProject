"""
System Health Page — Resource monitoring and performance metrics.

Shows CPU, memory, event throughput, latency trends,
and C++ DSP pipeline statistics.
"""
import streamlit as st
import plotly.graph_objects as go
from src.ui.styles import PLOTLY_DARK_THEME, COLORS


def render_system_health():
    """Render the system health monitoring page."""
    st.markdown("""
    <div class="page-header">
        <h2>💚 System Health</h2>
        <p>Resource usage, event throughput, latency, and DSP pipeline performance</p>
    </div>
    """, unsafe_allow_html=True)

    # --- Current Metrics ---
    health = st.session_state.health_data

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("CPU", f"{health.get('cpu', 0):.1f}%")
    with col2:
        st.metric("Memory", f"{health.get('memory', 0):.1f}%")
    with col3:
        st.metric("Events/s", f"{health.get('eps', 0):.0f}")
    with col4:
        st.metric("Latency", f"{health.get('latency', 0):.1f}ms")
    with col5:
        uptime = health.get("uptime", 0)
        mins = int(uptime // 60)
        secs = int(uptime % 60)
        st.metric("Uptime", f"{mins}m {secs}s")

    # --- C++ DSP Pipeline Stats ---
    if st.session_state.dsp_available:
        _render_dsp_stats()

    # --- Latency & EPS History ---
    _render_health_history()


def _render_dsp_stats():
    """Show C++ DSP pipeline performance stats."""
    st.markdown("#### ⚡ C++ DSP Pipeline")

    dsp = st.session_state.get("dsp_pipeline")
    if not dsp:
        st.caption("DSP pipeline not initialized.")
        return

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Samples", f"{dsp.total_samples:,}")
    with col2:
        st.metric("Total Trips", dsp.total_trips)
    with col3:
        st.metric("Avg Processing", f"{dsp.avg_processing_us:.1f}μs")


def _render_health_history():
    """Render CPU, EPS, and latency history charts."""
    history = st.session_state.get("health_history", [])
    if len(history) < 2:
        st.info("Accumulating health history...")
        return

    st.markdown("#### 📈 Performance Trends")

    col1, col2 = st.columns(2)

    with col1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=[h.get("eps", 0) for h in history[-50:]],
            mode="lines",
            name="Events/s",
            line=dict(color=COLORS["accent"], width=2),
        ))
        fig.update_layout(
            **PLOTLY_DARK_THEME,
            height=250,
            title="Events/s",
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=[h.get("latency", 0) for h in history[-50:]],
            mode="lines",
            name="Latency (ms)",
            line=dict(color=COLORS["warning"], width=2),
        ))
        fig.update_layout(
            **PLOTLY_DARK_THEME,
            height=250,
            title="Latency (ms)",
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)
