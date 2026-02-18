"""
Wavelet Inspector Page — DWT Coefficient & Energy Analysis.

Shows ACTUAL DWT coefficients (D1-D4, A4) from the processing pipeline,
NOT random noise. Includes energy history over time, coefficient
waveforms, and AI-generated fault diagnosis.
"""
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from src.ui.styles import PLOTLY_DARK_THEME, COLORS


def render_wavelet_inspector():
    """Render the wavelet inspector page."""
    st.markdown("""
    <div class="page-header">
        <h2>🔬 Wavelet Inspector</h2>
        <p>Real DWT coefficients (Daubechies-4) and energy spectrum analysis</p>
    </div>
    """, unsafe_allow_html=True)

    # --- Source Toggle ---
    col1, col2 = st.columns([3, 1])
    with col2:
        source = "C++ DSP" if st.session_state.dsp_available else "Python Engine"
        st.caption(f"Source: **{source}**")

    # --- Coefficient Waveforms ---
    _render_coefficient_plots()

    # --- Energy Over Time ---
    _render_energy_history()

    # --- AI Diagnosis ---
    _render_ai_diagnosis()


def _render_coefficient_plots():
    """Render actual DWT coefficient waveforms (D1-D4, A4)."""
    st.markdown("#### 📈 DWT Coefficients")

    # Get real coefficients from C++ DSP or Python pipeline
    coefficients = _get_real_coefficients()

    if not coefficients or len(coefficients) < 2:
        st.info("Waiting for DWT coefficients... Start the system and wait for data.")
        return

    # Create subplot for each level
    n_levels = min(len(coefficients), 5)
    labels = _get_level_labels(n_levels)

    fig = make_subplots(
        rows=n_levels, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=labels,
    )

    level_colors = [COLORS["a4"], COLORS["d4"], COLORS["d3"], COLORS["d2"], COLORS["d1"]]

    for i in range(n_levels):
        if i < len(coefficients):
            data = coefficients[i]
            color = level_colors[i] if i < len(level_colors) else COLORS["accent"]
            fig.add_trace(
                go.Scatter(
                    y=data,
                    mode="lines",
                    name=labels[i],
                    line=dict(color=color, width=1),
                ),
                row=i + 1, col=1
            )

    fig.update_layout(
        **PLOTLY_DARK_THEME,
        height=120 * n_levels + 60,
        showlegend=False,
        title_text="",
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_energy_history():
    """Render energy level history over time."""
    st.markdown("#### 📊 Energy History")

    history = st.session_state.get("energy_history", [])
    if not history:
        st.info("Accumulating energy history...")
        return

    fig = go.Figure()
    levels = ["D1", "D2", "D3", "D4", "A4"]
    level_colors = [COLORS["d1"], COLORS["d2"], COLORS["d3"], COLORS["d4"], COLORS["a4"]]

    for i, level in enumerate(levels):
        values = [h.get(level, 0) for h in history[-100:]]
        fig.add_trace(go.Scatter(
            y=values,
            mode="lines",
            name=level,
            line=dict(color=level_colors[i], width=1.5),
        ))

    fig.update_layout(
        **PLOTLY_DARK_THEME,
        height=300,
        xaxis_title="Time Step",
        yaxis_title="Energy",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_ai_diagnosis():
    """Render AI fault diagnosis results."""
    st.markdown("#### 🤖 AI Diagnosis")

    diagnosis = st.session_state.get("ai_diagnosis")
    causes = st.session_state.get("ai_probable_causes", [])

    if not diagnosis and not causes:
        if st.session_state.fault_active:
            st.warning("Fault detected — awaiting AI analysis...")
        else:
            st.success("System normal — no fault patterns detected.")
        return

    if diagnosis:
        st.markdown(f"""
        <div class="component-card">
            <div style="color: #e94560; font-size: 16px; font-weight: bold;">
                Diagnosis: {diagnosis}
            </div>
        </div>
        """, unsafe_allow_html=True)

    if causes:
        st.markdown("**Probable Causes:**")
        for cause in causes:
            st.markdown(f"- {cause}")


def _get_real_coefficients():
    """Get actual DWT coefficients from C++ DSP or Python pipeline.
    
    Returns list of coefficient arrays: [A4, D4, D3, D2, D1]
    (or fewer levels depending on configuration).
    """
    # Priority 1: C++ DSP pipeline coefficients
    dsp = st.session_state.get("dsp_pipeline")
    if dsp:
        try:
            coeffs = dsp.get_coefficients()
            if coeffs and len(coeffs) > 0:
                return coeffs
        except Exception:
            pass

    # Priority 2: Python pipeline coefficients from events
    coeffs = st.session_state.get("dwt_coefficients")
    if coeffs and len(coeffs) > 0:
        return coeffs

    return None


def _get_level_labels(n_levels):
    """Generate level labels like [A4, D4, D3, D2, D1]."""
    if n_levels == 5:
        return ["A4 (Approx)", "D4 (Detail)", "D3 (Detail)", "D2 (Detail)", "D1 (HF Detail)"]
    elif n_levels == 4:
        return ["A3 (Approx)", "D3 (Detail)", "D2 (Detail)", "D1 (HF Detail)"]
    else:
        labels = [f"Level {i}" for i in range(n_levels)]
        labels[0] = "Approximation"
        labels[-1] = "D1 (HF Detail)"
        return labels
