"""
Circuit Designer Page — View and edit the circuit model.

Displays the circuit schematic, bus/line parameters, and allows
loading/saving circuit configurations.
"""
import streamlit as st
from src.ui.system import add_log


def render_circuit_designer():
    """Render the circuit designer page."""
    st.markdown("""
    <div class="page-header">
        <h2>⚙️ Circuit Designer</h2>
        <p>View and configure the DC microgrid circuit model</p>
    </div>
    """, unsafe_allow_html=True)

    circuit = st.session_state.get("circuit_model")
    if not circuit:
        st.info("No circuit model loaded. Start the system to auto-load the reference grid.")
        return

    # --- Circuit Overview ---
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Buses", len(circuit.buses))
    with col2:
        st.metric("Lines", len(circuit.lines))
    with col3:
        st.metric("Generators", len(circuit.generators))

    # --- Bus Table ---
    st.markdown("#### 🔌 Buses")
    import pandas as pd
    bus_rows = []
    for bus in circuit.buses:
        bus_rows.append({
            "ID": bus.id,
            "Name": bus.name,
            "Voltage (kV)": bus.voltage_kv,
            "Type": bus.type,
        })
    st.dataframe(pd.DataFrame(bus_rows), use_container_width=True, hide_index=True)

    # --- Line Table ---
    st.markdown("#### 📏 Lines")
    line_rows = []
    for line in circuit.lines:
        line_rows.append({
            "ID": line.id,
            "From": line.from_bus,
            "To": line.to_bus,
            "R (Ω)": line.r_ohm,
            "X (Ω)": line.x_ohm,
            "Length (km)": line.length_km,
        })
    st.dataframe(pd.DataFrame(line_rows), use_container_width=True, hide_index=True)

    # --- Generator Table ---
    st.markdown("#### ⚡ Generators")
    gen_rows = []
    for gen in circuit.generators:
        gen_rows.append({
            "ID": gen.id,
            "Bus": gen.bus_id,
            "P (MW)": gen.p_mw,
        })
    st.dataframe(pd.DataFrame(gen_rows), use_container_width=True, hide_index=True)

    # --- Load Table ---
    st.markdown("#### 🏭 Loads")
    load_rows = []
    for load in circuit.loads:
        load_rows.append({
            "ID": load.id,
            "Bus": load.bus_id,
            "P (MW)": load.p_mw,
            "Priority": getattr(load, "priority", "—"),
        })
    st.dataframe(pd.DataFrame(load_rows), use_container_width=True, hide_index=True)
