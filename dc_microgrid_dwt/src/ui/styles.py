"""
CSS Styles Module — DC Microgrid Fault Detection Platform

Contains all custom CSS styling for the Streamlit application.
Separated from app.py for maintainability.
"""


def get_custom_css() -> str:
    """Return the complete CSS stylesheet for the application."""
    return """
    <style>
        /* ====== GLOBAL THEME ====== */
        .stApp {
            background: linear-gradient(135deg, #0a0a1a 0%, #1a1a2e 50%, #16213e 100%);
        }

        /* ====== METRIC CARDS ====== */
        .metric-card {
            background: linear-gradient(135deg, #1a1a3e 0%, #2a2a5e 100%);
            border: 1px solid rgba(233, 69, 96, 0.3);
            border-radius: 15px;
            padding: 20px;
            text-align: center;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }
        .metric-card:hover {
            transform: translateY(-3px);
            box-shadow: 0 8px 25px rgba(233, 69, 96, 0.2);
        }
        .metric-value {
            font-size: 32px;
            font-weight: bold;
            background: linear-gradient(135deg, #e94560, #ff6b6b);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin: 5px 0;
        }
        .metric-label {
            font-size: 12px;
            color: #8892b0;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        .metric-unit {
            font-size: 11px;
            color: #666;
        }

        /* ====== STATUS BADGES ====== */
        .status-badge {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: bold;
            text-transform: uppercase;
        }
        .status-normal {
            background: rgba(76, 175, 80, 0.2);
            color: #4caf50;
            border: 1px solid rgba(76, 175, 80, 0.3);
        }
        .status-fault {
            background: rgba(233, 69, 96, 0.2);
            color: #e94560;
            border: 1px solid rgba(233, 69, 96, 0.3);
            animation: pulse-red 1.5s infinite;
        }
        .status-warning {
            background: rgba(255, 152, 0, 0.2);
            color: #ff9800;
            border: 1px solid rgba(255, 152, 0, 0.3);
        }

        /* ====== FAULT ALERT BANNER ====== */
        .fault-alert {
            background: linear-gradient(135deg, rgba(233, 69, 96, 0.15), rgba(255, 107, 107, 0.1));
            border: 1px solid rgba(233, 69, 96, 0.4);
            border-radius: 12px;
            padding: 15px 20px;
            margin: 10px 0;
            animation: pulse-border 2s infinite;
        }

        /* ====== LOG VIEWER ====== */
        .log-entry {
            font-family: 'JetBrains Mono', 'Fira Code', monospace;
            font-size: 11px;
            padding: 3px 8px;
            border-left: 3px solid #333;
            margin: 2px 0;
            color: #a0a0a0;
        }
        .log-entry.warning {
            border-left-color: #ff9800;
            color: #ffb74d;
        }
        .log-entry.error {
            border-left-color: #e94560;
            color: #ff6b6b;
        }
        .log-entry.info {
            border-left-color: #4caf50;
            color: #81c784;
        }

        /* ====== PAGE HEADER ====== */
        .page-header {
            background: linear-gradient(135deg, #1a1a3e 0%, #2a2a5e 100%);
            border: 1px solid rgba(233, 69, 96, 0.2);
            border-radius: 12px;
            padding: 20px 25px;
            margin-bottom: 20px;
        }
        .page-header h2 {
            margin: 0;
            color: #e94560;
        }
        .page-header p {
            margin: 5px 0 0 0;
            color: #8892b0;
            font-size: 14px;
        }

        /* ====== SIDEBAR ====== */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0a0a1a 0%, #1a1a2e 100%);
            border-right: 1px solid rgba(233, 69, 96, 0.2);
        }

        /* ====== ANIMATIONS ====== */
        @keyframes pulse-red {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.7; }
        }
        @keyframes pulse-border {
            0%, 100% { border-color: rgba(233, 69, 96, 0.4); }
            50% { border-color: rgba(233, 69, 96, 0.8); }
        }

        /* ====== GRAPH CONTAINERS ====== */
        .graph-container {
            background: rgba(26, 26, 62, 0.5);
            border: 1px solid rgba(233, 69, 96, 0.15);
            border-radius: 10px;
            padding: 10px;
            margin: 8px 0;
        }

        /* ====== COMPONENT CARD ====== */
        .component-card {
            background: linear-gradient(135deg, #1a1a3e 0%, #252550 100%);
            border: 1px solid rgba(100, 100, 200, 0.2);
            border-radius: 10px;
            padding: 15px;
            margin: 5px 0;
        }
    </style>
    """


# Common Plotly layout theme for consistent graph styling
PLOTLY_DARK_THEME = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(10,10,30,0.5)",
    font=dict(color="#8892b0", family="Inter, sans-serif"),
    margin=dict(l=40, r=20, t=40, b=30),
    xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
    yaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
)

# Color palette for graphs
COLORS = {
    "primary": "#e94560",
    "secondary": "#ff6b6b",
    "accent": "#4fc3f7",
    "success": "#4caf50",
    "warning": "#ff9800",
    "danger": "#e94560",
    "d1": "#e94560",
    "d2": "#ff6b6b",
    "d3": "#4fc3f7",
    "d4": "#7c4dff",
    "a4": "#4caf50",
    "voltage": "#00bcd4",
    "current": "#ff9800",
}
