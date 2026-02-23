import streamlit as st


PRIMARY = "#2563EB"
BACKGROUND = "#020617"
SURFACE = "#0F172A"
ACCENT = "#22C55E"
ERROR = "#EF4444"
TEXT = "#E5E7EB"
SUBTEXT = "#9CA3AF"


def inject_theme():
    st.set_page_config(
        page_title="Pro Face Detection Studio",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(
        f"""
        <style>
        .stApp {{
            background: radial-gradient(circle at top left, #1d4ed8 0, {BACKGROUND} 45%);
            color: {TEXT};
            font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        }}
        section[data-testid="stSidebar"] {{
            background-color: {SURFACE};
            border-right: 1px solid rgba(148, 163, 184, 0.25);
        }}
        .block-container {{
            padding-top: 1.5rem;
            padding-bottom: 2rem;
        }}
        .metric-card {{
            background: linear-gradient(135deg, #020617 0%, #111827 50%, #0f172a 100%);
            border-radius: 18px;
            padding: 1rem 1.5rem;
            border: 1px solid rgba(148, 163, 184, 0.35);
            box-shadow: 0 18px 45px rgba(15, 23, 42, 0.9);
        }}
        .metric-label {{
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.12em;
            color: {SUBTEXT};
        }}
        .metric-value {{
            font-size: 1.4rem;
            font-weight: 600;
            color: {TEXT};
        }}
        .metric-hint {{
            font-size: 0.75rem;
            color: {SUBTEXT};
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def metric_card(label: str, value: str, hint: str = ""):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-hint">{hint}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

