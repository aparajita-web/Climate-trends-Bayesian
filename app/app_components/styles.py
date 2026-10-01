from pathlib import Path
import streamlit as st


DECADAL_COLORS = {
    "navy": "#121358",
    "primary": "#232F72",
    "blue": "#2F578A",
    "teal": "#36ADA3",
    "climatology": "#7B86A1",

    "background": "#121358",
    "surface": "#232F72",
    "surface_2": "#2F578A",
    "surface_3": "#192066",

    "text": "#F4F7FA",
    "text_secondary": "#D1D8E8",
    "muted": "#9FAAC1",
    "border": "rgba(255,255,255,0.13)",
    "axis": "#7F8BA7",

    "plot_grid": "rgba(255,255,255,0.10)",
}


MODEL_COLORS = {
    "sarima": "#f2ff43",
    "gb": "#41f718",
    "clim": "#fb0e0e",
}







def load_css():
    css_path = Path(__file__).resolve().parents[1] / "styles.css"

    css = css_path.read_text(encoding="utf-8")

    st.markdown(
        f"<style>{css}</style>",
        unsafe_allow_html=True,
    )
