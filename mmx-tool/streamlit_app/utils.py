import pandas as pd
from pathlib import Path
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data"

def load_csv(relative_path):
    path = DATA_PATH / relative_path
    if not path.exists():
        st.error(f"❌ File not found: {path}")
        st.stop()
    return pd.read_csv(path)

def load_parquet(relative_path):
    path = DATA_PATH / relative_path
    if not path.exists():
        st.error(f"❌ File not found: {path}")
        st.stop()
    return pd.read_parquet(path)
