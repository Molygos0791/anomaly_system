"""Streamlit entry point: open realtime monitoring first."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

st.switch_page("pages/1_📊_实时监控.py")
