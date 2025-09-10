

# ---------- streamlit_app.py ----------
import io
import re
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from dateutil.relativedelta import relativedelta

try:
    import gdown
except Exception:
    gdown = None

st.set_page_config(page_title="Crypto KPI Dashboard", page_icon="📈", layout="wide")
st.title("📈 Crypto KPI Dashboard – SOL/USDC")
st.caption("Incolla un link Google Drive (XLSX/Google Sheets) **oppure** carica il file .xlsx. Poi premi **Calcola KPI**.")

# ---------------------- Helpers ----------------------
PAIR_DEFAULT = "SOL/USDC"

def extract_drive_id(url: str) -> str | None:
    """Estrae il fileId da vari formati di URL Google Drive/Sheets."""
    if not url:
        return None
    patterns = [
        r"https?://drive\.google\.com/file/d/([\w-]+)/",
        r"https?://drive\.google\.com/open\?id=([\w-]+)",
        r"https?://docs\.google\.com/spreadsheets/d/([\w-]+)",
        r"id=([\w-]+)"
    ]
    for p in patterns:
        m = re.search(p, url)
        if m:
            return m.group(1)
    return None

@st.cache_data(show_spinner=False)
def download_drive_xlsx(file_id: str) -> bytes:
    if gdown is None:
        raise RuntimeError("gdown non disponibile: installa i requirements e riavvia.")
    url = f"https://drive.google.com/uc?id={file_id}"
    return gdown.download(url, None, quiet=True)

# (resto invariato) ...
