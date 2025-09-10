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

def _find_col(df: pd.DataFrame, candidates: List[str]) -> str:
    clean = {re.sub(r"\s+", "", c).lower(): c for c in df.columns}
    for cand in candidates:
        key = re.sub(r"\s+", "", cand).lower()
        if key in clean:
            return clean[key]
    for c in df.columns:
        normalized = re.sub(r"\s+", "", c).lower()
        if any(re.sub(r"\s+", "", cand).lower() in normalized for cand in candidates):
            return c
    raise KeyError(f"Colonna non trovata: {candidates}")

def sanitize_orders(df_raw: pd.DataFrame) -> pd.DataFrame:
    type_col = _find_col(df_raw, ["Type"])
    df = df_raw[df_raw[type_col].isin(["BUY", "SELL"])].copy()

    col_date = _find_col(df, ["Date(UTC)", "Date"])
    col_pair = _find_col(df, ["Pair"])
    col_filled = _find_col(df, ["Filled"])
    col_avg_price = _find_col(df, ["AvgTrading Price", "AvgTradingPrice", "Trading Price", "Price"])
    col_total = _find_col(df, ["Total"])

    df.rename(columns={
        col_date: "date",
        col_pair: "pair",
        type_col: "type",
        col_filled: "qty",
        col_avg_price: "price",
        col_total: "total"
    }, inplace=True)

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["qty"] = pd.to_numeric(df["qty"], errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["total"] = pd.to_numeric(df["total"], errors="coerce")

    df = df.dropna(subset=["date", "qty", "price"]).copy()
    df = df.sort_values("date").reset_index(drop=True)
    return df

def fifo_trades(orders: pd.DataFrame, pair: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    df = orders[orders["pair"] == pair].copy()
    if df.empty:
        return [], {"start": None, "end": None, "totalMoved": 0.0}

    buy_q: List[Dict[str, Any]] = []
    trades: List[Dict[str, Any]] = []

    for _, o in df.iterrows():
        if o["type"] == "BUY":
            buy_q.append(o.to_dict())
        elif o["type"] == "SELL":
            sell_qty = float(o["qty"])
            while sell_qty > 1e-12 and buy_q:
                b = buy_q[0]
                take = float(min(b["qty"], sell_qty))
                pnl = (float(o["price"]) - float(b["price"])) * take
                size_usdc = take * float(b["price"])  # size valorizzata a prezzo di carico
                dur_min = (o["date"] - b["date"]).total_seconds() / 60.0
                trades.append({
                    "buy_date": b["date"],
                    "sell_date": o["date"],
                    "qty": take,
                    "buy_price": float(b["price"]),
                    "sell_price": float(o["price"]),
                    "size_usdc": size_usdc,
                    "pnl": pnl,
                    "dur_min": dur_min,
                })
                b["qty"] -= take
                sell_qty -= take
                if b["qty"] <= 1e-12:
                    buy_q.pop(0)

    start = df["date"].min()
    end = df["date"].max()
    total_moved = float(df["total"].abs().sum())
    return trades, {"start": start, "end": end, "totalMoved": total_moved}

def compute_kpis(trades: List[Dict[str, Any]], meta: Dict[str, Any], residual: Dict[str, float] | None) -> Dict[str, Any]:
    total_pnl = float(sum(t["pnl"] for t in trades))
    trades_n = len(trades)
    best = max(trades, key=lambda t: t["pnl"]) if trades else None
    worst = min(trades, key=lambda t: t["pnl"]) if trades else None
    avg_size = float(np.mean([t["size_usdc"] for t in trades])) if trades else 0.0
    avg_pnl = float(np.mean([t["pnl"] for t in trades])) if trades else 0.0
    success_rate = float(100.0 * sum(1 for t in trades if t["pnl"] > 0) / trades_n) if trades_n else 0.0
    avg_dur = float(np.mean([t["dur_min"] for t in trades])) if trades else 0.0

    residual_block = {"cost": 0.0, "value": 0.0, "pnl": 0.0}
    if residual and residual.get("qty"):
        residual_block["cost"] = residual["qty"] * residual.get("avgCost", 0)
        residual_block["value"] = residual["qty"] * residual.get("targetPrice", 0)
        residual_block["pnl"] = residual_block["value"] - residual_block["cost"]

    total_with_residual = total_pnl + residual_block["pnl"]

    start = meta.get("start")
    end = meta.get("end")
    if start and end:
        months = (end.year - start.year) * 12 + (end.month - start.month) + 1
    else:
        months = 1
    monthly_avg = total_with_residual / max(1, months)

    # Ultimi 10 giorni basati sulla SELL date
    last10 = []
    last10_pnl = 0.0
    if end:
        cutoff = end - timedelta(days=10)
        last10 = [t for t in trades if t["sell_date"] >= cutoff]
        last10_pnl = float(sum(t["pnl"] for t in last10))

    return {
        "period": {"start": start, "end": end},
        "pnl": {"totalWithResidual": round(total_with_residual, 2), "residual": residual_block},
        "bestTrade": best,
        "worstTrade": worst,
        "counts": {"trades": trades_n, "successRatePct": round(success_rate, 2)},
        "sizes": {"avgTradeUSDC": round(avg_size, 2), "totalUSDCMoved": round(meta.get("totalMoved", 0.0), 2)},
        "perTrade": {"avgPnl": round(avg_pnl, 2), "avgDurationMin": round(avg_dur, 2)},
        "last10d": {"pnl": round(last10_pnl, 2), "trades": len(last10)},
        "monthlyAvgGain": round(monthly_avg, 2),
        "trades": trades,
    }

def format_number(x: float) -> str:
    return f"{x:,.2f}".replace(",", "@").replace(".", ",").replace("@", ".")

# ---------------------- UI ----------------------
st.sidebar.header("⚙️ Input")
drive_url = st.sidebar.text_input("Link Google Drive/Sheets del file XLSX", placeholder="https://docs.google.com/spreadsheets/d/.../edit")
uploaded = st.sidebar.file_uploader("oppure carica .xlsx", type=["xlsx"])
pair = st.sidebar.text_input("Coppia", value=PAIR_DEFAULT)
st.sidebar.divider()
st.sidebar.subheader("Residuo (opzionale)")
res_qty = st.sidebar.number_input("Qty residua", value=15.0, step=1.0)
res_avg = st.sidebar.number_input("Prezzo medio residuo", value=201.0, step=0.1)
res_target = st.sidebar.number_input("Prezzo target residuo", value=220.0, step=0.1)
st.sidebar.divider()
go = st.sidebar.button("Calcola KPI", type="primary")

if not go:
    st.info("⬅️ Inserisci link o carica il file e premi **Calcola KPI** dal menu a sinistra.")
else:
    # --- Caricamento file ---
    xlsx_bytes: bytes | None = None
    try:
        if uploaded is not None:
            xlsx_bytes = uploaded.read()
        elif drive_url:
            file_id = extract_drive_id(drive_url)
            if not file_id:
                st.error("Non riesco a estrarre il fileId dal link Drive. Controlla il formato del link.")
                st.stop()
            x = download_drive_xlsx(file_id)
            if isinstance(x, bytes):
                xlsx_bytes = x
            elif isinstance(x, str):
                with open(x, "rb") as f:
                    xlsx_bytes = f.read()
        else:
            st.warning("Inserisci un link Drive oppure carica un file .xlsx")
            st.stop()
    except Exception as e:
        st.error("Errore nel download del file da Drive")
        st.exception(e)
        st.stop()

    # --- Parsing ---
    try:
        xls = pd.ExcelFile(io.BytesIO(xlsx_bytes))
        sheet = "sheet1" if "sheet1" in xls.sheet_names else xls.sheet_names[0]
        raw = pd.read_excel(xls, sheet_name=sheet)
        orders = sanitize_orders(raw)
    except Exception as e:
        st.error("Errore nel parsing del file. Verifica che sia l'export ordini Binance.")
        st.exception(e)
        st.stop()

    # --- Calcolo trades e KPI ---
    trades, meta = fifo_trades(orders, pair)
    residual_cfg = {"qty": res_qty, "avgCost": res_avg, "targetPrice": res_target}
    result = compute_kpis(trades, meta, residual_cfg)

    # --- Output KPI ---
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("PNL totale (con residuo)", f"{format_number(result['pnl']['totalWithResidual'])} USDC")
        st.metric("Numero di trade", result["counts"]["trades"])
        st.metric("Success rate", f"{format_number(result['counts']['successRatePct'])}%")
    with col2:
        st.metric("Size media per trade", f"{format_number(result['sizes']['avgTradeUSDC'])} USDC")
        st.metric("PNL medio per trade", f"{format_number(result['perTrade']['avgPnl'])} USDC")
        st.metric("Totale USDC mossi", f"{format_number(result['sizes']['totalUSDCMoved'])} USDC")
    with col3:
        start = result['period']['start']
        end = result['period']['end']
        st.metric("Data inizio", start.strftime('%d/%m/%Y %H:%M') if start else '-')
        st.metric("Data fine", end.strftime('%d/%m/%Y %H:%M') if end else '-')
        st.metric("Guadagno mensile medio", f"{format_number(result['monthlyAvgGain'])} USDC")

    st.divider()

    # Miglior/Peggior trade
    b = result.get("bestTrade")
    w = result.get("worstTrade")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("🥇 Miglior trade")
        if b:
            st.write(f"PnL: **{format_number(b['pnl'])} USDC**  |  Size: {format_number(b['size_usdc'])} USDC")
            st.write(f"Buy: {pd.to_datetime(b['buy_date']).strftime('%d/%m/%Y %H:%M')}  →  Sell: {pd.to_datetime(b['sell_date']).strftime('%d/%m/%Y %H:%M')}")
            st.write(f"Durata: ~{format_number(b['dur_min'])} min")
        else:
            st.write("-")
    with c2:
        st.subheader("🥶 Peggior trade")
        if w:
            st.write(f"PnL: **{format_number(w['pnl'])} USDC**  |  Size: {format_number(w['size_usdc'])} USDC")
            st.write(f"Buy: {pd.to_datetime(w['buy_date']).strftime('%d/%m/%Y %H:%M')}  →  Sell: {pd.to_datetime(w['sell_date']).strftime('%d/%m/%Y %H:%M')}")
            st.write(f"Durata: ~{format_number(w['dur_min'])} min")
        else:
            st.write("-")

    st.divider()

    # Ultimi 10 giorni
    st.subheader("🗓️ Ultimi 10 giorni")
    end = result['period']['end']
    if end:
        cutoff = end - timedelta(days=10)
        last10 = [t for t in result['trades'] if t['sell_date'] >= cutoff]
        df10 = pd.DataFrame(last10)
        if not df10.empty:
            df10 = df10[["buy_date","sell_date","qty","buy_price","sell_price","size_usdc","pnl","dur_min"]].copy()
            df10.rename(columns={
                "buy_date":"Buy",
                "sell_date":"Sell",
                "qty":"Qty (SOL)",
                "buy_price":"Prezzo Buy",
                "sell_price":"Prezzo Sell",
                "size_usdc":"Size (USDC)",
                "pnl":"PnL (USDC)",
                "dur_min":"Durata (min)"
            }, inplace=True)
            st.dataframe(df10, use_container_width=True)
            st.info(f"PnL ultimi 10gg: **{format_number(result['last10d']['pnl'])} USDC** su {len(last10)} trade")
        else:
            st.write("Nessun trade chiuso negli ultimi 10 giorni.")
    else:
        st.write("Periodo non disponibile.")

    # Download JSON
    st.download_button("⬇️ Scarica JSON KPI", data=pd.Series(result).to_json(), file_name="kpi_report.json", mime="application/json")
