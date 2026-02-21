import warnings
import time
from datetime import datetime

warnings.filterwarnings(“ignore”)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use(“Agg”)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import streamlit as st
import yfinance as yf
import ta

# — constants —

PERIODS   = [“1mo”, “3mo”, “6mo”, “1y”, “2y”, “5y”]
INTERVALS = [“1d”, “1wk”, “1mo”]
PERIOD_LBL = {
“1mo”: “1M”, “3mo”: “3M”, “6mo”: “6M”,
“1y”:  “1Y”, “2y”:  “2Y”, “5y”:  “5Y”
}
INTERVAL_LBL = {“1d”: “Daily”, “1wk”: “Weekly”, “1mo”: “Monthly”}
UPDATE_OPTS  = {“5min”: 300, “10min”: 600, “15min”: 900, “30min”: 1800, “1hr”: 3600}

POPULAR = [
(“7203.T”, “Toyota”),  (“6758.T”, “Sony”),   (“9984.T”, “SBG”),
(“6861.T”, “Keyence”), (“8306.T”, “MUFG”),   (”^N225”,  “Nikkei225”),
(“AAPL”,   “Apple”),   (“NVDA”,   “NVIDIA”),  (”^GSPC”,  “S&P500”),
(“USDJPY=X”, “USDJPY”),
]

C = {
“bg”:    “#0d1117”, “panel”: “#161b22”, “grid”:  “#21262d”,
“text”:  “#e6edf3”, “sub”:   “#8b949e”,
“buy”:   “#3fb950”, “sell”:  “#f85149”, “neutral”: “#58a6ff”,
“sma25”: “#ffa657”, “sma75”: “#58a6ff”, “sma200”: “#bc8cff”,
“bb”:    “#388bfd”, “macd”:  “#58a6ff”, “msig”:  “#ffa657”,
“hup”:   “#3fb950”, “hdn”:   “#f85149”, “rsi”:   “#d2a8ff”,
“cup”:   “#3fb950”, “cdn”:   “#f85149”,
}

# — data fetch & analysis (cached 60s) —

@st.cache_data(ttl=60, show_spinner=False)
def fetch_and_analyze(code, period, interval):
try:
df = yf.download(code, period=period, interval=interval,
auto_adjust=True, progress=False)
if df is None or df.empty or len(df) < 30:
return None
df = df.dropna(subset=[“Close”, “Open”, “High”, “Low”, “Volume”])

```
    close = df["Close"].squeeze()
    high  = df["High"].squeeze()
    low   = df["Low"].squeeze()
    vol   = df["Volume"].squeeze()

    df["SMA25"]  = close.rolling(25).mean()
    df["SMA75"]  = close.rolling(75).mean()
    df["SMA200"] = close.rolling(200).mean()

    bb = ta.volatility.BollingerBands(close, 20, 2)
    df["BB_upper"] = bb.bollinger_hband()
    df["BB_mid"]   = bb.bollinger_mavg()
    df["BB_lower"] = bb.bollinger_lband()
    df["BB_width"] = (df["BB_upper"] - df["BB_lower"]) / df["BB_mid"]

    m = ta.trend.MACD(close, 26, 12, 9)
    df["MACD"]      = m.macd()
    df["MACD_sig"]  = m.macd_signal()
    df["MACD_hist"] = m.macd_diff()

    df["RSI"]     = ta.momentum.RSIIndicator(close, 14).rsi()
    df["ATR"]     = ta.volatility.AverageTrueRange(high, low, close, 14).average_true_range()
    adx           = ta.trend.ADXIndicator(high, low, close, 14)
    df["ADX"]     = adx.adx()
    st_obj        = ta.momentum.StochasticOscillator(high, low, close, 14, 3)
    df["Stoch_K"] = st_obj.stoch()
    df["Stoch_D"] = st_obj.stoch_signal()
    df["Vol_MA20"]= vol.rolling(20).mean()

    s = df
    mxu = (s["MACD"] > s["MACD_sig"]) & (s["MACD"].shift(1) <= s["MACD_sig"].shift(1))
    mxd = (s["MACD"] < s["MACD_sig"]) & (s["MACD"].shift(1) >= s["MACD_sig"].shift(1))
    sxu = (s["Stoch_K"] > s["Stoch_D"]) & (s["Stoch_K"].shift(1) <= s["Stoch_D"].shift(1))
    sxd = (s["Stoch_K"] < s["Stoch_D"]) & (s["Stoch_K"].shift(1) >= s["Stoch_D"].shift(1))

    buy_sc = (
        (s["SMA25"] > s["SMA75"]).astype(int) +
        mxu.astype(int) * 2 +
        ((s["RSI"] > 30) & (s["RSI"].shift(1) <= 30)).astype(int) +
        (close <= s["BB_lower"] * 1.01).astype(int) +
        (s["ADX"] > 25).astype(int) +
        (sxu & (s["Stoch_K"] < 25)).astype(int)
    )
    sell_sc = (
        (s["SMA25"] < s["SMA75"]).astype(int) +
        mxd.astype(int) * 2 +
        ((s["RSI"] < 70) & (s["RSI"].shift(1) >= 70)).astype(int) +
        (close >= s["BB_upper"] * 0.99).astype(int) +
        (s["ADX"] > 25).astype(int) +
        (sxd & (s["Stoch_K"] > 75)).astype(int)
    )

    df["buy_sc"]  = buy_sc
    df["sell_sc"] = sell_sc
    df["signal"]  = 0
    df.loc[buy_sc  >= 3, "signal"] =  1
    df.loc[sell_sc >= 3, "signal"] = -1

    df["trend"] = "Range"
    df.loc[
        (s["SMA25"] > s["SMA75"]) & (s["SMA75"] > s["SMA200"]) & (s["ADX"] > 20),
        "trend"
    ] = "Uptrend"
    df.loc[
        (s["SMA25"] < s["SMA75"]) & (s["SMA75"] < s["SMA200"]) & (s["ADX"] > 20),
        "trend"
    ] = "Downtrend"

    return {"df": df, "fetched_at": datetime.now().strftime("%Y/%m/%d %H:%M:%S")}
except Exception as e:
    return {"error": str(e)}
```

def sig_label(v):
if v == 1:
return “BUY”
elif v == -1:
return “SELL”
return “NEUTRAL”

# — chart —

def draw_candles(ax, df):
op = df[“Open”].squeeze().values
hi = df[“High”].squeeze().values
lo = df[“Low”].squeeze().values
cl = df[“Close”].squeeze().values
for i in range(len(df)):
col = C[“cup”] if cl[i] >= op[i] else C[“cdn”]
ax.plot([i, i], [lo[i], hi[i]], color=col, lw=0.7, zorder=2)
b0 = min(op[i], cl[i])
b1 = max(op[i], cl[i])
ax.bar(i, max(b1 - b0, 1e-6), bottom=b0, width=0.6,
color=col, linewidth=0, zorder=3)

def make_chart(df, title, mobile=False):
w, h = (9, 14) if mobile else (16, 13)
fig = plt.figure(figsize=(w, h), facecolor=C[“bg”])
fig.suptitle(title, fontsize=9 if mobile else 12, color=C[“text”],
fontweight=“bold”, x=0.02, ha=“left”, y=0.99)
gs = gridspec.GridSpec(
5, 1, figure=fig,
height_ratios=[4, 1, 1.3, 1.3, 1.3],
hspace=0.05, top=0.96, bottom=0.04,
left=0.10 if mobile else 0.07, right=0.97
)
axes = [fig.add_subplot(gs[i]) for i in range(5)]
for ax in axes:
ax.set_facecolor(C[“panel”])
ax.tick_params(colors=C[“sub”], labelsize=6 if mobile else 7.5)
for sp in ax.spines.values():
sp.set_edgecolor(C[“grid”])
ax.grid(axis=“y”, color=C[“grid”], lw=0.5, ls=”–”, alpha=0.7)
ax.grid(axis=“x”, color=C[“grid”], lw=0.3, alpha=0.4)

```
x  = np.arange(len(df))
ds = df.index.strftime("%m/%d").tolist()
step = max(1, len(df) // (6 if mobile else 10))
tks  = list(range(0, len(df), step))
for ax in axes:
    ax.set_xlim(-1, len(df))
    ax.set_xticks(tks)
for ax in axes[:-1]:
    ax.set_xticklabels([])
axes[-1].set_xticklabels(
    [ds[i] for i in tks], rotation=35, ha="right",
    fontsize=5 if mobile else 6.5
)

cv = df["Close"].squeeze().values

# panel 1: candle + MA + BB + signals
ax1 = axes[0]
draw_candles(ax1, df)
ax1.plot(x, df["SMA25"].values,  color=C["sma25"],  lw=1.0, label="SMA25")
ax1.plot(x, df["SMA75"].values,  color=C["sma75"],  lw=1.2, label="SMA75")
ax1.plot(x, df["SMA200"].values, color=C["sma200"], lw=1.3, ls="--", label="SMA200")
ax1.fill_between(x, df["BB_upper"].values, df["BB_lower"].values,
                 alpha=0.1, color=C["bb"])
ax1.plot(x, df["BB_upper"].values, color=C["bb"], lw=0.7, ls=":", alpha=0.8)
ax1.plot(x, df["BB_lower"].values, color=C["bb"], lw=0.7, ls=":", alpha=0.8)

bx = [df.index.get_loc(i) for i in df.index[df["signal"] == 1]]
sx = [df.index.get_loc(i) for i in df.index[df["signal"] == -1]]
by = [float(df.iloc[i]["Low"].squeeze())  * 0.985 for i in bx]
sy = [float(df.iloc[i]["High"].squeeze()) * 1.015 for i in sx]
ms = 60 if mobile else 100
ax1.scatter(bx, by, marker="^", s=ms, color=C["buy"],  zorder=6, label="BUY")
ax1.scatter(sx, sy, marker="v", s=ms, color=C["sell"], zorder=6, label="SELL")

tc = {"Uptrend": "#3fb95015", "Downtrend": "#f8514915", "Range": "none"}
prev, start = None, 0
for i, (_, row) in enumerate(df.iterrows()):
    t = row["trend"]
    if t != prev:
        if prev and tc[prev] != "none":
            ax1.axvspan(start, i, color=tc[prev], lw=0, zorder=0)
        start, prev = i, t
if prev and tc[prev] != "none":
    ax1.axvspan(start, len(df), color=tc[prev], lw=0, zorder=0)

tnow = df["trend"].iloc[-1]
tclr = {"Uptrend": C["buy"], "Downtrend": C["sell"], "Range": C["neutral"]}
ax1.text(0.995, 0.97, tnow, transform=ax1.transAxes,
         ha="right", va="top",
         fontsize=8 if mobile else 10,
         fontweight="bold", color=tclr[tnow])
ax1.set_ylabel("Price", color=C["sub"], fontsize=6)
ax1.legend(loc="upper left", fontsize=5.5, framealpha=0.3,
           facecolor=C["bg"], edgecolor=C["grid"], labelcolor=C["text"])

# panel 2: volume
ax2 = axes[1]
vc = [
    C["cup"] if cv[i] >= df["Open"].squeeze().values[i] else C["cdn"]
    for i in range(len(df))
]
ax2.bar(x, df["Volume"].squeeze().values, color=vc, alpha=0.7, width=0.8)
ax2.plot(x, df["Vol_MA20"].values, color=C["sma25"], lw=0.9)
ax2.set_ylabel("Volume", color=C["sub"], fontsize=6)
ax2.yaxis.set_major_formatter(
    plt.FuncFormatter(lambda v, _: f"{v/1e6:.0f}M" if v >= 1e6 else f"{int(v/1e3)}K")
)

# panel 3: MACD
ax3 = axes[2]
hist = df["MACD_hist"].values
ax3.bar(x, hist,
        color=[C["hup"] if h >= 0 else C["hdn"] for h in hist],
        alpha=0.8, width=0.8)
ax3.plot(x, df["MACD"].values,     color=C["macd"], lw=1.0, label="MACD")
ax3.plot(x, df["MACD_sig"].values, color=C["msig"], lw=0.9, ls="--", label="Signal")
ax3.axhline(0, color=C["grid"], lw=0.7)
ax3.set_ylabel("MACD", color=C["sub"], fontsize=6)
ax3.legend(loc="upper left", fontsize=5.5, framealpha=0.2,
           facecolor=C["bg"], edgecolor="none", labelcolor=C["text"])

# panel 4: RSI
ax4 = axes[3]
ax4.plot(x, df["RSI"].values, color=C["rsi"], lw=1.1)
ax4.axhline(70, color=C["sell"], lw=0.8, ls="--")
ax4.axhline(50, color=C["grid"], lw=0.5)
ax4.axhline(30, color=C["buy"],  lw=0.8, ls="--")
ax4.fill_between(x, df["RSI"].values, 70,
                 where=df["RSI"].values >= 70, alpha=0.2, color=C["sell"])
ax4.fill_between(x, df["RSI"].values, 30,
                 where=df["RSI"].values <= 30, alpha=0.2, color=C["buy"])
ax4.set_ylim(0, 100)
ax4.set_ylabel("RSI", color=C["sub"], fontsize=6)

# panel 5: stochastic
ax5 = axes[4]
ax5.plot(x, df["Stoch_K"].values, color=C["macd"], lw=1.1, label="%K")
ax5.plot(x, df["Stoch_D"].values, color=C["msig"], lw=0.9, ls="--", label="%D")
ax5.axhline(80, color=C["sell"], lw=0.8, ls="--")
ax5.axhline(20, color=C["buy"],  lw=0.8, ls="--")
ax5.fill_between(x, df["Stoch_K"].values, 80,
                 where=df["Stoch_K"].values >= 80, alpha=0.15, color=C["sell"])
ax5.fill_between(x, df["Stoch_K"].values, 20,
                 where=df["Stoch_K"].values <= 20, alpha=0.15, color=C["buy"])
ax5.set_ylim(0, 100)
ax5.set_ylabel("Stoch", color=C["sub"], fontsize=6)
ax5.legend(loc="upper left", fontsize=5.5, framealpha=0.2,
           facecolor=C["bg"], edgecolor="none", labelcolor=C["text"])

plt.close("all")
return fig
```

# — Streamlit UI —

st.set_page_config(
page_title=“Trend Signal”,
page_icon=“📈”,
layout=“wide”,
initial_sidebar_state=“collapsed”,
)

st.markdown(”””

<style>
.stApp,[data-testid="stAppViewContainer"]{background:#0d1117!important;color:#e6edf3;}
section[data-testid="stSidebar"]{background:#161b22!important;}
[data-testid="stHeader"]{background:#0d1117!important;}
.badge-buy{background:#1a3d24;color:#3fb950;border:1.5px solid #3fb950;
  border-radius:10px;padding:7px 16px;font-size:1.05rem;font-weight:bold;display:inline-block;}
.badge-sell{background:#3d1a1a;color:#f85149;border:1.5px solid #f85149;
  border-radius:10px;padding:7px 16px;font-size:1.05rem;font-weight:bold;display:inline-block;}
.badge-neutral{background:#1a1f3d;color:#58a6ff;border:1.5px solid #58a6ff;
  border-radius:10px;padding:7px 16px;font-size:1.05rem;font-weight:bold;display:inline-block;}
[data-testid="stMetricValue"]{color:#e6edf3!important;font-size:1.15rem!important;}
[data-testid="metric-container"]{background:#161b22;border-radius:10px;
  border:1px solid #21262d;padding:10px 14px;}
.stButton>button{background:#21262d;color:#e6edf3;border:1px solid #30363d;
  border-radius:8px;font-weight:600;}
.stButton>button:hover{background:#388bfd22;border-color:#58a6ff;}
.stTextInput>div>div>input{background:#161b22!important;color:#e6edf3!important;
  border:1px solid #30363d!important;border-radius:8px!important;font-size:1rem!important;}
.stTabs [data-baseweb="tab"]{color:#8b949e;font-size:0.9rem;}
.stTabs [aria-selected="true"]{color:#e6edf3!important;}
.utime{color:#8b949e;font-size:0.72rem;}
.log-row{border-bottom:1px solid #21262d;padding:3px 0;font-size:0.78rem;}
@media(max-width:768px){
  [data-testid="stMetricValue"]{font-size:0.95rem!important;}
  .badge-buy,.badge-sell,.badge-neutral{font-size:0.9rem;padding:5px 10px;}
  h1{font-size:1.3rem!important;}
  .block-container{padding:0.5rem 0.8rem!important;}
}
h1,h2,h3{color:#e6edf3!important;}
hr{border-color:#21262d;}
</style>

“””, unsafe_allow_html=True)

# session state init

defaults = {
“result”: None, “code”: “”, “period”: “1y”, “interval”: “1d”,
“auto_on”: False, “auto_interval”: “15min”,
“update_log”: [], “next_update”: None,
}
for k, v in defaults.items():
if k not in st.session_state:
st.session_state[k] = v

# — sidebar —

with st.sidebar:
st.markdown(”### Settings”)
period   = st.selectbox(“Period”,   PERIODS,   index=3,
format_func=lambda x: PERIOD_LBL[x])
interval = st.selectbox(“Interval”, INTERVALS, index=0,
format_func=lambda x: INTERVAL_LBL[x])

```
st.divider()
st.markdown("### Auto-Update")
auto_int = st.selectbox("Interval", list(UPDATE_OPTS.keys()), index=2,
                         disabled=st.session_state["auto_on"])

ac1, ac2 = st.columns(2)
with ac1:
    can_start = (not st.session_state["auto_on"]
                 and bool(st.session_state["code"]))
    if st.button("Start", use_container_width=True, disabled=not can_start):
        st.session_state.update({
            "auto_on": True,
            "auto_interval": auto_int,
            "next_update": time.time() + UPDATE_OPTS[auto_int],
        })
        st.rerun()
with ac2:
    if st.button("Stop", use_container_width=True,
                 disabled=not st.session_state["auto_on"]):
        st.session_state["auto_on"] = False
        st.rerun()

if st.session_state["auto_on"]:
    st.success(f"Running every {st.session_state['auto_interval']}")
else:
    st.caption("Stopped")

if st.session_state["update_log"]:
    st.divider()
    st.markdown("### Update Log")
    for e in st.session_state["update_log"][:8]:
        sig = e["signal"]
        col = "#3fb950" if sig == "BUY" else ("#f85149" if sig == "SELL" else "#58a6ff")
        st.markdown(
            f"<div class='log-row'>"
            f"<span style='color:#8b949e'>{e['time']}</span> "
            f"<b>{e['code']}</b> "
            f"<span style='color:{col}'>{sig}</span> {e['price']}"
            f"</div>",
            unsafe_allow_html=True
        )
```

# — main area —

st.markdown(”# Trend Signal Analyzer”)

# input bar

ic1, ic2, ic3 = st.columns([4, 1, 1])
with ic1:
typed = st.text_input(
“ticker”,
value=st.session_state.get(“code”, “”),
placeholder=“e.g. 7203.T  AAPL  ^N225  USDJPY=X”,
label_visibility=“collapsed”
)
with ic2:
fetch_btn = st.button(“Analyze”, use_container_width=True, type=“primary”)
with ic3:
refresh_btn = st.button(“Refresh”, use_container_width=True,
disabled=not st.session_state[“code”])

if fetch_btn and typed.strip():
code = typed.strip().upper()
fetch_and_analyze.clear()
with st.spinner(f”Fetching {code}…”):
res = fetch_and_analyze(code, period, interval)
if res and “df” in res:
st.session_state.update({
“result”: res, “code”: code,
“period”: period, “interval”: interval
})
st.rerun()
else:
err = res.get(“error”, “no data”) if res else “no data”
st.error(f”Failed: {err}”)

if refresh_btn and st.session_state[“code”]:
fetch_and_analyze.clear()
with st.spinner(“Refreshing…”):
res = fetch_and_analyze(
st.session_state[“code”],
st.session_state[“period”],
st.session_state[“interval”]
)
if res and “df” in res:
st.session_state[“result”] = res
df_r = res[“df”]
log  = st.session_state[“update_log”]
log.insert(0, {
“time”:   res[“fetched_at”][11:],
“code”:   st.session_state[“code”],
“signal”: sig_label(int(df_r[“signal”].iloc[-1])),
“price”:  f”{float(df_r[‘Close’].squeeze().iloc[-1]):,.1f}”,
})
st.session_state[“update_log”] = log[:30]
st.rerun()

# auto-update check

if (st.session_state[“auto_on”]
and st.session_state[“code”]
and st.session_state[“next_update”]
and time.time() >= st.session_state[“next_update”]):
fetch_and_analyze.clear()
with st.spinner(“Auto-updating…”):
res = fetch_and_analyze(
st.session_state[“code”],
st.session_state[“period”],
st.session_state[“interval”]
)
if res and “df” in res:
st.session_state[“result”] = res
df_r = res[“df”]
log  = st.session_state[“update_log”]
log.insert(0, {
“time”:   res[“fetched_at”][11:],
“code”:   st.session_state[“code”],
“signal”: sig_label(int(df_r[“signal”].iloc[-1])),
“price”:  f”{float(df_r[‘Close’].squeeze().iloc[-1]):,.1f}”,
})
st.session_state[“update_log”] = log[:30]
st.session_state[“next_update”] = (
time.time() + UPDATE_OPTS[st.session_state[“auto_interval”]]
)

# quick-select buttons (home screen)

result = st.session_state.get(“result”)
if result is None:
st.info(“Enter a ticker code above and click Analyze, or pick a popular one below.”)
st.divider()
cols = st.columns(5)
for i, (c, n) in enumerate(POPULAR):
if cols[i % 5].button(n, key=f”h_{c}”, use_container_width=True):
fetch_and_analyze.clear()
with st.spinner(f”Fetching {n}…”):
res = fetch_and_analyze(c, “1y”, “1d”)
if res and “df” in res:
st.session_state.update({
“result”: res, “code”: c,
“period”: “1y”, “interval”: “1d”
})
st.rerun()
st.stop()

# — result display —

df   = result[“df”]
code = st.session_state[“code”]
per  = st.session_state[“period”]
inv  = st.session_state[“interval”]
lat  = df.iloc[-1]

close_v = float(df[“Close”].squeeze().iloc[-1])
prev_v  = float(df[“Close”].squeeze().iloc[-2])
chg_pct = (close_v - prev_v) / prev_v * 100
chg_abs = close_v - prev_v
sig     = int(lat[“signal”])
trend   = lat[“trend”]
fat     = result[“fetched_at”]

st.markdown(
f”### {code}  {PERIOD_LBL[per]}/{INTERVAL_LBL[inv]}”
f”  <span class='utime'>Updated: {fat}</span>”,
unsafe_allow_html=True
)

sig_map = {
1:  (“badge-buy”,     “BUY Signal”),
-1: (“badge-sell”,    “SELL Signal”),
0:  (“badge-neutral”, “NEUTRAL”),
}
css_cls, sig_txt = sig_map[sig]
tclr = {“Uptrend”: “#3fb950”, “Downtrend”: “#f85149”, “Range”: “#58a6ff”}[trend]

r1, r2, r3 = st.columns([2, 2, 3])
with r1:
st.markdown(f”<div class='{css_cls}'>{sig_txt}</div>”, unsafe_allow_html=True)
with r2:
st.markdown(
f”<div style='font-size:0.95rem;margin-top:9px;"
f"color:{tclr};font-weight:bold'>{trend}</div>”,
unsafe_allow_html=True
)
with r3:
bsc = int(lat[“buy_sc”])
ssc = int(lat[“sell_sc”])
st.progress(bsc / 7, text=f”Buy score {bsc}/7”)
st.progress(ssc / 7, text=f”Sell score {ssc}/7”)

st.divider()

m1, m2, m3, m4 = st.columns(4)
m1.metric(“Price”,   f”{close_v:,.1f}”, f”{chg_pct:+.2f}%”)
m2.metric(“RSI”,     f”{float(lat[‘RSI’]):.1f}”,
“Overbought” if float(lat[“RSI”]) > 70
else (“Oversold” if float(lat[“RSI”]) < 30 else “-”))
m3.metric(“ADX”,     f”{float(lat[‘ADX’]):.1f}”,
“Trending” if float(lat[“ADX”]) > 25 else “Range”)
m4.metric(“MACD”,    f”{float(lat[‘MACD’]):.2f}”)

m5, m6, m7, m8 = st.columns(4)
m5.metric(“Change”,  f”{chg_abs:+,.1f}”)
m6.metric(“Stoch%K”, f”{float(lat[‘Stoch_K’]):.1f}”)
m7.metric(“SMA25”,   f”{float(lat[‘SMA25’]):.1f}”)
m8.metric(“BB Width”, f”{float(lat[‘BB_width’]):.4f}”)

st.divider()

tab1, tab2, tab3 = st.tabs([“Chart”, “Signal History”, “Raw Data”])

with tab1:
mobile_mode = st.toggle(“Mobile view (portrait)”, value=False)
with st.spinner(“Rendering…”):
title = f”{code}  {PERIOD_LBL[per]}/{INTERVAL_LBL[inv]}  ({len(df)} bars)”
fig = make_chart(df, title, mobile=mobile_mode)
st.pyplot(fig, use_container_width=True)

with tab2:
sig_df = df[df[“signal”] != 0].copy()
if not sig_df.empty:
rows = []
for idx_h, rh in sig_df.iterrows():
cl_v = float(df.loc[idx_h, “Close”].squeeze())
rows.append({
“Date”:    idx_h.strftime(”%Y/%m/%d”),
“Signal”:  sig_label(int(rh[“signal”])),
“Price”:   f”{cl_v:,.1f}”,
“BuySc”:   int(rh[“buy_sc”]),
“SellSc”:  int(rh[“sell_sc”]),
“RSI”:     round(float(rh[“RSI”]), 1),
“ADX”:     round(float(rh[“ADX”]), 1),
“Trend”:   rh[“trend”],
})
st.dataframe(
pd.DataFrame(rows).iloc[::-1].reset_index(drop=True),
use_container_width=True, height=400
)
else:
st.info(“No signals in this period. Try a longer period or different interval.”)

```
if st.session_state["update_log"]:
    st.markdown("#### Auto-Update Log")
    st.dataframe(
        pd.DataFrame(st.session_state["update_log"]),
        use_container_width=True
    )
```

with tab3:
show_cols = [“Open”, “High”, “Low”, “Close”, “Volume”,
“SMA25”, “SMA75”, “RSI”, “ADX”, “MACD”, “signal”, “trend”]
df_show = df[[c for c in show_cols if c in df.columns]].tail(60).copy()
df_show.columns = [
c if isinstance(c, str) else c[0] for c in df_show.columns
]
st.dataframe(
df_show.style.format(precision=2),
use_container_width=True, height=400
)

# auto-update countdown loop

if st.session_state[“auto_on”] and st.session_state[“next_update”]:
remaining = int(st.session_state[“next_update”] - time.time())
if remaining > 0:
st.caption(f”Next auto-update in {remaining}s”)
time.sleep(min(remaining, 30))
st.rerun()
else:
st.rerun()