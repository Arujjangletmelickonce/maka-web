import json
import os
import re
import textwrap
import tomllib
from pathlib import Path
from datetime import datetime, time as dtime
from zoneinfo import ZoneInfo

import requests
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import yfinance as yf
import pandas as pd
from openai import OpenAI


# =========================
# Config load
# =========================
def load_config(path: str = "config.toml") -> dict:
    with open(path, "rb") as f:
        return tomllib.load(f)

cfg_path = "config.public.toml" if os.path.exists("config.public.toml") else "config.toml"
config = load_config(cfg_path)

OPENAI_KEY = os.getenv("OPENAI_KEY") or config.get("api_keys", {}).get("openai_key", "")
POLYGON_KEY = os.getenv("POLYGON_KEY") or config.get("api_keys", {}).get("polygon_key", "")

if not OPENAI_KEY:
    raise RuntimeError("OPENAI_KEY is missing (set env OPENAI_KEY or api_keys.openai_key in config.toml)")
if not POLYGON_KEY:
    raise RuntimeError("POLYGON_KEY is missing (set env POLYGON_KEY or api_keys.polygon_key in config.toml)")

MY_MODEL_ID = config.get("models", {}).get("maka_ft_model", "")
PRO_MODEL_ID = config.get("models", {}).get("pro_model", "gpt-5.2")

if not MY_MODEL_ID:
    raise RuntimeError("maka_ft_model is missing in config.toml under [models]")

RUN_TICKER = config.get("run", {}).get("ticker", "QQQ")
PIVOT_DEFAULT = float(config.get("run", {}).get("pivot", 600.0))
OUTPUTS_DIR = str(config.get("run", {}).get("outputs_dir", "outputs"))

REQUEST_TIMEOUT = int(config.get("run", {}).get("request_timeout", 10))
OPTIONS_LIMIT = int(config.get("run", {}).get("options_limit", 250))
MAX_CONTRACTS = int(config.get("run", {}).get("max_contracts", 2000))

CONTRACT_MULTIPLIER = 100.0

client = OpenAI(api_key=OPENAI_KEY)

# Chart: keep English-only to avoid tofu(□□) in matplotlib text
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False


# =========================
# Systems
# =========================
KOREAN_SYSTEM = (
    "반드시 한국어로만 답변해라. "
    "영어/중국어/일본어 단어를 섞지 말고, 티커/숫자/약어(예: QQQ, GEX, OI, DEX)만 예외로 허용한다."
)

ANALYST_JSON_SYSTEM = (
    "너는 옵션 수급 분석가다. 입력(JSON)을 읽고 반드시 JSON만 출력해라. 설명 문장/코드블록 금지.\n"
    "목표: 마카가 '원문 제작자 톤'으로 게시글을 쓰기 좋게, 스토리 재료를 뽑아라.\n"
    "절대 금지:\n"
    "- 매매계획(평단/수익/손절/수량/매수·매도 지시)\n"
    "- 한국장/빅테크/다른 종목으로 빠지기(QQQ만)\n\n"
    "중요:\n"
    "- intraday(now_et/phase/flow, 오늘 고점/저점/오픈, 최근 30분 흐름)을 반드시 반영.\n"
    "- pivot은 '가격 숫자 반복' 대신 '그 선/그 자리/중심값/두꺼운 옵션벽' 같은 표현으로 설명하도록 재료를 줘라.\n"
    "- 핵심은 '옵션벽(풋월/콜월/맥스페인/피닝/감마 전환) + 마켓메이커 운전 + 심리전' 톤.\n"
    "- 결론은 단정형으로 만들되, 표현은 '가능성이 높다/확률이 우세하다' 같은 확률어는 허용.\n"
    "- 레벨은 4개까지만. watch 2개, risks 2개까지만.\n"
    "- chart_what_to_watch_en 은 최대 14단어.\n\n"
    "스키마:\n"
    "{\n"
    '  "spot": number,\n'
    '  "now_et": string,\n'
    '  "phase_et": string,\n'
    '  "flow_one_line": string,\n'
    '  "day_stats": {"open": number, "high": number, "low": number, "vwap_like": number},\n'
    '  "pivot": number,\n'
    '  "pivot_touches": number,\n'
    '  "one_liner": string,\n'
    '  "key_levels": [{"price": number, "role": "pivot"|"support"|"resistance", "why": string}],\n'
    '  "core_thesis": string,\n'
    '  "branches": {\n'
    '     "if_hold": {"condition": string, "expectation": string},\n'
    '     "if_break": {"condition": string, "expectation": string}\n'
    "  },\n"
    '  "watch": [string, string],\n'
    '  "risks": [string, string],\n'
    '  "chart_window_usd": number,\n'
    '  "chart_title_en": string,\n'
    '  "chart_what_to_watch_en": string\n'
    "}\n"
)

MAKA_WRITER_SYSTEM = (
    "너는 '마카'다. 글은 리포트가 아니라 '게시글'이다.\n"
    "금지:\n"
    "- 섹션 제목(오늘의 초점/핵심 레벨/리스크 등) 같은 헤더 금지\n"
    "- 인사/감사/댓글유도 금지\n"
    "- 한국장/다른 종목/빅테크 금지(QQQ만)\n"
    "- 매매계획(평단/수익/손절/수량/매수·매도 지시) 금지\n\n"
    "길이/리듬(필수):\n"
    "- 너무 짧게 쓰지 마라. 최소 10문단(각 문단 1~3문장), 총 900~1400자 정도.\n"
    "- 문단 사이에는 반드시 빈 줄(두 줄 띄기)로 리듬을 만든다.\n"
    "- 중후반부에 '(이미지 1)', '(이미지 2)'를 자연스럽게 1~2번 넣는다(실제 이미지는 차트 파일로 대체).\n\n"
    "반복 제어(필수):\n"
    "- pivot 숫자는 최대 3~4회만 직접 쓰고, 나머진 '그 선/그 자리/두꺼운 옵션벽/중심값'으로 치환.\n"
    "- 같은 뜻을 같은 리듬으로 반복 금지.\n\n"
    "톤(원문 제작자 톤):\n"
    "- '수많은 돈들의 이해관계', '심리싸움', '월가가 운전' 같은 어휘를 과하지 않게 섞어라.\n"
    "- 마켓메이커, 옵션월(풋월/콜월), 피닝, 감마(양수/음수) 같은 키워드는 짧게 직관적으로.\n"
    "- 독자와 한 번만 툭 던지듯 대화(‘눈 아프시죠?’ 같은) 허용.\n\n"
    "구성(자연스러운 흐름):\n"
    "1) one_liner로 시작 + now_et/phase_et를 1문장에 녹이기\n"
    "2) 오늘 흐름(day_stats/flow_one_line) 한 문단\n"
    "3) 차트 읽는 법 bullet 3개(put OI/red, call OI/blue, Net GEX/line, Net DEX/bottom)\n"
    "4) 레벨은 4개만: pivot / 아래 2개 / 위 1개(숫자 남발 금지)\n"
    "5) core_thesis를 이야기로 풀기(pivot_touches를 '여러 번 두드림'으로 서술)\n"
    "6) if_hold / if_break 각각 2문장 이내\n"
    "7) 오늘 체크 2개 + 리스크 2개\n"
    "8) 끝맺음: '오늘은 결국 뭐만 보면 된다' 톤으로 2~3문장 깔끔 요약(도돌이표 금지)\n"
)

SILENT_FIX_SYSTEM = (
    "너는 감수 편집자다. 글을 새로 쓰지 마라.\n"
    "표시 없이 반영 수정만 수행해라.\n"
    "중복 단어/중복 문장 리듬을 강하게 제거해라(특히 pivot 숫자 반복).\n"
    "후반부 2~3문단이 요약을 반복하면, 1문단(2~3문장)으로 압축해서 깔끔하게 끝내라.\n"
    "첫 문단에 now_et/phase_et는 반드시 남겨라.\n"
    "출력은 최종 본문만.\n"
)



# =========================
# File helpers
# =========================
def ensure_dir(p: str | Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p

def write_text(path: Path, text: str):
    path.write_text(text, encoding="utf-8")

def write_json(path: Path, obj: dict):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

def build_public_index(outputs_dir: str):
    """
    Create outputs/index.json with a list of posts for static website.
    """
    base = Path(outputs_dir)
    posts_dir = base / "posts"
    if not posts_dir.exists():
        return

    items = []
    for d in posts_dir.iterdir():
        if not d.is_dir():
            continue
        meta_p = d / "meta.json"
        post_p = d / "post.md"
        if meta_p.exists() and post_p.exists():
            try:
                meta = json.loads(meta_p.read_text(encoding="utf-8"))
                items.append(
                    {
                        "run_id": d.name,
                        "time_kst": meta.get("time_kst"),
                        "spot": meta.get("spot"),
                        "pivot": meta.get("pivot"),
                        "path": f"data/posts/{d.name}/meta.json",
                    }
                )
            except Exception:
                pass

    items.sort(key=lambda x: x.get("run_id", ""), reverse=True)

    out = {
        "count": len(items),
        "items": items,
    }
    (base / "index.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")


# =========================
# Helpers: normalize columns/series
# =========================
def flatten_columns_if_needed(df: pd.DataFrame) -> pd.DataFrame:
    if df is None:
        return df
    if hasattr(df.columns, "nlevels") and df.columns.nlevels > 1:
        df = df.copy()
        df.columns = [c[0] for c in df.columns]
    return df

def to_series(x):
    if isinstance(x, pd.DataFrame):
        return x.iloc[:, 0]
    return x


# =========================
# Session phase (ET)
# =========================
def get_session_phase_et(now_et: datetime) -> str:
    t = now_et.time()
    if dtime(4, 0) <= t < dtime(9, 30):
        return "premarket"
    if dtime(9, 30) <= t < dtime(10, 30):
        return "open_early"
    if dtime(10, 30) <= t < dtime(15, 30):
        return "mid_day"
    if dtime(15, 30) <= t < dtime(16, 0):
        return "power_hour"
    if dtime(16, 0) <= t < dtime(20, 0):
        return "afterhours"
    return "closed"


# =========================
# yfinance intraday DF + pivot touches
# =========================
def fetch_intraday_df(ticker: str):
    for interval in ("1m", "5m", "15m"):
        try:
            df = yf.download(ticker, period="1d", interval=interval, prepost=True, progress=False)
            df = flatten_columns_if_needed(df)
            if df is not None and len(df) > 30:
                return df.copy(), interval
        except Exception:
            pass
    return None, None

def count_pivot_touches(df: pd.DataFrame, pivot: float, band: float = 0.25) -> int:
    if df is None or len(df) == 0:
        return 0
    close = to_series(df["Close"]).astype(float).values
    in_band = np.abs(close - pivot) <= band
    touches = 0
    prev = False
    for x in in_band:
        if x and not prev:
            touches += 1
        prev = x
    return touches

def fetch_intraday_context(ticker: str = "QQQ", pivot: float = 600.0) -> dict:
    now_et = datetime.now(ZoneInfo("America/New_York"))
    phase = get_session_phase_et(now_et)

    df, interval = fetch_intraday_df(ticker)
    if df is None or len(df) == 0:
        return {
            "now_et": now_et.strftime("%Y-%m-%d %H:%M"),
            "phase_et": phase,
            "interval": None,
            "note": "intraday unavailable",
        }

    close = to_series(df["Close"]).astype(float)
    high = to_series(df["High"]).astype(float)
    low = to_series(df["Low"]).astype(float)
    vol = to_series(df["Volume"]).astype(float) if "Volume" in df.columns else None

    last = float(close.iloc[-1])
    day_open = float(close.iloc[0])
    day_high = float(high.max())
    day_low = float(low.min())

    if vol is not None:
        vol_sum = float(vol.sum())
        vwap_like = float((close * vol).sum() / vol_sum) if vol_sum > 0 else float(close.mean())
    else:
        vwap_like = float(close.mean())

    bars_30m = 30 if interval == "1m" else (6 if interval == "5m" else 2)
    if len(close) > bars_30m:
        r_30m = float((close.iloc[-1] / close.iloc[-1 - bars_30m] - 1.0) * 100.0)
    else:
        r_30m = 0.0

    if phase in ("closed", "afterhours", "premarket"):
        flow = "정규장 밖에서 " + ("위쪽이 상대적으로 단단한 흐름" if last >= vwap_like else "아래쪽이 무거운 흐름")
    else:
        if last >= vwap_like and r_30m >= 0:
            flow = "VWAP 위에서 버티는 중(상방 우위)"
        elif last >= vwap_like and r_30m < 0:
            flow = "VWAP 위지만 식는 중(되밀림 경계)"
        elif last < vwap_like and r_30m < 0:
            flow = "VWAP 아래로 밀리는 중(하방 우위)"
        else:
            flow = "VWAP 아래지만 되돌림 시도(휩쏘 가능)"

    touches = count_pivot_touches(df, pivot=pivot, band=0.25)

    return {
        "now_et": now_et.strftime("%Y-%m-%d %H:%M"),
        "phase_et": phase,
        "interval": interval,
        "last": last,
        "day_open": day_open,
        "day_high": day_high,
        "day_low": day_low,
        "vwap_like": vwap_like,
        "r_30m": r_30m,
        "flow_one_line": flow,
        "pivot": pivot,
        "pivot_touches": touches,
    }


def get_spot_robust(ticker: str, intraday: dict | None) -> tuple[float, str]:
    if intraday and isinstance(intraday.get("last"), (int, float)) and float(intraday["last"]) > 0:
        return float(intraday["last"]), "intraday_last"

    try:
        t = yf.Ticker(ticker)
        fi = getattr(t, "fast_info", None)
        if fi and fi.get("last_price") is not None and float(fi["last_price"]) > 0:
            return float(fi["last_price"]), "yfinance_fast_info"
    except Exception:
        pass

    for interval in ("1m", "5m", "15m"):
        try:
            df = yf.download(ticker, period="1d", interval=interval, prepost=True, progress=False)
            df = flatten_columns_if_needed(df)
            if df is not None and len(df) > 0:
                last = float(to_series(df["Close"]).iloc[-1])
                if last > 0:
                    return last, f"yfinance_{interval}_close"
        except Exception:
            pass

    try:
        df = yf.download(ticker, period="5d", interval="1d", progress=False)
        df = flatten_columns_if_needed(df)
        if df is not None and len(df) > 0:
            last = float(to_series(df["Close"]).iloc[-1])
            if last > 0:
                return last, "yfinance_1d_close"
    except Exception:
        pass

    raise RuntimeError("Spot price fetch failed")


# =========================
# Polygon options
# =========================
def fetch_options_snapshot_all(ticker: str, polygon_key: str, limit: int, max_contracts: int) -> list[dict]:
    url = f"https://api.polygon.io/v3/snapshot/options/{ticker}?limit={limit}&apiKey={polygon_key}"
    out: list[dict] = []
    while url and len(out) < max_contracts:
        r = requests.get(url, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        j = r.json() or {}
        out.extend(j.get("results") or [])
        next_url = j.get("next_url")
        if next_url:
            if "apiKey=" not in next_url:
                next_url = next_url + ("&" if "?" in next_url else "?") + f"apiKey={polygon_key}"
            url = next_url
        else:
            url = None
    return out[:max_contracts]


# =========================
# Aggregate by strike: call OI, put OI, netGEX, netDEX
# =========================
def aggregate_by_strike(contracts: list[dict], spot: float):
    agg = {}
    for c in contracts:
        k = float(c["strike"])
        t = c["type"]
        oi = float(c["oi"])
        gamma = float(c["gamma"])
        delta = float(c["delta"])

        if k not in agg:
            agg[k] = {"call_oi": 0.0, "put_oi": 0.0, "net_gex": 0.0, "net_dex": 0.0}

        if t == "call":
            agg[k]["call_oi"] += oi
            sign = +1.0
        else:
            agg[k]["put_oi"] += oi
            sign = -1.0

        agg[k]["net_dex"] += sign * delta * oi * CONTRACT_MULTIPLIER
        agg[k]["net_gex"] += sign * gamma * oi * (spot ** 2) * 0.01 * CONTRACT_MULTIPLIER

    strikes = np.array(sorted(agg.keys()), dtype=float)
    call_oi = np.array([agg[s]["call_oi"] for s in strikes], dtype=float)
    put_oi = np.array([agg[s]["put_oi"] for s in strikes], dtype=float)
    net_gex = np.array([agg[s]["net_gex"] for s in strikes], dtype=float)
    net_dex = np.array([agg[s]["net_dex"] for s in strikes], dtype=float)
    return strikes, call_oi, put_oi, net_gex, net_dex


def _clip_ylim(vals: np.ndarray, p_low=1, p_high=99, pad=1.15):
    if len(vals) == 0:
        return (-1.0, 1.0)
    lo = np.percentile(vals, p_low)
    hi = np.percentile(vals, p_high)
    if abs(hi - lo) < 1e-9:
        lo, hi = lo - 1.0, hi + 1.0
    mid = 0.5 * (lo + hi)
    half = 0.5 * (hi - lo) * pad
    return (mid - half, mid + half)


def _set_strike_ticks(ax, x: np.ndarray, max_ticks: int = 14):
    """
    Show strike labels outside plot box as x-axis ticks.
    To avoid clutter, downsample tick labels if needed.
    """
    if x is None or len(x) == 0:
        return
    n = len(x)
    step = max(1, int(np.ceil(n / max_ticks)))
    idx = np.arange(0, n, step, dtype=int)
    xt = x[idx]
    ax.set_xticks(xt)
    ax.set_xticklabels([f"{v:.0f}" for v in xt], rotation=0, fontsize=9)


# =========================
# Chart (English-only) + short title
# =========================
def generate_chart(
    spot: float,
    strikes: np.ndarray,
    call_oi: np.ndarray,
    put_oi: np.ndarray,
    net_gex: np.ndarray,
    net_dex: np.ndarray,
    analysis_json: dict,
    outfile="maka_chart.png",
) -> str | None:
    window = float(analysis_json.get("chart_window_usd", 45.0))
    window = max(25.0, min(80.0, window))
    xmin, xmax = spot - window, spot + window

    mask = (strikes >= xmin) & (strikes <= xmax)
    x = strikes[mask]
    cOI = call_oi[mask]
    pOI = put_oi[mask]
    g = net_gex[mask]
    d = net_dex[mask]

    if len(x) == 0:
        return None

    width = 1.0
    if len(x) >= 2:
        width = max(0.6, min(2.0, float(np.median(np.diff(x))) * 0.8))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 7.6), sharex=True)

    # Top: OI bars
    ax1.bar(x - width * 0.20, pOI, width=width * 0.40, alpha=0.75, label="Put OI", color="#e74c3c")
    ax1.bar(x + width * 0.20, cOI, width=width * 0.40, alpha=0.75, label="Call OI", color="#3498db")
    ax1.set_ylabel("Open Interest")
    ax1.grid(axis="y", linestyle="--", alpha=0.35)

    # Right axis: Net GEX line
    ax1b = ax1.twinx()
    ax1b.plot(x, g, linewidth=2.0, label="Net GEX", color="#f1c40f")
    ax1b.set_ylabel("Net GEX")
    ax1b.set_ylim(*_clip_ylim(g))

    # Spot
    ax1.axvline(x=spot, color="black", linestyle="--", linewidth=1.8)

    # Key levels: max 4, closest-to-spot first
    levels = analysis_json.get("key_levels", [])
    if isinstance(levels, list) and len(levels) > 0:
        def dist(lv):
            try:
                return abs(float(lv.get("price")) - spot)
            except Exception:
                return 1e9
        levels_sel = sorted(levels, key=dist)[:4]
        for lv in levels_sel:
            try:
                p = float(lv.get("price"))
                if xmin <= p <= xmax:
                    ax1.axvline(x=p, linestyle=":", linewidth=1.0, alpha=0.65)
            except Exception:
                pass

    # Short title: cap + wrap
    title = str(analysis_json.get("chart_title_en", "QQQ Options Positioning")).strip()[:55]
    what = str(analysis_json.get("chart_what_to_watch_en", "pivot retest")).strip()
    what = re.sub(r"\s+", " ", what)[:90]
    subtitle = textwrap.fill(f"What to watch: {what}", width=70)
    ax1.set_title(f"{title}\n{subtitle}", fontsize=12)

    # Merge legends
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax1b.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper right")

    # Bottom: Net DEX
    ax2.plot(x, d, linewidth=2.0, label="Net DEX", color="#2c3e50")
    ax2.axvline(x=spot, color="black", linestyle="--", linewidth=1.2, label=f"Spot {spot:.2f}")
    ax2.set_xlabel("Strike")
    ax2.set_ylabel("Net DEX")
    ax2.grid(axis="y", linestyle="--", alpha=0.35)
    ax2.set_ylim(*_clip_ylim(d))
    ax2.legend(loc="upper right")

    ax2.set_xlim(xmin, xmax)

    # ✅ Strike labels outside plot box: use ticks on bottom axis
    _set_strike_ticks(ax2, x, max_ticks=14)

    plt.tight_layout()
    plt.savefig(outfile, dpi=170)
    plt.close()
    return outfile


# =========================
# QQQ session chart (regular hours only, timestamps in KST)
# =========================
def generate_qqq_session_chart_kst(outfile="qqq_price.png") -> str | None:
    """
    Plot QQQ regular session (09:30~16:00 ET) for today.
    X-axis labels in KST.
    """
    tz_et = ZoneInfo("America/New_York")
    tz_kst = ZoneInfo("Asia/Seoul")

    # Prefer 1m, fallback to 5m
    df = None
    interval_used = None
    for interval in ("1m", "5m"):
        try:
            _df = yf.download("QQQ", period="1d", interval=interval, prepost=False, progress=False)
            _df = flatten_columns_if_needed(_df)
            if _df is not None and len(_df) >= 10:
                df = _df.copy()
                interval_used = interval
                break
        except Exception:
            pass

    if df is None or len(df) == 0:
        return None

    # Ensure tz-aware index in ET
    idx = df.index
    if getattr(idx, "tz", None) is None:
        # yfinance usually returns tz-aware, but guard anyway
        try:
            idx = idx.tz_localize(tz_et)
        except Exception:
            # if it fails, treat as ET naive
            idx = pd.DatetimeIndex(idx).tz_localize(tz_et)

    idx_et = idx.tz_convert(tz_et)

    # Filter regular session: 09:30~16:00 ET
    t_open = dtime(9, 30)
    t_close = dtime(16, 0)
    mask = (idx_et.time >= t_open) & (idx_et.time <= t_close)
    df_sess = df.loc[mask].copy()
    if df_sess is None or len(df_sess) == 0:
        return None

    close = to_series(df_sess["Close"]).astype(float)
    idx_kst = df_sess.index.tz_convert(tz_kst)

    fig, ax = plt.subplots(figsize=(13, 4.4))
    ax.plot(idx_kst, close.values, linewidth=2.0)

    # Title in English only (matplotlib)
    day_kst = idx_kst[-1].strftime("%Y-%m-%d")
    ax.set_title(f"QQQ Regular Session (KST) - {day_kst}  |  interval {interval_used}")
    ax.set_xlabel("KST Time")
    ax.set_ylabel("Price")
    ax.grid(True, linestyle="--", alpha=0.35)

    # Format x-axis: show HH:MM in KST
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=tz_kst))
    fig.autofmt_xdate(rotation=0)

    plt.tight_layout()
    plt.savefig(outfile, dpi=170)
    plt.close()
    return outfile


# =========================
# OpenAI helpers
# =========================
def _extract_first_json(text: str) -> dict:
    text = (text or "").strip()
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        raise ValueError("no JSON found")
    return json.loads(m.group(0))


# =========================
# Main pipeline
# =========================
def run(ticker: str = "QQQ", pivot: float = 600.0):
    time_kst = datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y-%m-%d %H:%M")

    # outputs folder per run
    run_id = datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y-%m-%d_%H%M")
    base = ensure_dir(Path(OUTPUTS_DIR) / "posts" / run_id)

    # Intraday first
    intraday = fetch_intraday_context(ticker, pivot=pivot)

    # Robust spot
    spot, spot_source = get_spot_robust(ticker, intraday)

    # Options
    try:
        raw_opts = fetch_options_snapshot_all(ticker, POLYGON_KEY, OPTIONS_LIMIT, MAX_CONTRACTS)
    except Exception as e:
        print(f"[warn] options snapshot failed: {e}")
        raw_opts = []

    contracts = []
    for opt in raw_opts:
        details = opt.get("details", {}) or {}
        greeks = opt.get("greeks", {}) or {}
        strike = details.get("strike_price")
        ctype = details.get("contract_type")
        oi = opt.get("open_interest")
        if strike is None or oi is None or ctype not in ("call", "put"):
            continue
        contracts.append(
            {
                "strike": float(strike),
                "type": ctype,
                "oi": float(oi),
                "gamma": float(greeks.get("gamma") or 0.0),
                "delta": float(greeks.get("delta") or 0.0),
            }
        )

    if not contracts:
        print("[warn] no option contracts pulled. check Polygon plan/entitlements.")
        return

    strikes, call_oi, put_oi, net_gex, net_dex = aggregate_by_strike(contracts, spot)

    # Summary for analysis
    top_gex_idx = np.argsort(np.abs(net_gex))[::-1][:10]
    top_dex_idx = np.argsort(np.abs(net_dex))[::-1][:10]
    top_oi_idx = np.argsort((call_oi + put_oi))[::-1][:12]
    summary = {
        "contracts": len(contracts),
        "top_gex": [{"strike": float(strikes[i]), "net_gex": float(net_gex[i])} for i in top_gex_idx],
        "top_dex": [{"strike": float(strikes[i]), "net_dex": float(net_dex[i])} for i in top_dex_idx],
        "top_oi": [{"strike": float(strikes[i]), "call_oi": float(call_oi[i]), "put_oi": float(put_oi[i])} for i in top_oi_idx],
    }

    # Stage 1: analysis JSON
    payload = {
        "time_kst": time_kst,
        "ticker": ticker,
        "spot": spot,
        "spot_source": spot_source,
        "intraday": intraday,
        "summary": summary,
        "note": "Story-style post. No trade plan. QQQ only.",
    }

    r1 = client.responses.create(
        model=PRO_MODEL_ID,
        input=[
            {"role": "system", "content": KOREAN_SYSTEM},
            {"role": "system", "content": ANALYST_JSON_SYSTEM},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
    )
    analysis_json = _extract_first_json((r1.output_text or "").strip())

    # Charts (saved into run folder)
    chart_file = generate_chart(
        spot=spot,
        strikes=strikes,
        call_oi=call_oi,
        put_oi=put_oi,
        net_gex=net_gex,
        net_dex=net_dex,
        analysis_json=analysis_json,
        outfile=str(base / "maka_chart.png"),
    )

    # ✅ QQQ session chart in KST
    qqq_price_file = generate_qqq_session_chart_kst(outfile=str(base / "qqq_price.png"))

    # Stage 2: Maka write
    user_prompt = (
        f"[TIME_KST] {time_kst}\n"
        f"[INTRADAY_ET] {json.dumps(intraday, ensure_ascii=False)}\n"
        f"[SPOT] {ticker} = {spot:.2f}\n"
        f"[PIVOT] {pivot:.2f}\n"
        f"[CHART_FILE] {chart_file}\n\n"
        f"{json.dumps(analysis_json, ensure_ascii=False)}"
    )

    r2 = client.chat.completions.create(
        model=MY_MODEL_ID,
        messages=[
            {"role": "system", "content": KOREAN_SYSTEM},
            {"role": "system", "content": MAKA_WRITER_SYSTEM},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.7,
    )
    maka_body = (r2.choices[0].message.content or "").strip()

    # Stage 3: Silent polish
    r3 = client.responses.create(
        model=PRO_MODEL_ID,
        input=[
            {"role": "system", "content": KOREAN_SYSTEM},
            {"role": "system", "content": SILENT_FIX_SYSTEM},
            {"role": "user", "content": json.dumps({"analysis": analysis_json, "text": maka_body}, ensure_ascii=False)},
        ],
    )
    final_body = (r3.output_text or "").strip()

    # Save text + meta + latest pointer
    write_text(base / "post.md", final_body)

    meta = {
        "run_id": run_id,
        "time_kst": time_kst,
        "ticker": ticker,
        "pivot": pivot,
        "spot": spot,
        "spot_source": spot_source,
        "intraday": intraday,
        "analysis_json": analysis_json,
        "files": {
            "post": "post.md",
            "maka_chart": "maka_chart.png" if chart_file else None,
            "qqq_price": "qqq_price.png" if qqq_price_file else None,
            "meta": "meta.json",
        },
    }
    write_json(base / "meta.json", meta)

    latest = {
        "latest_run_id": run_id,
        "updated_kst": time_kst,
        "meta_path": str((Path(OUTPUTS_DIR) / "posts" / run_id / "meta.json").as_posix()),
    }
    write_json(Path(OUTPUTS_DIR) / "latest.json", latest)

    # Console output (optional)
    print("\n" + "=" * 70)
    print(f"[Maka Body - {time_kst}]")
    print("=" * 70)
    print(final_body)
    print("=" * 70)
    print(f"[saved] {base}")
    print(f"[chart] {chart_file}")
    print(f"[qqq]   {qqq_price_file}")
    build_public_index(OUTPUTS_DIR)


if __name__ == "__main__":
    run(RUN_TICKER, PIVOT_DEFAULT)
