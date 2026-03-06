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
OUTPUTS_DIR = str(config.get("run", {}).get("outputs_dir", "web/data"))

REQUEST_TIMEOUT = int(config.get("run", {}).get("request_timeout", 10))
OPTIONS_LIMIT = int(config.get("run", {}).get("options_limit", 250))
MAX_CONTRACTS = int(config.get("run", {}).get("max_contracts", 2000))

CONTRACT_MULTIPLIER = 100.0

client = OpenAI(api_key=OPENAI_KEY)

# Chart: English-only for matplotlib text
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
    "너는 QQQ 옵션 구조와 참여자 심리를 해석하는 분석가다. "
    "입력(JSON)을 읽고 반드시 JSON만 출력해라. 설명 문장, 코드블록, 주석 금지.\n\n"
    "목표:\n"
    "- 마카가 게시글을 쓸 때 바로 사용할 수 있는 '스토리 재료'를 뽑아라.\n"
    "- 정답 예측이 목적이 아니다. '이 시점에 참여자들이 어떤 태도였는지'를 자연스럽게 설명할 재료를 만들어라.\n\n"
    "반드시 반영:\n"
    "- content_mode_kst\n"
    "- intraday(now_et, phase_et, flow_one_line, day_open, day_high, day_low, vwap_like, r_30m)\n"
    "- 옵션벽(콜월/풋월), 핀 후보, 자석 역할, 감마/델타의 압력\n"
    "- 마켓메이커 운전 가설, 고래/개미 심리\n\n"
    "방향:\n"
    "- 숫자 나열보다 '왜 그 선이 중요한지'를 설명하는 재료를 줘라.\n"
    "- pivot은 숫자를 반복하기보다 중심값, 두꺼운 옵션벽, 그 선, 그 자리 같은 서술로 풀 수 있게 만들어라.\n"
    "- key_levels는 최대 4개만.\n"
    "- watch 2개, risks 2개까지만.\n"
    "- 한 줄 요약(one_liner)은 게시글 첫 문장으로 바로 이어질 수 있게 짧고 강하게.\n"
    "- core_thesis는 문장 2~4개 분량의 압축된 생각거리여야 한다.\n"
    "- if_hold / if_break는 기계적 매매전략이 아니라 '심리와 흐름 변화'를 설명하라.\n"
    "- chart_title_en / chart_what_to_watch_en 은 matplotlib 용이므로 짧은 영어만 써라.\n\n"
    "절대 금지:\n"
    "- 매매지시, 손절/익절, 수량, 평단, 매수·매도 권유\n"
    "- QQQ 외 다른 종목/한국장/빅테크로 화제 전환\n"
    "- 장황한 거시경제 강의\n\n"
    "content_mode_kst별 우선순위:\n"
    "- overnight_recap: 전일/야간 흐름을 요약하고 오늘 밤의 심리선을 제시\n"
    "- noon_brief: 최근 며칠 흐름과 참여자 태도를 설명\n"
    "- premarket_preview: 개장 전 벽, 자석, 가짜 방향 가능성을 제시\n"
    "- intraday_live: 지금 가격 변화와 옵션 반응의 상호작용을 가장 우선\n"
    "- post_close_recap: 오늘 실제 운전과 벽의 효력을 복기하고 다음 세션 단서를 남김\n\n"
    "스키마:\n"
    "{\n"
    '  "content_mode_kst": string,\n'
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
    '  "participants_view": {\n'
    '     "market_maker": string,\n'
    '     "whales": string,\n'
    '     "retail": string\n'
    "  },\n"
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
    "너는 '마카'다. 글은 리포트가 아니라 커뮤니티 게시글이다.\n\n"
    "절대 금지:\n"
    "- 섹션 제목, 소제목, 번호 매기기, bullet list 금지\n"
    "- (이미지 1), (이미지 2) 같은 문구 출력 금지\n"
    "- 인사, 감사, 댓글 유도 금지\n"
    "- QQQ 외 다른 종목 언급 금지\n"
    "- 매매지시, 수량, 평단, 손절/익절, 매수·매도 권유 금지\n"
    "- 지나치게 보고서처럼 정리된 문체 금지\n\n"
    "문체 목표:\n"
    "- 로니 계열의 흐름을 참고하되 완전 복사하지 말고, 자연스러운 한국어 게시글로 써라.\n"
    "- '돈들의 이해관계', '운전', '심리전', '벽', '자석', '핀', '눌러두기' 같은 말을 과하지 않게 섞어라.\n"
    "- 숫자는 꼭 필요한 핵심값만 쓰고, 같은 숫자를 계속 반복하지 마라.\n"
    "- pivot 숫자는 최대 3회 정도만 직접 쓰고, 나머지는 그 선, 그 자리, 중심값, 두꺼운 옵션벽 등으로 치환하라.\n"
    "- 마켓메이커, 고래, 개미의 태도를 자연스럽게 녹여라.\n"
    "- 독자에게 한 번 정도 툭 던지는 말투는 허용되지만 과하지 마라.\n\n"
    "글 길이와 리듬:\n"
    "- 총 8~12문단 정도.\n"
    "- 각 문단은 1~3문장.\n"
    "- 문단 사이에는 빈 줄을 둬라.\n"
    "- 장중/장전은 조금 더 짧고 날카롭게, 정오/장후는 약간 더 설명적으로 써라.\n\n"
    "자연스러운 전개:\n"
    "1) one_liner를 바탕으로 강한 첫 문장으로 시작\n"
    "2) 지금 시간대(content_mode_kst, now_et, phase_et)에 맞는 판의 성격을 한 문단으로 설명\n"
    "3) 오늘 또는 최근 흐름(day_stats, flow_one_line)을 짧게 짚기\n"
    "4) 핵심 가격대 2~4개와 그 의미를 이야기로 풀기\n"
    "5) participants_view와 core_thesis를 활용해 누가 무엇을 원할지 설명\n"
    "6) if_hold / if_break를 이용해 위아래 시나리오를 짧게 정리\n"
    "7) watch / risks를 억지 목록처럼 쓰지 말고 본문 흐름 안에 녹여라\n"
    "8) 마지막은 '오늘 결국 뭘 봐야 하는지' 한 번에 닫아라\n\n"
    "시간대별 톤:\n"
    "- overnight_recap: 전일과 야간 흐름을 바탕으로 오늘 밤 심리선을 정리\n"
    "- noon_brief: 최근 며칠의 태도 변화와 누가 판을 쥐고 있는지 설명\n"
    "- premarket_preview: 개장 직전 어디가 함정이고 어디가 자석인지 경계감 있게\n"
    "- intraday_live: 지금 벌어지는 가격 반응과 옵션벽의 상호작용을 가장 생생하게\n"
    "- post_close_recap: 오늘 어떤 식으로 운전했는지 복기하고 내일로 넘길 포인트 제시\n\n"
    "중요:\n"
    "- 본문은 자연스럽게 이어지는 산문이어야 한다.\n"
    "- 체크리스트처럼 쓰지 마라.\n"
    "- '오늘은 결국 여기만 보면 된다' 같은 마무리는 가능하지만, 너무 도돌이표처럼 반복하지 마라.\n"
)

SILENT_FIX_SYSTEM = (
    "너는 감수 편집자다. 글을 새로 쓰지 마라.\n"
    "표시 없이 반영 수정만 수행해라.\n"
    "중복 단어, 중복 문장 리듬, 같은 숫자 반복을 강하게 줄여라.\n"
    "후반부가 같은 뜻을 반복하면 1문단으로 압축해라.\n"
    "첫 문단의 시간감은 유지해라.\n"
    "리스트처럼 보이는 부분이 있으면 산문으로 바꿔라.\n"
    "출력은 최종 본문만."
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
                        "content_mode_kst": meta.get("content_mode_kst"),
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
# Content mode (KST)
# =========================
def get_content_mode_kst(now_kst: datetime) -> str:
    """
    KST 기준 운영용 모드.
    워크플로 스케줄:
    06:30, 09:30, 12:30, 15:30, 18:30, 21:30, 00:30, 03:30
    """
    h = now_kst.hour
    m = now_kst.minute
    hm = h * 60 + m

    if 6 * 60 <= hm < 9 * 60:
        return "post_close_recap"
    if 9 * 60 <= hm < 12 * 60:
        return "overnight_recap"
    if 12 * 60 <= hm < 18 * 60:
        return "noon_brief"
    if 18 * 60 <= hm < 22 * 60 + 30:
        return "premarket_preview"
    if 22 * 60 + 30 <= hm or hm < 5 * 60:
        return "intraday_live"
    return "overnight_recap"


def get_mode_guide_kst(content_mode: str) -> str:
    guides = {
        "overnight_recap": "전일 장과 야간 흐름을 짚고 오늘 밤 어디가 심리선인지 정리한다.",
        "noon_brief": "최근 며칠간 흐름과 참여자 태도를 설명하고 오늘 밤의 중심 자리를 제시한다.",
        "premarket_preview": "개장 전 벽, 자석, 가짜 방향 가능성을 짚고 초반 함정을 경계한다.",
        "intraday_live": "지금 가격 변화와 옵션 반응의 상호작용을 가장 생생하게 해석한다.",
        "post_close_recap": "오늘 어떤 식으로 운전했는지 복기하고 다음 세션으로 이어질 포인트를 남긴다.",
    }
    return guides.get(content_mode, "현재 시점의 QQQ 옵션 구조와 참여자 심리를 자연스럽게 정리한다.")


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
        flow = "정규장 밖에서는 " + ("위쪽을 더 열어두는 흐름" if last >= vwap_like else "아래쪽을 더 무겁게 보는 흐름")
    else:
        if last >= vwap_like and r_30m >= 0:
            flow = "VWAP 위에서 버티며 상방 우위가 유지되는 흐름"
        elif last >= vwap_like and r_30m < 0:
            flow = "VWAP 위지만 탄력이 식으며 되밀림을 경계해야 하는 흐름"
        elif last < vwap_like and r_30m < 0:
            flow = "VWAP 아래로 눌리며 하방 우위가 이어지는 흐름"
        else:
            flow = "VWAP 아래지만 되돌림 시도가 나오며 휩쏘 가능성이 있는 흐름"

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
# Aggregate by strike
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
    if x is None or len(x) == 0:
        return
    n = len(x)
    step = max(1, int(np.ceil(n / max_ticks)))
    idx = np.arange(0, n, step, dtype=int)
    xt = x[idx]
    ax.set_xticks(xt)
    ax.set_xticklabels([f"{v:.0f}" for v in xt], rotation=0, fontsize=9)


def _annotate_key_lines(ax, levels: list, xmin: float, xmax: float, ymax_hint: float | None = None):
    if not isinstance(levels, list):
        return
    used = 0
    for lv in levels[:4]:
        try:
            p = float(lv.get("price"))
            if not (xmin <= p <= xmax):
                continue
            role = str(lv.get("role", "")).strip()
            label = f"{role} {p:.0f}" if role else f"{p:.0f}"
            y = ymax_hint if ymax_hint is not None else ax.get_ylim()[1]
            ax.text(
                p,
                y,
                label,
                fontsize=8,
                ha="center",
                va="bottom",
                rotation=0,
                alpha=0.85,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.55),
            )
            used += 1
            if used >= 4:
                break
        except Exception:
            pass


# =========================
# Chart
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

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 7.8), sharex=True)

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

    # Spot line
    ax1.axvline(x=spot, color="black", linestyle="--", linewidth=1.8)
    ax2.axvline(x=spot, color="black", linestyle="--", linewidth=1.2, label=f"Spot {spot:.2f}")

    # Key levels
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
                    ax1.axvline(x=p, linestyle=":", linewidth=1.0, alpha=0.65, color="#666666")
                    ax2.axvline(x=p, linestyle=":", linewidth=0.9, alpha=0.45, color="#666666")
            except Exception:
                pass
    else:
        levels_sel = []

    # Title
    title = str(analysis_json.get("chart_title_en", "QQQ Options Positioning")).strip()[:55]
    what = str(analysis_json.get("chart_what_to_watch_en", "pivot retest")).strip()
    what = re.sub(r"\s+", " ", what)[:90]
    subtitle = textwrap.fill(f"What to watch: {what}", width=70)
    ax1.set_title(f"{title}\n{subtitle}", fontsize=12)

    # Legends
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax1b.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper right")

    # Bottom: Net DEX
    ax2.plot(x, d, linewidth=2.0, label="Net DEX", color="#2c3e50")
    ax2.set_xlabel("Strike")
    ax2.set_ylabel("Net DEX")
    ax2.grid(axis="y", linestyle="--", alpha=0.35)
    ax2.set_ylim(*_clip_ylim(d))
    ax2.legend(loc="upper right")
    ax2.set_xlim(xmin, xmax)

    # Bottom ticks
    _set_strike_ticks(ax2, x, max_ticks=14)

    # Top panel also show strike labels
    ax1.tick_params(axis="x", labelbottom=True)
    _set_strike_ticks(ax1, x, max_ticks=14)

    # Spot / levels labels
    y1 = ax1.get_ylim()[1] * 0.98
    ax1.text(
        spot,
        y1,
        f"spot {spot:.2f}",
        fontsize=8,
        ha="center",
        va="top",
        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7),
    )
    _annotate_key_lines(ax1, levels_sel, xmin, xmax, ymax_hint=ax1.get_ylim()[1] * 0.90)

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

    idx = df.index
    if getattr(idx, "tz", None) is None:
        try:
            idx = idx.tz_localize(tz_et)
        except Exception:
            idx = pd.DatetimeIndex(idx).tz_localize(tz_et)

    idx_et = idx.tz_convert(tz_et)

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

    day_kst = idx_kst[-1].strftime("%Y-%m-%d")
    ax.set_title(f"QQQ Regular Session (KST) - {day_kst}  |  interval {interval_used}")
    ax.set_xlabel("KST Time")
    ax.set_ylabel("Price")
    ax.grid(True, linestyle="--", alpha=0.35)

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


def normalize_analysis_json(analysis_json: dict, content_mode_kst: str, intraday: dict, pivot: float, spot: float) -> dict:
    if not isinstance(analysis_json, dict):
        analysis_json = {}

    analysis_json["content_mode_kst"] = analysis_json.get("content_mode_kst") or content_mode_kst
    analysis_json["spot"] = float(analysis_json.get("spot") or spot)
    analysis_json["now_et"] = analysis_json.get("now_et") or intraday.get("now_et", "")
    analysis_json["phase_et"] = analysis_json.get("phase_et") or intraday.get("phase_et", "")
    analysis_json["flow_one_line"] = analysis_json.get("flow_one_line") or intraday.get("flow_one_line", "")
    analysis_json["pivot"] = float(analysis_json.get("pivot") or pivot)
    analysis_json["pivot_touches"] = int(analysis_json.get("pivot_touches") or intraday.get("pivot_touches", 0))

    if "day_stats" not in analysis_json or not isinstance(analysis_json["day_stats"], dict):
        analysis_json["day_stats"] = {
            "open": float(intraday.get("day_open", spot)),
            "high": float(intraday.get("day_high", spot)),
            "low": float(intraday.get("day_low", spot)),
            "vwap_like": float(intraday.get("vwap_like", spot)),
        }

    if "participants_view" not in analysis_json or not isinstance(analysis_json["participants_view"], dict):
        analysis_json["participants_view"] = {
            "market_maker": "중심값 근처에서 가격을 붙들며 양쪽 심리를 흔들려는 태도",
            "whales": "방향성 확신보다는 두꺼운 벽 주변에서 효율적인 자리만 노리는 태도",
            "retail": "눈에 잘 보이는 돌파와 이탈에 쉽게 끌릴 수 있는 상태",
        }

    chart_title_default = {
        "overnight_recap": "QQQ Overnight Positioning",
        "noon_brief": "QQQ Multi-Day Positioning",
        "premarket_preview": "QQQ Pre-Open Positioning",
        "intraday_live": "QQQ Live Positioning",
        "post_close_recap": "QQQ Session Recap Positioning",
    }.get(content_mode_kst, "QQQ Options Positioning")

    chart_watch_default = {
        "overnight_recap": "last session pressure and next key magnet",
        "noon_brief": "multi-day bias around heavy strike clusters",
        "premarket_preview": "open reaction near key strike walls",
        "intraday_live": "live reaction around pivot and nearby walls",
        "post_close_recap": "which wall controlled the session",
    }.get(content_mode_kst, "pivot retest and nearby walls")

    analysis_json["chart_title_en"] = str(analysis_json.get("chart_title_en") or chart_title_default)[:55]
    analysis_json["chart_what_to_watch_en"] = str(analysis_json.get("chart_what_to_watch_en") or chart_watch_default)[:90]
    analysis_json["chart_window_usd"] = float(analysis_json.get("chart_window_usd") or 45.0)

    if not isinstance(analysis_json.get("watch"), list):
        analysis_json["watch"] = []
    if not isinstance(analysis_json.get("risks"), list):
        analysis_json["risks"] = []

    analysis_json["watch"] = analysis_json["watch"][:2]
    analysis_json["risks"] = analysis_json["risks"][:2]

    if not isinstance(analysis_json.get("key_levels"), list):
        analysis_json["key_levels"] = []

    return analysis_json


# =========================
# Main pipeline
# =========================
def run(ticker: str = "QQQ", pivot: float = 600.0):
    now_kst = datetime.now(ZoneInfo("Asia/Seoul"))
    time_kst = now_kst.strftime("%Y-%m-%d %H:%M")
    content_mode_kst = get_content_mode_kst(now_kst)
    mode_guide_kst = get_mode_guide_kst(content_mode_kst)

    run_id = now_kst.strftime("%Y-%m-%d_%H%M")
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
        "content_mode_kst": content_mode_kst,
        "mode_guide_kst": mode_guide_kst,
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
    analysis_json = normalize_analysis_json(
        analysis_json=analysis_json,
        content_mode_kst=content_mode_kst,
        intraday=intraday,
        pivot=pivot,
        spot=spot,
    )

    # Charts
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

    qqq_price_file = generate_qqq_session_chart_kst(outfile=str(base / "qqq_price.png"))

    # Stage 2: Maka write
    user_prompt = (
        f"[TIME_KST] {time_kst}\n"
        f"[CONTENT_MODE_KST] {content_mode_kst}\n"
        f"[MODE_GUIDE_KST] {mode_guide_kst}\n"
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
        "content_mode_kst": content_mode_kst,
        "mode_guide_kst": mode_guide_kst,
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
        "content_mode_kst": content_mode_kst,
    }
    write_json(Path(OUTPUTS_DIR) / "latest.json", latest)

    print("\n" + "=" * 70)
    print(f"[Maka Body - {time_kst} | {content_mode_kst}]")
    print("=" * 70)
    print(final_body)
    print("=" * 70)
    print(f"[saved] {base}")
    print(f"[chart] {chart_file}")
    print(f"[qqq]   {qqq_price_file}")

    build_public_index(OUTPUTS_DIR)


if __name__ == "__main__":
    run(RUN_TICKER, PIVOT_DEFAULT)