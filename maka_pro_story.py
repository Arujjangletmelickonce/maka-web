import json
import os
import re
import textwrap
import tomllib
from copy import deepcopy
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
def load_toml(path: str) -> dict:
    with open(path, "rb") as f:
        return tomllib.load(f)


def deep_merge(base: dict, override: dict) -> dict:
    out = deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_merged_config() -> tuple[dict, list[str]]:
    loaded_paths = []
    base_cfg = {}

    if os.path.exists("config.toml"):
        base_cfg = load_toml("config.toml")
        loaded_paths.append("config.toml")

    override_name = None
    for candidate in ("config.public.toml", "public.config.toml"):
        if os.path.exists(candidate):
            override_name = candidate
            break

    if override_name:
        override_cfg = load_toml(override_name)
        loaded_paths.append(override_name)
        return deep_merge(base_cfg, override_cfg), loaded_paths

    if not base_cfg:
        raise RuntimeError(
            "No config file found. Expected config.toml and optionally config.public.toml/public.config.toml."
        )

    return base_cfg, loaded_paths


config, loaded_config_paths = load_merged_config()

OPENAI_KEY = (os.getenv("OPENAI_KEY") or config.get("api_keys", {}).get("openai_key", "")).strip()
POLYGON_KEY = (os.getenv("POLYGON_KEY") or config.get("api_keys", {}).get("polygon_key", "")).strip()

if not OPENAI_KEY:
    raise RuntimeError(
        f"OPENAI_KEY is missing. Checked env OPENAI_KEY and merged config from: {', '.join(loaded_config_paths)}"
    )
if not POLYGON_KEY:
    raise RuntimeError(
        f"POLYGON_KEY is missing. Checked env POLYGON_KEY and merged config from: {', '.join(loaded_config_paths)}"
    )

MY_MODEL_ID = config.get("models", {}).get("maka_ft_model", "").strip()
PRO_MODEL_ID = (
    config.get("models", {}).get("pro_model")
    or config.get("models", {}).get("model")
    or "gpt-5.2"
)

if not MY_MODEL_ID:
    raise RuntimeError(f"maka_ft_model is missing in merged config from: {', '.join(loaded_config_paths)}")

RUN_TICKER = config.get("run", {}).get("ticker", "QQQ")
PIVOT_DEFAULT = float(config.get("run", {}).get("pivot", 600.0))
PIVOT_MODE = str(config.get("run", {}).get("pivot_mode", "manual")).strip().lower()
OUTPUTS_DIR = str(config.get("run", {}).get("outputs_dir", "web/data"))

REQUEST_TIMEOUT = int(config.get("run", {}).get("request_timeout", 10))
OPTIONS_LIMIT = int(config.get("run", {}).get("options_limit", 250))
MAX_CONTRACTS = int(config.get("run", {}).get("max_contracts", 2000))
TREND_MODE = str(config.get("run", {}).get("trend_mode", "medium")).strip().lower()

if TREND_MODE not in {"off", "low", "medium", "high"}:
    TREND_MODE = "medium"

CONTRACT_MULTIPLIER = 100.0

client = OpenAI(api_key=OPENAI_KEY)

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False


# =========================
# Systems
# =========================
KOREAN_SYSTEM = (
    "반드시 한국어로만 답변해라. "
    "영어/중국어/일본어 단어를 섞지 말고, 티커/숫자/약어(예: QQQ, OI, GEX, DEX)만 예외로 허용한다."
)

ANALYST_JSON_SYSTEM = (
    "너는 QQQ 옵션 구조와 참여자 심리를 해석하는 분석가다. "
    "입력(JSON)을 읽고 반드시 JSON만 출력해라. 설명 문장, 코드블록, 주석 금지.\n\n"

    "목표:\n"
    "- 정답 예측이 아니라, 이 시점에 시장참여자들이 어떤 태도였는지를 이야기할 재료를 뽑아라.\n"
    "- 게시글 작성자가 '오늘 판의 속내'를 바로 말할 수 있게 재료를 만들어라.\n"
    "- 숫자를 많이 주는 것보다, 왜 그 숫자가 방패/미끼/자석/가속 자리인지 설명할 수 있게 만들어라.\n\n"

    "핵심 방향:\n"
    "- 옵션벽, 핀, 자석, 눌러두기, 가짜 이탈, 위아래 흔들기, 방패가 가속 페달로 바뀌는 순간을 우선하라.\n"
    "- one_liner는 제목이 아니라 본문 첫 문장으로 바로 써도 될 정도로 강해야 한다.\n"
    "- core_thesis는 친절한 설명이 아니라 운전 가설이어야 한다.\n"
    "- participants_view는 누가 어디서 무엇을 팔고 싶은지 드러나야 한다.\n"
    "- key_levels는 가격 사전이 아니라, 왜 그 자리가 심리전의 무대인지 드러내야 한다.\n"
    "- 문장을 예쁘게 정리하려 하지 말고, 판의 중심축이 먼저 보이게 재료를 줘라.\n\n"

    "반드시 반영:\n"
    "- content_mode_kst\n"
    "- intraday(now_et, phase_et, flow_one_line, day_open, day_high, day_low, vwap_like, r_30m)\n"
    "- summary(top_gex, top_dex, top_oi)\n"
    "- trend_mode\n\n"

    "표현 가이드:\n"
    "- 기본은 단정적으로 써라.\n"
    "- '~가능성이 높다'는 꼭 필요할 때만 써라.\n"
    "- '오늘은 방향보다 순서가 중요하다', '방패가 가속 페달로 바뀐다', '공포를 파는 자리 / 희망을 파는 자리' 같은 압축된 사고를 선호한다.\n"
    "- 같은 뜻을 반복하지 마라.\n"
    "- key_levels는 최대 4개.\n"
    "- watch 2개, risks 2개까지만.\n"
    "- trend_mode가 off면 추이 해석 비중을 낮추고, high면 최근 흐름 변화와 반응성을 더 강조해라.\n\n"

    "절대 금지:\n"
    "- 매매지시, 손절/익절, 수량, 평단, 매수·매도 권유\n"
    "- QQQ 외 다른 종목/한국장/빅테크로 화제 전환\n"
    "- 장황한 거시경제 강의\n"
    "- 지나치게 무난하고 안전한 설명문\n\n"

    "content_mode_kst별 우선순위:\n"
    "- overnight_recap: 전일/야간 흐름과 오늘 밤 심리선 정리\n"
    "- noon_brief: 최근 며칠 흐름과 참여자 태도 설명\n"
    "- premarket_preview: 개장 전 벽, 자석, 가짜 방향 가능성 제시\n"
    "- intraday_live: 지금 가격 변화와 옵션 반응의 상호작용을 최우선\n"
    "- post_close_recap: 오늘 실제 운전과 벽의 효력을 복기\n\n"

    "출력 스키마:\n"
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

MAKA_WRITER_SYSTEM = """
You are 'Maka', a Korean writer who explains QQQ market structure.

Write only from the raw data in the brief.
Do not add macro events, news, earnings, Fed, calendar items, or any outside context unless they are explicitly present in the brief.

Primary goals:
- Explain today's market structure, not what the reader should buy or sell.
- Stay readable for a general QQQ or U.S. stock investor.
- Sound gentle, considerate, and calm.
- Build the explanation in steps instead of dumping all conclusions at once.

Writing rules:
- Korean output only.
- Start with "{TIME_KST_SHORT} KST / {TICKER} {SPOT}".
- Write 4 to 6 short paragraphs, with 1 to 3 sentences per paragraph.
- Paragraph flow: today's mood -> why the center price matters -> upper pressure -> lower pressure -> soft wrap-up.
- Avoid repeating the same idea, same price, or same jargon across paragraphs.
- Do not give trade instructions, entry ideas, stop ideas, target ideas, or checklist-style triggers.
- Prefer soft, considerate wording.
- If a term may be hard for a general investor, mark it in the body as `term*`.
- Add glossary lines only for terms that were actually marked with `*` in the body.
- Keep glossary explanations very short and friendly.
""".strip()

SILENT_FIX_SYSTEM = (
    "너는 감수 편집자다. 글을 새로 쓰지 마라.\n"
    "표시 없이 반영 수정만 수행해라.\n"
    "중복 단어, 중복 문장 리듬, 같은 숫자 반복을 강하게 줄여라.\n"
    "지나치게 친절한 설명문처럼 보이면 더 압축해라.\n"
    "문장이 너무 무난하면 더 단정적으로 다듬어라.\n"
    "후반부가 같은 뜻을 반복하면 1문단으로 압축해라.\n"
    "첫 줄의 시간/스팟 라인은 반드시 유지해라.\n"
    "리스트처럼 보이는 부분이 있으면 산문으로 바꿔라.\n"
    "출력은 최종 본문만."
)


MAKA_FINAL_FIX_SYSTEM = """
You are the final Korean editor for a preview draft.

Your job is to keep the writer's voice, but make the result gentler, clearer, and safer.

Edit rules:
- Keep the text in Korean.
- Do not add any new market facts or outside context.
- Remove action-guiding phrasing such as entry, target, trigger, confirm, chase, follow, or checklist-like if/then wording.
- Keep the build-up across paragraphs instead of collapsing everything into one dense paragraph.
- Make the tone more considerate and less blunt.
- It is good to keep soft, considerate phrasing when natural.
- If the body uses `term*`, keep only glossary lines for those exact starred terms.
- If a term is not starred in the body, do not explain it in the glossary.
- If no starred terms are used, remove the glossary entirely.
- Keep the first line time/ticker header.
- Output only the final polished text.
""".strip()

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
                        "trend_mode": meta.get("trend_mode"),
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
# Helpers
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
    return guides.get(content_mode, "현재 시점의 QQQ 옵션 구조와 참여자 심리를 정리한다.")


def get_trend_mode_guide(trend_mode: str) -> str:
    guides = {
        "off": "옵션 추이 해석은 최소화하고 현재 구조와 핵심 벽 중심으로만 본다.",
        "low": "옵션 추이는 약하게 반영하고 현재 구조를 우선한다.",
        "medium": "옵션 추이와 현재 구조를 균형 있게 본다.",
        "high": "최근 옵션 반응과 흐름 변화를 강하게 반영한다.",
    }
    return guides.get(trend_mode, "옵션 추이와 현재 구조를 균형 있게 본다.")


# =========================
# yfinance intraday
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
        flow = "정규장 밖에서는 위아래를 다 열어두되, 눈에 잘 보이는 자리부터 심리를 흔들기 좋은 흐름"
    else:
        if last >= vwap_like and r_30m >= 0:
            flow = "VWAP 위에서 버티며 위를 열어두는 흐름"
        elif last >= vwap_like and r_30m < 0:
            flow = "VWAP 위지만 탄력이 식으면서 흔들기 좋은 흐름"
        elif last < vwap_like and r_30m < 0:
            flow = "VWAP 아래로 눌리며 공포를 팔기 좋은 흐름"
        else:
            flow = "VWAP 아래지만 되감기 여지를 남긴 흐름"

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


def _nice_5_strike_ticks(xmin: float, xmax: float, step: int = 5):
    start = int(np.floor(xmin / step) * step)
    end = int(np.ceil(xmax / step) * step)
    return np.arange(start, end + step, step, dtype=float)


def _nice_window_around_spot(spot: float, half_window: float = 25.0):
    xmin = np.floor((spot - half_window) / 5.0) * 5.0
    xmax = np.ceil((spot + half_window) / 5.0) * 5.0
    return float(xmin), float(xmax)


def _annotate_key_lines(ax, levels: list, xmin: float, xmax: float, ymax_hint: float | None = None):
    if not isinstance(levels, list):
        return
    used = 0
    for lv in levels[:3]:
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
            if used >= 3:
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
    window = float(analysis_json.get("chart_window_usd", 25.0))
    window = max(20.0, min(40.0, window))
    xmin, xmax = _nice_window_around_spot(spot, half_window=window)

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

    ax1.bar(x - width * 0.20, pOI, width=width * 0.40, alpha=0.75, label="Put OI", color="#e74c3c")
    ax1.bar(x + width * 0.20, cOI, width=width * 0.40, alpha=0.75, label="Call OI", color="#3498db")
    ax1.set_ylabel("Open Interest")
    ax1.grid(axis="y", linestyle="--", alpha=0.35)

    ax1b = ax1.twinx()
    ax1b.plot(x, g, linewidth=2.0, label="Net GEX", color="#f1c40f")
    ax1b.set_ylabel("Net GEX")
    ax1b.set_ylim(*_clip_ylim(g))

    ax1.axvline(x=spot, color="black", linestyle="--", linewidth=1.8)
    ax2.axvline(x=spot, color="black", linestyle="--", linewidth=1.2, label=f"Spot {spot:.2f}")

    levels = analysis_json.get("key_levels", [])
    if isinstance(levels, list) and len(levels) > 0:
        def dist(lv):
            try:
                return abs(float(lv.get("price")) - spot)
            except Exception:
                return 1e9

        levels_sel = sorted(levels, key=dist)[:3]
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

    title = str(analysis_json.get("chart_title_en", "QQQ Options Positioning")).strip()[:48]
    what = str(analysis_json.get("chart_what_to_watch_en", "watch key strike reaction")).strip()
    what = re.sub(r"\s+", " ", what)[:64]
    subtitle = textwrap.fill(f"What to watch: {what}", width=62)
    ax1.set_title(f"{title}\n{subtitle}", fontsize=12)

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax1b.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper right")

    ax2.plot(x, d, linewidth=2.0, label="Net DEX", color="#2c3e50")
    ax2.set_xlabel("Strike")
    ax2.set_ylabel("Net DEX")
    ax2.grid(axis="y", linestyle="--", alpha=0.35)
    ax2.set_ylim(*_clip_ylim(d))
    ax2.legend(loc="upper right")
    ax2.set_xlim(xmin, xmax)

    ticks = _nice_5_strike_ticks(xmin, xmax, step=5)

    ax2.set_xticks(ticks)
    ax2.set_xticklabels([f"{v:.0f}" for v in ticks], fontsize=9)

    ax1.tick_params(axis="x", labelbottom=True)
    ax1.set_xticks(ticks)
    ax1.set_xticklabels([f"{v:.0f}" for v in ticks], fontsize=9)

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
# QQQ session chart
# =========================
def generate_qqq_session_chart_kst(outfile="qqq_price.png") -> str | None:
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


def normalize_analysis_json(
    analysis_json: dict,
    content_mode_kst: str,
    intraday: dict,
    pivot: float,
    spot: float,
    trend_mode: str,
) -> dict:
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
            "market_maker": "눈에 잘 보이는 가격대를 먼저 흔들며 양쪽 심리를 다 소진시키려는 태도",
            "whales": "두꺼운 벽이 있는 자리에서 유리한 재배치를 노리는 태도",
            "retail": "눈에 띄는 돌파와 명확한 지지 서사에 쉽게 끌릴 수 있는 상태",
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
    }.get(content_mode_kst, "watch key strike reaction")

    if trend_mode == "off":
        chart_watch_default = "current wall map around spot"
    elif trend_mode == "high" and content_mode_kst == "intraday_live":
        chart_watch_default = "live reaction and fast strike rotation"

    analysis_json["chart_title_en"] = str(analysis_json.get("chart_title_en") or chart_title_default)[:48]
    analysis_json["chart_what_to_watch_en"] = str(analysis_json.get("chart_what_to_watch_en") or chart_watch_default)[:64]
    analysis_json["chart_window_usd"] = float(analysis_json.get("chart_window_usd") or 25.0)

    if not isinstance(analysis_json.get("watch"), list):
        analysis_json["watch"] = []
    if not isinstance(analysis_json.get("risks"), list):
        analysis_json["risks"] = []

    analysis_json["watch"] = analysis_json["watch"][:2]
    analysis_json["risks"] = analysis_json["risks"][:2]

    if not isinstance(analysis_json.get("key_levels"), list):
        analysis_json["key_levels"] = []

    analysis_json["key_levels"] = analysis_json["key_levels"][:4]

    return analysis_json



def _fmt_price(value: object) -> str:
    try:
        number = float(value)
    except Exception:
        return "n/a"
    if abs(number - round(number)) < 0.005:
        return str(int(round(number)))
    return f"{number:.2f}"


def _fmt_signed_pct(value: object) -> str:
    try:
        return f"{float(value):+.2f}%"
    except Exception:
        return "n/a"


def _fmt_big_number(value: object) -> str:
    try:
        number = float(value)
    except Exception:
        return "n/a"
    if abs(number) >= 1_000_000_000:
        return f"{number / 1_000_000_000:+.2f}B"
    if abs(number) >= 1_000_000:
        return f"{number / 1_000_000:+.2f}M"
    if abs(number) >= 1_000:
        return f"{number / 1_000:+.1f}K"
    return f"{number:+.0f}"


def _clean_text(value: object, fallback: str = "") -> str:
    text = str(value or "").strip()
    return text or fallback


def _build_strike_surface_lines(
    strikes,
    call_oi,
    put_oi,
    net_gex,
    net_dex,
    *,
    spot: float,
    max_gap_pct: float = 0.10,
    min_total_oi: float = 1000.0,
) -> list[str]:
    rows: list[str] = []
    for strike, c_oi, p_oi, gex, dex in zip(strikes, call_oi, put_oi, net_gex, net_dex):
        total_oi = float(c_oi) + float(p_oi)
        if total_oi <= 0:
            continue
        gap = abs(float(strike) - spot)
        if spot > 0 and gap / spot > max_gap_pct:
            continue
        if total_oi < min_total_oi and gap > 10.0:
            continue
        rows.append(
            "- "
            + " | ".join(
                [
                    f"price {_fmt_price(strike)}",
                    f"spot_gap {float(strike) - spot:+.2f}",
                    f"call_oi {int(round(float(c_oi))):,}",
                    f"put_oi {int(round(float(p_oi))):,}",
                    f"total_oi {int(round(total_oi)):,}",
                    f"net_gex {_fmt_big_number(gex)}",
                    f"net_dex {_fmt_big_number(dex)}",
                ]
            )
        )
    return rows or ["- none"]


def choose_story_pivot(
    *,
    intraday_df,
    strikes,
    call_oi,
    put_oi,
    net_gex,
    net_dex,
    spot: float,
    day_high: float,
    day_low: float,
    pivot_mode: str,
    manual_pivot: float | None,
) -> tuple[float, str, str]:
    manual_value = float(manual_pivot) if manual_pivot is not None else None
    mode = (pivot_mode or "manual").strip().lower()

    if mode != "auto":
        if manual_value is not None:
            return manual_value, "manual", "manual pivot from config"
        return float(spot), "spot_fallback", "manual pivot missing; used spot as fallback"

    if spot <= 0:
        if manual_value is not None:
            return manual_value, "manual_fallback", "auto pivot unavailable because spot was invalid"
        return 0.0, "empty_fallback", "auto pivot unavailable because spot was invalid"

    window = max(6.0, min(15.0, abs(day_high - day_low) + 2.0))
    band_low = min(day_low, spot) - 1.0
    band_high = max(day_high, spot) + 1.0

    candidates: list[tuple[float, float, float, float, float, float]] = []
    for strike, c_oi, p_oi, gex, dex in zip(strikes, call_oi, put_oi, net_gex, net_dex):
        strike_f = float(strike)
        total_oi = float(c_oi) + float(p_oi)
        if total_oi <= 0:
            continue
        gap = abs(strike_f - spot)
        if gap > window and not (band_low <= strike_f <= band_high):
            continue
        touch_count = 0.0
        if intraday_df is not None and len(intraday_df) > 0:
            try:
                recent_df = intraday_df.tail(90)
                high = to_series(recent_df["High"]).astype(float).values
                low = to_series(recent_df["Low"]).astype(float).values
                band = max(0.45, min(0.90, abs(spot) * 0.001))
                touch_hits = ((low - band) <= strike_f) & ((high + band) >= strike_f)
                touch_count = float(touch_hits.sum())
            except Exception:
                touch_count = 0.0
        candidates.append((strike_f, total_oi, abs(float(gex)), abs(float(dex)), gap, touch_count))

    if not candidates:
        if manual_value is not None:
            return manual_value, "manual_fallback", "no near-spot pivot candidate met the auto filter"
        return float(spot), "spot_fallback", "no near-spot pivot candidate met the auto filter"

    max_touch = max(item[5] for item in candidates)
    if max_touch <= 0:
        if manual_value is not None:
            return manual_value, "manual_fallback", "auto pivot found no recent touch cluster"
        return float(spot), "spot_fallback", "auto pivot found no recent touch cluster"

    max_oi = max(item[1] for item in candidates) or 1.0
    max_abs_gex = max(item[2] for item in candidates) or 1.0
    max_abs_dex = max(item[3] for item in candidates) or 1.0

    best_score = float("-inf")
    best_strike = None
    for strike_f, total_oi, abs_gex, abs_dex, gap, touch_count in candidates:
        round_bonus = 0.08 if abs(strike_f - round(strike_f / 5.0) * 5.0) < 0.01 else 0.0
        score = (
            0.58 * (touch_count / max_touch)
            + 0.20 * (total_oi / max_oi)
            + 0.10 * (abs_dex / max_abs_dex)
            + 0.05 * (abs_gex / max_abs_gex)
            + 0.07 * max(0.0, 1.0 - (gap / max(window, 1.0)))
            + round_bonus
        )
        if score > best_score:
            best_score = score
            best_strike = strike_f

    if best_strike is None:
        if manual_value is not None:
            return manual_value, "manual_fallback", "auto pivot scoring did not produce a winner"
        return float(spot), "spot_fallback", "auto pivot scoring did not produce a winner"

    return float(best_strike), "auto", f"auto pivot from recent touch cluster plus near-spot options surface within +/-{window:.2f}"


def build_writer_brief(
    *,
    time_kst: str,
    time_kst_short: str,
    ticker: str,
    spot: float,
    pivot: float,
    pivot_source: str,
    content_mode_kst: str,
    mode_guide_kst: str,
    intraday: dict,
    summary: dict,
    strike_surface_lines: list[str],
) -> str:
    day_open = float(intraday.get("day_open") or spot)
    day_high = float(intraday.get("day_high") or spot)
    day_low = float(intraday.get("day_low") or spot)
    vwap_like = float(intraday.get("vwap_like") or spot)
    open_move_pct = ((spot / day_open) - 1.0) * 100.0 if day_open else 0.0

    phase_et = _clean_text(intraday.get("phase_et"), "unknown")
    now_et = _clean_text(intraday.get("now_et"), "unknown")
    flow_one_line = _clean_text(intraday.get("flow_one_line"), "No intraday flow note.")

    qqq_price_summary = (
        f"{ticker} {spot:.2f} (open {day_open:.2f}, high {day_high:.2f}, "
        f"low {day_low:.2f}, VWAP_like {vwap_like:.2f}), "
        f"{_fmt_signed_pct(open_move_pct)} from open; ET {phase_et}, 30m {_fmt_signed_pct(intraday.get('r_30m'))}."
    )

    session_mapping = (
        f"Current ET {now_et}. KST mode {content_mode_kst}; {mode_guide_kst} "
        f"Current flow note: {flow_one_line}"
    )

    options_structure = (
        f"Pivot {_fmt_price(pivot)} has {_clean_text(intraday.get('pivot_touches'), '0')} nearby touches. "
        f"Contracts pulled: {summary.get('contracts', 'n/a')}. "
        "The strike ladder below is limited to about +/-10% around spot. "
        "It is raw surface data: price, distance from spot, call OI, put OI, total OI, net GEX, and net DEX."
    )

    sections = [
        "[Sample Type]\nmain",
        f"[T in KST]\n{time_kst}",
        f"[QQQ Price Summary]\n{qqq_price_summary}",
        f"[Session Context]\n{session_mapping}",
        f"[Options Structure]\n{options_structure}",
        "[Strike Ladder Raw]\n" + "\n".join(strike_surface_lines),
        (
            "[Session Anchors Raw]\n"
            f"- spot {spot:.2f}\n"
            f"- pivot {pivot:.2f}\n"
            f"- pivot_source {pivot_source}\n"
            f"- vwap_like {vwap_like:.2f}\n"
            f"- day_high {day_high:.2f}\n"
            f"- day_low {day_low:.2f}"
        ),
        (
            "[Task]\n"
            f"Write the Maka-style QQQ market-structure commentary from the raw data above. "
            f"Start with \"{time_kst_short} KST / {ticker} {spot:.2f}\" and keep it to 4-6 short paragraphs. "
            "Focus on explaining the structure in a way that a general investor can follow. "
            "Let the explanation build gradually instead of dropping every conclusion in the first paragraph. "
            "Use a gentle, considerate tone. "
            "Do not give trade instructions. Do not turn price levels into targets, triggers, or action plans. "
            "If a term is hard, mark it with * in the body and explain only those starred terms in short glossary lines at the end. "
            "If no term needs explanation, do not add a glossary. "
            "Avoid repeating the same point. Do not add macro events, news, or calendar items unless they are explicitly present above."
        ),
    ]
    return "\n\n".join(sections)


# =========================
# Main pipeline
# =========================
def run(ticker: str = "QQQ", pivot: float | None = None, pivot_mode: str | None = None):
    manual_pivot = float(PIVOT_DEFAULT if pivot is None else pivot)
    pivot_mode_value = str(pivot_mode or PIVOT_MODE or "manual").strip().lower()
    if pivot_mode_value not in {"auto", "manual"}:
        pivot_mode_value = "manual"

    now_kst = datetime.now(ZoneInfo("Asia/Seoul"))
    time_kst = now_kst.strftime("%Y-%m-%d %H:%M")
    time_kst_short = now_kst.strftime("%H:%M")
    content_mode_kst = get_content_mode_kst(now_kst)
    mode_guide_kst = get_mode_guide_kst(content_mode_kst)
    trend_mode_guide = get_trend_mode_guide(TREND_MODE)

    run_id = now_kst.strftime("%Y-%m-%d_%H%M")
    base = ensure_dir(Path(OUTPUTS_DIR) / "posts" / run_id)

    intraday_df, intraday_interval = fetch_intraday_df(ticker)
    intraday = fetch_intraday_context(ticker, pivot=manual_pivot)
    spot, spot_source = get_spot_robust(ticker, intraday)

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

    top_gex_idx = np.argsort(np.abs(net_gex))[::-1][:10]
    top_dex_idx = np.argsort(np.abs(net_dex))[::-1][:10]
    top_oi_idx = np.argsort((call_oi + put_oi))[::-1][:12]
    summary = {
        "contracts": len(contracts),
        "top_gex": [{"strike": float(strikes[i]), "net_gex": float(net_gex[i])} for i in top_gex_idx],
        "top_dex": [{"strike": float(strikes[i]), "net_dex": float(net_dex[i])} for i in top_dex_idx],
        "top_oi": [{"strike": float(strikes[i]), "call_oi": float(call_oi[i]), "put_oi": float(put_oi[i])} for i in top_oi_idx],
    }

    pivot, pivot_source, pivot_reason = choose_story_pivot(
        intraday_df=intraday_df,
        strikes=strikes,
        call_oi=call_oi,
        put_oi=put_oi,
        net_gex=net_gex,
        net_dex=net_dex,
        spot=spot,
        day_high=float(intraday.get("day_high") or spot),
        day_low=float(intraday.get("day_low") or spot),
        pivot_mode=pivot_mode_value,
        manual_pivot=manual_pivot,
    )

    intraday = fetch_intraday_context(ticker, pivot=pivot)

    payload = {
        "time_kst": time_kst,
        "content_mode_kst": content_mode_kst,
        "mode_guide_kst": mode_guide_kst,
        "trend_mode": TREND_MODE,
        "trend_mode_guide": trend_mode_guide,
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
        trend_mode=TREND_MODE,
    )

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
    strike_surface_lines = _build_strike_surface_lines(
        strikes,
        call_oi,
        put_oi,
        net_gex,
        net_dex,
        spot=spot,
    )
    user_prompt = build_writer_brief(
        time_kst=time_kst,
        time_kst_short=time_kst_short,
        ticker=ticker,
        spot=spot,
        pivot=pivot,
        pivot_source=pivot_source,
        content_mode_kst=content_mode_kst,
        mode_guide_kst=mode_guide_kst,
        intraday=intraday,
        summary=summary,
        strike_surface_lines=strike_surface_lines,
    )

    r2 = client.chat.completions.create(
        model=MY_MODEL_ID,
        messages=[
            {"role": "system", "content": KOREAN_SYSTEM},
            {"role": "system", "content": MAKA_WRITER_SYSTEM},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.8,
    )
    maka_body = (r2.choices[0].message.content or "").strip()

    r3 = client.responses.create(
        model=PRO_MODEL_ID,
        input=[
            {"role": "system", "content": KOREAN_SYSTEM},
            {"role": "system", "content": SILENT_FIX_SYSTEM},
            {"role": "system", "content": MAKA_FINAL_FIX_SYSTEM},
            {"role": "user", "content": json.dumps({"brief": user_prompt, "text": maka_body}, ensure_ascii=False)},
        ],
    )
    final_body = (r3.output_text or "").strip()

    write_text(base / "post.md", final_body)

    meta = {
        "run_id": run_id,
        "time_kst": time_kst,
        "ticker": ticker,
        "pivot": pivot,
        "pivot_source": pivot_source,
        "pivot_reason": pivot_reason,
        "pivot_mode": pivot_mode_value,
        "spot": spot,
        "spot_source": spot_source,
        "content_mode_kst": content_mode_kst,
        "mode_guide_kst": mode_guide_kst,
        "trend_mode": TREND_MODE,
        "trend_mode_guide": trend_mode_guide,
        "intraday_interval": intraday_interval,
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
        "trend_mode": TREND_MODE,
        "pivot": pivot,
        "pivot_source": pivot_source,
    }
    write_json(Path(OUTPUTS_DIR) / "latest.json", latest)

    print("\n" + "=" * 70)
    print(f"[Maka Body - {time_kst} | {content_mode_kst} | trend={TREND_MODE} | pivot={pivot:.2f} ({pivot_source})]")
    print("=" * 70)
    print(final_body)
    print("=" * 70)
    print(f"[saved] {base}")
    print(f"[chart] {chart_file}")
    print(f"[qqq]   {qqq_price_file}")

    build_public_index(OUTPUTS_DIR)


if __name__ == "__main__":
    run(RUN_TICKER, PIVOT_DEFAULT, PIVOT_MODE)
