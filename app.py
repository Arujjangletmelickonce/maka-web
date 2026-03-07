import json
from pathlib import Path
from datetime import datetime
import streamlit as st

st.set_page_config(page_title="QQQ Maka", layout="wide")

# 실제 운영 경로에 맞게 필요하면 outputs -> web/data 로 바꾸세요.
OUT_DIR = Path("outputs")
LATEST = OUT_DIR / "latest.json"
POSTS = OUT_DIR / "posts"


def read_json(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def read_text(p: Path):
    return p.read_text(encoding="utf-8")


def get_week_label_from_time_kst(time_kst: str) -> str:
    """
    time_kst 예: '2026-03-07 03:30:12'
    또는    '2026-03-07 03:30'
    """
    if not time_kst:
        return "기타"

    dt = None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
        try:
            dt = datetime.strptime(time_kst, fmt)
            break
        except Exception:
            pass

    if dt is None:
        return "기타"

    day = dt.day

    if 1 <= day <= 7:
        week_no = 1
    elif 8 <= day <= 14:
        week_no = 2
    elif 15 <= day <= 21:
        week_no = 3
    elif 22 <= day <= 28:
        week_no = 4
    else:
        week_no = 5

    return f"{dt.year}년 {dt.month}월 {week_no}주차"


def make_post_label(meta: dict, run_id: str) -> str:
    time_kst = str(meta.get("time_kst", ""))
    spot = meta.get("spot")
    mode = str(meta.get("content_mode_kst", "")).strip()

    mmdd_hhmm = time_kst
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
        try:
            dt = datetime.strptime(time_kst, fmt)
            mmdd_hhmm = dt.strftime("%m-%d %H:%M")
            break
        except Exception:
            pass

    if isinstance(spot, (int, float)):
        spot_text = f"{spot:.2f}"
    else:
        try:
            spot_text = f"{float(spot):.2f}"
        except Exception:
            spot_text = "-"

    if not mode:
        mode = "mode?"

    return f"{mmdd_hhmm} / QQQ {spot_text} / {mode}"


def list_runs_with_meta():
    if not POSTS.exists():
        return []

    items = []
    run_dirs = [d for d in POSTS.iterdir() if d.is_dir()]
    run_dirs.sort(key=lambda x: x.name, reverse=True)

    for d in run_dirs:
        meta_path = d / "meta.json"
        if not meta_path.exists():
            continue

        try:
            meta = read_json(meta_path)
        except Exception:
            continue

        week_label = get_week_label_from_time_kst(str(meta.get("time_kst", "")))
        post_label = make_post_label(meta, d.name)

        items.append(
            {
                "run_id": d.name,
                "meta": meta,
                "week_label": week_label,
                "post_label": post_label,
            }
        )

    return items


def group_runs_by_week(items: list[dict]):
    grouped = {}

    for item in items:
        key = item["week_label"]
        if key not in grouped:
            grouped[key] = []
        grouped[key].append(item)

    # 주차 그룹도 최신순
    week_keys = sorted(grouped.keys(), reverse=True)

    return grouped, week_keys


def render_run(run_id: str):
    base = POSTS / run_id
    meta = read_json(base / "meta.json")

    st.caption(
        f"KST {meta.get('time_kst')} · Spot {meta.get('spot')} ({meta.get('spot_source')})"
    )

    files = meta.get("files") or {}

    if files.get("maka_chart"):
        p = base / files["maka_chart"]
        if p.exists():
            st.image(str(p), use_container_width=True)

    if files.get("qqq_price"):
        p = base / files["qqq_price"]
        if p.exists():
            st.image(str(p), use_container_width=True)

    st.markdown(read_text(base / "post.md"))


tab1, tab2 = st.tabs(["최신", "아카이브"])

with tab1:
    if not LATEST.exists():
        st.info("아직 최신 결과가 없습니다. 먼저 maka_pro_story.py를 한 번 실행하세요.")
    else:
        rid = read_json(LATEST).get("latest_run_id")
        if rid:
            render_run(rid)
        else:
            st.error("latest.json 형식이 이상합니다.")

with tab2:
    items = list_runs_with_meta()

    if not items:
        st.info("아카이브가 비어 있습니다.")
    else:
        grouped, week_keys = group_runs_by_week(items)

        selected_week = st.selectbox(
            "주차 선택",
            week_keys,
            index=0
        )

        week_items = grouped[selected_week]

        label_to_run_id = {
            item["post_label"]: item["run_id"]
            for item in week_items
        }

        selected_label = st.selectbox(
            "게시글 선택",
            list(label_to_run_id.keys()),
            index=0
        )

        selected_run_id = label_to_run_id[selected_label]
        render_run(selected_run_id)