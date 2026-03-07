import json
import os
import re
import shutil
import sys
from pathlib import Path


BASE_DIR = Path("web/data")
POSTS_DIR = BASE_DIR / "posts"
INDEX_PATH = BASE_DIR / "index.json"
LATEST_PATH = BASE_DIR / "latest.json"


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: dict):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def normalize_run_ids(raw: str) -> list[str]:
    """
    쉼표, 줄바꿈, 공백으로 섞여 들어와도 run_id 목록으로 정리
    """
    if not raw:
        return []

    # 쉼표/줄바꿈/탭/여러 공백 전부 분리
    parts = re.split(r"[\s,]+", raw.strip())

    # 빈값 제거
    parts = [p.strip() for p in parts if p.strip()]

    # 중복 제거, 순서 유지
    seen = set()
    result = []
    for x in parts:
        if x not in seen:
            seen.add(x)
            result.append(x)

    return result


def delete_runs(run_ids: list[str]):
    deleted = []
    missing = []

    for run_id in run_ids:
        target = POSTS_DIR / run_id
        if target.exists() and target.is_dir():
            shutil.rmtree(target)
            deleted.append(run_id)
        else:
            missing.append(run_id)

    return deleted, missing


def collect_items():
    items = []

    if not POSTS_DIR.exists():
        return items

    for d in POSTS_DIR.iterdir():
        if not d.is_dir():
            continue

        meta_path = d / "meta.json"
        post_path = d / "post.md"

        if not meta_path.exists() or not post_path.exists():
            continue

        try:
            meta = read_json(meta_path)
        except Exception:
            continue

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

    items.sort(key=lambda x: x.get("run_id", ""), reverse=True)
    return items


def rebuild_index_and_latest():
    items = collect_items()

    index_obj = {
        "count": len(items),
        "items": items,
    }
    write_json(INDEX_PATH, index_obj)

    if items:
        latest_run_id = items[0]["run_id"]
        latest_meta_path = POSTS_DIR / latest_run_id / "meta.json"

        latest_meta = {}
        if latest_meta_path.exists():
            try:
                latest_meta = read_json(latest_meta_path)
            except Exception:
                latest_meta = {}

        latest_obj = {
            "latest_run_id": latest_run_id,
            "updated_kst": latest_meta.get("time_kst"),
            "meta_path": f"data/posts/{latest_run_id}/meta.json",
            "content_mode_kst": latest_meta.get("content_mode_kst"),
        }
    else:
        latest_obj = {
            "latest_run_id": None,
            "updated_kst": None,
            "meta_path": None,
            "content_mode_kst": None,
        }

    write_json(LATEST_PATH, latest_obj)


def main():
    raw = os.getenv("RUN_IDS", "").strip()

    if not raw and len(sys.argv) > 1:
        raw = " ".join(sys.argv[1:]).strip()

    run_ids = normalize_run_ids(raw)

    if not run_ids:
        print("No run_ids provided.")
        sys.exit(1)

    if not POSTS_DIR.exists():
        print(f"Posts directory does not exist: {POSTS_DIR}")
        sys.exit(1)

    print("Targets:")
    for rid in run_ids:
        print(" -", rid)

    deleted, missing = delete_runs(run_ids)

    print("\nDeleted:")
    for rid in deleted:
        print(" -", rid)

    print("\nMissing:")
    for rid in missing:
        print(" -", rid)

    rebuild_index_and_latest()

    print("\nRebuilt:")
    print(" -", INDEX_PATH)
    print(" -", LATEST_PATH)

    if not deleted:
        print("\nNo posts were deleted.")
        sys.exit(1)

    print("\nDone.")


if __name__ == "__main__":
    main()