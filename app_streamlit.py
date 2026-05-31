import ast
import json
import os
import re
import subprocess
import sys
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import streamlit as st

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(BASE_DIR, "input_videos")
OUTPUT_DIR = os.path.join(BASE_DIR, "output_videos")
SUMMARY_JSON = os.path.join(OUTPUT_DIR, "firefly_summary.json")
PROJECT_SCRIPT = os.path.join(BASE_DIR, "project.py")
VIDEO_EXTENSIONS = [".mp4", ".avi", ".mov", ".MTS"]

TEXT_MENU_BAR = "\u529f\u80fd\u5217"
TEXT_FILE = "\u6a94\u6848"
TEXT_SETTINGS = "\u8a2d\u5b9a"
TEXT_REFRESH = "\u91cd\u65b0\u6574\u7406"
TEXT_ANALYZE = "\u5206\u6790\u5f71\u7247"
TEXT_VIDEO_LIST = "\u5f71\u7247\u5217\u8868"
TEXT_FIREFLY_LIST = "\u87a2\u706b\u87f2\u5217\u8868"
TEXT_PLAY = "\u64ad\u653e"
TEXT_PAUSE = "\u66ab\u505c"
TEXT_SPEED = "\u8abf\u6574\u5f71\u7247\u64ad\u653e\u901f\u5ea6"
TEXT_VIDEO_PROGRESS = "\u5f71\u7247\u9032\u5ea6\u689d"
TEXT_BROWSE = "\u700f\u89bd"
TEXT_DETAIL_TITLE = "\u87a2\u706b\u87f2\u8cc7\u6599"
TEXT_PATH_CHART = "\u8def\u5f91\u5716"
TEXT_DETAIL_INFO = "\u8a73\u7d30\u8cc7\u6599"
TEXT_DURATION = "\u6642\u9577"
TEXT_EMPTY_FIREFLY = "\u5c1a\u7121\u87a2\u706b\u87f2\u8cc7\u6599\uff0c\u8acb\u5148\u5206\u6790\u5f71\u7247"
TEXT_EMPTY_VIDEO = "\u627e\u4e0d\u5230\u5f71\u7247\uff0c\u8acb\u5c07\u6a94\u6848\u653e\u5165 input_videos"

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=Noto+Sans+TC:wght@400;600&display=swap');
:root {
  --bg-0: #0f1117;
  --bg-1: #171b26;
  --panel: #151a23;
  --accent: #3ddc97;
  --accent-2: #f9c74f;
  --text: #e6edf3;
  --muted: #9aa4b2;
}
body, .stApp {
  font-family: 'Space Grotesk', 'Noto Sans TC', sans-serif;
  color: var(--text);
  background: radial-gradient(circle at 10% 10%, #1d2332, #0f1117 50%, #0b0e14 100%);
}
.section-card {
  background: linear-gradient(135deg, rgba(61,220,151,0.08), rgba(25,30,45,0.9));
  border: 1px solid rgba(61,220,151,0.2);
  border-radius: 16px;
  padding: 16px;
  box-shadow: 0 10px 30px rgba(0,0,0,0.2);
}
.topbar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
  padding: 12px 16px;
  background: rgba(15,17,23,0.85);
  border-radius: 16px;
  border: 1px solid rgba(249,199,79,0.2);
}
.menu-label {
  font-size: 12px;
  color: var(--muted);
  text-transform: uppercase;
  letter-spacing: 0.08em;
}
.menu-title {
  font-size: 20px;
  font-weight: 700;
}
.card {
  background: rgba(23,27,38,0.8);
  border-radius: 12px;
  padding: 12px;
  border: 1px solid rgba(255,255,255,0.08);
  margin-bottom: 12px;
}
.card h4 {
  margin: 0 0 6px 0;
  font-size: 16px;
}
.card p {
  margin: 0;
  color: var(--muted);
}
</style>
"""


def scan_videos(input_dir: str) -> List[str]:
    items: List[str] = []
    if not os.path.isdir(input_dir):
        return items
    for name in os.listdir(input_dir):
        ext = os.path.splitext(name)[1]
        if ext in VIDEO_EXTENSIONS:
            items.append(os.path.join(input_dir, name))
    items.sort()
    return items


def parse_project_output(lines: List[str]) -> Dict[str, List[dict]]:
    records: Dict[str, List[dict]] = {}
    current_video: Optional[str] = None
    pending: Optional[dict] = None

    start_marker = "\u958b\u59cb\u8655\u7406\u5f71\u7247"
    dur_marker = "\u6642\u9577"
    path_marker = "\u79fb\u52d5\u8ecc\u8de1"
    bright_marker = "\u6700\u4eae BGR"

    for raw in lines:
        line = raw.strip()
        if start_marker in line:
            current_video = line.split(":")[-1].strip()
            records.setdefault(current_video, [])
            pending = None
            continue

        if "[Track Summary]" in line:
            match = re.search(r"ID:\s*(\d+)", line)
            if match:
                pending = {
                    "track_id": int(match.group(1)),
                    "duration_s": None,
                    "path": [],
                    "brightest_bgr": None,
                }
            continue

        if pending and line.startswith(dur_marker):
            value = line.split(":")[-1].strip().replace("s", "")
            try:
                pending["duration_s"] = float(value)
            except ValueError:
                pending["duration_s"] = None
            continue

        if pending and line.startswith(path_marker):
            payload = line.split(":", 1)[-1].strip()
            try:
                pending["path"] = ast.literal_eval(payload)
            except (ValueError, SyntaxError):
                pending["path"] = []
            continue

        if pending and line.startswith(bright_marker):
            payload = line.split(":", 1)[-1].strip()
            try:
                pending["brightest_bgr"] = tuple(ast.literal_eval(payload))
            except (ValueError, SyntaxError):
                pending["brightest_bgr"] = None
            if current_video is None:
                current_video = "unknown"
                records.setdefault(current_video, [])
            records[current_video].append(pending)
            pending = None

    return records


def load_summary_json() -> Dict[str, List[dict]]:
    if not os.path.isfile(SUMMARY_JSON):
        return {}
    try:
        with open(SUMMARY_JSON, "r", encoding="utf-8") as handle:
            data = json.load(handle)
            if isinstance(data, dict):
                return data
    except (OSError, json.JSONDecodeError):
        pass
    return {}


def save_summary_json(data: Dict[str, List[dict]]) -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(SUMMARY_JSON, "w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)


def run_project() -> Tuple[Dict[str, List[dict]], List[str]]:
    if not os.path.isfile(PROJECT_SCRIPT):
        return {}, ["project.py not found"]

    cmd = [sys.executable, PROJECT_SCRIPT]
    result = subprocess.run(
        cmd,
        cwd=BASE_DIR,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    output = (result.stdout or "").splitlines()
    if result.stderr:
        output.extend(result.stderr.splitlines())
    return parse_project_output(output), output


def normalize_records(data: Dict[str, List[dict]]) -> Dict[str, List[dict]]:
    cleaned: Dict[str, List[dict]] = {}
    for video_name, records in data.items():
        cleaned[video_name] = []
        for record in records:
            path = [tuple(p) for p in record.get("path", [])]
            cleaned[video_name].append(
                {
                    "track_id": int(record.get("track_id", 0)),
                    "duration_s": record.get("duration_s"),
                    "path": path,
                    "brightest_bgr": record.get("brightest_bgr"),
                }
            )
    return cleaned


st.set_page_config(page_title="Firefly Tracker", layout="wide")
st.markdown(CSS, unsafe_allow_html=True)

if "firefly_db" not in st.session_state:
    st.session_state.firefly_db = normalize_records(load_summary_json())

if "selected_firefly" not in st.session_state:
    st.session_state.selected_firefly = None

videos = scan_videos(INPUT_DIR)
video_names = [os.path.basename(path) for path in videos]

with st.container():
    st.markdown(
        f"<div class='topbar'><div class='menu-title'>{TEXT_MENU_BAR}</div>"
        "<div style='display:flex; gap:12px; align-items:center;'>"
        f"<span class='menu-label'>{TEXT_FILE}</span>"
        f"<span class='menu-label'>{TEXT_SETTINGS}</span>"
        "</div></div>",
        unsafe_allow_html=True,
    )

left, right = st.columns([3, 2], gap="large")

selected_video: Optional[str]
if video_names:
    selected_video = video_names[0]
else:
    selected_video = None

with left:
    st.markdown(f"<div class='section-card'><h3>{TEXT_VIDEO_PROGRESS}</h3>", unsafe_allow_html=True)

    if selected_video:
        selected_path = os.path.join(INPUT_DIR, selected_video)
        if os.path.isfile(selected_path):
            st.video(selected_path)
    else:
        st.info(TEXT_EMPTY_VIDEO)

    st.slider(TEXT_VIDEO_PROGRESS, 0, 100, value=0)
    play_col, pause_col, speed_col = st.columns([1, 1, 2])
    play_col.button(TEXT_PLAY)
    pause_col.button(TEXT_PAUSE)
    speed_col.selectbox(TEXT_SPEED, ["0.25x", "0.5x", "1x", "1.5x", "2x"], index=2)
    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown(f"<div class='section-card'><h3>{TEXT_VIDEO_LIST}</h3>", unsafe_allow_html=True)

    if video_names:
        selected_video = st.selectbox(TEXT_VIDEO_LIST, video_names, key="video_select")
    else:
        selected_video = None

    refresh_clicked = st.button(TEXT_REFRESH)
    analyze_clicked = st.button(TEXT_ANALYZE)

    if refresh_clicked:
        st.session_state.firefly_db = normalize_records(load_summary_json())

    if analyze_clicked:
        with st.spinner(TEXT_ANALYZE):
            records, logs = run_project()
            st.session_state.firefly_db = normalize_records(records)
            save_summary_json(records)
            if logs:
                st.code("\n".join(logs[-200:]))

    st.markdown(f"</div><div class='section-card'><h3>{TEXT_FIREFLY_LIST}</h3>", unsafe_allow_html=True)

    firefly_records = st.session_state.firefly_db.get(selected_video or "", [])
    if not firefly_records:
        st.info(TEXT_EMPTY_FIREFLY)
    else:
        for record in firefly_records:
            st.markdown(
                "<div class='card'>"
                f"<h4>ID {record['track_id']}</h4>"
                f"<p>{TEXT_DURATION}: {record['duration_s']}</p>"
                "</div>",
                unsafe_allow_html=True,
            )
            if st.button(TEXT_BROWSE, key=f"browse_{selected_video}_{record['track_id']}"):
                st.session_state.selected_firefly = record

    st.markdown("</div>", unsafe_allow_html=True)

if st.session_state.selected_firefly:
    record = st.session_state.selected_firefly
    st.markdown(f"<div class='section-card'><h3>{TEXT_DETAIL_TITLE}</h3>", unsafe_allow_html=True)
    col_a, col_b = st.columns([2, 1])

    with col_a:
        path = record.get("path", [])
        fig, ax = plt.subplots(figsize=(6, 4))
        if path:
            xs = [p[0] for p in path]
            ys = [p[1] for p in path]
            ax.plot(xs, ys, color="#3ddc97")
            ax.scatter([xs[0]], [ys[0]], color="#f9c74f")
            ax.scatter([xs[-1]], [ys[-1]], color="#f94144")
            ax.set_title(TEXT_PATH_CHART)
        else:
            ax.text(0.5, 0.5, TEXT_PATH_CHART, ha="center", va="center")
        ax.set_facecolor("#0f1117")
        ax.tick_params(colors="#9aa4b2")
        st.pyplot(fig)

    with col_b:
        st.markdown(f"**ID**: {record['track_id']}")
        st.markdown(f"**{TEXT_DURATION}**: {record['duration_s']}")
        st.markdown(f"**BGR**: {record.get('brightest_bgr')}")

    st.markdown("</div>", unsafe_allow_html=True)
