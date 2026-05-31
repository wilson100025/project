import ast
import json
import os
import re
import subprocess
import sys
from typing import Dict, List, Optional, Tuple

import gradio as gr
import matplotlib.pyplot as plt

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
body {
  font-family: 'Space Grotesk', 'Noto Sans TC', sans-serif;
  background: radial-gradient(circle at 10% 10%, #1d2332, #0f1117 50%, #0b0e14 100%);
}
.topbar {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 12px 16px;
  border-radius: 16px;
  background: rgba(15,17,23,0.85);
  border: 1px solid rgba(249,199,79,0.2);
}
.card {
  background: rgba(23,27,38,0.8);
  border-radius: 12px;
  padding: 12px;
  border: 1px solid rgba(255,255,255,0.08);
  margin-bottom: 10px;
}
.card-title {
  font-weight: 600;
  margin-bottom: 4px;
}
.card-meta {
  color: var(--muted);
  font-size: 12px;
}
.empty-state {
  color: var(--muted);
  padding: 12px;
}
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


def render_firefly_cards(records: List[dict]) -> str:
    if not records:
        return f"<div class='empty-state'>{TEXT_EMPTY_FIREFLY}</div>"
    blocks: List[str] = []
    for record in records:
        duration = record.get("duration_s")
        duration_text = f"{duration:.2f}s" if duration is not None else "-"
        blocks.append(
            "<div class='card'>"
            f"<div class='card-title'>ID {record['track_id']}</div>"
            f"<div class='card-meta'>{TEXT_DURATION}: {duration_text}</div>"
            "</div>"
        )
    return "\n".join(blocks)


def build_path_plot(record: dict) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(5, 3))
    path = record.get("path", [])
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
    return fig


def get_video_names() -> List[str]:
    return [os.path.basename(path) for path in scan_videos(INPUT_DIR)]


def refresh_videos(firefly_db: Dict[str, List[dict]]):
    video_names = get_video_names()
    selected = video_names[0] if video_names else None
    path = os.path.join(INPUT_DIR, selected) if selected else None
    records = firefly_db.get(selected, []) if selected else []
    firefly_choices = [str(r["track_id"]) for r in records]
    return (
        gr.Dropdown.update(choices=video_names, value=selected),
        path,
        gr.Dropdown.update(choices=firefly_choices, value=firefly_choices[0] if firefly_choices else None),
        render_firefly_cards(records),
    )


def on_video_change(video_name: str, firefly_db: Dict[str, List[dict]]):
    if not video_name:
        return None, gr.Dropdown.update(choices=[], value=None), render_firefly_cards([])
    path = os.path.join(INPUT_DIR, video_name)
    records = firefly_db.get(video_name, [])
    firefly_choices = [str(r["track_id"]) for r in records]
    return (
        path,
        gr.Dropdown.update(choices=firefly_choices, value=firefly_choices[0] if firefly_choices else None),
        render_firefly_cards(records),
    )


def analyze_videos(firefly_db: Dict[str, List[dict]]):
    records, logs = run_project()
    firefly_db = normalize_records(records)
    save_summary_json(records)

    video_names = get_video_names()
    selected = video_names[0] if video_names else None
    path = os.path.join(INPUT_DIR, selected) if selected else None
    selected_records = firefly_db.get(selected, []) if selected else []
    firefly_choices = [str(r["track_id"]) for r in selected_records]
    status = "\n".join(logs[-50:]) if logs else ""

    return (
        firefly_db,
        gr.Dropdown.update(choices=video_names, value=selected),
        path,
        gr.Dropdown.update(choices=firefly_choices, value=firefly_choices[0] if firefly_choices else None),
        render_firefly_cards(selected_records),
        status,
    )


def browse_firefly(video_name: str, firefly_id: str, firefly_db: Dict[str, List[dict]]):
    if not video_name or not firefly_id:
        return TEXT_DETAIL_TITLE, None, TEXT_EMPTY_FIREFLY

    records = firefly_db.get(video_name, [])
    target = None
    for record in records:
        if str(record.get("track_id")) == str(firefly_id):
            target = record
            break

    if not target:
        return TEXT_DETAIL_TITLE, None, TEXT_EMPTY_FIREFLY

    duration = target.get("duration_s")
    duration_text = f"{duration:.2f}s" if duration is not None else "-"
    details = (
        f"**ID**: {target['track_id']}\n\n"
        f"**{TEXT_DURATION}**: {duration_text}\n\n"
        f"**BGR**: {target.get('brightest_bgr')}"
    )
    return TEXT_DETAIL_TITLE, build_path_plot(target), details


initial_db = normalize_records(load_summary_json())
initial_videos = get_video_names()
initial_video = initial_videos[0] if initial_videos else None
initial_cards = render_firefly_cards(initial_db.get(initial_video, [])) if initial_video else render_firefly_cards([])
initial_fireflies = [str(r["track_id"]) for r in initial_db.get(initial_video, [])] if initial_video else []
initial_path = os.path.join(INPUT_DIR, initial_video) if initial_video else None

with gr.Blocks(css=CSS) as demo:
    gr.HTML(
        f"<div class='topbar'><div><strong>{TEXT_MENU_BAR}</strong></div>"
        f"<div>{TEXT_FILE} | {TEXT_SETTINGS}</div></div>"
    )

    with gr.Row():
        with gr.Column(scale=3):
            video_view = gr.Video(value=initial_path, label=TEXT_VIDEO_PROGRESS)
            progress = gr.Slider(0, 100, value=0, label=TEXT_VIDEO_PROGRESS)
            with gr.Row():
                play_btn = gr.Button(TEXT_PLAY)
                pause_btn = gr.Button(TEXT_PAUSE)
                speed = gr.Dropdown(["0.25x", "0.5x", "1x", "1.5x", "2x"], value="1x", label=TEXT_SPEED)

        with gr.Column(scale=2):
            video_dd = gr.Dropdown(choices=initial_videos, value=initial_video, label=TEXT_VIDEO_LIST)
            refresh_btn = gr.Button(TEXT_REFRESH)
            analyze_btn = gr.Button(TEXT_ANALYZE)
            firefly_dd = gr.Dropdown(choices=initial_fireflies, label=TEXT_FIREFLY_LIST)
            browse_btn = gr.Button(TEXT_BROWSE)
            firefly_cards = gr.HTML(value=initial_cards)
            status_box = gr.Textbox(label="Log", lines=6)

    detail_title = gr.Markdown(TEXT_DETAIL_TITLE)
    with gr.Row():
        path_plot = gr.Plot()
        detail_info = gr.Markdown(TEXT_EMPTY_FIREFLY)

    firefly_state = gr.State(initial_db)

    refresh_btn.click(
        refresh_videos,
        inputs=[firefly_state],
        outputs=[video_dd, video_view, firefly_dd, firefly_cards],
    )

    video_dd.change(
        on_video_change,
        inputs=[video_dd, firefly_state],
        outputs=[video_view, firefly_dd, firefly_cards],
    )

    analyze_btn.click(
        analyze_videos,
        inputs=[firefly_state],
        outputs=[firefly_state, video_dd, video_view, firefly_dd, firefly_cards, status_box],
    )

    browse_btn.click(
        browse_firefly,
        inputs=[video_dd, firefly_dd, firefly_state],
        outputs=[detail_title, path_plot, detail_info],
    )


if __name__ == "__main__":
    demo.launch()
