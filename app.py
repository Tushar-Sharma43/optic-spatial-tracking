import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import tempfile
import torch
import os
import time

st.set_page_config(
    page_title="OPTIC",
    page_icon="◎",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500&family=IBM+Plex+Mono:wght@300;400;500&display=swap');

:root {
    --bg:        #0C0C0E;
    --surface:   #111114;
    --border:    #1E1E24;
    --border2:   #2A2A32;
    --muted:     #3A3A48;
    --subtle:    #6B6B80;
    --text:      #E2E2EE;
    --text-dim:  #8888A0;
    --accent:    #7B68EE;     
    --accent-dim:#3D3580;
    --sans:      'Inter', sans-serif;
    --mono:      'IBM Plex Mono', monospace;
}

*, *::before, *::after { box-sizing: border-box; }
html, body, [class*="css"] {
    font-family: var(--sans);
    background: var(--bg);
    color: var(--text);
}
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2.5rem 2.5rem 1rem; max-width: 100%; }

[data-testid="stSidebar"] { background: var(--surface); border-right: 1px solid var(--border); }
[data-testid="stSidebar"] .block-container { padding: 2.5rem 1.8rem; }

[data-testid="stRadio"] > div { gap: 4px !important; }
[data-testid="stRadio"] label span {
    font-family: var(--mono) !important; font-size: 11px !important;
    color: var(--subtle) !important; letter-spacing: 0.04em;
}
[data-testid="stRadio"] label:has(input:checked) span { color: var(--text) !important; }

div[data-baseweb="select"] > div {
    background-color: transparent !important;
    border: 1px dashed var(--border2) !important;
    color: var(--text) !important;
    font-family: var(--mono) !important;
    font-size: 10px !important;
}
div[data-baseweb="select"] span { color: var(--text-dim) !important; }

[data-testid="stSlider"] > label {
    font-family: var(--mono) !important; font-size: 10px !important;
    letter-spacing: 0.12em; color: var(--subtle) !important; text-transform: uppercase;
}
[data-testid="stSlider"] [data-testid="stTickBarMin"],
[data-testid="stSlider"] [data-testid="stTickBarMax"] {
    font-family: var(--mono) !important; font-size: 9px !important; color: var(--muted) !important;
}
[data-testid="stSlider"] > div > div > div > div:first-child {
    background: var(--border2) !important; height: 2px !important; border-radius: 2px !important;
}
[data-testid="stSlider"] > div > div > div > div:nth-child(2) {
    background: var(--accent) !important; height: 2px !important;
}
[data-testid="stSlider"] > div > div > div > div[role="slider"] {
    background: var(--accent) !important; border: 2px solid var(--bg) !important;
    width: 14px !important; height: 14px !important; border-radius: 50% !important;
    box-shadow: 0 0 0 2px var(--accent-dim) !important; cursor: pointer !important;
}

div.stButton > button:first-child {
    background: transparent; color: var(--text-dim); border: 1px solid var(--border2);
    font-family: var(--mono); font-size: 10px; letter-spacing: 0.16em; text-transform: uppercase;
    border-radius: 6px; padding: 13px 10px; width: 100%; transition: all 0.18s ease;
}
div.stButton > button:first-child:hover {
    border-color: var(--accent); color: var(--accent); background: rgba(123,104,238,0.05);
    box-shadow: 0 0 0 1px var(--accent-dim);
}

div[data-testid="metric-container"] {
    background: var(--surface); border: 1px solid var(--border); border-radius: 10px;
    padding: 20px 20px 16px; position: relative; overflow: hidden;
}
div[data-testid="metric-container"]::after {
    content: ''; position: absolute; bottom: 0; left: 0; right: 0; height: 1px;
    background: linear-gradient(90deg, var(--accent-dim), transparent);
}
div[data-testid="metric-container"] [data-testid="stMetricLabel"] {
    font-family: var(--mono) !important; font-size: 9px !important;
    letter-spacing: 0.18em !important; color: var(--muted) !important; text-transform: uppercase;
}
div[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-family: var(--sans) !important; font-size: 36px !important; font-weight: 300 !important;
    color: var(--text) !important; letter-spacing: -0.02em; line-height: 1.1 !important;
}
div[data-testid="metric-container"] [data-testid="stMetricDelta"] {
    font-family: var(--mono) !important; font-size: 9px !important; letter-spacing: 0.08em !important;
}
div[data-testid="metric-container"] [data-testid="stMetricDelta"] svg { display: none; }

[data-testid="stFileUploader"] { background: transparent; border: 1px dashed var(--border2); border-radius: 6px; padding: 6px; }
[data-testid="stFileUploader"] * { font-family: var(--mono) !important; font-size: 10px !important; color: var(--muted) !important; }
[data-testid="stAlert"], [data-testid="stSpinner"] p { font-family: var(--mono); font-size: 10px; color: var(--muted); }
[data-testid="stImage"] img { border-radius: 8px; }

.page-header { display: flex; align-items: center; justify-content: space-between; padding-bottom: 2rem; margin-bottom: 0.5rem; border-bottom: 1px solid var(--border); }
.logotype { font-family: var(--sans); font-size: 22px; font-weight: 300; color: var(--text); letter-spacing: -0.03em; }
.logotype b { font-weight: 500; color: var(--accent); }
.header-right { display: flex; align-items: center; gap: 20px; }
.header-pill { display: inline-flex; align-items: center; gap: 6px; background: rgba(123,104,238,0.08); border: 1px solid rgba(123,104,238,0.18); border-radius: 20px; padding: 5px 12px; font-family: var(--mono); font-size: 9px; letter-spacing: 0.12em; color: var(--accent); text-transform: uppercase; }
.pip { width: 5px; height: 5px; border-radius: 50%; background: var(--accent); animation: pip-pulse 2.5s ease-in-out infinite; }
@keyframes pip-pulse { 0%,100% { opacity: 1; box-shadow: 0 0 0 0 rgba(123,104,238,0.4); } 50% { opacity: 0.6; box-shadow: 0 0 0 4px rgba(123,104,238,0); } }
.header-meta { font-family: var(--mono); font-size: 9px; letter-spacing: 0.1em; color: var(--muted); text-transform: uppercase; text-align: right; line-height: 1.9; }

.sb-brand { display: flex; align-items: center; gap: 10px; padding-bottom: 2rem; margin-bottom: 2rem; border-bottom: 1px solid var(--border); }
.sb-icon { width: 30px; height: 30px; border-radius: 8px; background: rgba(123,104,238,0.1); border: 1px solid rgba(123,104,238,0.2); display: flex; align-items: center; justify-content: center; font-size: 14px; color: var(--accent); flex-shrink: 0; }
.sb-name { font-size: 14px; font-weight: 400; color: var(--text); letter-spacing: -0.02em; }
.sb-name b { color: var(--accent); font-weight: 500; }
.sb-sub { font-family: var(--mono); font-size: 9px; color: var(--muted); letter-spacing: 0.1em; text-transform: uppercase; margin-top: 1px; }
.sb-sec { font-family: var(--mono); font-size: 9px; letter-spacing: 0.2em; color: var(--muted); text-transform: uppercase; margin: 1.8rem 0 0.7rem; }
.sb-divider { border: none; border-top: 1px solid var(--border); margin: 1.4rem 0; }
.sb-note { font-family: var(--mono); font-size: 10px; color: var(--muted); letter-spacing: 0.06em; padding: 8px 10px; border: 1px solid var(--border); border-radius: 6px; margin-top: 6px; }

.tel { width: 100%; border-top: 1px solid var(--border); margin-top: 4px; }
.tel-row { display: flex; justify-content: space-between; align-items: center; padding: 8px 0; border-bottom: 1px solid var(--border); }
.tel-k { font-family: var(--mono); font-size: 9px; letter-spacing: 0.14em; color: var(--muted); text-transform: uppercase; }
.tel-v { font-family: var(--mono); font-size: 10px; font-weight: 400; color: var(--subtle); }
.tel-v.hi { color: var(--accent); }
.tel-v.ok { color: #5cb87a; }
.col-label { font-family: var(--mono); font-size: 9px; letter-spacing: 0.18em; color: var(--muted); text-transform: uppercase; margin-bottom: 10px; }

.offline-wrap { width: 100%; aspect-ratio: 16/9; background: var(--surface); border: 1px solid var(--border); border-radius: 8px; display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 10px; position: relative; }
.offline-icon { font-size: 32px; color: var(--border2); line-height: 1; font-weight: 200; }
.offline-label { font-family: var(--mono); font-size: 10px; letter-spacing: 0.2em; color: var(--muted); text-transform: uppercase; }
.offline-corner { position: absolute; font-family: var(--mono); font-size: 8px; letter-spacing: 0.1em; color: var(--border2); text-transform: uppercase; }
.tl { top: 14px; left: 16px; } .tr { top: 14px; right: 16px; } .bl { bottom: 14px; left: 16px; } .br { bottom: 14px; right: 16px; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return YOLO('yolov8n.pt')

@st.cache_resource
def get_device():
    if torch.cuda.is_available(): return "cuda"
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(): return "mps"
    return "cpu"

model = load_model()
compute_device = get_device()

DEMO_SCENES = {
    "01. Demo 1": "demo1.mp4",
    "02. Demo 2": "demo2.mp4",
    "03. Demo 3": "demo3.mp4",
    "04. Demo 4": "demo4.mp4",
    "05. Demo 5": "demo5.mp4"
}

st.markdown(f"""
<div class="page-header">
    <div class="logotype">OP<b>TIC</b></div>
    <div class="header-right">
        <div class="header-meta">
            YOLOv8n · ByteTrack · {compute_device.upper()}<br>
            Made by <strong style="color:var(--text)">Tushar Sharma</strong>
        </div>
        <div class="header-pill">
            <div class="pip"></div>
            Online
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("""
    <div class="sb-brand">
        <div class="sb-icon">◎</div>
        <div>
            <div class="sb-name">OP<b>TIC</b></div>
            <div class="sb-sub">Spatial Tracking Engine</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sb-sec">Source</div>', unsafe_allow_html=True)
    video_source = st.radio("src", ("System Demo", "Upload File"), label_visibility="collapsed")

    if video_source == "Upload File":
        st.markdown('<div style="height:6px"></div>', unsafe_allow_html=True)
        video_file = st.file_uploader("Upload", type=['mp4','mov','avi'], label_visibility="collapsed")
        selected_demo = None 
    else:
        video_file = None
        st.markdown('<div style="height:6px"></div>', unsafe_allow_html=True)
        selected_demo = st.selectbox("Select Demo Scene", list(DEMO_SCENES.keys()), label_visibility="collapsed")
        st.markdown(f'<div class="sb-note">Active: {selected_demo.split(". ")[1]}</div>', unsafe_allow_html=True)

    st.markdown('<div class="sb-sec">Parameters</div>', unsafe_allow_html=True)
    conf_threshold = st.slider(
        "Confidence", min_value=0.10, max_value=1.0, value=0.45, step=0.05
    )
    zone_position = st.slider(
        "Tripwire Y-axis", min_value=10, max_value=90, value=60, step=5
    )

    st.markdown('<hr class="sb-divider">', unsafe_allow_html=True)
    start_processing = st.button("◎  Initialize Pipeline", use_container_width=True)

    st.markdown('<hr class="sb-divider">', unsafe_allow_html=True)
    st.markdown('<div class="sb-sec">System</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="tel">
        <div class="tel-row"><span class="tel-k">Compute</span><span class="tel-v hi">{compute_device.upper()}</span></div>
        <div class="tel-row"><span class="tel-k">Model</span><span class="tel-v">YOLOv8 Nano</span></div>
        <div class="tel-row"><span class="tel-k">Tracker</span><span class="tel-v">ByteTrack</span></div>
        <div class="tel-row"><span class="tel-k">Status</span><span class="tel-v ok">Operational</span></div>
    </div>
    """, unsafe_allow_html=True)

col_feed, col_stats = st.columns([3, 1], gap="large")
source_display = "Upload" if video_source == "Upload File" else f"Demo {selected_demo.split('.')[0]}"

with col_stats:
    st.markdown('<div class="col-label">Detection</div>', unsafe_allow_html=True)
    m_count = st.empty()
    m_count.metric("Entities in Zone", "—", "Standby", label_visibility="visible")

    st.markdown('<div style="height:0.5rem"></div>', unsafe_allow_html=True)
    m_lat = st.empty()
    m_lat.metric("Latency", "—", "—", label_visibility="visible")

    st.markdown('<div style="height:0.5rem"></div>', unsafe_allow_html=True)
    m_frame = st.empty()
    m_frame.metric("Frame", "—", label_visibility="visible")

    st.markdown('<div style="height:1.4rem"></div>', unsafe_allow_html=True)
    st.markdown('<div class="col-label">Config</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="tel">
        <div class="tel-row"><span class="tel-k">Confidence</span><span class="tel-v hi">{conf_threshold:.2f}</span></div>
        <div class="tel-row"><span class="tel-k">Tripwire</span><span class="tel-v hi">{zone_position}%</span></div>
        <div class="tel-row"><span class="tel-k">Source</span><span class="tel-v">{source_display}</span></div>
    </div>
    """, unsafe_allow_html=True)

with col_feed:
    stframe = st.empty()

    if not start_processing:
        stframe.markdown("""
        <div class="offline-wrap">
            <div class="offline-icon">◎</div>
            <div class="offline-label">Awaiting initialization</div>
            <span class="offline-corner tl">Feed · Offline</span>
            <span class="offline-corner tr">YOLOv8n</span>
            <span class="offline-corner bl">ByteTrack</span>
            <span class="offline-corner br">OPTIC</span>
        </div>
        """, unsafe_allow_html=True)

    if start_processing:
        with st.spinner("Initializing pipeline..."):
            if video_source == "Upload File" and video_file is not None:
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                tfile.write(video_file.read())
                cap = cv2.VideoCapture(tfile.name)
            
            elif video_source == "System Demo":
                demo_path = DEMO_SCENES[selected_demo]
                
                if not os.path.exists(demo_path):
                    st.error(f"Missing File: '{demo_path}' not found in the project directory.")
                    st.stop()
                cap = cv2.VideoCapture(demo_path)
            
            else:
                st.warning("No video source selected.")
                st.stop()

            W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            unique_ids = set()

        frame_n = 0
        ui_update_interval = 5 
        last_count = -1

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                stframe.markdown("""
                <div class="offline-wrap">
                    <div class="offline-icon" style="color:#5cb87a">✓</div>
                    <div class="offline-label">Stream complete</div>
                </div>
                """, unsafe_allow_html=True)
                break

            frame_n += 1
            t0 = time.time()

            zy = int(H * (zone_position / 100))
            zone = np.array([[0, zy], [W, zy], [W, H], [0, H]], np.int32)

            results = model.track(
                frame, persist=True, tracker="bytetrack.yaml",
                conf=conf_threshold, classes=[0],
                device=compute_device, verbose=False
            )

            COL_IDLE   = (195, 160, 160)   
            COL_ACTIVE = (238, 104, 123)   
            COL_TEXT   = (245, 230, 230)   
            
            overlay = frame.copy()
            cv2.fillPoly(overlay, [zone], COL_ACTIVE)
            cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)
            
            cv2.line(frame, (0, zy), (W, zy), COL_ACTIVE, 2)

            if results[0].boxes is not None and results[0].boxes.id is not None:
                boxes     = results[0].boxes.xyxy.cpu().numpy()
                track_ids = results[0].boxes.id.cpu().numpy()

                for box, tid in zip(boxes, track_ids):
                    x1, y1, x2, y2 = map(int, box)
                    tid = int(tid)
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    in_zone = cv2.pointPolygonTest(zone, (cx, cy), False) >= 0

                    if in_zone:
                        unique_ids.add(tid)

                    col = COL_ACTIVE if in_zone else COL_IDLE
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), col, 1)
                    cv2.putText(frame, f"TID:{tid}", (x1 + 3, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.42, COL_TEXT, 1, cv2.LINE_AA)

                    if in_zone:
                        cv2.circle(frame, (cx, cy), 4, COL_ACTIVE, -1)

            hud = f"SYS.FRAME: {frame_n:04d} | ROI.COUNT: {len(unique_ids):02d} | CONF: {conf_threshold:.2f}"
            cv2.putText(frame, hud, (15, H - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            stframe.image(frame_rgb, channels="RGB", use_container_width=True)

            if frame_n % ui_update_interval == 0 or len(unique_ids) != last_count:
                ms = int((time.time() - t0) * 1000)
                m_count.metric("Entities in Zone", len(unique_ids),
                               "Active" if unique_ids else "Clear")
                m_lat.metric("Latency", f"{ms} ms", "Live")
                m_frame.metric("Frame", frame_n)
                last_count = len(unique_ids)

        cap.release()