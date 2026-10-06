import io
import tempfile
 
import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from PIL import Image
from ultralytics import YOLO
 
# -----------------------------------------------------------------------------
# 1. PAGE CONFIG
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="SmartPark AI | Vision Core",
    page_icon="🅿️",
    layout="wide",
    initial_sidebar_state="expanded",
)
 
# Streamlit renamed use_container_width -> width="stretch" in newer versions.
try:
    _ver = tuple(int(x) for x in st.__version__.split(".")[:2])
except Exception:
    _ver = (1, 0)
FULL = {"width": "stretch"} if _ver >= (1, 50) else {"use_container_width": True}
 
# -----------------------------------------------------------------------------
# 2. THEME  (asphalt + road-paint palette)
#    free = signal green · occupied = coral · accent = road-line amber
# -----------------------------------------------------------------------------
if "theme" not in st.session_state:
    st.session_state.theme = "dark"
 
 
def toggle_theme():
    st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"
 
 
dark = st.session_state.theme == "dark"
 
P = {
    "bg":        "#0F1318" if dark else "#EEF1F4",   # asphalt / concrete
    "surface":   "#171C23" if dark else "#FFFFFF",
    "surface2":  "#1E252E" if dark else "#F6F8FA",
    "line":      "#2A323D" if dark else "#DCE1E7",
    "text":      "#EEF2F6" if dark else "#14202E",
    "muted":     "#8B97A6" if dark else "#5F6E80",
    "accent":    "#FFC83D" if dark else "#E8A500",   # road-line amber
    "free":      "#2EE6A6" if dark else "#0E9F6E",
    "occ":       "#FF6B6B" if dark else "#E03E3E",
    "info":      "#5AB0FF" if dark else "#1F6FEB",
    "warn":      "#FFB547" if dark else "#C77700",
}
 
st.markdown(
    f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,700;12..96,800&family=DM+Sans:wght@400;500;600;700&display=swap');
 
html, body, [class*="css"], .stApp {{ font-family: 'DM Sans', sans-serif; }}
.stApp {{ background: {P['bg']}; color: {P['text']}; }}
header[data-testid="stHeader"] {{ background: transparent; }}
section[data-testid="stSidebar"] {{
    background: {P['surface']};
    border-right: 1px solid {P['line']};
}}
h1,h2,h3,h4 {{ font-family: 'Bricolage Grotesque', sans-serif; letter-spacing: -0.01em; color: {P['text']}; }}
 
/* Hero: a parking bay seen from above */
.hero {{
    position: relative;
    background: {P['surface']};
    border: 1px solid {P['line']};
    border-radius: 20px;
    padding: 28px 34px;
    margin-bottom: 22px;
    overflow: hidden;
}}
.hero::before {{            /* amber bay line */
    content: "";
    position: absolute; left: 0; top: 18px; bottom: 18px; width: 5px;
    border-radius: 0 4px 4px 0;
    background: {P['accent']};
}}
.hero-title {{
    font-family: 'Bricolage Grotesque', sans-serif;
    font-weight: 800; font-size: 2.3rem; line-height: 1.1;
    color: {P['text']}; margin: 0 0 6px 0;
}}
.hero-sub {{ color: {P['muted']}; font-size: 0.98rem; margin: 0 0 14px 0; max-width: 62ch; }}
.chip {{
    display: inline-flex; align-items: center; gap: 6px;
    padding: 5px 12px; margin: 0 6px 6px 0;
    border-radius: 999px; font-size: 0.78rem; font-weight: 600;
    background: {P['surface2']}; color: {P['text']};
    border: 1px solid {P['line']};
}}
.dot {{ width: 7px; height: 7px; border-radius: 50%; background: {P['free']}; display:inline-block; }}
.dot.live {{ animation: pulse 1.6s ease-in-out infinite; }}
@keyframes pulse {{ 0%,100% {{ opacity: 1; }} 50% {{ opacity: .25; }} }}
@media (prefers-reduced-motion: reduce) {{ .dot.live {{ animation: none; }} }}
 
/* Metrics */
.metrics {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; }}
.metric {{
    background: {P['surface']};
    border: 1px solid {P['line']};
    border-radius: 14px;
    padding: 16px 18px;
}}
.metric .label {{ font-size: 0.8rem; font-weight: 600; color: {P['muted']}; }}
.metric .value {{
    font-family: 'Bricolage Grotesque', sans-serif;
    font-size: 2.4rem; font-weight: 800; line-height: 1.1; margin-top: 4px;
}}
.metric.total {{ border-top: 3px solid {P['info']}; }}  .metric.total .value {{ color: {P['info']}; }}
.metric.occ   {{ border-top: 3px solid {P['occ']}; }}   .metric.occ .value   {{ color: {P['occ']}; }}
.metric.free  {{ border-top: 3px solid {P['free']}; }}  .metric.free .value  {{ color: {P['free']}; }}
 
/* Status banner */
.status {{
    display: flex; align-items: center; gap: 10px;
    padding: 14px 18px; margin: 14px 0;
    border-radius: 12px; font-weight: 600; font-size: 0.95rem;
    border: 1px solid transparent;
}}
.status.ok   {{ background: {P['free']}1F; color: {P['free']}; border-color: {P['free']}55; }}
.status.warn {{ background: {P['warn']}1F; color: {P['warn']}; border-color: {P['warn']}55; }}
.status.full {{ background: {P['occ']}1F;  color: {P['occ']};  border-color: {P['occ']}55; }}
 
/* Panels */
.panel-title {{ font-family:'Bricolage Grotesque',sans-serif; font-weight:700; font-size:1.15rem; margin: 4px 0 10px 0; }}
.empty {{
    text-align: center; padding: 56px 20px; margin-top: 8px;
    background: {P['surface']}; border: 1.5px dashed {P['line']}; border-radius: 18px;
}}
.empty h3 {{ margin: 8px 0 6px 0; }}
.empty p {{ color: {P['muted']}; margin: 0 auto; max-width: 440px; }}
 
/* Tabs */
.stTabs [data-baseweb="tab-list"] {{ gap: 6px; border-bottom: 1px solid {P['line']}; }}
.stTabs [data-baseweb="tab"] {{ font-weight: 600; color: {P['muted']}; padding: 10px 16px; }}
.stTabs [aria-selected="true"] {{ color: {P['text']}; }}
.stTabs [data-baseweb="tab-highlight"] {{ background-color: {P['accent']}; }}
 
/* Buttons */
.stButton > button, .stDownloadButton > button {{
    border-radius: 10px; font-weight: 600;
    border: 1px solid {P['line']}; background: {P['surface2']}; color: {P['text']};
}}
.stButton > button:hover, .stDownloadButton > button:hover {{
    border-color: {P['accent']}; color: {P['accent']};
}}
.stButton > button:focus-visible, .stDownloadButton > button:focus-visible {{
    outline: 2px solid {P['accent']}; outline-offset: 2px;
}}
[data-testid="stFileUploaderDropzone"] {{
    background: {P['surface']}; border: 1.5px dashed {P['line']}; border-radius: 16px;
}}
[data-testid="stExpander"] {{
    background: {P['surface']}; border: 1px solid {P['line']}; border-radius: 14px;
}}
</style>
""",
    unsafe_allow_html=True,
)
 
# -----------------------------------------------------------------------------
# 3. MODEL
# -----------------------------------------------------------------------------
@st.cache_resource
def load_yolo_model(path: str = "best.pt"):
    return YOLO(path)
 
 
try:
    model = load_yolo_model("best.pt")
except Exception as e:
    st.error(f"Could not load `best.pt`. Make sure it is in the app folder. Details: {e}")
    st.stop()
 
# -----------------------------------------------------------------------------
# 4. SIDEBAR
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Appearance")
    st.button(
        "☀️ Light mode" if dark else "🌙 Dark mode",
        on_click=toggle_theme,
        **FULL,
    )
    st.divider()
    st.markdown("### Detection")
    conf_threshold = st.slider(
        "Confidence threshold", 0.05, 0.90, 0.25, 0.05,
        help="Higher = fewer false positives. Lower = catches faint slots.",
    )
    show_labels = st.toggle("Show labels on image", value=False)
    st.divider()
    st.markdown("### Video")
    frame_skip = st.slider("Analyze every Nth frame", 1, 10, 3,
                           help="Higher = faster playback, fewer updates.")
    max_frames = st.slider("Max frames to process", 30, 1000, 300, 30)
    st.divider()
    st.caption("Detects free and occupied slots, updates occupancy live, and exports results as image or CSV.")
 
# -----------------------------------------------------------------------------
# 5. HELPERS
# -----------------------------------------------------------------------------
EMPTY_WORDS = {"empty", "space-empty", "vacant", "free", "empty-space", "free-space", "unoccupied"}
OCC_WORDS = {"occupied", "space-occupied", "filled", "taken", "car", "vehicle", "busy", "parked", "occupied-space"}
 
 
def normalize(name: str) -> str:
    n = name.strip().lower()
    if n == "0":
        return "empty"
    if n == "1":
        return "occupied"
    if n in EMPTY_WORDS or any(w in n for w in ("empty", "vacant", "free")):
        return "empty"
    if n in OCC_WORDS or any(w in n for w in ("occ", "car", "vehicle", "taken")):
        return "occupied"
    return "empty"
 
 
def hex_to_rgb(h: str):
    h = h.lstrip("#")[:6]
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))
 
 
def analyze(rgb: np.ndarray):
    res = model.predict(rgb, conf=conf_threshold, verbose=False)[0]
    names = res.names
    classes = [normalize(names[int(c)]) for c in res.boxes.cls]
    return {
        "classes": classes,
        "conf": [float(c) for c in res.boxes.conf],
        "boxes": res.boxes.xyxy.tolist(),
    }
 
 
def draw(rgb: np.ndarray, det: dict) -> np.ndarray:
    """Draw green (free) / coral (occupied) boxes."""
    out = rgb.copy()
    h, w = out.shape[:2]
    thick = max(2, int(round(min(h, w) / 300)))
    col = {"empty": hex_to_rgb("#2EE6A6"), "occupied": hex_to_rgb("#FF6B6B")}
    for cls, score, (x1, y1, x2, y2) in zip(det["classes"], det["conf"], det["boxes"]):
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cv2.rectangle(out, (x1, y1), (x2, y2), col[cls], thick)
        if show_labels:
            txt = f"{'free' if cls == 'empty' else 'taken'} {score:.0%}"
            fs = max(0.4, min(h, w) / 1400)
            (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
            cv2.rectangle(out, (x1, max(0, y1 - th - 6)), (x1 + tw + 6, y1), col[cls], -1)
            cv2.putText(out, txt, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, fs, (15, 19, 24), 1, cv2.LINE_AA)
    return out
 
 
def counts(det):
    e = det["classes"].count("empty")
    o = det["classes"].count("occupied")
    return e, o, e + o
 
 
def metrics_html(e, o, t):
    return (
        '<div class="metrics">'
        f'<div class="metric total"><div class="label">Total slots</div><div class="value">{t}</div></div>'
        f'<div class="metric occ"><div class="label">Occupied</div><div class="value">{o}</div></div>'
        f'<div class="metric free"><div class="label">Available</div><div class="value">{e}</div></div>'
        "</div>"
    )
 
 
def status_html(e, o, t):
    if t == 0:
        return '<div class="status warn">No slots detected. Try lowering the confidence threshold.</div>'
    ratio = o / t
    if e == 0:
        return '<div class="status full">🚨 Parking full · 0 slots available</div>'
    if ratio >= 0.8:
        return f'<div class="status warn">⚠️ Near capacity · only {e} slot(s) left</div>'
    return f'<div class="status ok">✅ Spots available · {e} slot(s) free</div>'
 
 
def gauge(pct: float):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pct,
        number={"suffix": "%", "font": {"size": 34, "color": P["text"], "family": "Bricolage Grotesque"}},
        title={"text": "Occupancy", "font": {"size": 13, "color": P["muted"]}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": P["line"], "tickfont": {"color": P["muted"]}},
            "bar": {"color": P["accent"], "thickness": 0.28},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 50],   "color": P["free"] + "33"},
                {"range": [50, 80],  "color": P["warn"] + "33"},
                {"range": [80, 100], "color": P["occ"] + "33"},
            ],
        },
    ))
    fig.update_layout(height=200, margin=dict(l=24, r=24, t=30, b=6),
                      paper_bgcolor="rgba(0,0,0,0)", font={"family": "DM Sans"})
    return fig
 
 
def trend(history):
    fig = go.Figure(go.Scatter(
        y=history, mode="lines", line=dict(color=P["accent"], width=2.5),
        fill="tozeroy", fillcolor=P["accent"] + "26",
    ))
    fig.update_layout(
        height=150, margin=dict(l=8, r=8, t=8, b=8),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(range=[0, 100], gridcolor=P["line"], ticksuffix="%", tickfont=dict(color=P["muted"])),
        xaxis=dict(visible=False),
    )
    return fig
 
 
def telemetry_table(det):
    rows = []
    for i, (cls, score, box) in enumerate(zip(det["classes"], det["conf"], det["boxes"])):
        x1, y1, x2, y2 = [int(v) for v in box]
        rows.append({
            "Slot": f"Space #{i + 1:02d}",
            "Status": "🟢 Available" if cls == "empty" else "🔴 Occupied",
            "Confidence": score,
            "Box [x1, y1, x2, y2]": f"[{x1}, {y1}, {x2}, {y2}]",
            "Size": f"{x2 - x1}×{y2 - y1} px",
        })
    return pd.DataFrame(rows)
 
 
def render_static(rgb: np.ndarray, key: str):
    """Analyze one image and render the full dashboard."""
    with st.spinner("Analyzing parking lot…"):
        det = analyze(rgb)
    e, o, t = counts(det)
    annotated = draw(rgb, det)
 
    left, right = st.columns([1.6, 1], gap="large")
    with left:
        st.markdown('<div class="panel-title">Detection feed</div>', unsafe_allow_html=True)
        st.image(annotated, **FULL)
        buf = io.BytesIO()
        Image.fromarray(annotated).save(buf, format="JPEG", quality=92)
        st.download_button("📥 Download annotated image", buf.getvalue(),
                           "smartpark_detection.jpg", "image/jpeg", key=f"dl_img_{key}", **FULL)
    with right:
        st.markdown('<div class="panel-title">Space analytics</div>', unsafe_allow_html=True)
        st.markdown(metrics_html(e, o, t), unsafe_allow_html=True)
        st.markdown(status_html(e, o, t), unsafe_allow_html=True)
        st.plotly_chart(gauge((o / t * 100) if t else 0), key=f"g_{key}", **FULL)
 
    with st.expander("Slot telemetry & export"):
        if t:
            df = telemetry_table(det)
            st.dataframe(
                df,
                column_config={"Confidence": st.column_config.ProgressColumn(
                    "AI confidence", format="percent", min_value=0.0, max_value=1.0)},
                hide_index=True,
                **FULL,
            )
            st.download_button("📥 Export CSV", df.to_csv(index=False).encode("utf-8"),
                               "parking_telemetry.csv", "text/csv", key=f"dl_csv_{key}")
        else:
            st.write("No slots detected at this sensitivity.")
 
 
def placeholder_card(icon: str, title: str, text: str):
    st.markdown(
        f'<div class="empty"><div style="font-size:2.4rem">{icon}</div><h3>{title}</h3><p>{text}</p></div>',
        unsafe_allow_html=True,
    )
 
 
# -----------------------------------------------------------------------------
# 6. HERO
# -----------------------------------------------------------------------------
st.markdown(
    """
<div class="hero">
  <div class="hero-title">SmartPark Vision Core</div>
  <p class="hero-sub">Know which parking slots are free the moment you look. Upload a photo, snap one with your camera, or play a video and watch occupancy update live.</p>
  <span class="chip"><span class="dot live"></span> Live occupancy</span>
  <span class="chip">YOLOv8</span>
  <span class="chip">Photo · Camera · Video</span>
</div>
""",
    unsafe_allow_html=True,
)
 
# -----------------------------------------------------------------------------
# 7. MODES
# -----------------------------------------------------------------------------
tab_photo, tab_cam, tab_video = st.tabs(["📷 Photo", "🎥 Camera", "▶️ Live video"])
 
with tab_photo:
    up = st.file_uploader("Upload parking photo", type=["jpg", "jpeg", "png", "webp"],
                          label_visibility="collapsed", key="photo")
    if up:
        render_static(np.array(Image.open(up).convert("RGB")), "photo")
    else:
        placeholder_card("🅿️", "Drop a parking lot photo",
                         "JPG, PNG or WebP. Results appear here with free and occupied slots marked.")
 
with tab_cam:
    shot = st.camera_input("Point your camera at the lot", label_visibility="collapsed")
    if shot:
        render_static(np.array(Image.open(shot).convert("RGB")), "cam")
    else:
        placeholder_card("🎥", "Take a snapshot",
                         "Allow camera access, frame the parking area, and capture.")
 
with tab_video:
    vid = st.file_uploader("Upload parking video", type=["mp4", "mov", "avi", "mkv"],
                           label_visibility="collapsed", key="video")
    if not vid:
        placeholder_card("▶️", "Upload a video to start",
                         "Each analyzed frame refreshes the counts, gauge and occupancy trend in real time.")
    else:
        run = st.button("▶️ Start live analysis", type="primary", **FULL)
        if run:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tmp.write(vid.read())
            tmp.flush()
 
            cap = cv2.VideoCapture(tmp.name)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or max_frames
            limit = min(total_frames, max_frames)
 
            left, right = st.columns([1.6, 1], gap="large")
            with left:
                st.markdown('<div class="panel-title">Live feed</div>', unsafe_allow_html=True)
                frame_slot = st.empty()
                progress = st.progress(0.0)
            with right:
                st.markdown('<div class="panel-title">Live analytics</div>', unsafe_allow_html=True)
                metric_slot = st.empty()
                status_slot = st.empty()
                gauge_slot = st.empty()
                trend_slot = st.empty()
 
            history, idx = [], 0
            while cap.isOpened() and idx < limit:
                ok, frame = cap.read()
                if not ok:
                    break
                if idx % frame_skip == 0:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    det = analyze(rgb)
                    e, o, t = counts(det)
                    pct = (o / t * 100) if t else 0
                    history.append(pct)
 
                    frame_slot.image(draw(rgb, det), **FULL)
                    metric_slot.markdown(metrics_html(e, o, t), unsafe_allow_html=True)
                    status_slot.markdown(status_html(e, o, t), unsafe_allow_html=True)
                    gauge_slot.plotly_chart(gauge(pct), **FULL)
                    trend_slot.plotly_chart(trend(history[-60:]), **FULL)
                idx += 1
                progress.progress(min(idx / limit, 1.0))
 
            cap.release()
            progress.empty()
            st.success(f"Finished. Analyzed {len(history)} frames.")
 