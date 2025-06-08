import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
import plotly.graph_objects as go
import base64

# ── 1) STREAMLIT CONFIG ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Interactive Office Heatmap",
    layout="wide",
    initial_sidebar_state="collapsed"
)
st.title("📊 Interactive Office Utilization Heatmap")

# ── 2) FILEPATHS ──────────────────────────────────────────────────────────────
BOOKING_CSV   = "booking_data.csv"
COORD_CSV     = "coordinate_mapping.csv"
FLOORPLAN_IMG = "office_floorplan.jpg"

# ── 3) LOAD DATA ───────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    bookings_df = pd.read_csv(BOOKING_CSV, parse_dates=["Booking_Timestamp"])
    coords_df   = pd.read_csv(COORD_CSV)
    return bookings_df, coords_df

bookings, coords = load_data()

# ── 4) DATE RANGE PICKER (MAIN PAGE, SAFE) ────────────────────────────────────
min_date = bookings["Booking_Timestamp"].dt.date.min()
max_date = bookings["Booking_Timestamp"].dt.date.max()

st.header("📅 Filter Bookings by Date")
date_sel = st.date_input(
    "Select date range:",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
    key="date_range"
)
if not (isinstance(date_sel, tuple) and len(date_sel) == 2):
    st.info("👉 Please select *both* a start **and** end date to view the heatmap.")
    st.stop()
start_date, end_date = date_sel
if start_date > end_date:
    st.error("↩️ Start date must be before end date.")
    st.stop()

filtered = bookings.loc[
    (bookings["Booking_Timestamp"].dt.date >= start_date) &
    (bookings["Booking_Timestamp"].dt.date <= end_date)
].copy()

st.write(f"Total bookings in range: **{len(filtered):,}**")
if filtered.empty:
    st.warning("⚠️ No booking data in this date range.")
    st.stop()

# ── 5) AGGREGATE UTILIZATION ──────────────────────────────────────────────────
filtered["ID"] = filtered["Desk_ID"].fillna(filtered["Meeting_Room_ID"])
agg = (
    filtered
    .groupby("ID")["Duration"]
    .sum()
    .reset_index(name="Total_Duration")
)
agg["Utilization_Score"] = agg["Total_Duration"] / 480.0  # minutes → 8h days

# merge with coords, fill missing
df = coords.merge(agg, on="ID", how="left").fillna({
    "Total_Duration":    0.0,
    "Utilization_Score": 0.0
})

# ── 6) LOAD FLOORPLAN & SET EXTENTS ────────────────────────────────────────────
try:
    img_pil = Image.open(FLOORPLAN_IMG)
    img_width, img_height = img_pil.size
    x_min, x_max = 0, img_width
    y_min, y_max = 0, img_height
except FileNotFoundError:
    x_min = min(df["X_Pixels"] - df["Width_Pixels"])
    x_max = max(df["X_Pixels"] + df["Width_Pixels"])
    y_min = min(df["Y_Pixels"] - df["Height_Pixels"])
    y_max = max(df["Y_Pixels"] + df["Height_Pixels"])
    img_width, img_height = int(x_max - x_min), int(y_max - y_min)

w, h = img_width, img_height

# ── 7) BUILD INTENSITY GRID ───────────────────────────────────────────────────
RES_X = min(int(w), 400)
RES_Y = min(int(h), 300)
xi    = np.linspace(x_min, x_max, RES_X)
yi    = np.linspace(y_min, y_max, RES_Y)
intensity = np.zeros((RES_Y, RES_X))

# weights & radii
DESK_WEIGHT       = 1.0
DESK_HALO_RATIO   = 0.2
ROOM_WEIGHT       = 1.0
ROOM_HALO_RATIO   = 1.0    # bigger halo
ROOM_HALO_RADIUS  = 5     # spread out two cells

for _, row in df.iterrows():
    score = row["Utilization_Score"]
    if score <= 0:
        continue

    # determine center cell
    if row["Type"] == "desk":
        cx, cy = row["X_Pixels"], row["Y_Pixels"]
        halo_ratio, halo_radius = DESK_HALO_RATIO, 1
        weight = DESK_WEIGHT
    else:
        # use geometric center of room
        cx = row["X_Pixels"] + row["Width_Pixels"]/2
        cy = row["Y_Pixels"] + row["Height_Pixels"]/2
        halo_ratio, halo_radius = ROOM_HALO_RATIO, ROOM_HALO_RADIUS
        weight = ROOM_WEIGHT

    # map to grid index
    ix = int(np.clip((cx-x_min)/(x_max-x_min)*(RES_X-1), 0, RES_X-1))
    iy = int(np.clip((cy-y_min)/(y_max-y_min)*(RES_Y-1), 0, RES_Y-1))

    # bump center
    intensity[iy, ix] += score * weight

    # bump neighbors within halo_radius
    for dy in range(-halo_radius, halo_radius+1):
        for dx in range(-halo_radius, halo_radius+1):
            if dx == 0 and dy == 0:
                continue
            ny, nx = iy+dy, ix+dx
            if 0 <= ny < RES_Y and 0 <= nx < RES_X:
                # distance‐based falloff (optional)
                dist = max(abs(dx), abs(dy))
                falloff = 1.0 / dist   # or simply 1.0
                intensity[ny, nx] += score * weight * halo_ratio * falloff

# ── 8) GAUSSIAN BLUR ───────────────────────────────────────────────────────────
sigma_x = max(3, RES_X/40)
sigma_y = max(3, RES_Y/40)
blurred = gaussian_filter(intensity, sigma=(sigma_y, sigma_x))

# ── 9) DYNAMIC ZMAX & COLORSCALE ─────────────────────────────────────────────
ZMAX = np.percentile(blurred, 95)

# custom 7-color ramp
MY_PALETTE = [
    "#001f3f", "#0074D9", "#7FDBFF",
    "#2ECC40", "#FFDC00", "#FF851B", "#FF4136",
]
colorscale = [[i/(len(MY_PALETTE)-1), c] for i, c in enumerate(MY_PALETTE)]

# ── 10) PLOTLY HEATMAP ─────────────────────────────────────────────────────────
fig = go.Figure()
fig.add_trace(
    go.Heatmap(
        z=np.flipud(blurred),
        x=xi,
        y=yi,
        colorscale=colorscale,
        zmin=0,
        zmax=ZMAX,
        zsmooth="best",
        showscale=False,
        opacity=0.6
    )
)

# floorplan underlay
try:
    with open(FLOORPLAN_IMG, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    fig.update_layout(images=[{
        "xref":"x","yref":"y",
        "x":x_min,"y":y_max,
        "sizex":x_max-x_min,"sizey":y_max-y_min,
        "sizing":"stretch","opacity":1.0,"layer":"below",
        "source":f"data:image/png;base64,{b64}"
    }])
except FileNotFoundError:
    pass

fig.update_layout(
    xaxis=dict(visible=False, showgrid=False, zeroline=False, range=[x_min, x_max]),
    yaxis=dict(visible=False, showgrid=False, zeroline=False,
               scaleanchor="x", scaleratio=1, range=[y_min, y_max]),
    margin=dict(l=0, r=0, t=0, b=0),
    dragmode="zoom"
)

st.plotly_chart(fig, use_container_width=True, height=600)

# ── 11) LEGEND & SUMMARY ───────────────────────────────────────────────────────
st.markdown("---")
st.header("📖 Legend & Explanation")

st.markdown("""
**🖥️ How to interact with the heatmap**  
- **Zoom/Pan**: Hover near the edges to reveal the zoom icons, or simply click-and-drag to draw a zoom box.  
- **Pan**: After zooming in, click the pan‐hand icon (🤚) in the top‐right toolbar to move around.  
- **Reset View**: Click the home‐shaped icon to zoom back out to full office view.

---

**🗺️ What you’re seeing**  
- The grayscale image underneath is your full‐floor floorplan.  
- Over it, you’ll see **red‐yellow “hot” blobs** wherever desks/rooms have been used in the selected date range.  
- Desks appear as **blue circles** (light‐blue fill if that desk has > 0 utilization; grey if unused).  
- Meeting rooms appear as **red rectangles** (light‐coral fill if > 0 utilization; grey if unused).  
- Every desk/room has its `ID` centered over it, with a yellow utilization‐score (fraction of an 8-hour day) printed just below the icon.

---

**🎯 How utilization is calculated**  
1. We group all bookings by `Desk_ID` or `Meeting_Room_ID` in your selected date range.  
2. We sum `Duration` (minutes) per ID.  
3. We divide by **480 minutes** (8 h) to get a fractional “Utilization Score.”  
   - e.g. if a desk was booked for 240 min total, it’s shown as `0.50`.  
   - Anything with 0.00 means “unused” in that time span.

---

**❓ Tips**  
- If no desks/rooms appear “hot,” check that your date range actually contains bookings.  
- If the floorplan looks misaligned, double-check that your `coordinate_mapping.csv` pixel coordinates match exactly the dimensions of `office_floorplan.jpg`.  
- Feel free to change the blur/sigma parameters (in code) if you want tighter or looser heat‐“clouds.”  
- You can scroll horizontally/vertically on the Streamlit page if you zoom in very far.

""")

# ── 12) SUMMARY STATISTICS ─────────────────────────────────────────────────────
st.header("📊 Utilization Summary Statistics")
summary_stats = df.groupby("Type").agg({
    "Utilization_Score": ["count", "mean", "max", "sum"]
}).round(3)
summary_stats.columns = ["Count", "Mean", "Max", "Total"]
st.dataframe(summary_stats.reset_index().rename(columns={"Type":"Item Type"}), use_container_width=True)

st.subheader("🏆 Top 5 Most Utilized Items")
top5 = df.nlargest(5, "Utilization_Score")[["ID","Type","Utilization_Score","Total_Duration"]]
top5 = top5.rename(columns={
    "ID":"Item ID",
    "Utilization_Score":"Util Score",
    "Total_Duration":"Total Minutes"
})
st.table(top5)

st.subheader("⛔ Unused Items")
unused = df[df["Utilization_Score"] == 0]["ID"].tolist()
if unused:
    st.write(f"Total unused items: **{len(unused):,}**")
    display_list = unused[:20] + (["…"] if len(unused)>20 else [])
    st.write(display_list)
else:
    st.write("None – all desks & rooms have some utilization in this range!")
