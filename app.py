import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
import plotly.graph_objects as go
import base64

# ── 1) CONFIG ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Office Heatmap", layout="wide")
st.title("📊 Interactive Office Utilization Heatmap")

# ── 2) PATHS ──────────────────────────────────────────────────────────────────
BOOKING_CSV   = "booking_data.csv"
COORD_CSV     = "coordinate_mapping.csv"
FLOORPLAN_IMG = "office_floorplan.jpg"

# ── 3) LOAD ───────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    b = pd.read_csv(BOOKING_CSV, parse_dates=["Booking_Timestamp"])
    c = pd.read_csv(COORD_CSV)
    return b, c

bookings, coords = load_data()

# ── 4) DATE PICKER ────────────────────────────────────────────────────────────
min_date = bookings["Booking_Timestamp"].dt.date.min()
max_date = bookings["Booking_Timestamp"].dt.date.max()
st.header("📅 Filter Bookings by Date")
date_sel = st.date_input("Select date range:",
                         value=(min_date, max_date),
                         min_value=min_date,
                         max_value=max_date)
if not (isinstance(date_sel, tuple) and len(date_sel) == 2):
    st.info("Select *both* start and end dates.")
    st.stop()
start_date, end_date = date_sel
if start_date > end_date:
    st.error("Start must be before end.")
    st.stop()

f = bookings.loc[
    (bookings["Booking_Timestamp"].dt.date >= start_date) &
    (bookings["Booking_Timestamp"].dt.date <= end_date)
]
st.write(f"Total bookings: {len(f):,}")
if f.empty:
    st.warning("No data in this range.")
    st.stop()

# ── 5) AGGREGATE ──────────────────────────────────────────────────────────────
f["ID"] = f["Desk_ID"].fillna(f["Meeting_Room_ID"])
agg = (f.groupby("ID")["Duration"]
         .sum()
         .reset_index(name="Total_Duration"))
agg["Utilization_Score"] = agg["Total_Duration"] / 480.0

df = coords.merge(agg, on="ID", how="left").fillna(0)
# we’ll fill missing columns automatically:
df["Utilization_Score"] = df["Utilization_Score"].astype(float)

# ── 6) FLOORPLAN & EXTENTS ───────────────────────────────────────────────────
try:
    img = Image.open(FLOORPLAN_IMG)
    w, h = img.size
    x_min, x_max = 0, w
    y_min, y_max = 0, h
except FileNotFoundError:
    x_min = min(df["X_Pixels"] - df["Width_Pixels"])
    x_max = max(df["X_Pixels"] + df["Width_Pixels"])
    y_min = min(df["Y_Pixels"] - df["Height_Pixels"])
    y_max = max(df["Y_Pixels"] + df["Height_Pixels"])
    w, h = x_max - x_min, y_max - y_min

# ── 7) HEAT GRID ──────────────────────────────────────────────────────────────
RES_X, RES_Y = min(int(w), 400), min(int(h), 300)
xi = np.linspace(x_min, x_max, RES_X)
yi = np.linspace(y_min, y_max, RES_Y)
intensity = np.zeros((RES_Y, RES_X))

DESK_WEIGHT     = 1.0
NEIGHBOR_RATIO  = 0.2

for _, row in df.iterrows():
    score = row["Utilization_Score"]
    if score <= 0:
        continue

    cx, cy = row["X_Pixels"], row["Y_Pixels"]
    w_px, h_px = row["Width_Pixels"], row["Height_Pixels"]

    if row["Type"] == "desk":
        ix = int(np.clip((cx-x_min)/(x_max-x_min)*(RES_X-1), 0, RES_X-1))
        iy = int(np.clip((cy-y_min)/(y_max-y_min)*(RES_Y-1), 0, RES_Y-1))
        intensity[iy, ix] += score * DESK_WEIGHT
        for dy in (-1,0,1):
            for dx in (-1,0,1):
                if dx==0 and dy==0: continue
                ny, nx = iy+dy, ix+dx
                if 0 <= ny < RES_Y and 0 <= nx < RES_X:
                    intensity[ny, nx] += score * DESK_WEIGHT * NEIGHBOR_RATIO

    else:  # meeting room
        x1, x2 = cx, cx + w_px
        y1, y2 = cy, cy + h_px
        ix1 = int(np.clip((x1-x_min)/(x_max-x_min)*(RES_X-1), 0, RES_X-1))
        ix2 = int(np.clip((x2-x_min)/(x_max-x_min)*(RES_X-1), 0, RES_X-1))
        iy1 = int(np.clip((y1-y_min)/(y_max-y_min)*(RES_Y-1), 0, RES_Y-1))
        iy2 = int(np.clip((y2-y_min)/(y_max-y_min)*(RES_Y-1), 0, RES_Y-1))
        ix2, iy2 = max(ix1+1, min(ix2, RES_X)), max(iy1+1, min(iy2, RES_Y))
        area = (ix2-ix1)*(iy2-iy1)
        if area>0:
            # spread one DESK_WEIGHT across the room footprint
            per_cell = (score * DESK_WEIGHT) / area
            intensity[iy1:iy2, ix1:ix2] += per_cell

# blur
sigma = (max(3, RES_Y/40), max(3, RES_X/40))
blurred = gaussian_filter(intensity, sigma=sigma)

# ── 8) DYNAMIC ZMAX ──────────────────────────────────────────────────────────
ZMAX = np.percentile(blurred, 95)

# ── 9) PLOTLY HEATMAP ────────────────────────────────────────────────────────
# custom 7-color ramp
PALETTE = ["#001f3f","#0074D9","#7FDBFF","#2ECC40","#FFDC00","#FF851B","#FF4136"]
cs = [[i/(len(PALETTE)-1), c] for i, c in enumerate(PALETTE)]

fig = go.Figure()
fig.add_trace(go.Heatmap(
    z=np.flipud(blurred),
    x=xi, y=yi,
    colorscale=cs,
    zmin=0, zmax=ZMAX,
    showscale=False, opacity=0.6,
    zsmooth="best"
))

# add floorplan underlay
try:
    with open(FLOORPLAN_IMG, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    fig.update_layout(images=[{
        "xref":"x","yref":"y","x":x_min,"y":y_max,
        "sizex":x_max-x_min,"sizey":y_max-y_min,
        "sizing":"stretch","opacity":1,"layer":"below",
        "source":f"data:image/png;base64,{b64}"
    }])
except FileNotFoundError:
    pass

fig.update_layout(
    xaxis=dict(visible=False, showgrid=False, zeroline=False, range=[x_min,x_max]),
    yaxis=dict(visible=False, showgrid=False, zeroline=False, range=[y_min,y_max],
               scaleanchor="x", scaleratio=1),
    margin=dict(l=0,r=0,t=0,b=0), dragmode="zoom"
)

st.plotly_chart(fig, use_container_width=True, height=600)

# ── 10) LEGEND & STATS ────────────────────────────────────────────────────────
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

# ── 11) SUMMARY STATISTICS ─────────────────────────────────────────────────────
st.header("📊 Utilization Summary Statistics")
summary_stats = df.groupby("Type").agg({
    "Utilization_Score": ["count", "mean", "max", "sum"]
}).round(3)
summary_stats.columns = ["Count", "Mean", "Max", "Total"]
summary_stats = summary_stats.reset_index().rename(columns={"Type": "Item Type"})
st.dataframe(summary_stats, use_container_width=True)

st.subheader("🏆 Top 5 Most Utilized Items")
top5 = df.nlargest(5, "Utilization_Score")[["ID","Type","Utilization_Score","Total_Duration"]]
top5 = top5.rename(columns={
    "ID": "Item ID", "Utilization_Score": "Util Score", "Total_Duration": "Total Minutes"
})
st.table(top5)

st.subheader("⛔ Unused Items")
unused = df[df["Utilization_Score"]==0]["ID"].tolist()
if unused:
    st.write(f"Total unused: {len(unused)}")
    st.write(unused[:20] + (["…"] if len(unused)>20 else []))
else:
    st.write("None — all used!")
