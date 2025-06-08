import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
import plotly.graph_objects as go
import base64
import io

# ── 1) STREAMLIT CONFIG ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Interactive Office Heatmap",
    layout="wide",
    initial_sidebar_state="collapsed"
)
st.title("📊 Interactive Office Utilization Heatmap")

# ── 2) FILEPATHS ──────────────────────────────────────────────────────────────
BOOKING_CSV    = "booking_data.csv"
COORD_CSV      = "coordinate_mapping.csv"
FLOORPLAN_IMG  = "office_floorplan.jpg"

# ── 3) LOAD DATA (cached) ─────────────────────────────────────────────────────
@st.cache_data
def load_data():
    bookings_df = pd.read_csv(BOOKING_CSV, parse_dates=["Booking_Timestamp"])
    coords_df   = pd.read_csv(COORD_CSV)
    return bookings_df, coords_df

bookings, coords = load_data()
coords = coords.copy()

# ── 4) DATE RANGE PICKER (MAIN PAGE, SAFELY HANDLED) ─────────────────────────
# First compute your min/max dates from the booking timestamps:
min_date = bookings["Booking_Timestamp"].dt.date.min()
max_date = bookings["Booking_Timestamp"].dt.date.max()

st.header("📅 Filter Bookings by Date")

# Call date_input a single time
date_sel = st.date_input(
    "Select date range:",
    value=(min_date, max_date),   # this default is a 2-tuple
    min_value=min_date,
    max_value=max_date,
    key="date_range"
)

# Only unpack once we actually have two dates
if isinstance(date_sel, tuple) and len(date_sel) == 2:
    start_date, end_date = date_sel
else:
    st.info("👉 Please select *both* a start **and** end date to view the heatmap.")
    st.stop()

# Now it’s safe to validate and filter
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

# ── 5) AGGREGATE UTILIZATION SCORES ────────────────────────────────────────────
filtered["ID"] = filtered["Desk_ID"].fillna(filtered["Meeting_Room_ID"])
agg = (
    filtered
    .groupby("ID")["Duration"]
    .sum()
    .reset_index(name="Total_Duration")
)
agg["Utilization_Score"] = agg["Total_Duration"] / 480.0  # normalize to 8-hour day

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
    x_min = min(df["X_Pixels"].min(), (df["X_Pixels"] - df["Width_Pixels"]).min())
    x_max = max(df["X_Pixels"].max(), (df["X_Pixels"] + df["Width_Pixels"]).max())
    y_min = min(df["Y_Pixels"].min(), (df["Y_Pixels"] - df["Height_Pixels"]).min())
    y_max = max(df["Y_Pixels"].max(), (df["Y_Pixels"] + df["Height_Pixels"]).max())
    img_width, img_height = int(x_max - x_min), int(y_max - y_min)

# ── 7) BUILD A DOWN-SAMPLED HEAT GRID ─────────────────────────────────────────
HEAT_RES_X = min(int(img_width), 400)
HEAT_RES_Y = min(int(img_height), 300)

xi = np.linspace(x_min, x_max, HEAT_RES_X)
yi = np.linspace(y_min, y_max, HEAT_RES_Y)
intensity = np.zeros((HEAT_RES_Y, HEAT_RES_X))

for _, row in df.iterrows():
    cx, cy = row["X_Pixels"], row["Y_Pixels"]
    w, h   = row["Width_Pixels"], row["Height_Pixels"]
    score  = row["Utilization_Score"]
    if score <= 0:
        continue

    if row["Type"] == "desk":
        ix = int(np.clip((cx - x_min)/(x_max - x_min)*(HEAT_RES_X-1), 0, HEAT_RES_X-1))
        iy = int(np.clip((cy - y_min)/(y_max - y_min)*(HEAT_RES_Y-1), 0, HEAT_RES_Y-1))
        intensity[iy, ix] += score * 3.0
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx==0 and dy==0: continue
                ny, nx = iy+dy, ix+dx
                if 0 <= ny < HEAT_RES_Y and 0 <= nx < HEAT_RES_X:
                    intensity[ny, nx] += score * 0.5
    else:
        x1, x2 = cx, cx + w
        y1, y2 = cy, cy + h
        ix1 = int(np.clip((x1 - x_min)/(x_max - x_min)*(HEAT_RES_X-1), 0, HEAT_RES_X-1))
        ix2 = int(np.clip((x2 - x_min)/(x_max - x_min)*(HEAT_RES_X-1), 0, HEAT_RES_X-1))
        iy1 = int(np.clip((y1 - y_min)/(y_max - y_min)*(HEAT_RES_Y-1), 0, HEAT_RES_Y-1))
        iy2 = int(np.clip((y2 - y_min)/(y_max - y_min)*(HEAT_RES_Y-1), 0, HEAT_RES_Y-1))
        ix2, iy2 = max(ix1+1, min(ix2, HEAT_RES_X)), max(iy1+1, min(iy2, HEAT_RES_Y))
        if (ix2-ix1)*(iy2-iy1) > 0:
            intensity[iy1:iy2, ix1:ix2] += score * 0.3

sigma_x = max(3, HEAT_RES_X/40)
sigma_y = max(3, HEAT_RES_Y/40)
blurred = gaussian_filter(intensity, sigma=(sigma_y, sigma_x))

# ── 8) BACKGROUND IMAGE → BASE64 ──────────────────────────────────────────────
def _img_to_base64(path):
    with open(path, "rb") as f:
        data = f.read()
    return f"data:image/png;base64,{base64.b64encode(data).decode()}"

try:
    img_b64 = _img_to_base64(FLOORPLAN_IMG)
except FileNotFoundError:
    img_b64 = None

# ── 9) PLOTLY FIGURE WITH CUSTOM COLORSCALE ──────────────────────────────────
# Your 7-color palette: dark-blue → red
MY_PALETTE = [
    "#001f3f",  # deep navy (unused areas)
    "#0074D9",
    "#7FDBFF",
    "#2ECC40",
    "#FFDC00",
    "#FF851B",
    "#FF4136",  # hottest
]
colorscale = [[i/(len(MY_PALETTE)-1), c] for i, c in enumerate(MY_PALETTE)]

blurred_flipped = np.flipud(blurred)
fig = go.Figure()

fig.add_trace(
    go.Heatmap(
        z=blurred_flipped,
        x=xi,
        y=yi,
        colorscale=colorscale,
        zsmooth="best",
        showscale=False,
        opacity=0.6
    )
)

if img_b64:
    fig.update_layout(
        images=[{
            "xref": "x",
            "yref": "y",
            "x": x_min,
            "y": y_max,
            "sizex": x_max - x_min,
            "sizey": y_max - y_min,
            "sizing": "stretch",
            "opacity": 1.0,
            "layer": "below",
            "source": img_b64
        }]
    )

fig.update_layout(
    xaxis=dict(range=[x_min, x_max], showgrid=False, zeroline=False, visible=False),
    yaxis=dict(range=[y_min, y_max], showgrid=False, zeroline=False, scaleanchor="x", scaleratio=1, visible=False),
    margin=dict(l=0, r=0, t=0, b=0),
    dragmode="zoom"
)

st.plotly_chart(fig, use_container_width=True, height=600)

# ── 10) EXPLANATORY TEXT & LEGEND ──────────────────────────────────────────────
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
unused_ids = df.loc[df["Utilization_Score"] == 0, "ID"].tolist()
if unused_ids:
    st.write(f"Total unused items: **{len(unused_ids):,}**")
    display_list = unused_ids[:20] + (["…"] if len(unused_ids)>20 else [])
    st.write(display_list)
else:
    st.write("None – all desks & rooms have some utilization in this range!")
