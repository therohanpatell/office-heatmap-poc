import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import io
import os
from datetime import date, timedelta

st.set_page_config(layout="wide", page_title="Office Heatmap Dashboard")


# --- Core Logic ---

@st.cache_data
def get_custom_colormap():
    """Creates and returns a custom Green -> Yellow -> Red colormap."""
    colors = [(0, "#20ff40"), (0.5, "#ffff00"), (1, "#ff2020")]
    return LinearSegmentedColormap.from_list("custom_gyr", colors)


@st.cache_data
def generate_legend_image(_colormap):
    """Creates a legend image to display on the webpage."""
    fig, ax = plt.subplots(figsize=(4, 1.5))
    gradient = np.linspace(0, 1, 256).reshape(1, 256)
    ax.imshow(gradient, aspect='auto', cmap=_colormap)
    ax.set_title("Usage Intensity")
    ax.set_xticks([0, 128, 255])
    ax.set_xticklabels(['Low', 'Mid', 'High'])
    ax.set_yticks([])
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.1, transparent=True)
    buf.seek(0)
    return buf


@st.cache_data
def generate_heatmap_image(_floor_plan_img, _df_coords, _df_bookings, start_date, end_date, desk_intensity,
                           room_intensity, opacity, colormap_name, show_labels_on_map):
    """
    Generates the heatmap and optionally overlays asset mappings on top.
    """
    # --- 1. Generate Heatmap ---
    df_bookings['BookingDate'] = pd.to_datetime(df_bookings['BookingDate']).dt.date
    mask = (df_bookings['BookingDate'] >= start_date) & (df_bookings['BookingDate'] <= end_date)
    filtered_bookings = df_bookings.loc[mask].copy()

    desk_usage = \
    filtered_bookings[filtered_bookings['ID'].isin(_df_coords[_df_coords['Type'] == 'desk']['ID'])].groupby('ID')[
        'BookingDate'].nunique()
    room_usage = \
    filtered_bookings[filtered_bookings['ID'].isin(_df_coords[_df_coords['Type'] == 'meeting-room']['ID'])].groupby(
        'ID')['MinutesBooked'].sum()

    _df_coords['raw_usage'] = _df_coords['ID'].map(desk_usage).fillna(_df_coords['ID'].map(room_usage)).fillna(0)

    heat_array = np.zeros((_floor_plan_img.height, _floor_plan_img.width))
    max_desk_usage = _df_coords[_df_coords['Type'] == 'desk']['raw_usage'].max()
    max_room_usage = _df_coords[_df_coords['Type'] == 'meeting-room']['raw_usage'].max()

    for _, row in _df_coords.iterrows():
        intensity = 0.0
        if row['Type'] == 'desk' and max_desk_usage > 0:
            intensity = (row['raw_usage'] / max_desk_usage) * desk_intensity
        elif row['Type'] == 'meeting-room' and max_room_usage > 0:
            intensity = (row['raw_usage'] / max_room_usage) * room_intensity
        if intensity > 0:
            x, y, w, h = int(row['X_Pixels']), int(row['Y_Pixels']), int(row['Width_Pixels']), int(row['Height_Pixels'])
            heat_array[y:y + h, x:x + w] += intensity

    blurred_heat = gaussian_filter(heat_array, sigma=40)
    if blurred_heat.max() > 0:
        normed_heat = (blurred_heat - blurred_heat.min()) / (blurred_heat.max() - blurred_heat.min())
    else:
        normed_heat = blurred_heat

    colormap = get_custom_colormap() if colormap_name == 'custom_gyr' else plt.get_cmap(colormap_name)
    heatmap_image_arr = (colormap(normed_heat)[:, :, :3] * 255).astype(np.uint8)
    heatmap_image = Image.fromarray(heatmap_image_arr)

    alpha = Image.new('L', _floor_plan_img.size, color=int(255 * opacity))
    heatmap_image.putalpha(alpha)

    final_image = Image.alpha_composite(_floor_plan_img.convert('RGBA'), heatmap_image)

    # --- 2. Optionally Draw Asset Mappings on Top ---
    if show_labels_on_map:
        draw = ImageDraw.Draw(final_image)
        try:
            font = ImageFont.truetype("arial.ttf", 12)
        except IOError:
            font = ImageFont.load_default()

        desk_color = "#3b82f6"
        room_color = "#8b5cf6"

        for _, row in _df_coords.iterrows():
            x, y, w, h = int(row['X_Pixels']), int(row['Y_Pixels']), int(row['Width_Pixels']), int(row['Height_Pixels'])
            item_id = str(row['ID'])
            outline_color = desk_color if row['Type'] == 'desk' else room_color

            draw.rectangle([x, y, x + w, y + h], outline=outline_color, width=2)
            text_bbox = draw.textbbox((x + 4, y + 4), item_id, font=font)
            text_bg_bbox = (text_bbox[0] - 2, text_bbox[1] - 2, text_bbox[2] + 2, text_bbox[3] + 2)
            draw.rectangle(text_bg_bbox, fill=outline_color)
            draw.text((x + 4, y + 4), item_id, fill="white", font=font)

    return final_image, colormap


# The 'get_underutilized_assets' function is no longer needed for now.
# def get_underutilized_assets(df_with_usage, start_date, end_date):
#     ...

# --- Streamlit UI ---

st.title("🏢 Interactive Office Utilization Heatmap")

# --- Sidebar for Controls and File Uploads ---
st.sidebar.header("⚙️ Controls & Settings")

# Use default files if they exist, otherwise show uploaders
default_floorplan = "office_floorplan.jpg"
default_coords = "coordinate_mapping.csv"
default_bookings = "bookings_realistic_simple.csv"

use_default_files = os.path.exists(default_floorplan) and os.path.exists(default_coords) and os.path.exists(
    default_bookings)

if use_default_files:
    st.sidebar.success("Default data loaded successfully!")
    with st.sidebar.expander("Override Default Files"):
        uploaded_floorplan = st.file_uploader("Upload New Floor Plan", type=['png', 'jpg'])
        uploaded_coords = st.file_uploader("Upload New Coordinates CSV", type=['csv'])
        uploaded_bookings = st.file_uploader("Upload New Bookings CSV", type=['csv'])

    floor_plan_img = Image.open(uploaded_floorplan or default_floorplan)
    df_coords = pd.read_csv(uploaded_coords or default_coords)
    df_bookings = pd.read_csv(uploaded_bookings or default_bookings)

else:
    st.sidebar.warning("Default files not found.")
    uploaded_floorplan = st.sidebar.file_uploader("1. Upload Floor Plan Image", type=['png', 'jpg'])
    uploaded_coords = st.sidebar.file_uploader("2. Upload Coordinates CSV", type=['csv'])
    uploaded_bookings = st.sidebar.file_uploader("3. Upload Bookings CSV", type=['csv'])

    if not (uploaded_floorplan and uploaded_coords and uploaded_bookings):
        st.info("Please upload all three required files to begin.")
        st.stop()

    floor_plan_img = Image.open(uploaded_floorplan)
    df_coords = pd.read_csv(uploaded_coords)
    df_bookings = pd.read_csv(uploaded_bookings)

# --- Main Page Layout ---

st.header("📅 Select Date Range")
min_date = pd.to_datetime(df_bookings['BookingDate']).dt.date.min()
max_date = pd.to_datetime(df_bookings['BookingDate']).dt.date.max()
# Presets are relative to the most recent data point for relevance
last_data_date = max_date

# Initialize session state for dates if they don't exist
if 'start_date' not in st.session_state:
    st.session_state.start_date = min_date
if 'end_date' not in st.session_state:
    st.session_state.end_date = max_date


def set_date_range(start, end):
    """Callback function to update session state for dates."""
    # Clamp the calculated dates to be within the min/max bounds of the data
    st.session_state.start_date = max(min_date, start)
    st.session_state.end_date = min(max_date, end)


# --- Preset Buttons ---
st.write("Select a preset or define a custom range below.")
cols = st.columns(5)
with cols[0]:
    st.button("Last 7 Days", on_click=set_date_range, args=(last_data_date - timedelta(days=6), last_data_date),
              use_container_width=True)
with cols[1]:
    st.button("Last 30 Days", on_click=set_date_range, args=(last_data_date - timedelta(days=29), last_data_date),
              use_container_width=True)
with cols[2]:
    st.button("Last 90 Days", on_click=set_date_range, args=(last_data_date - timedelta(days=89), last_data_date),
              use_container_width=True)
with cols[3]:
    year_to_date_start = last_data_date.replace(month=1, day=1)
    st.button(f"{last_data_date.year} to Date", on_click=set_date_range, args=(year_to_date_start, last_data_date),
              use_container_width=True, disabled=(year_to_date_start < min_date))
with cols[4]:
    st.button("All Time", on_click=set_date_range, args=(min_date, max_date), use_container_width=True)

# --- Custom Date Range Selector ---
# The date_input's state is now controlled by st.session_state
selected_dates = st.date_input(
    "Custom Date Range",
    value=(st.session_state.start_date, st.session_state.end_date),
    min_value=min_date,
    max_value=max_date,
)

# If the user manually changes the date input, its return value `selected_dates` will be different.
# We then update the session state to reflect this manual change.
if len(selected_dates) == 2:
    start, end = selected_dates
    if start != st.session_state.start_date or end != st.session_state.end_date:
        st.session_state.start_date = start
        st.session_state.end_date = end
        st.experimental_rerun()  # Rerun to ensure all widgets are in sync
else:
    st.warning("Please select a valid date range (start and end date).")
    st.stop()

# The single source of truth for the date range is now session state
start_date, end_date = st.session_state.start_date, st.session_state.end_date

# Heatmap Controls in Sidebar
st.sidebar.subheader("Heatmap Controls")
desk_intensity = st.sidebar.slider("Desk Intensity", 0.1, 5.0, 1.2, 0.1)
room_intensity = st.sidebar.slider("Room Intensity", 0.1, 5.0, 0.4, 0.1)
opacity = st.sidebar.slider("Heatmap Opacity", 0.1, 1.0, 0.7, 0.05)
colormap_name = st.sidebar.selectbox("Color Scheme", ['custom_gyr', 'hot', 'YlOrRd', 'jet'],
                                     help="`custom_gyr` is the recommended Green-Yellow-Red map.")

st.sidebar.subheader("Display Options")
show_labels_on_heatmap = st.sidebar.toggle("Show Labels on Heatmap",
                                           help="Overlay desk and room IDs and boundaries on the heatmap.")

# --- Generation and Display ---
with st.spinner("Generating heatmap..."):
    final_image, colormap = generate_heatmap_image(
        floor_plan_img, df_coords, df_bookings,
        start_date, end_date,
        desk_intensity, room_intensity, opacity, colormap_name,
        show_labels_on_map=show_labels_on_heatmap
    )

    st.header("🔥 Heatmap Visualization")
    st.image(final_image, caption="Office Utilization Heatmap", use_column_width=True)

    # Display Legend in Sidebar
    st.sidebar.subheader("Color Legend")
    legend_img = generate_legend_image(colormap)
    st.sidebar.image(legend_img)

# The Underutilized Assets Table is commented out as requested.
# st.header("📊 Underutilized Assets (<50% Usage)")
# under_desks, under_rooms = get_underutilized_assets(df_with_usage, start_date, end_date)
#
# col1, col2 = st.columns(2)
# with col1:
#     st.subheader(f"Desks ({len(under_desks)})")
#     st.dataframe(under_desks)
# with col2:
#     st.subheader(f"Meeting Rooms ({len(under_rooms)})")
#     st.dataframe(under_rooms)