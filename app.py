import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
import plotly.graph_objects as go
import plotly.express as px
import base64
import logging
from datetime import datetime, timedelta
from pathlib import Path
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION & CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

class Config:
    """Application configuration constants"""
    BOOKING_CSV = "booking_data.csv"
    COORD_CSV = "coordinate_mapping.csv"
    FLOORPLAN_IMG = "office_floorplan.jpg"

    # Heatmap parameters
    MAX_RESOLUTION_X = 400
    MAX_RESOLUTION_Y = 300
    WORKING_HOURS_PER_DAY = 8
    MINUTES_PER_WORKING_DAY = 480

    # Visualization weights
    DESK_WEIGHT = 1.0
    ROOM_WEIGHT = 1.0

    # Thresholds
    LOW_UTILIZATION_THRESHOLD = 0.2  # 20% or less
    HIGH_UTILIZATION_THRESHOLD = 0.8  # 80% or more

    # Colorscale for heatmap from low (blue) to high (red)
    COLOR_PALETTE = [
        "#001f3f", "#0074D9", "#7FDBFF",
        "#2ECC40", "#FFDC00", "#FF851B", "#FF4136"
    ]


# ═══════════════════════════════════════════════════════════════════════════════
# STREAMLIT PAGE CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Office Utilization Analytics",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': "Office Utilization Analytics Dashboard v2.0"
    }
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main > div {
        padding-top: 2rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ff6b6b;
    }
    .metric-container {
        display: flex;
        justify-content: space-around;
        margin: 1rem 0;
    }
    .section-header {
        color: #1f2937;
        border-bottom: 2px solid #e5e7eb;
        padding-bottom: 0.5rem;
        margin: 2rem 0 1rem 0;
    }
    .alert-warning {
        background-color: #fef3c7;
        border: 1px solid #f59e0b;
        border-radius: 0.375rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .alert-info {
        background-color: #dbeafe;
        border: 1px solid #3b82f6;
        border-radius: 0.375rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 0px 24px;
        background-color: #f1f5f9;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #ffffff;
        border-bottom: 2px solid #ff6b6b;
    }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_and_validate_data():
    """Load and validate all required data files with comprehensive error handling"""
    try:
        # Check if files exist
        files_status = {}
        for file_path in [Config.BOOKING_CSV, Config.COORD_CSV, Config.FLOORPLAN_IMG]:
            files_status[file_path] = Path(file_path).exists()

        if not all(files_status.values()):
            missing_files = [f for f, exists in files_status.items() if not exists]
            st.error(f"❌ Missing required files: {', '.join(missing_files)}")
            st.info("Please ensure all required data files are in the application directory.")
            st.stop()

        # Load booking data
        bookings_df = pd.read_csv(Config.BOOKING_CSV, parse_dates=["Booking_Timestamp"])

        # Validate booking data structure
        required_booking_cols = ["Booking_Timestamp", "Duration", "Desk_ID", "Meeting_Room_ID"]
        missing_cols = [col for col in required_booking_cols if col not in bookings_df.columns]
        if missing_cols:
            st.error(f"❌ Missing required columns in booking data: {', '.join(missing_cols)}")
            st.stop()

        # Load coordinate mapping
        coords_df = pd.read_csv(Config.COORD_CSV)

        # Validate coordinate data structure
        required_coord_cols = ["ID", "Type", "X_Pixels", "Y_Pixels"]
        missing_coord_cols = [col for col in required_coord_cols if col not in coords_df.columns]
        if missing_coord_cols:
            st.error(f"❌ Missing required columns in coordinate data: {', '.join(missing_coord_cols)}")
            st.stop()

        # Data quality checks
        if bookings_df.empty:
            st.warning("⚠️ No booking data found in the dataset.")
            st.stop()

        if coords_df.empty:
            st.warning("⚠️ No coordinate mapping data found.")
            st.stop()

        # Log successful data loading
        logger.info(f"Successfully loaded {len(bookings_df)} bookings and {len(coords_df)} coordinate mappings")

        return bookings_df, coords_df, files_status

    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        st.info("Please check your data files and try again.")
        st.stop()


def create_utilization_metrics(df, daily_bookings=None, analysis_period=None):
    """Create comprehensive utilization metrics."""
    total_items = len(df)
    utilized_items = len(df[df["Utilization_Score"] > 0])
    max_utilization = df["Utilization_Score"].max()

    low_util_items = len(
        df[(df["Utilization_Score"] > 0) & (df["Utilization_Score"] <= Config.LOW_UTILIZATION_THRESHOLD)])
    high_util_items = len(df[df["Utilization_Score"] >= Config.HIGH_UTILIZATION_THRESHOLD])

    # Calculate average daily utilization if booking data is provided
    avg_daily_utilization = np.nan
    if daily_bookings is not None and analysis_period is not None:
        start_date, end_date = analysis_period
        # Ensure date column exists
        daily_bookings["Booking_Date"] = daily_bookings["Booking_Timestamp"].dt.date
        daily_counts = (
            daily_bookings.groupby("Booking_Date")["ID"].nunique()
        )
        # Ensure all days in the period are represented
        all_days = pd.date_range(start_date, end_date, freq="D").date
        daily_counts = daily_counts.reindex(all_days, fill_value=0)
        daily_utilization = daily_counts / total_items if total_items > 0 else 0
        avg_daily_utilization = daily_utilization.mean()

    return {
        "total_items": total_items,
        "utilized_items": utilized_items,
        "utilization_rate": (utilized_items / total_items) * 100 if total_items > 0 else 0,
        "avg_utilization": avg_daily_utilization,
        "max_utilization": max_utilization,
        "low_util_items": low_util_items,
        "high_util_items": high_util_items,
        "unused_items": total_items - utilized_items
    }


def generate_heatmap_data(df, img_dimensions):
    """Generate heatmap intensity grid with optimized performance"""
    x_min, x_max, y_min, y_max = img_dimensions
    w, h = int(x_max - x_min), int(y_max - y_min)

    # Optimize resolution based on image size
    RES_X = min(int(w), Config.MAX_RESOLUTION_X)
    RES_Y = min(int(h), Config.MAX_RESOLUTION_Y)

    xi = np.linspace(x_min, x_max, RES_X)
    yi = np.linspace(y_min, y_max, RES_Y)
    intensity = np.zeros((RES_Y, RES_X))

    # Vectorized intensity calculation for better performance
    for _, row in df.iterrows():
        score = row["Utilization_Score"]
        if score <= 0:
            continue

        # Determine parameters based on type
        if row["Type"] == "desk":
            cx, cy = row["X_Pixels"], row["Y_Pixels"]
            weight = Config.DESK_WEIGHT
        else:
            cx = row["X_Pixels"] + row.get("Width_Pixels", 0) / 2
            cy = row["Y_Pixels"] + row.get("Height_Pixels", 0) / 2
            weight = Config.ROOM_WEIGHT

        # Map to grid coordinates
        ix = int(np.clip((cx - x_min) / (x_max - x_min) * (RES_X - 1), 0, RES_X - 1))
        iy = int(np.clip((cy - y_min) / (y_max - y_min) * (RES_Y - 1), 0, RES_Y - 1))

        # Add intensity without halo effect
        intensity[iy, ix] += score * weight

    # Apply Gaussian blur
    sigma_x = max(3, RES_X / 40)
    sigma_y = max(3, RES_Y / 40)
    blurred = gaussian_filter(intensity, sigma=(sigma_y, sigma_x))

    return blurred, xi, yi


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN APPLICATION
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    # Header
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0;">
        <h1 style="color: #1f2937; margin-bottom: 0.5rem;">🏢 Office Utilization Analytics</h1>
        <p style="color: #6b7280; font-size: 1.2rem;">Interactive workspace optimization dashboard</p>
    </div>
    """, unsafe_allow_html=True)

    # Load data
    with st.spinner("🔄 Loading and validating data..."):
        bookings, coords, file_status = load_and_validate_data()

    # Sidebar filters
    with st.sidebar:
        st.markdown("## 🎛️ Filters & Controls")

        # Date range selection
        min_date = bookings["Booking_Timestamp"].dt.date.min()
        max_date = bookings["Booking_Timestamp"].dt.date.max()

        st.markdown("### 📅 Date Range")
        date_range = st.date_input(
            "Select analysis period:",
            value=(max_date - timedelta(days=30), max_date),
            min_value=min_date,
            max_value=max_date,
            help="Choose the date range for utilization analysis"
        )

        # Utilization threshold controls
        st.markdown("### ⚙️ Analysis Settings")
        low_util_threshold = st.slider(
            "Low Utilization Threshold (%)",
            min_value=5,
            max_value=50,
            value=int(Config.LOW_UTILIZATION_THRESHOLD * 100),
            step=5,
            help="Items below this threshold are considered underutilized"
        ) / 100

        high_util_threshold = st.slider(
            "High Utilization Threshold (%)",
            min_value=60,
            max_value=100,
            value=int(Config.HIGH_UTILIZATION_THRESHOLD * 100),
            step=5,
            help="Items above this threshold are considered highly utilized"
        ) / 100

        # Display options
        st.markdown("### 🎨 Display Options")
        show_heatmap_overlay = st.checkbox("Show Heatmap Overlay", value=True)
        heatmap_opacity = st.slider("Heatmap Opacity", 0.1, 1.0, 0.6, 0.1)

    # Validate date range
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = date_range
        if start_date > end_date:
            st.error("❌ Start date must be before end date.")
            st.stop()
    else:
        st.info("👉 Please select both start and end dates to continue.")
        st.stop()

    # Filter data
    with st.spinner("🔍 Processing data..."):
        filtered_bookings = bookings[
            (bookings["Booking_Timestamp"].dt.date >= start_date) &
            (bookings["Booking_Timestamp"].dt.date <= end_date)
            ].copy()

        if filtered_bookings.empty:
            st.warning("⚠️ No booking data found in the selected date range.")
            st.stop()

        # Aggregate utilization data
        filtered_bookings["ID"] = filtered_bookings["Desk_ID"].fillna(filtered_bookings["Meeting_Room_ID"])

        utilization_agg = (
            filtered_bookings
            .groupby("ID")["Duration"]
            .agg(["sum", "count"])
            .reset_index()
        )
        utilization_agg.columns = ["ID", "Total_Duration", "Booking_Count"]
        utilization_agg["Utilization_Score"] = utilization_agg["Total_Duration"] / Config.MINUTES_PER_WORKING_DAY

        # Merge with coordinates
        analysis_df = coords.merge(utilization_agg, on="ID", how="left").fillna({
            "Total_Duration": 0.0,
            "Utilization_Score": 0.0,
            "Booking_Count": 0
        })

    # Calculate metrics, including average daily utilization
    metrics = create_utilization_metrics(
        analysis_df,
        daily_bookings=filtered_bookings,
        analysis_period=(start_date, end_date)
    )

    # Display key metrics
    st.markdown('<h2 class="section-header">📊 Key Performance Indicators</h2>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(
            "Total Workspace Items",
            f"{metrics['total_items']:,}",
            help="Total number of desks and meeting rooms"
        )
    with col2:
        st.metric(
            "Utilization Rate",
            f"{metrics['utilization_rate']:.1f}%",
            delta=f"{metrics['utilized_items']}/{metrics['total_items']} utilized",
            help="Percentage of items that were used at least once"
        )
    with col3:
        st.metric(
            "Average Daily Utilization",
            f"{metrics['avg_utilization']:.1%}",
            help="Average percentage of seats booked each day"
        )
    with col4:
        st.metric(
            "Peak Utilization",
            f"{metrics['max_utilization']:.1%}",
            help="Highest utilization score achieved"
        )

    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(
        ["🗺️ Interactive Heatmap", "📈 Analytics", "⚠️ Underutilized Spaces", "📋 Detailed Reports"])

    with tab1:
        st.markdown("### Interactive Office Utilization Heatmap")

        # Load and process floorplan
        try:
            img_pil = Image.open(Config.FLOORPLAN_IMG)
            img_width, img_height = img_pil.size
            img_dimensions = (0, img_width, 0, img_height)

            with open(Config.FLOORPLAN_IMG, "rb") as f:
                img_b64 = base64.b64encode(f.read()).decode()
        except FileNotFoundError:
            # Fallback to calculated dimensions
            img_dimensions = (
                analysis_df["X_Pixels"].min() - 50,
                analysis_df["X_Pixels"].max() + 50,
                analysis_df["Y_Pixels"].min() - 50,
                analysis_df["Y_Pixels"].max() + 50
            )
            img_b64 = None

        # Generate heatmap
        if show_heatmap_overlay and not analysis_df.empty:
            with st.spinner("🎨 Generating heatmap visualization..."):
                heatmap_data, xi, yi = generate_heatmap_data(analysis_df, img_dimensions)

                # Create Plotly figure
                fig = go.Figure()

                # Add heatmap
                zmax = np.percentile(heatmap_data, 95) if heatmap_data.max() > 0 else 1
                colorscale = [[i / (len(Config.COLOR_PALETTE) - 1), c]
                             for i, c in enumerate(Config.COLOR_PALETTE)]

                fig.add_trace(
                    go.Heatmap(
                        z=np.flipud(heatmap_data),
                        x=xi,
                        y=yi,
                        colorscale=colorscale,
                        zmin=0,
                        zmax=zmax,
                        zsmooth="best",
                        showscale=True,
                        opacity=heatmap_opacity,
                        colorbar=dict(
                            title="Utilization<br>Intensity",
                            thickness=20,
                            len=0.7,
                            x=1.02
                        )
                    )
                )

                # Add floorplan background
                if img_b64:
                    x_min, x_max, y_min, y_max = img_dimensions
                    fig.update_layout(
                        images=[{
                            "xref": "x", "yref": "y",
                            "x": x_min, "y": y_max,
                            "sizex": x_max - x_min,
                            "sizey": y_max - y_min,
                            "sizing": "stretch",
                            "opacity": 0.8,
                            "layer": "below",
                            "source": f"data:image/png;base64,{img_b64}"
                        }]
                    )

                # Configure layout
                fig.update_layout(
                    title="Office Utilization Heatmap",
                    xaxis=dict(visible=False, showgrid=False, zeroline=False,
                               range=[img_dimensions[0], img_dimensions[1]]),
                    yaxis=dict(visible=False, showgrid=False, zeroline=False,
                               scaleanchor="x", scaleratio=1, range=[img_dimensions[2], img_dimensions[3]]),
                    margin=dict(l=0, r=0, t=40, b=0),
                    dragmode="zoom",
                    height=600
                )

                st.plotly_chart(fig, use_container_width=True)

                # Interactive controls help
                with st.expander("🎮 How to use the interactive heatmap"):
                    st.markdown("""
                    - **Zoom**: Click and drag to select an area, or use the zoom tools in the toolbar
                    - **Pan**: After zooming, click the pan tool (hand icon) to move around  
                    - **Reset**: Click the home icon to return to full view
                    - **Heatmap Colors**: Red/orange areas indicate high utilization, blue areas show low utilization
                    - **Opacity**: Adjust the heatmap opacity using the sidebar slider
                    """)

        else:
            st.info("Enable 'Show Heatmap Overlay' in the sidebar to view the utilization heatmap.")

    with tab2:
        st.markdown("### 📈 Utilization Analytics")

        # Utilization distribution
        col1, col2 = st.columns(2)

        with col1:
            # Utilization histogram
            fig_hist = px.histogram(
                analysis_df[analysis_df["Utilization_Score"] > 0],
                x="Utilization_Score",
                nbins=20,
                title="Utilization Score Distribution",
                labels={"Utilization_Score": "Utilization Score", "count": "Number of Items"},
                color_discrete_sequence=["#ff6b6b"]
            )
            fig_hist.update_layout(height=400)
            st.plotly_chart(fig_hist, use_container_width=True)

        with col2:
            # Type comparison
            type_summary = analysis_df.groupby("Type").agg({
                "Utilization_Score": ["count", "mean", "max"]
            }).round(3)
            type_summary.columns = ["Count", "Avg Utilization", "Max Utilization"]

            fig_bar = px.bar(
                type_summary.reset_index(),
                x="Type",
                y="Avg Utilization",
                title="Average Utilization by Type",
                color="Type",
                color_discrete_sequence=["#4ecdc4", "#ff6b6b"]
            )
            fig_bar.update_layout(height=400)
            st.plotly_chart(fig_bar, use_container_width=True)

        # Detailed statistics table
        st.markdown("#### 📊 Summary Statistics")
        st.dataframe(
            type_summary.reset_index().rename(columns={"Type": "Item Type"}),
            use_container_width=True
        )

    with tab3:
        st.markdown("### ⚠️ Underutilized Workspace Analysis")

        # Filter underutilized items
        low_util_df = analysis_df[
            (analysis_df["Utilization_Score"] > 0) &
            (analysis_df["Utilization_Score"] <= low_util_threshold)
            ].sort_values("Utilization_Score")

        unused_df = analysis_df[analysis_df["Utilization_Score"] == 0]

        # Summary cards
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                "Underutilized Items",
                len(low_util_df),
                delta=f"{len(low_util_df) / len(analysis_df) * 100:.1f}% of total",
                help=f"Items with utilization ≤ {low_util_threshold:.0%}"
            )
        with col2:
            st.metric(
                "Completely Unused",
                len(unused_df),
                delta=f"{len(unused_df) / len(analysis_df) * 100:.1f}% of total",
                help="Items with zero utilization"
            )
        with col3:
            potential_savings = (len(low_util_df) + len(unused_df)) * 0.1  # Assumed cost per item
            st.metric(
                "Optimization Potential",
                f"{len(low_util_df) + len(unused_df)} items",
                help="Items that could be optimized or repurposed"
            )

        # Detailed breakdown
        if not low_util_df.empty:
            st.markdown("#### 🔍 Underutilized Items (Low Usage)")
            st.markdown(f"*Items with utilization between 0.1% and {low_util_threshold:.0%}*")

            display_cols = ["ID", "Type", "Utilization_Score", "Total_Duration", "Booking_Count"]
            low_util_display = low_util_df[display_cols].copy()
            low_util_display["Utilization_Score"] = low_util_display["Utilization_Score"].apply(lambda x: f"{x:.1%}")
            low_util_display["Total_Duration"] = low_util_display["Total_Duration"].apply(lambda x: f"{x:.0f} min")
            low_util_display.columns = ["Item ID", "Type", "Utilization %", "Total Minutes", "Bookings"]

            st.dataframe(low_util_display, use_container_width=True, height=300)

            # Export option
            csv = low_util_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Underutilized Items Report",
                data=csv,
                file_name=f"underutilized_items_{start_date}_{end_date}.csv",
                mime="text/csv"
            )

        if not unused_df.empty:
            st.markdown("#### ⛔ Completely Unused Items")
            st.markdown("*Items with zero utilization in the selected period*")

            unused_summary = unused_df.groupby("Type").size().reset_index(name="Count")

            col1, col2 = st.columns([1, 2])
            with col1:
                st.dataframe(unused_summary, use_container_width=True)

            with col2:
                if len(unused_df) <= 50:  # Show full list if reasonable size
                    unused_list = unused_df[["ID", "Type"]].copy()
                    unused_list.columns = ["Item ID", "Type"]
                    st.dataframe(unused_list, use_container_width=True, height=200)
                else:
                    st.info(f"📋 {len(unused_df)} unused items total. Download the full report for complete details.")

            # Export unused items
            csv_unused = unused_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Unused Items Report",
                data=csv_unused,
                file_name=f"unused_items_{start_date}_{end_date}.csv",
                mime="text/csv"
            )

        # Recommendations
        with st.expander("💡 Optimization Recommendations"):
            st.markdown("""
            **For Underutilized Items:**
            - Consider relocating high-demand equipment to these spaces
            - Evaluate if the location affects accessibility
            - Review booking processes and user awareness
            - Consider alternative uses (storage, collaboration zones)

            **For Unused Items:**
            - Investigate potential causes (equipment issues, location, access)
            - Consider repurposing for high-demand activities
            - Evaluate removal or consolidation opportunities
            - Check if items are properly listed in booking systems

            **Cost Optimization:**
            - Calculate maintenance costs for unused items
            - Consider subletting or repurposing unused spaces
            - Optimize cleaning and utility costs for underutilized areas
            """)

    with tab4:
        st.markdown("### 📋 Detailed Utilization Reports")

        # Top performers
        st.markdown("#### 🏆 Most Utilized Items")
        top_utilized = analysis_df.nlargest(10, "Utilization_Score")[
            ["ID", "Type", "Utilization_Score", "Total_Duration", "Booking_Count"]
        ].copy()

        if not top_utilized.empty:
            top_utilized["Utilization_Score"] = top_utilized["Utilization_Score"].apply(lambda x: f"{x:.1%}")
            top_utilized["Total_Duration"] = top_utilized["Total_Duration"].apply(lambda x: f"{x:.0f} min")
            top_utilized.columns = ["Item ID", "Type", "Utilization %", "Total Minutes", "Bookings"]
            st.dataframe(top_utilized, use_container_width=True)

        # Full dataset export
        st.markdown("#### 📊 Complete Dataset Export")
        st.markdown("Download the complete analysis dataset for further processing:")

        export_df = analysis_df.copy()
        export_df["Analysis_Period"] = f"{start_date} to {end_date}"
        export_df["Low_Utilization_Threshold"] = low_util_threshold
        export_df["High_Utilization_Threshold"] = high_util_threshold

        csv_complete = export_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Complete Analysis Report",
            data=csv_complete,
            file_name=f"office_utilization_analysis_{start_date}_{end_date}.csv",
            mime="text/csv",
            help="Downloads all utilization data with analysis parameters"
        )

        # Summary report
        with st.expander("📈 Executive Summary"):
            st.markdown(f"""
            **Analysis Period:** {start_date} to {end_date} ({(end_date - start_date).days + 1} days)

            **Key Findings:**
            - **Total Workspace Items:** {metrics['total_items']:,}
            - **Overall Utilization Rate:** {metrics['utilization_rate']:.1f}%
            - **Average Daily Utilization:** {metrics['avg_utilization']:.1%}
            - **Items Needing Attention:** {metrics['low_util_items'] + metrics['unused_items']} ({(metrics['low_util_items'] + metrics['unused_items']) / metrics['total_items'] * 100:.1f}% of total)

            **Breakdown by Category:**
            - High Utilization (≥{high_util_threshold:.0%}): {metrics['high_util_items']} items
            - Normal Utilization: {metrics['utilized_items'] - metrics['high_util_items'] - metrics['low_util_items']} items
            - Low Utilization (≤{low_util_threshold:.0%}): {metrics['low_util_items']} items
            - Unused: {metrics['unused_items']} items

            **Recommendations:**
            1. **Immediate Action Required:** Focus on {metrics['unused_items']} completely unused items
            2. **Optimization Opportunity:** Review {metrics['low_util_items']} underutilized items
            3. **Capacity Planning:** Monitor {metrics['high_util_items']} high-demand items for potential expansion needs
            4. **Cost Savings Potential:** Consider consolidating or repurposing {metrics['low_util_items'] + metrics['unused_items']} underperforming assets
            """)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #6b7280; padding: 2rem 0;">
        <p><strong>Office Utilization Analytics Dashboard v2.0</strong></p>
        <p>📊 Data-driven workspace optimization • 🏢 Maximize efficiency • 💰 Reduce costs</p>
        <p><em>Last updated: {}</em></p>
    </div>
    """.format(datetime.now().strftime("%Y-%m-%d %H:%M")), unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# ERROR HANDLING & MONITORING
# ═══════════════════════════════════════════════════════════════════════════════

def setup_error_monitoring():
    """Setup comprehensive error monitoring and logging"""
    import sys
    import traceback

    def handle_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return

        logger.error(
            "Uncaught exception",
            exc_info=(exc_type, exc_value, exc_traceback)
        )

        st.error("""
        ❌ **An unexpected error occurred**

        The application encountered an error while processing your request. 
        Please try refreshing the page or contact support if the problem persists.
        """)

    sys.excepthook = handle_exception


# ═══════════════════════════════════════════════════════════════════════════════
# PERFORMANCE MONITORING
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data
def get_performance_stats():
    """Get application performance statistics"""
    return {
        "cache_hits": 0,  # Would be populated by actual cache metrics
        "load_time": 0,  # Would be populated by actual timing
        "data_size": 0  # Would be populated by actual data size
    }


# ═══════════════════════════════════════════════════════════════════════════════
# APPLICATION ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    try:
        setup_error_monitoring()
        main()
    except Exception as e:
        st.error(f"""
        🚨 **Critical Application Error**

        The application failed to start properly. Please check:
        - All required data files are present
        - File permissions are correct
        - Dependencies are installed

        Error details: `{str(e)}`
        """)
        logger.error(f"Critical application error: {str(e)}", exc_info=True)