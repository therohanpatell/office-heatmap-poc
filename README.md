# Office Utilization Analytics Dashboard - Complete Documentation

## Table of Contents
1. [Overview](#overview)
2. [Application Architecture](#application-architecture)
3. [File Structure & Requirements](#file-structure--requirements)
4. [Data Flow](#data-flow)
5. [Code Structure Breakdown](#code-structure-breakdown)
6. [Configuration Settings](#configuration-settings)
7. [User Interface Components](#user-interface-components)
8. [Key Functions Explained](#key-functions-explained)
9. [Customization Guide](#customization-guide)
10. [Troubleshooting](#troubleshooting)
11. [Performance Optimization](#performance-optimization)
12. [Deployment Guide](#deployment-guide)

---

## Overview

### What This Application Does
The Office Utilization Analytics Dashboard is a web-based tool that helps organizations:
- **Visualize** how office spaces (desks and meeting rooms) are being used
- **Analyze** utilization patterns with interactive heatmaps
- **Identify** underutilized or unused spaces
- **Generate** reports for space optimization decisions
- **Save costs** by optimizing workspace allocation

### Technology Stack
- **Frontend**: Streamlit (Python web framework)
- **Data Processing**: Pandas, NumPy
- **Visualizations**: Plotly (interactive charts and heatmaps)
- **Image Processing**: PIL (Python Imaging Library), SciPy
- **Styling**: Custom CSS with Streamlit

---

## Application Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    USER INTERFACE                           │
│  ┌─────────────┬─────────────┬─────────────┬─────────────┐  │
│  │  Heatmap    │  Analytics  │ Under-util  │   Reports   │  │
│  │     Tab     │     Tab     │     Tab     │     Tab     │  │
│  └─────────────┴─────────────┴─────────────┴─────────────┘  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   DATA PROCESSING                           │
│  ┌─────────────┬─────────────┬─────────────────────────────┐ │
│  │   Load &    │  Filter &   │      Generate               │ │
│  │  Validate   │  Aggregate  │     Heatmap                 │ │
│  │    Data     │    Data     │                             │ │
│  └─────────────┴─────────────┴─────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    DATA SOURCES                             │
│  ┌─────────────┬─────────────┬─────────────────────────────┐ │
│  │  Booking    │ Coordinate  │      Floor Plan             │ │
│  │   Data      │  Mapping    │       Image                 │ │
│  │  (.csv)     │   (.csv)    │      (.jpg)                 │ │
│  └─────────────┴─────────────┴─────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

---

## File Structure & Requirements

### Required Files
Your application directory must contain these files:

```
project_folder/
├── app.py                 # Main application code
├── booking_data.csv           # Booking history data
├── coordinate_mapping.csv     # Space coordinates
├── office_floorplan.jpg       # Floor plan image
└── requirements.txt           # Python dependencies
```

### Required Python Packages
```txt
streamlit
pandas
numpy
plotly
pillow
scipy
```

### Data File Formats

#### 1. booking_data.csv
**Purpose**: Contains all booking/usage records
```csv
Booking_Timestamp,Duration,Desk_ID,Meeting_Room_ID
2024-01-15 09:00:00,120,DESK_001,
2024-01-15 10:30:00,60,,ROOM_A01
2024-01-15 14:00:00,90,DESK_002,
```

**Columns Explained**:
- `Booking_Timestamp`: When the booking started (YYYY-MM-DD HH:MM:SS)
- `Duration`: How long the space was used (in minutes)
- `Desk_ID`: Desk identifier (leave empty if it's a room booking)
- `Meeting_Room_ID`: Room identifier (leave empty if it's a desk booking)

#### 2. coordinate_mapping.csv
**Purpose**: Maps each space to its physical location on the floor plan
```csv
ID,Type,X_Pixels,Y_Pixels,Width_Pixels,Height_Pixels
DESK_001,desk,150,200,,
DESK_002,desk,300,200,,
ROOM_A01,room,450,150,100,80
```

**Columns Explained**:
- `ID`: Unique identifier matching Desk_ID or Meeting_Room_ID
- `Type`: Either "desk" or "room"
- `X_Pixels`: Horizontal position on floor plan image
- `Y_Pixels`: Vertical position on floor plan image
- `Width_Pixels`: Width of room (optional, for rooms only)
- `Height_Pixels`: Height of room (optional, for rooms only)

#### 3. office_floorplan.jpg
**Purpose**: Background image showing the office layout
- Should be a clear floor plan image
- Coordinates in CSV should correspond to pixel positions on this image
- Supported formats: JPG, PNG

---

## Data Flow

### Step-by-Step Process

1. **Application Startup**
   ```python
   # Application starts and loads configuration
   Config class → Sets all parameters and thresholds
   ```

2. **Data Loading & Validation**
   ```python
   load_and_validate_data() → Checks files exist → Loads CSVs → Validates structure
   ```

3. **User Input Processing**
   ```python
   Sidebar filters → Date range selection → Threshold adjustments
   ```

4. **Data Filtering & Aggregation**
   ```python
   Filter by date range → Group bookings by space → Calculate utilization scores
   ```

5. **Visualization Generation**
   ```python
   Create heatmap data → Generate Plotly charts → Display results
   ```

6. **Report Generation**
   ```python
   Identify problem areas → Generate recommendations → Create export files
   ```

---

## Code Structure Breakdown

### 1. Configuration Section (Lines 1-60)
```python
class Config:
    """Application configuration constants"""
```
**Purpose**: Central place for all settings
**What you can change**:
- File names
- Heatmap resolution
- Color schemes
- Utilization thresholds

### 2. Streamlit Setup (Lines 61-120)
```python
st.set_page_config(...)
st.markdown("""<style>...""")
```
**Purpose**: Configures the web interface appearance
**What you can change**:
- Page title and icon
- CSS styling
- Layout settings

### 3. Utility Functions (Lines 121-250)
```python
@st.cache_data
def load_and_validate_data():
```
**Purpose**: Core data processing functions
**Key Functions**:
- `load_and_validate_data()`: Loads and checks all data files
- `create_utilization_metrics()`: Calculates key statistics
- `generate_heatmap_data()`: Creates visualization data

### 4. Main Application (Lines 251-600)
```python
def main():
```
**Purpose**: The main application logic and user interface
**Sections**:
- Header and data loading
- Sidebar controls
- Four main tabs with different views

### 5. Error Handling (Lines 601-650)
```python
def setup_error_monitoring():
```
**Purpose**: Catches and displays errors gracefully

---

## Configuration Settings

### Key Settings You Can Modify

#### File Names (Config class)
```python
BOOKING_CSV = "booking_data.csv"        # Change your booking data file name
COORD_CSV = "coordinate_mapping.csv"    # Change your coordinate file name  
FLOORPLAN_IMG = "office_floorplan.jpg"  # Change your floor plan image name
```

#### Heatmap Resolution
```python
MAX_RESOLUTION_X = 400  # Higher = more detailed heatmap (slower)
MAX_RESOLUTION_Y = 300  # Lower = faster processing (less detail)
```

#### Working Hours Settings
```python
WORKING_HOURS_PER_DAY = 8      # Adjust for your office hours
MINUTES_PER_WORKING_DAY = 480  # 8 hours × 60 minutes
```

#### Utilization Thresholds
```python
LOW_UTILIZATION_THRESHOLD = 0.2   # 20% or less = underutilized
HIGH_UTILIZATION_THRESHOLD = 0.8  # 80% or more = highly utilized
```

#### Colors
```python
COLOR_PALETTE = [
    "#001f3f",  # Dark blue (low utilization)
    "#0074D9",  # Blue
    "#7FDBFF",  # Light blue  
    "#2ECC40",  # Green
    "#FFDC00",  # Yellow
    "#FF851B",  # Orange
    "#FF4136"   # Red (high utilization)
]
```

---

## User Interface Components

### Sidebar Controls
- **Date Range Picker**: Select analysis period
- **Utilization Thresholds**: Adjust what's considered low/high usage
- **Display Options**: Control heatmap appearance

### Main Content Tabs

#### Tab 1: Interactive Heatmap 🗺️
**What it shows**: Visual overlay of utilization on floor plan
**How to use**: 
- Zoom and pan around the office layout
- Red/orange areas = high utilization
- Blue areas = low utilization
- Adjust opacity with sidebar slider

#### Tab 2: Analytics 📈
**What it shows**: Charts and statistics
**Components**:
- Utilization distribution histogram
- Comparison between desks and rooms
- Summary statistics table

#### Tab 3: Underutilized Spaces ⚠️
**What it shows**: Spaces that need attention
**Components**:
- Count of underutilized items
- List of specific problem spaces
- Downloadable reports
- Optimization recommendations

#### Tab 4: Detailed Reports 📋
**What it shows**: Comprehensive data exports
**Components**:
- Top performing spaces
- Complete dataset download
- Executive summary

---

## Key Functions Explained

### 1. Data Loading Function
```python
@st.cache_data(ttl=3600)  # Caches data for 1 hour for performance
def load_and_validate_data():
```
**What it does**:
- Checks if all required files exist
- Loads CSV files into pandas DataFrames
- Validates that required columns are present
- Returns clean, validated data

**To modify**: Change file validation logic or add new data sources

### 2. Utilization Calculation
```python
def create_utilization_metrics(df):
```
**What it does**:
- Calculates how many spaces are used vs. total spaces
- Computes average utilization across all spaces
- Identifies underutilized and highly utilized spaces

**Formula**: `Utilization Score = Total Duration Used / Total Available Time`

### 3. Heatmap Generation
```python
def generate_heatmap_data(df, img_dimensions):
```
**What it does**:
- Creates a grid overlay for the floor plan
- Assigns intensity values based on utilization
- Applies gaussian blur for smooth visualization
- Handles different weights for desks vs. rooms

**To modify**: Adjust blur radius, grid resolution, or intensity calculations

### 4. Main Application Loop
```python
def main():
```
**What it does**:
- Sets up the user interface
- Processes user inputs from sidebar
- Calls data processing functions
- Displays results in tabs

---

## Customization Guide

### Adding New Metrics

1. **Add to Config class**:
```python
class Config:
    NEW_THRESHOLD = 0.5  # Your new threshold
```

2. **Modify metrics function**:
```python
def create_utilization_metrics(df):
    # Add your new calculation
    new_metric = df[df["Utilization_Score"] > Config.NEW_THRESHOLD].count()
    
    return {
        # ... existing metrics ...
        "new_metric": new_metric
    }
```

3. **Display in interface**:
```python
st.metric("New Metric", f"{metrics['new_metric']:,}")
```

### Changing Colors

**For heatmap colors**:
```python
COLOR_PALETTE = [
    "#your_color_1",  # Low utilization
    "#your_color_2",  # Medium-low
    "#your_color_3",  # Medium
    "#your_color_4",  # Medium-high
    "#your_color_5"   # High utilization
]
```

**For interface colors**:
Modify the CSS in the `st.markdown()` section:
```python
st.markdown("""
<style>
    .stMetric {
        border-left: 4px solid #your_new_color;
    }
</style>
""")
```

### Adding New Data Sources

1. **Add to Config**:
```python
NEW_DATA_CSV = "new_data.csv"
```

2. **Modify loading function**:
```python
def load_and_validate_data():
    # ... existing code ...
    new_df = pd.read_csv(Config.NEW_DATA_CSV)
    return bookings, coords, new_df, files_status
```

3. **Update main function**:
```python
def main():
    bookings, coords, new_data, file_status = load_and_validate_data()
    # Process new_data as needed
```

### Modifying Calculations

**To change utilization formula**:
```python
# Current formula in aggregation section:
utilization_agg["Utilization_Score"] = utilization_agg["Total_Duration"] / Config.MINUTES_PER_WORKING_DAY

# Example alternative (percentage of bookings):
utilization_agg["Utilization_Score"] = utilization_agg["Booking_Count"] / total_possible_bookings
```

---

## Troubleshooting

### Common Issues and Solutions

#### 1. "Missing required files" Error
**Problem**: Application can't find data files
**Solution**: 
- Check file names match exactly in Config class
- Ensure files are in same directory as Python script
- Check file permissions

#### 2. "Missing required columns" Error
**Problem**: CSV files don't have expected column names
**Solution**:
- Check CSV headers match exactly: `Booking_Timestamp`, `Duration`, `Desk_ID`, `Meeting_Room_ID`
- Ensure no extra spaces in column names
- Use a CSV editor to verify file structure

#### 3. Heatmap Not Displaying
**Problem**: No heatmap visible on floor plan
**Solutions**:
- Check "Show Heatmap Overlay" is enabled in sidebar
- Verify coordinate data has valid X_Pixels and Y_Pixels values
- Ensure floor plan image loads correctly
- Check that some bookings exist in selected date range

#### 4. Performance Issues (Slow Loading)
**Solutions**:
- Reduce heatmap resolution in Config:
  ```python
  MAX_RESOLUTION_X = 200  # Reduce from 400
  MAX_RESOLUTION_Y = 150  # Reduce from 300
  ```
- Filter data to smaller date ranges
- Check data file sizes (very large files slow processing)

#### 5. Date Range Issues
**Problem**: "No booking data found in selected date range"
**Solutions**:
- Check date format in CSV is YYYY-MM-DD HH:MM:SS
- Verify dates in CSV fall within selectable range
- Check for timezone issues in timestamps

### Debug Mode
To enable detailed error information, add at the top of your script:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

---

## Performance Optimization

### For Large Datasets

#### 1. Data Caching
The application uses Streamlit's caching:
```python
@st.cache_data(ttl=3600)  # Cache for 1 hour
```
**To adjust**: Change `ttl` value (time in seconds)

#### 2. Reduce Heatmap Resolution
```python
# In Config class - lower values = faster processing
MAX_RESOLUTION_X = 200  # Instead of 400
MAX_RESOLUTION_Y = 150  # Instead of 300
```

#### 3. Optimize Data Loading
```python
# Add these parameters to pandas read_csv for large files:
pd.read_csv(file, 
    parse_dates=["Booking_Timestamp"],
    dtype={"Desk_ID": "category", "Meeting_Room_ID": "category"}  # Saves memory
)
```

#### 4. Limit Data Display
```python
# Show only top N items in reports
TOP_N_ITEMS = 50
top_items = df.head(TOP_N_ITEMS)
```

### Memory Management
```python
# Clear large variables when done
del large_dataframe
import gc
gc.collect()
```

---

## Deployment Guide

### Running Locally
1. **Install Python 3.8+**
2. **Install dependencies**:
   ```bash
   pip install streamlit pandas numpy plotly pillow scipy
   ```
3. **Prepare data files** in same directory
4. **Run application**:
   ```bash
   streamlit run app.py
   ```
5. **Open browser** to displayed URL (usually http://localhost:8501)

### Cloud Deployment Options

#### Streamlit Cloud (Recommended)
1. **Upload code to GitHub**
2. **Connect to Streamlit Cloud** (share.streamlit.io)
3. **Deploy directly** from repository
4. **Upload data files** through Streamlit interface

#### Other Platforms
- **Heroku**: Requires Procfile and requirements.txt
- **AWS EC2**: Install Python and dependencies, run with screen/tmux
- **Docker**: Create Dockerfile with Python environment

### Security Considerations
- **Don't include sensitive data** in public repositories
- **Use environment variables** for sensitive configuration
- **Implement access controls** if deploying publicly
- **Regularly update dependencies** for security patches

---

## Advanced Customizations

### Adding New Visualizations

#### Example: Adding a Time Series Chart
```python
# In Tab 2 (Analytics), add this code:
daily_usage = filtered_bookings.groupby(
    filtered_bookings["Booking_Timestamp"].dt.date
)["Duration"].sum().reset_index()

fig_timeline = px.line(
    daily_usage, 
    x="Booking_Timestamp", 
    y="Duration",
    title="Daily Usage Over Time"
)
st.plotly_chart(fig_timeline, use_container_width=True)
```

### Custom Filters

#### Example: Adding Department Filter
```python
# In sidebar section:
if "Department" in bookings.columns:
    departments = st.multiselect(
        "Select Departments:",
        options=bookings["Department"].unique(),
        default=bookings["Department"].unique()
    )
    
    # In filtering section:
    filtered_bookings = filtered_bookings[
        filtered_bookings["Department"].isin(departments)
    ]
```

### Integration with External Systems

#### Example: Database Connection
```python
import sqlite3

@st.cache_data
def load_from_database():
    conn = sqlite3.connect("office_data.db")
    bookings = pd.read_sql_query("SELECT * FROM bookings", conn)
    coords = pd.read_sql_query("SELECT * FROM coordinates", conn)
    conn.close()
    return bookings, coords
```

### Custom Export Formats

#### Example: Excel Export with Multiple Sheets
```python
import openpyxl
from io import BytesIO

def create_excel_report(analysis_df, metrics):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        analysis_df.to_excel(writer, sheet_name='Full_Analysis', index=False)
        pd.DataFrame([metrics]).to_excel(writer, sheet_name='Summary', index=False)
    
    return output.getvalue()

# In reports tab:
excel_data = create_excel_report(analysis_df, metrics)
st.download_button(
    label="📊 Download Excel Report",
    data=excel_data,
    file_name="office_analysis.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)
```

---

## Maintenance and Updates

### Regular Maintenance Tasks

1. **Data Quality Checks**
   - Verify booking data completeness
   - Check for duplicate entries
   - Validate coordinate accuracy

2. **Performance Monitoring**
   - Monitor load times
   - Check memory usage
   - Review user feedback

3. **Dependency Updates**
   ```bash
   pip list --outdated
   pip install --upgrade package_name
   ```

### Version Control Best Practices
- Keep data files separate from code
- Use meaningful commit messages
- Tag releases with version numbers
- Maintain a changelog

### Backup Strategy
- **Code**: Use Git repository
- **Data**: Regular automated backups
- **Configuration**: Document all custom settings

---

## Support and Resources

### Getting Help
1. **Streamlit Documentation**: https://docs.streamlit.io
2. **Plotly Documentation**: https://plotly.com/python
3. **Pandas Documentation**: https://pandas.pydata.org/docs
4. **GitHub Issues**: For bug reports and feature requests

### Extending the Application
This documentation provides the foundation for understanding and modifying the Office Utilization Analytics Dashboard. The modular structure makes it easy to add new features, modify existing functionality, or integrate with other systems.

Remember to test thoroughly after making changes and keep backups of working versions.

---

*Last Updated: 2024*
*Version: 2.0*