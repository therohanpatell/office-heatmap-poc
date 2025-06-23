# Office Utilization Heatmap Dashboard

## Overview

This is a web-based analytics dashboard designed to visualize office space utilization. It generates an interactive heatmap overlay on an office floor plan, allowing users to quickly identify which desks and meeting rooms are most frequently used.

The application helps facility managers and team leaders understand usage patterns, identify underutilized assets, and make data-driven decisions about workspace allocation.

### Key Features

- **Interactive Heatmap:** A "blob-style" heatmap visually represents usage intensity across the floor plan.
- **Dynamic Filtering:** Filter utilization data by custom date ranges or use convenient presets (e.g., "Last 7 Days," "Last 30 Days").
- **Customizable Visualization:** Adjust heatmap parameters like desk/room intensity, opacity, and color schemes.
- **Asset Labels:** Toggle labels on the map to see the specific ID for each desk and room.
- **Responsive UI:** Built with Dash Bootstrap Components for a clean and professional user interface.
- **Robust Data Handling:** The app gracefully handles missing data files and provides clear error messages.

### Technology Stack

- **Web Framework:** [Dash](https://dash.plotly.com/)
- **UI Components:** [Dash Bootstrap Components](https://dash-bootstrap-components.opensource.faculty.ai/)
- **Data Processing:** Pandas, NumPy
- **Image Processing & Visualization:** Pillow, SciPy, Matplotlib

---

## Getting Started

### Prerequisites

- Python 3.8+
- An environment manager like `venv` or `conda` is recommended.

### 1. Installation

Clone the repository and install the required Python packages.

```bash
# Clone this repository
git clone <repository-url>
cd office-heatmap-poc

# (Recommended) Create and activate a virtual environment
python -m venv venv
# On Windows:
# venv\Scripts\activate
# On macOS/Linux:
# source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Required Data Files

Make sure the following files are present in the root directory of the project. The application depends on them to load the floor plan and usage data.

```
office-heatmap-poc/
├── dash_app.py                      # Main application script
├── office_floorplan.jpg             # The background image for the heatmap
├── coordinate_mapping.csv           # Maps asset IDs to pixel coordinates on the floor plan
├── bookings_realistic_simple.csv    # Contains the booking/usage data
└── requirements.txt                 # Python dependencies
```

### 3. Running the Application

Once the dependencies are installed and the data files are in place, you can run the application with a single command:

```bash
python dash_app.py
```

Open your web browser and navigate to `http://127.0.0.1:8050/` to view the dashboard.

---

## Data File Format

### `coordinate_mapping.csv`

This file maps each office asset (desk or meeting room) to its physical location and size on the `office_floorplan.jpg` image.

-   **ID**: A unique identifier for the asset (e.g., `DESK_001`, `ROOM_A`). This must match the ID in the bookings data.
-   **Type**: The type of asset. Must be either `desk` or `meeting-room`.
-   **X_Pixels**, **Y_Pixels**: The top-left corner (x, y) coordinates of the asset on the floor plan image.
-   **Width_Pixels**, **Height_Pixels**: The width and height of the asset's bounding box in pixels.

**Example:**
```csv
ID,Type,X_Pixels,Y_Pixels,Width_Pixels,Height_Pixels
DESK_001,desk,150,200,30,30
ROOM_A01,meeting-room,450,150,100,80
```

### `bookings_realistic_simple.csv`

This file contains the historical booking data for all assets. The application uses this data to calculate utilization.

-   **ID**: The unique identifier of the booked asset. Must match an ID in `coordinate_mapping.csv`.
-   **BookingDate**: The date of the booking in `YYYY-MM-DD` format.
-   **MinutesBooked**: The duration of the booking in minutes. This is primarily used for `meeting-room` utilization calculations.

**Example:**
```csv
ID,BookingDate,MinutesBooked
DESK_001,2024-01-15,480
ROOM_A01,2024-01-15,60
DESK_002,2024-01-16,480
```