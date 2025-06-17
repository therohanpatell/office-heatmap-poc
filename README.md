# Office Utilization Analytics Dashboard

## Overview
The Office Utilization Analytics Dashboard is a Streamlit application that visualizes workspace usage. It reads booking history for desks and meeting rooms and overlays utilization metrics on top of your office floor plan. The project was created for personal and educational use.

## Features
- Interactive heatmap showing desk and room utilization
- Filter by date range and adjust utilization thresholds
- Analytics tab with usage statistics and distribution charts
- List of underutilized spaces and downloadable reports
- Modular design that makes it easy to extend with new metrics

## Quick Start
1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```
2. **Prepare Data**
   - `booking_data.csv` – Historical bookings (Desk_ID/Meeting_Room_ID, timestamp, duration)
   - `coordinate_mapping.csv` – Pixel coordinates for each space on your floor plan
   - `office_floorplan.jpg` – Image of the office layout used as the heatmap background
3. **Run the Application**
   ```bash
   streamlit run app.py
   ```
   The dashboard will open in your browser.

## Repository Layout
```
app.py                 # Streamlit application
booking_data.csv       # Example booking records
coordinate_mapping.csv # Example coordinates
office_floorplan.jpg   # Floor plan image
requirements.txt       # Python dependencies
scripts/               # Utility scripts
```

### Scripts
- `scripts/generate_bookings.py` – Creates sample booking data for testing
- `scripts/Heat_map.py` – Stand‑alone heatmap generation example

## Data Format
### booking_data.csv
```
Desk_ID,Meeting_Room_ID,Employee_Name,Booking_Timestamp,Duration
D-015,,Kristen Walker,2025-10-05 21:32:40.713829,60
```
- `Desk_ID` or `Meeting_Room_ID` identifies the space
- `Booking_Timestamp` is a datetime value
- `Duration` is measured in minutes

### coordinate_mapping.csv
```
ID,Type,X_Pixels,Y_Pixels,Width_Pixels,Height_Pixels
D-001,desk,86,65,20,20
```
- `ID` matches the booking identifiers
- `Type` is either `desk` or `meeting-room`
- Pixel coordinates should align with your floor plan image

## Contributing
Pull requests are welcome! If you plan a larger contribution, please open an issue to discuss your proposal first.

1. Fork the repository and create a new branch.
2. Make your changes following the existing code style.
3. Run linting or tests if available.
4. Open a pull request with a clear description of your changes.

## License
This project is distributed under a custom personal license. Commercial or professional use is prohibited without explicit written permission. See [LICENSE.md](LICENSE.md) for the full text.

## Acknowledgements
The dashboard uses Streamlit, Pandas, Plotly and Pillow. The included floor plan and sample data are placeholders to demonstrate functionality.
