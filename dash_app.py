"""
This script creates an interactive web-based dashboard using Dash and Plotly
to visualize office space utilization. It generates a heatmap overlay on an
office floorplan to show areas of high and low usage based on booking data.

The application is structured as follows:
1.  Core Logic: Functions for data processing and image generation.
2.  Data Loading: Loads data from CSV files and the floorplan image.
3.  Dash App Initialization: Sets up the Dash application.
4.  UI Layout: Defines the user interface using Dash Bootstrap Components.
5.  Callbacks: Contains the logic that makes the dashboard interactive.
"""

import dash
from dash import dcc, html, Input, Output, State, no_update
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import io
import os
from datetime import date, timedelta, datetime
import base64


# --- Core Logic ---

def get_custom_colormap():
    """
    Creates and returns a custom LinearSegmentedColormap.
    This provides a visually intuitive Green -> Yellow -> Red gradient for the heatmap.
    
    Returns:
        LinearSegmentedColormap: A Matplotlib colormap object.
    """
    colors = [(0, "#20ff40"), (0.5, "#ffff00"), (1, "#ff2020")]  # Green at 0, Yellow at 0.5, Red at 1.0
    return LinearSegmentedColormap.from_list("custom_gyr", colors)


def generate_legend_image(colormap_name):
    """
    Generates a PNG image of the color legend for the selected colormap.
    This is displayed in the UI to help users understand the heatmap colors.

    Args:
        colormap_name (str): The name of the colormap to render.

    Returns:
        str: A base64 encoded PNG image string.
    """
    # Create a new Matplotlib figure and axes for the legend.
    # The small size is optimized for display below the heatmap.
    fig, ax = plt.subplots(figsize=(6, 1.2))

    # Get the colormap object, either our custom one or a standard one.
    colormap = get_custom_colormap() if colormap_name == 'custom_gyr' else plt.get_cmap(colormap_name)

    # Create a horizontal gradient to represent the colormap.
    gradient = np.linspace(0, 1, 256).reshape(1, 256)
    ax.imshow(gradient, aspect='auto', cmap=colormap)

    # Configure the legend's appearance.
    ax.set_title("Usage Intensity", fontsize=12, pad=10)
    ax.set_xticks([0, 128, 255])
    ax.set_xticklabels(['Low', 'Medium', 'High'])
    ax.set_yticks([])  # Hide the y-axis ticks.

    # Save the figure to an in-memory buffer as a PNG.
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.1, transparent=True)
    plt.close(fig)  # Close the figure to free up memory.

    # Convert the PNG buffer to a base64 string for embedding in HTML.
    return pil_to_b64(Image.open(buf))


def generate_heatmap_image(floor_plan_img, df_coords, df_bookings, start_date, end_date, desk_intensity, room_intensity,
                           opacity, colormap_name, show_labels, show_desks, show_rooms):
    """
    This is the main image generation function. It takes all the raw data and user inputs
    to produce the final heatmap visualization.

    Args:
        floor_plan_img (PIL.Image): The base floor plan image.
        df_coords (pd.DataFrame): DataFrame with asset coordinates.
        df_bookings (pd.DataFrame): DataFrame with booking data.
        start_date (str): The start date for filtering bookings ('YYYY-MM-DD').
        end_date (str): The end date for filtering bookings ('YYYY-MM-DD').
        desk_intensity (float): A multiplier for desk usage visualization.
        room_intensity (float): A multiplier for meeting room usage visualization.
        opacity (float): The desired opacity for the heatmap layer.
        colormap_name (str): The name of the colormap to use.
        show_labels (bool): If True, asset labels will be drawn on the image.
        show_desks (bool): If True, desk usage will be included in the heatmap.
        show_rooms (bool): If True, meeting room usage will be included in the heatmap.

    Returns:
        str: A base64 encoded PNG image string of the final visualization.
    """
    # Gracefully handle cases where data might not be loaded yet.
    if not all([floor_plan_img, df_coords is not None, df_bookings is not None, start_date, end_date]):
        return pil_to_b64(Image.new('RGBA', (800, 600), (240, 240, 240, 255)))

    # --- 1. Data Preparation and Filtering ---
    df_bookings['BookingDate'] = pd.to_datetime(df_bookings['BookingDate']).dt.date
    start_date = datetime.strptime(start_date, '%Y-%m-%d').date()
    end_date = datetime.strptime(end_date, '%Y-%m-%d').date()

    # Filter bookings to the selected date range.
    mask = (df_bookings['BookingDate'] >= start_date) & (df_bookings['BookingDate'] <= end_date)
    filtered_bookings = df_bookings.loc[mask].copy()

    # --- 2. Calculate Usage ---
    # For desks, usage is the number of unique days it was booked.
    desk_usage = \
    filtered_bookings[filtered_bookings['ID'].isin(df_coords[df_coords['Type'] == 'desk']['ID'])].groupby('ID')[
        'BookingDate'].nunique()
    # For rooms, usage is the total minutes it was booked.
    room_usage = \
    filtered_bookings[filtered_bookings['ID'].isin(df_coords[df_coords['Type'] == 'meeting-room']['ID'])].groupby('ID')[
        'MinutesBooked'].sum()

    # Merge the calculated usage back into the coordinates DataFrame.
    df_coords['raw_usage'] = df_coords['ID'].map(desk_usage).fillna(df_coords['ID'].map(room_usage)).fillna(0)

    # --- 3. Heatmap Array Generation ---
    # Create a 2D numpy array with the same dimensions as the floor plan.
    heat_array = np.zeros((floor_plan_img.height, floor_plan_img.width))

    # Normalize desk and room usage independently to balance their visual impact.
    # This prevents a very busy meeting room from overpowering all the desks.
    max_desk_usage = df_coords[df_coords['Type'] == 'desk']['raw_usage'].max()
    max_room_usage = df_coords[df_coords['Type'] == 'meeting-room']['raw_usage'].max()

    # Iterate over each asset and add its "heat" to the heat_array.
    for _, row in df_coords.iterrows():
        intensity = 0.0
        # Calculate normalized intensity for desks (only if desks are enabled).
        if row['Type'] == 'desk' and show_desks and max_desk_usage > 0:
            intensity = (row['raw_usage'] / max_desk_usage) * desk_intensity
        # Calculate normalized intensity for meeting rooms (only if rooms are enabled).
        elif row['Type'] == 'meeting-room' and show_rooms and max_room_usage > 0:
            intensity = (row['raw_usage'] / max_room_usage) * room_intensity

        # Add the calculated intensity to the corresponding area in the heat array.
        if intensity > 0:
            x, y, w, h = int(row['X_Pixels']), int(row['Y_Pixels']), int(row['Width_Pixels']), int(row['Height_Pixels'])
            heat_array[y:y + h, x:x + w] += intensity

    # --- 4. Image Processing ---
    # Apply a Gaussian blur to create the smooth, "blob-like" heatmap effect.
    # Sigma is the standard deviation for the Gaussian kernel, controlling the smoothness.
    blurred_heat = gaussian_filter(heat_array, sigma=40)

    # Normalize the blurred array to a 0-1 range to be mapped to colors.
    normed_heat = (blurred_heat - blurred_heat.min()) / (
                blurred_heat.max() - blurred_heat.min()) if blurred_heat.max() > 0 else blurred_heat

    # Map the normalized heat values to RGB colors using the chosen colormap.
    colormap = get_custom_colormap() if colormap_name == 'custom_gyr' else plt.get_cmap(colormap_name)
    heatmap_image_arr = (colormap(normed_heat)[:, :, :3] * 255).astype(np.uint8)
    heatmap_image = Image.fromarray(heatmap_image_arr)

    # Create a new layer for opacity and apply it to the heatmap image.
    alpha = Image.new('L', floor_plan_img.size, color=int(255 * opacity))
    heatmap_image.putalpha(alpha)

    # Composite the heatmap onto the original floor plan.
    final_image = Image.alpha_composite(floor_plan_img.convert('RGBA'), heatmap_image)

    # --- 5. Draw Labels (Optional) ---
    # If requested, draw the asset IDs on top of the final image.
    if show_labels:
        draw = ImageDraw.Draw(final_image)
        try:
            # Use a common font like Arial if available, otherwise fall back to default.
            font = ImageFont.truetype("arial.ttf", 12)
        except IOError:
            font = ImageFont.load_default()

        # Filter coordinates based on what's being shown
        coords_to_label = df_coords.copy()
        if not show_desks:
            coords_to_label = coords_to_label[coords_to_label['Type'] != 'desk']
        if not show_rooms:
            coords_to_label = coords_to_label[coords_to_label['Type'] != 'meeting-room']

        for _, row in coords_to_label.iterrows():
            x, y, w, h = int(row['X_Pixels']), int(row['Y_Pixels']), int(row['Width_Pixels']), int(row['Height_Pixels'])
            item_id = str(row['ID'])

            # Draw a semi-transparent white box behind the text to ensure it's readable
            # regardless of the background color.
            text_bbox = draw.textbbox((x + 2, y + 2), item_id, font=font)
            text_bg_bbox = (text_bbox[0] - 2, text_bbox[1] - 2, text_bbox[2] + 2, text_bbox[3] + 2)
            draw.rectangle(text_bg_bbox, fill=(255, 255, 255, 180))  # Semi-transparent white
            draw.text((x + 2, y + 2), item_id, fill="black", font=font)

    return pil_to_b64(final_image)


def pil_to_b64(image):
    """Converts a Pillow (PIL) Image object to a base64 encoded string."""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffered.getvalue()).decode()


# --- Load Default Data & Configuration ---
# Define paths for the required data files.
DEFAULT_FLOORPLAN_PATH = "office_floorplan.jpg"
DEFAULT_COORDS_PATH = "coordinate_mapping.csv"
DEFAULT_BOOKINGS_PATH = "bookings_realistic_simple.csv"

# Attempt to load all data upon application startup.
# A global flag is used to determine if the main app or an error screen should be shown.
DATA_LOADED_SUCCESSFULLY = False
FLOOR_PLAN_IMG, DF_COORDS, DF_BOOKINGS, MIN_DATE, MAX_DATE = (None,) * 5

try:
    FLOOR_PLAN_IMG = Image.open(DEFAULT_FLOORPLAN_PATH)
    DF_COORDS = pd.read_csv(DEFAULT_COORDS_PATH)
    DF_BOOKINGS = pd.read_csv(DEFAULT_BOOKINGS_PATH)

    # Calculate and store the overall min and max dates from the booking data.
    min_date_obj = pd.to_datetime(DF_BOOKINGS['BookingDate']).dt.date.min()
    max_date_obj = pd.to_datetime(DF_BOOKINGS['BookingDate']).dt.date.max()
    MIN_DATE = min_date_obj.isoformat()
    MAX_DATE = max_date_obj.isoformat()

    DATA_LOADED_SUCCESSFULLY = True
except (FileNotFoundError, pd.errors.EmptyDataError, KeyError) as e:
    # If any file is missing or corrupt, the app will display an error message.
    print(f"Error loading initial data: {e}")

# --- App Initialization ---
# Initialize the Dash application with a Dash Bootstrap Components theme.
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.LUX])
server = app.server


# --- UI Components & Layout ---

def create_main_layout():
    """
    Constructs the main layout of the dashboard, including all UI components.
    This function is called to build the page that users see.
    """
    # Define the collapsible sidebar (Offcanvas) for visualization controls.
    # This keeps the main view clean and focused on the data.
    controls_sidebar = dbc.Offcanvas(
        dbc.Card([
            dbc.CardBody([
                dbc.Label("Desk Intensity"),
                dcc.Slider(0.1, 5.0, 0.1, value=1.2, id='desk-intensity-slider', marks=None,
                           tooltip={"placement": "bottom", "always_visible": True}),
                dbc.Label("Room Intensity", className="mt-3"),
                dcc.Slider(0.1, 5.0, 0.1, value=0.4, id='room-intensity-slider', marks=None,
                           tooltip={"placement": "bottom", "always_visible": True}),
                dbc.Label("Heatmap Opacity", className="mt-3"),
                dcc.Slider(0.1, 1.0, 0.05, value=0.7, id='opacity-slider', marks=None,
                           tooltip={"placement": "bottom", "always_visible": True}),
                html.Hr(),
                dbc.Label("Color Scheme"),
                dcc.Dropdown(
                    id='colormap-dropdown',
                    options=[
                        {'label': 'Green-Yellow-Red (Recommended)', 'value': 'custom_gyr'},
                        {'label': 'Hot', 'value': 'hot'},
                        {'label': 'Yellow-Orange-Red', 'value': 'YlOrRd'},
                        {'label': 'Jet', 'value': 'jet'},
                    ],
                    value='custom_gyr',
                    clearable=False
                ),
                html.Hr(),
                dbc.Checklist(
                    options=[{"label": "Show Labels on Heatmap", "value": 1}],
                    value=[],
                    id="labels-switch",
                    switch=True,
                ),
            ])
        ]),
        id="offcanvas-controls",
        title="⚙️ Visualization Controls",
        is_open=False,
    )

    # Define the card containing all date-related filters.
    # Placing this above the heatmap makes the workflow more intuitive for users.
    date_controls = dbc.Card([
        dbc.CardBody([
            dbc.Label("Date Range Presets", className="fw-bold"),
            html.Br(),
            dbc.ButtonGroup([
                dbc.Button("Last 7 Days", id="btn-7-days", n_clicks=0),
                dbc.Button("Last 30 Days", id="btn-30-days", n_clicks=0),
                dbc.Button("Last 90 Days", id="btn-90-days", n_clicks=0),
                dbc.Button("All Time", id="btn-all-time", n_clicks=0),
            ], className="w-100"),
            html.Hr(),
            dbc.Label("Custom Date Range", className="fw-bold"),
            dcc.DatePickerRange(
                id='date-picker-range',
                min_date_allowed=MIN_DATE,
                max_date_allowed=MAX_DATE,
                start_date=MIN_DATE,
                end_date=MAX_DATE,
                display_format='MMM D, YYYY',
                className="w-100 mt-2"
            ),
        ])
    ])

    # Define the card containing asset type filters (NEW)
    asset_filter_controls = dbc.Card([
        dbc.CardBody([
            dbc.Label("Show Asset Types", className="fw-bold mb-3"),
            dbc.Row([
                dbc.Col([
                    dbc.Checklist(
                        options=[
                            {"label": "🖥️ Desks", "value": "desks"},
                            {"label": "🏢 Meeting Rooms", "value": "rooms"}
                        ],
                        value=["desks", "rooms"],  # Both selected by default
                        id="asset-type-filter",
                        switch=True,
                    ),
                ], width=12),
            ]),
        ])
    ])

    # Assemble the final page layout using a fluid Bootstrap container.
    return dbc.Container([
        controls_sidebar,
        dbc.Row([
            dbc.Col(html.H1("🏢 Office Utilization Dashboard"), width=10, className="my-4"),
            dbc.Col(dbc.Button("⚙️ Controls", id="btn-open-controls", n_clicks=0, className="my-4"), width=2)
        ]),
        dbc.Row([
            dbc.Col(date_controls, width=8, className="mb-4"),
            dbc.Col(asset_filter_controls, width=4, className="mb-4"),
        ]),
        dbc.Row([
            dbc.Col(
                # The Loading component shows a spinner while the heatmap is being generated.
                dcc.Loading(
                    id="loading-heatmap",
                    type="default",
                    children=html.Img(id='heatmap-image', style={'width': '100%', 'height': 'auto'})
                ),
                width=12
            )
        ]),
        dbc.Row([
            dbc.Col(
                html.Div(
                    html.Img(id='legend-image', src=generate_legend_image('custom_gyr'),
                             style={'width': '100%', 'max-width': '600px', 'height': 'auto', 'display': 'block', 'margin': '0 auto'}),
                    className="text-center mt-3"
                ),
                width=12
            )
        ]),
    ], fluid=True)


def create_error_layout():
    """
    Creates a simple layout to display an error message when data files cannot be found.
    This prevents the app from crashing and provides helpful feedback to the user.
    """
    return dbc.Container([
        dbc.Row(dbc.Col(html.H1("🏢 Office Utilization Dashboard"), width=12, className="text-center my-4")),
        dbc.Row(dbc.Col(dbc.Alert(
            [
                html.H4("Error: Data Files Not Found!", className="alert-heading"),
                html.P(
                    f"Please ensure {DEFAULT_FLOORPLAN_PATH}, {DEFAULT_COORDS_PATH}, and {DEFAULT_BOOKINGS_PATH} are in the same directory."),
            ], color="danger"
        ), width=12, lg={'size': 8, 'offset': 2})),
    ], fluid=True)


# The main layout of the app is determined here. If data loaded successfully,
# the main dashboard is shown. Otherwise, the error layout is displayed.
app.layout = create_main_layout() if DATA_LOADED_SUCCESSFULLY else create_error_layout()


# --- Callbacks ---
# Callbacks are functions that are automatically called by Dash whenever a
# user interacts with a component (e.g., clicks a button, adjusts a slider).

@app.callback(
    Output('heatmap-image', 'src'),
    [
        Input('date-picker-range', 'start_date'),
        Input('date-picker-range', 'end_date'),
        Input('desk-intensity-slider', 'value'),
        Input('room-intensity-slider', 'value'),
        Input('opacity-slider', 'value'),
        Input('colormap-dropdown', 'value'),
        Input('labels-switch', 'value'),
        Input('asset-type-filter', 'value')  # NEW: Asset type filter input
    ],
    # This prevents the callback from running on initial page load if data isn't ready.
    prevent_initial_call=not DATA_LOADED_SUCCESSFULLY
)
def update_heatmap(start_date, end_date, desk_intensity, room_intensity, opacity, colormap, show_labels, asset_types):
    """
    This is the primary callback. It listens for changes to any of the control
    components and regenerates the heatmap image accordingly.
    """
    # The 'show_labels' value is a list; we convert it to a simple boolean.
    show_labels_bool = True if 1 in show_labels else False

    # Convert asset type filter to boolean flags
    show_desks = "desks" in asset_types if asset_types else False
    show_rooms = "rooms" in asset_types if asset_types else False

    # Create safe copies of the dataframes to prevent modifying the original data.
    df_coords_copy = DF_COORDS.copy() if DF_COORDS is not None else None
    df_bookings_copy = DF_BOOKINGS.copy() if DF_BOOKINGS is not None else None

    # Call the main generation function with all the current parameters.
    return generate_heatmap_image(
        FLOOR_PLAN_IMG, df_coords_copy, df_bookings_copy,
        start_date, end_date,
        desk_intensity, room_intensity, opacity, colormap, show_labels_bool,
        show_desks, show_rooms  # NEW: Pass the asset type filters
    )


@app.callback(
    Output('legend-image', 'src'),
    Input('colormap-dropdown', 'value')
)
def update_legend(colormap):
    """Updates the legend image whenever the user selects a new color scheme."""
    return generate_legend_image(colormap)


@app.callback(
    Output("offcanvas-controls", "is_open"),
    Input("btn-open-controls", "n_clicks"),
    [State("offcanvas-controls", "is_open")],
)
def toggle_offcanvas(n1, is_open):
    """Toggles the visibility of the collapsible controls sidebar."""
    if n1:
        return not is_open
    return is_open


@app.callback(
    Output('date-picker-range', 'start_date'),
    Output('date-picker-range', 'end_date'),
    Input('btn-7-days', 'n_clicks'),
    Input('btn-30-days', 'n_clicks'),
    Input('btn-90-days', 'n_clicks'),
    Input('btn-all-time', 'n_clicks'),
    prevent_initial_call=True
)
def update_date_picker_from_presets(btn7, btn30, btn90, btn_all):
    """
    Updates the start and end dates of the DatePickerRange when a preset button is clicked.
    """
    # `dash.callback_context` tells us which users which button triggered the callback.
    ctx = dash.callback_context
    if not ctx.triggered or not DATA_LOADED_SUCCESSFULLY or not MIN_DATE or not MAX_DATE:
        return no_update, no_update

    button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    # Convert the stored min/max date strings back to date objects for calculations.
    end_date_obj = date.fromisoformat(MAX_DATE)
    min_date_obj = date.fromisoformat(MIN_DATE)

    start_date_obj = end_date_obj

    # Calculate the new start date based on which button was pressed.
    if button_id == 'btn-7-days':
        start_date_obj = max(min_date_obj, end_date_obj - timedelta(days=6))
    elif button_id == 'btn-30-days':
        start_date_obj = max(min_date_obj, end_date_obj - timedelta(days=29))
    elif button_id == 'btn-90-days':
        start_date_obj = max(min_date_obj, end_date_obj - timedelta(days=89))
    elif button_id == 'btn-all-time':
        start_date_obj = min_date_obj
    else:
        # If no button was triggered (should not happen), do nothing.
        return no_update, no_update

    # Return the new date range in ISO format, which the DatePickerRange component expects.
    return start_date_obj.isoformat(), end_date_obj.isoformat()


# This is the standard entry point for running a Python script.
# The code inside this block will only run when the script is executed directly.
if __name__ == '__main__':
    # For production, debug=False is crucial as it disables verbose error messages
    # that could be a security risk. A production-grade WSGI server like Gunicorn
    # should be used to run the app instead of Dash's built-in development server.
    app.run(debug=False)