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
    colors = [(0, "#20ff40"), (0.5, "#ffff00"), (1, "#ff2020")]
    return LinearSegmentedColormap.from_list("custom_gyr", colors)


def generate_legend_image(colormap_name):
    fig, ax = plt.subplots(figsize=(4, 1.5))
    colormap = get_custom_colormap() if colormap_name == 'custom_gyr' else plt.get_cmap(colormap_name)
    gradient = np.linspace(0, 1, 256).reshape(1, 256)
    ax.imshow(gradient, aspect='auto', cmap=colormap)
    ax.set_title("Usage Intensity")
    ax.set_xticks([0, 128, 255])
    ax.set_xticklabels(['Low', 'Mid', 'High'])
    ax.set_yticks([])
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.1, transparent=True)
    plt.close(fig)
    return pil_to_b64(Image.open(buf))


def generate_heatmap_image(floor_plan_img, df_coords, df_bookings, start_date, end_date, desk_intensity, room_intensity,
                           opacity, colormap_name, show_labels):
    if not all([floor_plan_img, df_coords is not None, df_bookings is not None, start_date, end_date]):
        return pil_to_b64(Image.new('RGBA', (800, 600), (240, 240, 240, 255)))

    df_bookings['BookingDate'] = pd.to_datetime(df_bookings['BookingDate']).dt.date
    start_date = datetime.strptime(start_date, '%Y-%m-%d').date()
    end_date = datetime.strptime(end_date, '%Y-%m-%d').date()

    mask = (df_bookings['BookingDate'] >= start_date) & (df_bookings['BookingDate'] <= end_date)
    filtered_bookings = df_bookings.loc[mask].copy()

    desk_usage = \
    filtered_bookings[filtered_bookings['ID'].isin(df_coords[df_coords['Type'] == 'desk']['ID'])].groupby('ID')[
        'BookingDate'].nunique()
    room_usage = \
    filtered_bookings[filtered_bookings['ID'].isin(df_coords[df_coords['Type'] == 'meeting-room']['ID'])].groupby('ID')[
        'MinutesBooked'].sum()

    df_coords['raw_usage'] = df_coords['ID'].map(desk_usage).fillna(df_coords['ID'].map(room_usage)).fillna(0)

    heat_array = np.zeros((floor_plan_img.height, floor_plan_img.width))
    max_desk_usage = df_coords[df_coords['Type'] == 'desk']['raw_usage'].max()
    max_room_usage = df_coords[df_coords['Type'] == 'meeting-room']['raw_usage'].max()

    for _, row in df_coords.iterrows():
        intensity = 0.0
        if row['Type'] == 'desk' and max_desk_usage > 0:
            intensity = (row['raw_usage'] / max_desk_usage) * desk_intensity
        elif row['Type'] == 'meeting-room' and max_room_usage > 0:
            intensity = (row['raw_usage'] / max_room_usage) * room_intensity
        if intensity > 0:
            x, y, w, h = int(row['X_Pixels']), int(row['Y_Pixels']), int(row['Width_Pixels']), int(row['Height_Pixels'])
            heat_array[y:y + h, x:x + w] += intensity

    blurred_heat = gaussian_filter(heat_array, sigma=40)
    normed_heat = (blurred_heat - blurred_heat.min()) / (
                blurred_heat.max() - blurred_heat.min()) if blurred_heat.max() > 0 else blurred_heat

    colormap = get_custom_colormap() if colormap_name == 'custom_gyr' else plt.get_cmap(colormap_name)
    heatmap_image_arr = (colormap(normed_heat)[:, :, :3] * 255).astype(np.uint8)
    heatmap_image = Image.fromarray(heatmap_image_arr)

    alpha = Image.new('L', floor_plan_img.size, color=int(255 * opacity))
    heatmap_image.putalpha(alpha)

    final_image = Image.alpha_composite(floor_plan_img.convert('RGBA'), heatmap_image)

    if show_labels:
        draw = ImageDraw.Draw(final_image)
        try:
            font = ImageFont.truetype("arial.ttf", 12)
        except IOError:
            font = ImageFont.load_default()

        for _, row in df_coords.iterrows():
            x, y, w, h = int(row['X_Pixels']), int(row['Y_Pixels']), int(row['Width_Pixels']), int(row['Height_Pixels'])
            item_id = str(row['ID'])

            # Draw text with a semi-transparent background for readability
            text_bbox = draw.textbbox((x + 2, y + 2), item_id, font=font)
            text_bg_bbox = (text_bbox[0] - 2, text_bbox[1] - 2, text_bbox[2] + 2, text_bbox[3] + 2)
            draw.rectangle(text_bg_bbox, fill=(255, 255, 255, 180))  # Semi-transparent white
            draw.text((x + 2, y + 2), item_id, fill="black", font=font)

    return pil_to_b64(final_image)


def pil_to_b64(image):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffered.getvalue()).decode()


# --- Load Default Data & Configuration ---
DEFAULT_FLOORPLAN_PATH = "office_floorplan.jpg"
DEFAULT_COORDS_PATH = "coordinate_mapping.csv"
DEFAULT_BOOKINGS_PATH = "bookings_realistic_simple.csv"

DATA_LOADED_SUCCESSFULLY = False
FLOOR_PLAN_IMG, DF_COORDS, DF_BOOKINGS, MIN_DATE, MAX_DATE = (None,) * 5

try:
    FLOOR_PLAN_IMG = Image.open(DEFAULT_FLOORPLAN_PATH)
    DF_COORDS = pd.read_csv(DEFAULT_COORDS_PATH)
    DF_BOOKINGS = pd.read_csv(DEFAULT_BOOKINGS_PATH)
    min_date_obj = pd.to_datetime(DF_BOOKINGS['BookingDate']).dt.date.min()
    max_date_obj = pd.to_datetime(DF_BOOKINGS['BookingDate']).dt.date.max()
    MIN_DATE = min_date_obj.isoformat()
    MAX_DATE = max_date_obj.isoformat()
    DATA_LOADED_SUCCESSFULLY = True
except (FileNotFoundError, pd.errors.EmptyDataError, KeyError) as e:
    print(f"Error loading initial data: {e}")

# --- App Initialization ---
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.LUX])
server = app.server


# --- UI Components & Layout ---

def create_main_layout():
    # Heatmap/visualization controls are moved to a collapsible sidebar
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
                html.Img(id='legend-image', src=generate_legend_image('custom_gyr'),
                         style={'width': '100%', 'margin-top': '15px'}),
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

    # Date controls are placed in a card above the heatmap
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

    return dbc.Container([
        controls_sidebar,
        dbc.Row([
            dbc.Col(html.H1("🏢 Office Utilization Dashboard"), width=10, className="my-4"),
            dbc.Col(dbc.Button("⚙️ Controls", id="btn-open-controls", n_clicks=0, className="my-4"), width=2)
        ]),
        dbc.Row([
            dbc.Col(date_controls, width=12, className="mb-4")
        ]),
        dbc.Row([
            dbc.Col(
                dcc.Loading(
                    id="loading-heatmap",
                    type="default",
                    children=html.Img(id='heatmap-image', style={'width': '100%', 'height': 'auto'})
                ),
                width=12
            )
        ]),
    ], fluid=True)


def create_error_layout():
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


app.layout = create_main_layout() if DATA_LOADED_SUCCESSFULLY else create_error_layout()


# --- Callbacks ---
@app.callback(
    Output('heatmap-image', 'src'),
    [
        Input('date-picker-range', 'start_date'),
        Input('date-picker-range', 'end_date'),
        Input('desk-intensity-slider', 'value'),
        Input('room-intensity-slider', 'value'),
        Input('opacity-slider', 'value'),
        Input('colormap-dropdown', 'value'),
        Input('labels-switch', 'value')
    ],
    prevent_initial_call=not DATA_LOADED_SUCCESSFULLY
)
def update_heatmap(start_date, end_date, desk_intensity, room_intensity, opacity, colormap, show_labels):
    show_labels_bool = True if 1 in show_labels else False

    df_coords_copy = DF_COORDS.copy() if DF_COORDS is not None else None
    df_bookings_copy = DF_BOOKINGS.copy() if DF_BOOKINGS is not None else None

    return generate_heatmap_image(
        FLOOR_PLAN_IMG, df_coords_copy, df_bookings_copy,
        start_date, end_date,
        desk_intensity, room_intensity, opacity, colormap, show_labels_bool
    )


@app.callback(
    Output('legend-image', 'src'),
    Input('colormap-dropdown', 'value')
)
def update_legend(colormap):
    return generate_legend_image(colormap)


@app.callback(
    Output("offcanvas-controls", "is_open"),
    Input("btn-open-controls", "n_clicks"),
    [State("offcanvas-controls", "is_open")],
)
def toggle_offcanvas(n1, is_open):
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
    ctx = dash.callback_context
    if not ctx.triggered or not DATA_LOADED_SUCCESSFULLY:
        return no_update, no_update

    button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    end_date_obj = date.fromisoformat(MAX_DATE)
    min_date_obj = date.fromisoformat(MIN_DATE)

    start_date_obj = end_date_obj

    if button_id == 'btn-7-days':
        start_date_obj = max(min_date_obj, end_date_obj - timedelta(days=6))
    elif button_id == 'btn-30-days':
        start_date_obj = max(min_date_obj, end_date_obj - timedelta(days=29))
    elif button_id == 'btn-90-days':
        start_date_obj = max(min_date_obj, end_date_obj - timedelta(days=89))
    elif button_id == 'btn-all-time':
        start_date_obj = min_date_obj
    else:
        return no_update, no_update

    return start_date_obj.isoformat(), end_date_obj.isoformat()


if __name__ == '__main__':
    app.run(debug=False) 