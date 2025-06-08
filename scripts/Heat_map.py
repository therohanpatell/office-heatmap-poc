import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.ndimage import gaussian_filter
import matplotlib.image as mpimg
from PIL import Image

# ── 1) FILEPATHS ─────────────────────────────────────────────────────────────
BOOKING_CSV = 'booking_data.csv'  # your booking data
COORD_CSV = 'coordinate_mapping.csv'  # your desk/room coords (pixel-based)
FLOORPLAN_IMG = 'office_floorplan.jpg'  # your floor plan image

# ──────────────────────────────────────────────────────────────────────────────
# 2) LOAD & AGGREGATE BOOKING DATA
print("Loading booking data...")
bookings = pd.read_csv(BOOKING_CSV, parse_dates=['Booking_Timestamp'])
coords = pd.read_csv(COORD_CSV)

print(f"Bookings shape: {bookings.shape}")
print(f"Coordinates shape: {coords.shape}")

# Unify ID column from bookings (combine Desk_ID and Meeting_Room_ID)
bookings['ID'] = bookings['Desk_ID'].fillna(bookings['Meeting_Room_ID'])

# Sum durations by ID
agg = (bookings
       .groupby('ID')['Duration']
       .sum()
       .reset_index(name='Total_Duration'))

# Normalize by an 8-hour day (480 min) to get utilization score
agg['Utilization_Score'] = agg['Total_Duration'] / 480.0

print(f"Aggregated bookings: {len(agg)} unique IDs")

# Merge coordinate data with booking data, fill unmapped IDs with zero utilization
df = coords.merge(agg, on='ID', how='left').fillna(0.0)

print(f"Final dataset: {len(df)} items with coordinates")
print("\nSample coordinate data:")
print(df.head())

# ──────────────────────────────────────────────────────────────────────────────
# 3) GET IMAGE DIMENSIONS AND COMPUTE PIXEL EXTENTS
try:
    # Load image to get actual dimensions
    img = Image.open(FLOORPLAN_IMG)
    img_width, img_height = img.size
    print(f"Image dimensions: {img_width} x {img_height} pixels")

    # Use full image dimensions for extents
    x_min, x_max = 0, img_width
    y_min, y_max = 0, img_height

except FileNotFoundError:
    print(f"Warning: {FLOORPLAN_IMG} not found. Using coordinate bounds.")
    # Fallback to coordinate bounds if image not available
    x_min = min(df['X_Pixels'].min(), (df['X_Pixels'] - df['Width_Pixels']).min())
    x_max = max(df['X_Pixels'].max(), (df['X_Pixels'] + df['Width_Pixels']).max())
    y_min = min(df['Y_Pixels'].min(), (df['Y_Pixels'] - df['Height_Pixels']).min())
    y_max = max(df['Y_Pixels'].max(), (df['Y_Pixels'] + df['Height_Pixels']).max())
    img_width, img_height = int(x_max - x_min), int(y_max - y_min)

# ──────────────────────────────────────────────────────────────────────────────
# 4) BUILD HEAT-GRID IN PIXEL COORDINATES
print("Building heatmap grid...")

# Use image dimensions for heat grid resolution (or scale down for performance)
HEAT_RES_X = min(img_width, 400)  # Reduced for better performance
HEAT_RES_Y = min(img_height, 300)  # Reduced for better performance

# Create coordinate arrays
xi = np.linspace(x_min, x_max, HEAT_RES_X)
yi = np.linspace(y_min, y_max, HEAT_RES_Y)

# Initialize intensity grid
intensity = np.zeros((HEAT_RES_Y, HEAT_RES_X))

# Add intensity at each desk/room location
for _, row in df.iterrows():
    cx, cy = row['X_Pixels'], row['Y_Pixels']
    w, h = row['Width_Pixels'], row['Height_Pixels']
    score = row['Utilization_Score']

    if score <= 0:  # Skip items with no utilization
        continue

    print(f"Adding heat for {row['ID']}: pos=({cx}, {cy}), size=({w}, {h}), score={score:.3f}")

    if row['Type'] == 'desk':
        # For desks (point locations), add heat at center point
        # Find grid indices for center point
        ix = int(np.clip((cx - x_min) / (x_max - x_min) * (HEAT_RES_X - 1), 0, HEAT_RES_X - 1))
        iy = int(np.clip((cy - y_min) / (y_max - y_min) * (HEAT_RES_Y - 1), 0, HEAT_RES_Y - 1))

        # INCREASED: Boost desk heat intensity to compete with meeting rooms
        intensity[iy, ix] += score * 3.0  # Increased from 0.7 to 3.0

        # Add some spread to neighboring pixels
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                ny, nx = iy + dy, ix + dx
                if 0 <= ny < HEAT_RES_Y and 0 <= nx < HEAT_RES_X:
                    intensity[ny, nx] += score * 0.5  # Increased from 0.1 to 0.5

    else:  # For meeting rooms, distribute heat across the area
        # Calculate area boundaries (cx, cy is top-left corner from your mapping tool)
        x1, x2 = cx, cx + w
        y1, y2 = cy, cy + h

        # Convert to grid indices
        ix1 = int(np.clip((x1 - x_min) / (x_max - x_min) * (HEAT_RES_X - 1), 0, HEAT_RES_X - 1))
        ix2 = int(np.clip((x2 - x_min) / (x_max - x_min) * (HEAT_RES_X - 1), 0, HEAT_RES_X - 1))
        iy1 = int(np.clip((y1 - y_min) / (y_max - y_min) * (HEAT_RES_Y - 1), 0, HEAT_RES_Y - 1))
        iy2 = int(np.clip((y2 - y_min) / (y_max - y_min) * (HEAT_RES_Y - 1), 0, HEAT_RES_Y - 1))

        # Ensure we have at least 1 pixel and proper bounds
        ix2 = max(ix1 + 1, min(ix2, HEAT_RES_X))
        iy2 = max(iy1 + 1, min(iy2, HEAT_RES_Y))

        print(f"  Meeting room grid bounds: ix=[{ix1}:{ix2}], iy=[{iy1}:{iy2}]")

        # BALANCED: Apply moderate heat density for meeting rooms
        area_pixels = (ix2 - ix1) * (iy2 - iy1)
        if area_pixels > 0:
            # Use a balanced approach: don't divide by full area, but don't use full score either
            room_heat_density = score * 0.3  # Reduced intensity to balance with boosted desks
            intensity[iy1:iy2, ix1:ix2] += room_heat_density

# Apply Gaussian blur for smooth heatmap blobs
# Adjust sigma for spread (higher = more blur/spread)
sigma_x = max(3, HEAT_RES_X / 40)  # Slightly increased for better blending
sigma_y = max(3, HEAT_RES_Y / 40)
blurred = gaussian_filter(intensity, sigma=[sigma_y, sigma_x])

print(f"Heatmap intensity range: {blurred.min():.3f} to {blurred.max():.3f}")

# ──────────────────────────────────────────────────────────────────────────────
# 5) PLOT EVERYTHING
print("Creating visualization...")

fig, ax = plt.subplots(figsize=(16, 12))

# a) Background floorplan image
try:
    bg = mpimg.imread(FLOORPLAN_IMG)
    # FIXED: Use 'lower' origin to match coordinate system
    ax.imshow(bg,
              extent=[x_min, x_max, y_max, y_min],  # FIXED: Flipped y extent
              origin='lower',  # FIXED: Changed to 'lower'
              aspect='equal')
    print("Floor plan image loaded as background")
except FileNotFoundError:
    ax.set_facecolor('#f8f9fa')
    print("Using plain background (image not found)")

# b) Heatmap overlay
if blurred.max() > 0:  # Only show heatmap if there's data
    # FIXED: Match the background image coordinate system
    cax = ax.imshow(
        blurred,
        extent=[x_min, x_max, y_max, y_min],  # FIXED: Flipped y extent to match background
        origin='lower',  # FIXED: Changed to 'lower'
        cmap='hot',
        alpha=0.6,
        interpolation='bilinear',
        vmin=0,
        vmax=blurred.max()
    )

    # Add colorbar
    cb = fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label('Utilization Score (fraction of 8-hour day)', fontsize=12)
else:
    print("Warning: No heat data to display!")

# c) Draw rectangles/circles for each desk/room with their actual pixel coordinates
for _, row in df.iterrows():
    cx, cy = row['X_Pixels'], row['Y_Pixels']
    w, h = row['Width_Pixels'], row['Height_Pixels']
    score = row['Utilization_Score']
    item_type = row['Type']

    if item_type == 'desk':
        # Draw desks as circles (since they're point locations)
        # FIXED: For desks in top-left coordinate system, convert Y coordinate
        desk_y = img_height - cy  # Convert from top-left to bottom-left origin

        circle = patches.Circle(
            (cx, desk_y),
            radius=max(8, w / 4),  # Smaller radius for better visibility
            linewidth=2,
            edgecolor='blue',
            facecolor='lightblue' if score > 0 else 'lightgray',
            alpha=0.5
        )
        ax.add_patch(circle)

        # Add ID label
        ax.text(cx, desk_y, row['ID'],
                ha='center', va='center',
                fontsize=7,
                fontweight='bold',
                color='black')

    else:
        # Draw meeting rooms as rectangles
        # FIXED: For meeting rooms, convert coordinates from top-left to bottom-left origin
        room_y = img_height - cy - h  # Convert top-left corner to bottom-left origin

        rect = patches.Rectangle(
            (cx, room_y),  # Bottom-left corner in matplotlib coordinates
            w, h,  # Width and height
            linewidth=2,
            edgecolor='red',
            facecolor='lightcoral' if score > 0 else 'lightgray',
            alpha=0.6
        )
        ax.add_patch(rect)

        # Add ID label at center of rectangle
        center_x = cx + w / 2
        center_y = room_y + h / 2  # Center in matplotlib coordinates
        ax.text(center_x, center_y, row['ID'],
                ha='center', va='center',
                fontsize=8,
                fontweight='bold',
                color='white',
                bbox=dict(boxstyle='round,pad=0.2',
                          facecolor='black',
                          alpha=0.7))

    # Add utilization score if > 0
    if score > 0:
        if item_type == 'desk':
            label_x = cx
            label_y = img_height - cy - 15  # Convert and offset for label
        else:
            label_x = cx + w / 2
            label_y = room_y + h / 2 + 15  # Center + offset for label

        ax.text(label_x, label_y, f'{score:.2f}',
                ha='center', va='center',
                fontsize=8,
                color='yellow',
                fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.1',
                          facecolor='black',
                          alpha=0.8))

# d) Set plot properties
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)  # FIXED: Normal y-axis limits (bottom to top)
ax.set_aspect('equal')
ax.set_xlabel('X Pixels', fontsize=12)
ax.set_ylabel('Y Pixels', fontsize=12)
ax.set_title('Office Utilization Heatmap (Fixed Coordinate System)', fontsize=16, fontweight='bold')

# Add grid for reference
ax.grid(True, alpha=0.3, linestyle='--')

# Add legend
legend_elements = [
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightblue',
               markersize=10, label='Desk (Active)', markeredgecolor='blue'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgray',
               markersize=10, label='Desk (Unused)', markeredgecolor='blue'),
    plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='lightcoral',
               markersize=10, label='Meeting Room (Active)', markeredgecolor='red'),
    plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='lightgray',
               markersize=10, label='Meeting Room (Unused)', markeredgecolor='red')
]
ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98))

plt.tight_layout()

# Save the plot
plt.savefig('office_utilization_heatmap_corrected.png', dpi=300, bbox_inches='tight')
print("Heatmap saved as 'office_utilization_heatmap_corrected.png'")

plt.show()

# ──────────────────────────────────────────────────────────────────────────────
# 6) PRINT SUMMARY STATISTICS
print("\n" + "=" * 60)
print("UTILIZATION SUMMARY")
print("=" * 60)

summary_stats = df.groupby('Type').agg({
    'Utilization_Score': ['count', 'mean', 'max', 'sum']
}).round(3)

print(summary_stats)

print(f"\nTop 5 Most Utilized Items:")
top_utilized = df.nlargest(5, 'Utilization_Score')[['ID', 'Type', 'Utilization_Score', 'Total_Duration']]
for _, row in top_utilized.iterrows():
    print(f"  {row['ID']} ({row['Type']}): {row['Utilization_Score']:.2f} score, {row['Total_Duration']:.0f} minutes")

print(f"\nUnused Items ({len(df[df['Utilization_Score'] == 0])} total):")
unused = df[df['Utilization_Score'] == 0]['ID'].tolist()
if unused:
    print(f"  {', '.join(unused[:10])}" + ("..." if len(unused) > 10 else ""))
else:
    print("  None - all items have some utilization!")

print("=" * 60)

# Debug information
print("\n" + "=" * 60)
print("DEBUG INFORMATION")
print("=" * 60)
print(f"Image bounds: X=[{x_min}, {x_max}], Y=[{y_min}, {y_max}]")
print(f"Heat grid resolution: {HEAT_RES_X} x {HEAT_RES_Y}")
print(f"Items with utilization > 0: {len(df[df['Utilization_Score'] > 0])}")
print("\nCoordinate samples (after conversion):")
for _, row in df.head(3).iterrows():
    if row['Type'] == 'desk':
        converted_y = img_height - row['Y_Pixels']
        print(
            f"  {row['ID']} (DESK): Original=({row['X_Pixels']}, {row['Y_Pixels']}) -> Matplotlib=({row['X_Pixels']}, {converted_y})")
    else:
        converted_y = img_height - row['Y_Pixels'] - row['Height_Pixels']
        print(
            f"  {row['ID']} (ROOM): Original=({row['X_Pixels']}, {row['Y_Pixels']}) -> Matplotlib=({row['X_Pixels']}, {converted_y})")