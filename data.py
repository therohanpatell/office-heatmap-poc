#!/usr/bin/env python3
"""
Generate realistic booking data for desks and meeting rooms over a date range,
excluding weekends. Outputs a CSV with columns: ID, BookingDate, MinutesBooked.

- Desks (D-001 to D-111): each business day, each desk has a probability of being "booked"
  (representing occupancy) with MinutesBooked = 0; unused desks are omitted.
- Meeting Rooms (MR-001 to MR-100): each business day, each room has a probability of being used.
  If used, compute a total daily MinutesBooked by picking one or more slot durations (30, 60, 90, 120),
  favoring shorter slots, stopping early with high probability, capped at 420 minutes.
- Date range: 2024-06-01 to 2025-06-01 inclusive, weekdays only.
"""

import csv
import random
from datetime import datetime, timedelta
import math


def generate_pattern_based_booking_data(
        desk_range=(1, 78),
        mr_range=(1, 9),
        start_date_str="2024-06-01",
        end_date_str="2025-06-01",
        desk_prefix="D-",
        mr_prefix="MR-",
        output_csv_path="bookings_realistic_simple.csv"
):
    """
    Generates highly realistic, pattern-based booking data for an office over a one-year period.
    - Simulates weekly rhythms (busier mid-week) and seasonal lulls (summer/holidays).
    - Designates certain assets as "popular" with higher-than-normal usage.
    - Simulates desk clustering to represent teams sitting together.
    """
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d").date()
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d").date()

    desk_ids = [f"{desk_prefix}{i:03d}" for i in range(desk_range[0], desk_range[1] + 1)]
    mr_ids = [f"{mr_prefix}{i:03d}" for i in range(mr_range[0], mr_range[1] + 1)]

    # --- Tier and Asset Personality Setup ---
    usage_tiers = {"high": 0.85, "medium": 0.50, "low": 0.15, "unused": 0.0}
    tier_distribution = {"high": 0.2, "medium": 0.4, "low": 0.3, "unused": 0.1}

    popular_desks = {f"{desk_prefix}{i:03d}" for i in [10, 25, 26, 55]}  # e.g., window seats
    popular_mr = f"{mr_prefix}001"  # e.g., best conference room

    asset_tiers = {}
    for asset_id in desk_ids + mr_ids:
        if asset_id in popular_desks or asset_id == popular_mr:
            asset_tiers[asset_id] = "high"
        else:
            tier_choice = random.choices(list(tier_distribution.keys()), list(tier_distribution.values()), k=1)[0]
            asset_tiers[asset_id] = tier_choice

    mr_slots = (30, 60, 90)
    mr_daily_cap = 420

    with open(output_csv_path, mode="w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["ID", "BookingDate", "MinutesBooked"])

        current = start_date
        while current <= end_date:
            if current.weekday() >= 5:  # Skip weekends
                current += timedelta(days=1)
                continue

            # --- Calculate Daily Modifiers ---
            # 1. Weekly Rhythm Modifier (busier mid-week)
            weekday_mod = 1.0
            if current.weekday() in [0, 4]:  # Mon/Fri
                weekday_mod = 0.8
            elif current.weekday() in [1, 2, 3]:  # Tue/Wed/Thu
                weekday_mod = 1.2

            # 2. Seasonal Modifier (quieter summer/holidays)
            seasonal_mod = 1.0
            if current.month in [7, 8] or (current.month == 12 and current.day > 15):
                seasonal_mod = 0.6

            booking_date_str = current.strftime("%Y-%m-%d")

            # --- Desk Bookings with Clustering ---
            last_desk_booked = False
            for desk_id in desk_ids:
                tier = asset_tiers[desk_id]
                base_prob = usage_tiers[tier]

                # Apply modifiers
                final_prob = base_prob * weekday_mod * seasonal_mod

                # Clustering bonus: if the previous desk was booked, this one is more likely to be.
                if last_desk_booked:
                    final_prob *= 1.5  # 50% higher chance

                final_prob = min(final_prob, 1.0)  # Cap probability at 100%

                if random.random() < final_prob:
                    writer.writerow([desk_id, booking_date_str, 0])
                    last_desk_booked = True
                else:
                    last_desk_booked = False

            # --- Meeting Room Bookings ---
            for room_id in mr_ids:
                tier = asset_tiers[room_id]
                base_prob = usage_tiers[tier]

                # Special boost for the popular meeting room
                if room_id == popular_mr:
                    base_prob *= 1.2

                final_prob = min(base_prob * weekday_mod * seasonal_mod, 1.0)

                if random.random() < final_prob:
                    total_minutes = 0
                    while total_minutes < mr_daily_cap:
                        if random.random() < 0.5 and total_minutes > 0: break

                        valid_slots = [s for s in mr_slots if s <= (mr_daily_cap - total_minutes)]
                        if not valid_slots: break

                        weights = [0.6, 0.3, 0.1]
                        chosen_slot = \
                        random.choices(valid_slots, [w for s, w in zip(mr_slots, weights) if s in valid_slots], k=1)[0]
                        total_minutes += chosen_slot

                    if total_minutes > 0:
                        writer.writerow([room_id, booking_date_str, total_minutes])

            current += timedelta(days=1)

    print(f"Generated pattern-based bookings for {len(desk_ids)} desks and {len(mr_ids)} rooms.")
    print("Simulation included weekly/seasonal rhythms, popular assets, and desk clustering.")
    print(f"Output written to: {output_csv_path}")


# --- Example Usage ---
generate_pattern_based_booking_data(
    desk_range=(1, 78),
    mr_range=(1, 9),
    start_date_str="2024-06-01",
    end_date_str="2025-06-01"
)
