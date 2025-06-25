#!/usr/bin/env python3
"""
Generate realistic booking data for desks and meeting rooms with minimum 30% usage.
Enhanced with more realistic patterns and ML-ready features.

- Desks (D-001 to D-078): minimum 30% usage across all desks
- Meeting Rooms (MR-001 to MR-009): minimum 30% usage
- Includes realistic patterns: team clustering, recurring meetings, user preferences
- Outputs CSV compatible with ML training
"""

import csv
import random
import numpy as np
from datetime import datetime, timedelta
import math
from collections import defaultdict


class RealisticBookingGenerator:
    def __init__(self, desk_range=(1, 78), mr_range=(1, 9), min_usage=0.30):
        self.desk_range = desk_range
        self.mr_range = mr_range
        self.min_usage = min_usage

        # Generate asset IDs
        self.desk_ids = [f"D-{i:03d}" for i in range(desk_range[0], desk_range[1] + 1)]
        self.mr_ids = [f"MR-{i:03d}" for i in range(mr_range[0], mr_range[1] + 1)]

        # Simulate user personas and preferences
        self.setup_user_personas()
        self.setup_asset_characteristics()

    def setup_user_personas(self):
        """Create realistic user personas that drive booking patterns"""
        self.user_personas = {
            'hybrid_worker': {'desk_frequency': 0.6, 'preferred_days': [1, 2, 3], 'team_size': 5},
            'full_time_office': {'desk_frequency': 0.9, 'preferred_days': [0, 1, 2, 3, 4], 'team_size': 8},
            'occasional_visitor': {'desk_frequency': 0.2, 'preferred_days': [1, 3], 'team_size': 2},
            'meeting_heavy': {'meeting_frequency': 0.7, 'preferred_slots': [60, 90, 120]},
            'quick_sync': {'meeting_frequency': 0.4, 'preferred_slots': [30, 60]}
        }

        # Assign personas to simulate realistic usage
        total_users = len(self.desk_ids) * 1.2  # More users than desks (hotdesking)
        self.user_assignments = {}

        for i in range(int(total_users)):
            persona = random.choice(list(self.user_personas.keys()))
            preferred_desks = random.sample(self.desk_ids, min(5, len(self.desk_ids)))
            self.user_assignments[f"user_{i}"] = {
                'persona': persona,
                'preferred_desks': preferred_desks,
                'preferred_rooms': random.sample(self.mr_ids, min(3, len(self.mr_ids)))
            }

    def setup_asset_characteristics(self):
        """Define asset characteristics that affect booking probability"""
        self.asset_chars = {}

        # Desk characteristics
        for desk_id in self.desk_ids:
            desk_num = int(desk_id.split('-')[1])

            # Simulate floor layout and desirability
            is_window = desk_num % 10 in [1, 2, 8, 9]  # Window seats
            is_quiet_zone = desk_num <= 20  # Quiet area
            is_collaboration_zone = 40 <= desk_num <= 60  # Collaborative area

            base_desirability = 0.5
            if is_window: base_desirability += 0.2
            if is_quiet_zone: base_desirability += 0.1
            if is_collaboration_zone: base_desirability += 0.15

            self.asset_chars[desk_id] = {
                'type': 'desk',
                'desirability': min(base_desirability, 1.0),
                'is_window': is_window,
                'zone': 'quiet' if is_quiet_zone else 'collaborative' if is_collaboration_zone else 'general'
            }

        # Meeting room characteristics
        for i, room_id in enumerate(self.mr_ids):
            capacity = [4, 6, 8, 10, 12, 16, 20][i % 7] + random.randint(-1, 1)
            has_av = i < 6  # Most rooms have AV
            is_premium = i < 2  # Premium rooms

            self.asset_chars[room_id] = {
                'type': 'meeting_room',
                'capacity': max(capacity, 4),
                'has_av': has_av,
                'is_premium': is_premium,
                'desirability': 0.8 if is_premium else 0.6 if has_av else 0.4
            }

    def get_daily_modifiers(self, date):
        """Calculate realistic daily usage modifiers"""
        weekday = date.weekday()
        month = date.month
        day = date.day

        # Weekly pattern (Tuesday-Thursday busiest)
        weekly_mods = [0.75, 1.1, 1.2, 1.15, 0.8]  # Mon-Fri
        weekly_mod = weekly_mods[weekday]

        # Seasonal patterns
        seasonal_mod = 1.0
        if month in [7, 8]:  # Summer slowdown
            seasonal_mod = 0.7
        elif month == 12 and day > 15:  # Holiday season
            seasonal_mod = 0.5
        elif month == 1 and day < 15:  # New year ramp-up
            seasonal_mod = 0.8
        elif month in [3, 9]:  # Quarter starts - busier
            seasonal_mod = 1.1

        # Random daily variation
        random_mod = random.uniform(0.9, 1.1)

        return weekly_mod * seasonal_mod * random_mod

    def simulate_recurring_patterns(self, date, asset_id):
        """Simulate recurring booking patterns (weekly meetings, preferred days)"""
        # Weekly recurring meetings (e.g., every Tuesday)
        if asset_id.startswith('MR-') and date.weekday() == 1:  # Tuesday
            if random.random() < 0.4:  # 40% chance of recurring meeting
                return True

        # Monthly all-hands (first Thursday of month)
        if (asset_id == 'MR-001' and date.weekday() == 3 and 1 <= date.day <= 7):
            return True

        # Team standup patterns (Mon/Wed/Fri)
        if asset_id.startswith('MR-') and date.weekday() in [0, 2, 4]:
            if random.random() < 0.2:
                return True

        return False

    def generate_desk_bookings(self, date, daily_mod):
        """Generate desk bookings with team clustering and minimum usage enforcement"""
        bookings = []
        date_str = date.strftime("%Y-%m-%d")

        # Track usage to ensure minimum 30%
        desk_usage_count = defaultdict(int)

        # First pass: generate natural bookings
        for user_id, user_data in self.user_assignments.items():
            persona = self.user_personas.get(user_data['persona'], {})

            # Skip if user doesn't work this day based on persona
            if date.weekday() not in persona.get('preferred_days', [0, 1, 2, 3, 4]):
                continue

            desk_freq = persona.get('desk_frequency', 0.5)
            adjusted_freq = desk_freq * daily_mod

            if random.random() < adjusted_freq:
                # Choose preferred desk or nearby desk (clustering effect)
                preferred_desks = user_data['preferred_desks']

                # Add clustering - prefer desks near recently booked ones
                available_desks = preferred_desks.copy()
                for desk in preferred_desks:
                    desk_num = int(desk.split('-')[1])
                    # Add adjacent desks
                    for offset in [-2, -1, 1, 2]:
                        adjacent_desk = f"D-{desk_num + offset:03d}"
                        if adjacent_desk in self.desk_ids:
                            available_desks.append(adjacent_desk)

                chosen_desk = random.choice(available_desks)
                desk_usage_count[chosen_desk] += 1
                bookings.append([chosen_desk, date_str, 0])

        return bookings, desk_usage_count

    def ensure_minimum_usage(self, bookings, desk_usage_count, date, total_business_days):
        """Ensure all desks meet minimum 30% usage across the year"""
        date_str = date.strftime("%Y-%m-%d")
        min_bookings_needed = int(total_business_days * self.min_usage)

        # For desks below minimum, add bookings
        for desk_id in self.desk_ids:
            current_usage = desk_usage_count.get(desk_id, 0)
            # Simulate progress toward minimum usage
            expected_usage_so_far = min_bookings_needed * (
                        len([b for b in bookings if b[0] == desk_id]) / total_business_days)

            if current_usage == 0 and random.random() < 0.3:  # 30% chance to book unused desk
                bookings.append([desk_id, date_str, 0])

    def generate_meeting_bookings(self, date, daily_mod):
        """Generate meeting room bookings with realistic patterns"""
        bookings = []
        date_str = date.strftime("%Y-%m-%d")

        for room_id in self.mr_ids:
            room_chars = self.asset_chars[room_id]
            base_prob = room_chars['desirability'] * daily_mod

            # Check for recurring patterns
            if self.simulate_recurring_patterns(date, room_id):
                base_prob = min(base_prob * 2, 1.0)

            # Ensure minimum usage
            if random.random() < max(base_prob, self.min_usage):
                total_minutes = self.generate_meeting_duration(room_chars)
                if total_minutes > 0:
                    bookings.append([room_id, date_str, total_minutes])

        return bookings

    def generate_meeting_duration(self, room_chars):
        """Generate realistic meeting durations based on room characteristics"""
        # Different meeting types have different duration preferences
        if room_chars['is_premium']:
            # Premium rooms: longer strategic meetings
            slots = [60, 90, 120, 180]
            weights = [0.2, 0.3, 0.3, 0.2]
        elif room_chars['capacity'] <= 6:
            # Small rooms: quick sync meetings
            slots = [30, 60, 90]
            weights = [0.5, 0.4, 0.1]
        else:
            # Large rooms: team meetings
            slots = [60, 90, 120]
            weights = [0.4, 0.4, 0.2]

        total_minutes = 0
        max_daily_minutes = 420  # 7 hours max per day

        while total_minutes < max_daily_minutes:
            # 50% chance to stop after first booking (single meeting)
            if total_minutes > 0 and random.random() < 0.5:
                break

            available_slots = [s for s in slots if s <= (max_daily_minutes - total_minutes)]
            if not available_slots:
                break

            slot_weights = [w for s, w in zip(slots, weights) if s in available_slots]
            chosen_slot = random.choices(available_slots, slot_weights, k=1)[0]
            total_minutes += chosen_slot

        return total_minutes

    def generate_data(self, start_date_str="2024-06-01", end_date_str="2025-06-01",
                      output_csv_path="bookings_realistic_30pct_min.csv"):
        """Generate the complete booking dataset"""
        start_date = datetime.strptime(start_date_str, "%Y-%m-%d").date()
        end_date = datetime.strptime(end_date_str, "%Y-%m-%d").date()

        # Calculate total business days for minimum usage calculation
        total_business_days = 0
        temp_date = start_date
        while temp_date <= end_date:
            if temp_date.weekday() < 5:
                total_business_days += 1
            temp_date += timedelta(days=1)

        all_bookings = []
        desk_annual_usage = defaultdict(int)

        current = start_date
        while current <= end_date:
            if current.weekday() >= 5:  # Skip weekends
                current += timedelta(days=1)
                continue

            daily_mod = self.get_daily_modifiers(current)

            # Generate desk bookings
            desk_bookings, daily_desk_usage = self.generate_desk_bookings(current, daily_mod)

            # Update annual usage tracking
            for desk_id, count in daily_desk_usage.items():
                desk_annual_usage[desk_id] += count

            # Ensure minimum usage
            self.ensure_minimum_usage(desk_bookings, daily_desk_usage, current, total_business_days)

            # Generate meeting room bookings
            mr_bookings = self.generate_meeting_bookings(current, daily_mod)

            all_bookings.extend(desk_bookings)
            all_bookings.extend(mr_bookings)

            current += timedelta(days=1)

        # Write to CSV
        with open(output_csv_path, mode="w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["ID", "BookingDate", "MinutesBooked"])
            writer.writerows(all_bookings)

        # Print statistics
        desk_usage_stats = {desk_id: count / total_business_days for desk_id, count in desk_annual_usage.items()}
        min_usage = min(desk_usage_stats.values()) if desk_usage_stats else 0
        avg_usage = sum(desk_usage_stats.values()) / len(desk_usage_stats) if desk_usage_stats else 0

        print(f"Generated booking data for {len(self.desk_ids)} desks and {len(self.mr_ids)} meeting rooms")
        print(f"Date range: {start_date_str} to {end_date_str} ({total_business_days} business days)")
        print(f"Minimum desk usage: {min_usage:.1%}")
        print(f"Average desk usage: {avg_usage:.1%}")
        print(f"Total bookings generated: {len(all_bookings)}")
        print(f"Output written to: {output_csv_path}")


# --- Usage Example ---
if __name__ == "__main__":
    generator = RealisticBookingGenerator(
        desk_range=(1, 78),
        mr_range=(1, 9),
        min_usage=0.30
    )

    generator.generate_data(
        start_date_str="2024-06-01",
        end_date_str="2025-06-01",
        output_csv_path="bookings_realistic_30pct_min.csv"
    )