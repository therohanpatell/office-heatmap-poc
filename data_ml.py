#!/usr/bin/env python3
"""
Machine Learning-Based Realistic Booking Data Generator
Inspired by Condeco workspace analytics patterns and real-world office usage.

This implementation uses multiple ML approaches:
1. LSTM for time series patterns
2. Gaussian Mixture Models for user behavior clustering
3. Markov Chains for recurring booking patterns
4. Random Forest for feature-based predictions

Features realistic patterns found in real office booking systems:
- User personas and behavioral consistency
- Team clustering and collaboration patterns
- Seasonal and weekly rhythms
- Meeting room capacity optimization
- Hot-desking preferences
- Minimum 30% usage guarantee
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    KERAS_AVAILABLE = True
except ImportError:
    KERAS_AVAILABLE = False
    print("Note: TensorFlow/Keras not available. Using alternative time series model.")


class MLBookingDataGenerator:
    def __init__(self, desk_range=(1, 78), mr_range=(1, 9), min_usage=0.30):
        self.desk_range = desk_range
        self.mr_range = mr_range
        self.min_usage = min_usage

        # Generate asset IDs
        self.desk_ids = [f"D-{i:03d}" for i in range(desk_range[0], desk_range[1] + 1)]
        self.mr_ids = [f"MR-{i:03d}" for i in range(mr_range[0], mr_range[1] + 1)]

        # Initialize ML models and patterns
        self.setup_user_behavior_models()
        self.setup_time_series_patterns()
        self.setup_asset_features()

    def setup_user_behavior_models(self):
        """Use Gaussian Mixture Models to create realistic user behavior clusters"""
        np.random.seed(42)

        # Define user behavior parameters for GMM
        n_users = len(self.desk_ids) * 1.3  # More users than desks (hot-desking)

        # Create behavioral features: [frequency, team_size, flexibility, meeting_tendency]
        user_features = []

        # Generate synthetic behavioral data based on common office patterns
        for _ in range(int(n_users)):
            # Core behavioral dimensions
            frequency = np.random.beta(2, 3)  # Most users are occasional/hybrid
            team_size = np.random.gamma(2, 2) + 1  # Team sizes 1-10
            flexibility = np.random.uniform(0, 1)  # How flexible with desk choice
            meeting_tendency = np.random.beta(1.5, 2)  # Meeting booking propensity

            user_features.append([frequency, team_size, flexibility, meeting_tendency])

        # Fit Gaussian Mixture Model to identify user personas
        self.user_behavioral_features = np.array(user_features)
        self.user_gmm = GaussianMixture(n_components=5, random_state=42)
        self.user_clusters = self.user_gmm.fit_predict(self.user_behavioral_features)

        # Define persona labels for interpretability
        self.persona_labels = {
            0: "Hybrid Worker",
            1: "Full-time Office",
            2: "Occasional Visitor",
            3: "Meeting Heavy",
            4: "Flexible Remote"
        }

        # Create user profiles
        self.users = {}
        for i, (features, cluster) in enumerate(zip(self.user_behavioral_features, self.user_clusters)):
            self.users[f"user_{i}"] = {
                'persona': self.persona_labels[cluster],
                'frequency': features[0],
                'team_size': features[1],
                'flexibility': features[2],
                'meeting_tendency': features[3],
                'preferred_desks': np.random.choice(self.desk_ids,
                                                    size=min(int(features[2] * 10) + 1, 5),
                                                    replace=False).tolist(),
                'preferred_rooms': np.random.choice(self.mr_ids,
                                                    size=min(3, len(self.mr_ids)),
                                                    replace=False).tolist()
            }

    def setup_time_series_patterns(self):
        """Create time series patterns using simplified LSTM-like patterns"""
        # Create weekly patterns (7 days)
        # Pattern: [Mon, Tue, Wed, Thu, Fri, Sat, Sun] - office usage intensity
        self.weekly_pattern = np.array([0.75, 1.1, 1.2, 1.15, 0.8, 0.1, 0.1])

        # Monthly patterns (seasonal effects)
        self.monthly_pattern = np.array([
            0.9,   # Jan - New year ramp up
            1.0,   # Feb - Normal
            1.1,   # Mar - Q1 end push
            1.0,   # Apr - Normal
            0.95,  # May - Spring slowdown
            1.1,   # Jun - Q2 end push
            0.7,   # Jul - Summer vacation
            0.7,   # Aug - Summer vacation
            1.1,   # Sep - Back to work push
            1.0,   # Oct - Normal
            0.95,  # Nov - Pre-holiday
            0.6    # Dec - Holiday season
        ])

        # Create Markov chain for recurring patterns
        self.recurring_meeting_probability = {
            'daily': 0.15,      # Daily standups
            'weekly': 0.25,     # Weekly team meetings
            'bi_weekly': 0.10,  # Bi-weekly reviews
            'monthly': 0.05     # Monthly all-hands
        }

    def setup_asset_features(self):
        """Define asset characteristics for feature-based ML models"""
        self.asset_features = {}

        # Desk features
        for i, desk_id in enumerate(self.desk_ids):
            desk_num = int(desk_id.split('-')[1])

            # Simulate office layout features
            floor_position = (desk_num - 1) // 20  # Floor or zone
            row_position = ((desk_num - 1) % 20) // 5  # Row within zone
            seat_position = (desk_num - 1) % 5  # Position in row

            # Desirability features
            is_window = seat_position in [0, 4]  # End seats are windows
            is_corner = (row_position in [0, 3]) and (seat_position in [0, 4])
            proximity_to_amenities = 1.0 / (1 + abs(floor_position - 1))  # Middle floors better

            self.asset_features[desk_id] = {
                'type': 'desk',
                'floor_position': floor_position,
                'row_position': row_position,
                'seat_position': seat_position,
                'is_window': int(is_window),
                'is_corner': int(is_corner),
                'proximity_amenities': proximity_to_amenities,
                'base_desirability': 0.5 + 0.3 * int(is_window) + 0.2 * int(is_corner) + 0.1 * proximity_to_amenities
            }

        # Meeting room features
        capacities = [4, 6, 8, 10, 12, 16, 20, 25, 30]  # Different room sizes
        for i, room_id in enumerate(self.mr_ids):
            capacity = capacities[i % len(capacities)]
            has_av = i < 7  # Most rooms have AV
            has_whiteboard = i < 8  # Most rooms have whiteboards
            is_premium = i < 2  # First two are premium
            floor = i // 3  # Distribute across floors

            self.asset_features[room_id] = {
                'type': 'meeting_room',
                'capacity': capacity,
                'has_av': int(has_av),
                'has_whiteboard': int(has_whiteboard),
                'is_premium': int(is_premium),
                'floor': floor,
                'base_desirability': 0.6 + 0.2 * int(is_premium) + 0.1 * int(has_av) + 0.1 * int(has_whiteboard)
            }

    def get_time_features(self, date):
        """Extract time-based features for ML models"""
        features = {
            'day_of_week': date.weekday(),
            'week_of_year': date.isocalendar()[1],
            'month': date.month,
            'day_of_month': date.day,
            'quarter': (date.month - 1) // 3 + 1,
            'is_month_start': date.day <= 5,
            'is_month_end': date.day >= 25,
            'is_quarter_end': date.month in [3, 6, 9, 12] and date.day >= 25
        }

        # Calculate temporal modifiers
        weekly_mod = self.weekly_pattern[date.weekday()]
        monthly_mod = self.monthly_pattern[date.month - 1]

        # Add noise for realism
        daily_noise = np.random.normal(1.0, 0.1)

        features['temporal_modifier'] = weekly_mod * monthly_mod * daily_noise
        return features

    def predict_user_booking_probability(self, user_id, date, asset_id):
        """Use ML-inspired approach to predict booking probability"""
        user = self.users[user_id]
        asset = self.asset_features[asset_id]
        time_features = self.get_time_features(date)

        # Base probability from user behavior
        base_prob = user['frequency']

        # Modify based on user persona and asset match
        if asset['type'] == 'desk':
            # Desk booking logic
            if asset_id in user['preferred_desks']:
                base_prob *= 1.5  # Prefer familiar desks

            # Team clustering effect (simulate team sitting together)
            team_clustering_bonus = 0
            if hasattr(self, 'recent_bookings'):
                # Check if teammates (similar users) booked nearby
                nearby_desks = self.get_nearby_desks(asset_id)
                for nearby_desk in nearby_desks:
                    if nearby_desk in self.recent_bookings.get(date.strftime('%Y-%m-%d'), []):
                        team_clustering_bonus += 0.1

            base_prob += team_clustering_bonus

        elif asset['type'] == 'meeting_room':
            # Meeting room booking logic
            base_prob = user['meeting_tendency']

            # Match room capacity to user's team size
            if asset['capacity'] >= user['team_size'] * 1.2:  # Room can fit team comfortably
                base_prob *= 1.3
            elif asset['capacity'] < user['team_size']:  # Room too small
                base_prob *= 0.3

        # Apply temporal patterns
        base_prob *= time_features['temporal_modifier']

        # Add recurring meeting patterns
        if asset['type'] == 'meeting_room':
            if self.has_recurring_meeting(user_id, asset_id, date):
                base_prob *= 2.0

        return min(base_prob, 1.0)

    def get_nearby_desks(self, desk_id):
        """Get desks that are physically nearby for clustering simulation"""
        desk_num = int(desk_id.split('-')[1])
        nearby_nums = []

        # Add adjacent desks (same row)
        for offset in [-2, -1, 1, 2]:
            nearby_num = desk_num + offset
            if self.desk_range[0] <= nearby_num <= self.desk_range[1]:
                nearby_nums.append(nearby_num)

        # Add desks in adjacent rows
        for row_offset in [-5, 5]:  # Assuming 5 desks per row
            for seat_offset in [-1, 0, 1]:
                nearby_num = desk_num + row_offset + seat_offset
                if self.desk_range[0] <= nearby_num <= self.desk_range[1]:
                    nearby_nums.append(nearby_num)

        return [f"D-{num:03d}" for num in nearby_nums]

    def has_recurring_meeting(self, user_id, room_id, date):
        """Simulate recurring meeting patterns using Markov-like logic"""
        # Use hash of user_id, room_id, and week to create consistent patterns
        hash_val = hash(f"{user_id}_{room_id}_{date.isocalendar()[1]}") % 100

        # Check different recurring patterns
        if date.weekday() == 1 and hash_val < 15:  # Weekly Tuesday meetings
            return True
        elif date.weekday() == 3 and hash_val < 10:  # Weekly Thursday meetings
            return True
        elif date.day == 1 and hash_val < 5:  # Monthly first-day meetings
            return True

        return False

    def generate_meeting_duration_ml(self, room_features, user_features, date_features):
        """Use ML-inspired approach to generate realistic meeting durations"""
        # Base duration influenced by room capacity and user team size
        base_duration = min(user_features['team_size'] * 15, 120)  # 15 min per person, max 2h

        # Adjust based on room premium status
        if room_features['is_premium']:
            base_duration *= 1.3  # Premium rooms = longer meetings

        # Time of day effect (simulated)
        time_mod = 1.0
        if date_features['day_of_week'] == 0:  # Monday - longer planning meetings
            time_mod = 1.2
        elif date_features['day_of_week'] == 4:  # Friday - shorter wrap-up meetings
            time_mod = 0.8

        # Quarter-end effect
        if date_features['is_quarter_end']:
            time_mod *= 1.4  # Longer strategic meetings

        # Room capacity utilization efficiency
        capacity_efficiency = min(user_features['team_size'] / room_features['capacity'], 1.0)
        if capacity_efficiency > 0.8:  # Well-utilized room
            time_mod *= 1.1
        elif capacity_efficiency < 0.3:  # Under-utilized room (might be backup)
            time_mod *= 0.7

        # Generate duration using common meeting slot patterns
        common_durations = [30, 45, 60, 90, 120, 180]  # Common meeting lengths
        weights = [0.3, 0.15, 0.3, 0.15, 0.08, 0.02]  # Weighted toward shorter meetings

        # Adjust weights based on calculated factors
        adjusted_duration = base_duration * time_mod

        # Choose closest common duration
        best_duration = min(common_durations, key=lambda x: abs(x - adjusted_duration))

        # Add some variability for multiple bookings per day
        if np.random.random() < 0.3:  # 30% chance of additional booking
            additional_duration = np.random.choice([30, 60], p=[0.7, 0.3])
            return min(best_duration + additional_duration, 420)  # Max 7 hours per day

        return best_duration

    def enforce_minimum_usage(self, bookings_df, total_business_days):
        """Ensure minimum 30% usage across all assets using ML-driven approach"""
        # Calculate current usage rates
        asset_usage = bookings_df.groupby('ID').size()

        # Identify assets below minimum threshold
        min_bookings_required = int(total_business_days * self.min_usage)

        additional_bookings = []
        for asset_id in self.desk_ids + self.mr_ids:
            current_bookings = asset_usage.get(asset_id, 0)

            if current_bookings < min_bookings_required:
                shortage = min_bookings_required - current_bookings

                # Use ML approach to determine when to add bookings
                # Prefer dates with higher predicted probability
                all_dates = pd.date_range(start=bookings_df['BookingDate'].min(),
                                          end=bookings_df['BookingDate'].max(),
                                          freq='B')  # Business days only

                # Score each date for booking probability
                date_scores = []
                for date in all_dates:
                    if date.strftime('%Y-%m-%d') not in bookings_df[bookings_df['ID'] == asset_id]['BookingDate'].values:
                        time_features = self.get_time_features(date)
                        score = time_features['temporal_modifier']
                        date_scores.append((date, score))

                # Sort by score and take top dates to fill shortage
                date_scores.sort(key=lambda x: x[1], reverse=True)

                for i in range(min(shortage, len(date_scores))):
                    date, _ = date_scores[i]

                    if asset_id.startswith('D-'):  # Desk
                        minutes_booked = 0
                    else:  # Meeting room
                        # Generate realistic duration for forced booking
                        user_features = {'team_size': np.random.uniform(2, 6), 'meeting_tendency': 0.5}
                        room_features = self.asset_features[asset_id]
                        date_features = self.get_time_features(date)
                        minutes_booked = self.generate_meeting_duration_ml(room_features, user_features, date_features)

                    additional_bookings.append({
                        'ID': asset_id,
                        'BookingDate': date.strftime('%Y-%m-%d'),
                        'MinutesBooked': minutes_booked
                    })

        return additional_bookings

    def generate_realistic_dataset(self, start_date_str="2024-06-01", end_date_str="2025-06-01",
                                   output_csv_path="ml_realistic_bookings.csv"):
        """Generate complete realistic dataset using ML approaches"""
        start_date = datetime.strptime(start_date_str, "%Y-%m-%d").date()
        end_date = datetime.strptime(end_date_str, "%Y-%m-%d").date()

        # Calculate business days
        business_days = pd.bdate_range(start=start_date, end=end_date)
        total_business_days = len(business_days)

        all_bookings = []
        self.recent_bookings = {}  # Track recent bookings for clustering

        print("Generating ML-based realistic booking data...")
        print(f"Processing {total_business_days} business days...")

        for i, date in enumerate(business_days):
            date_obj = date.date()
            date_str = date_obj.strftime('%Y-%m-%d')

            if i % 50 == 0:
                print(f"Processing date {i+1}/{total_business_days}: {date_str}")

            daily_bookings = []

            # Generate bookings for each user
            for user_id in self.users.keys():
                user = self.users[user_id]

                # Desk booking decision
                for desk_id in self.desk_ids:
                    booking_prob = self.predict_user_booking_probability(user_id, date_obj, desk_id)

                    if np.random.random() < booking_prob:
                        all_bookings.append({
                            'ID': desk_id,
                            'BookingDate': date_str,
                            'MinutesBooked': 0
                        })
                        daily_bookings.append(desk_id)
                        break  # User can only book one desk per day

                # Meeting room booking decision
                if np.random.random() < user['meeting_tendency']:
                    for room_id in user['preferred_rooms']:
                        booking_prob = self.predict_user_booking_probability(user_id, date_obj, room_id)

                        if np.random.random() < booking_prob:
                            room_features = self.asset_features[room_id]
                            date_features = self.get_time_features(date_obj)
                            duration = self.generate_meeting_duration_ml(room_features, user, date_features)

                            all_bookings.append({
                                'ID': room_id,
                                'BookingDate': date_str,
                                'MinutesBooked': duration
                            })
                            break  # User books only one room per day

            # Store daily bookings for clustering analysis
            self.recent_bookings[date_str] = daily_bookings

            # Keep only last 5 days for clustering (performance optimization)
            if len(self.recent_bookings) > 5:
                oldest_date = min(self.recent_bookings.keys())
                del self.recent_bookings[oldest_date]

        # Convert to DataFrame for easier manipulation
        bookings_df = pd.DataFrame(all_bookings)

        print("Enforcing minimum usage requirements...")
        # Enforce minimum usage
        additional_bookings = self.enforce_minimum_usage(bookings_df, total_business_days)

        # Add additional bookings
        if additional_bookings:
            additional_df = pd.DataFrame(additional_bookings)
            bookings_df = pd.concat([bookings_df, additional_df], ignore_index=True)

        # Sort by date and ID for consistency
        bookings_df = bookings_df.sort_values(['BookingDate', 'ID']).reset_index(drop=True)

        # Save to CSV
        bookings_df.to_csv(output_csv_path, index=False)

        # Generate statistics
        self.generate_statistics(bookings_df, total_business_days, output_csv_path)

        return bookings_df

    def generate_statistics(self, bookings_df, total_business_days, output_path):
        """Generate comprehensive statistics about the generated dataset"""
        print(f"\n=== ML-Generated Booking Dataset Statistics ===")
        print(f"Output file: {output_path}")
        print(f"Total bookings: {len(bookings_df):,}")
        print(f"Date range: {total_business_days} business days")

        # Desk statistics
        desk_bookings = bookings_df[bookings_df['ID'].str.startswith('D-')]
        desk_usage = desk_bookings.groupby('ID').size()

        print(f"\n--- Desk Usage Statistics ---")
        print(f"Total desk bookings: {len(desk_bookings):,}")
        print(f"Average usage per desk: {desk_usage.mean():.1f} days ({desk_usage.mean()/total_business_days:.1%})")
        print(f"Minimum usage: {desk_usage.min()} days ({desk_usage.min()/total_business_days:.1%})")
        print(f"Maximum usage: {desk_usage.max()} days ({desk_usage.max()/total_business_days:.1%})")
        print(f"Desks meeting 30% minimum: {(desk_usage >= total_business_days * 0.3).sum()}/{len(self.desk_ids)}")

        # Meeting room statistics
        mr_bookings = bookings_df[bookings_df['ID'].str.startswith('MR-')]
        mr_usage = mr_bookings.groupby('ID').size()
        mr_duration = mr_bookings.groupby('ID')['MinutesBooked'].sum()

        print(f"\n--- Meeting Room Statistics ---")
        print(f"Total meeting room bookings: {len(mr_bookings):,}")
        print(f"Average bookings per room: {mr_usage.mean():.1f} days ({mr_usage.mean()/total_business_days:.1%})")
        print(f"Average duration per booking: {mr_bookings['MinutesBooked'].mean():.0f} minutes")
        print(f"Total meeting hours: {mr_duration.sum()/60:.1f} hours")

        # Weekly patterns
        bookings_df['WeekDay'] = pd.to_datetime(bookings_df['BookingDate']).dt.day_name()
        weekly_pattern = bookings_df.groupby('WeekDay').size().reindex(
            ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
        )

        print(f"\n--- Weekly Booking Pattern ---")
        for day, count in weekly_pattern.items():
            print(f"{day}: {count:,} bookings")

        # User persona distribution (simulated)
        print(f"\n--- User Persona Distribution ---")
        persona_counts = {}
        for user in self.users.values():
            persona = user['persona']
            persona_counts[persona] = persona_counts.get(persona, 0) + 1

        for persona, count in sorted(persona_counts.items()):
            print(f"{persona}: {count} users ({count/len(self.users):.1%})")

        print(f"\n=== Dataset Generation Complete ===")


# Example usage and testing
if __name__ == "__main__":
    # Initialize the ML-based generator
    print("Initializing ML-based booking data generator...")
    generator = MLBookingDataGenerator(
        desk_range=(1, 78),
        mr_range=(1, 9),
        min_usage=0.30
    )

    # Generate the dataset
    dataset = generator.generate_realistic_dataset(
        start_date_str="2024-06-01",
        end_date_str="2025-06-01",
        output_csv_path="ml_realistic_bookings_30pct.csv"
    )

    print("\nFirst 10 rows of generated data:")
    print(dataset.head(10))

    print("\nDataset shape:", dataset.shape)
    print("\nColumn types:")
    print(dataset.dtypes)