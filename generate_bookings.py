#!/usr/bin/env python3
import pandas as pd
import numpy as np
from faker import Faker
from datetime import datetime, timedelta
import argparse
import csv
import random

fake = Faker()

def parse_args():
    p = argparse.ArgumentParser(
        description="Generate synthetic desk/room booking data."
    )
    p.add_argument("--mapping", default="coordinate_mapping.csv",
                   help="Path to your ID mapping CSV.")
    p.add_argument("--start", type=str, required=True,
                   help="Start date, YYYY-MM-DD")
    p.add_argument("--end", type=str, required=True,
                   help="End date,   YYYY-MM-DD")
    p.add_argument("--rows", type=int, default=5000,
                   help="Approximate number of bookings to generate")
    p.add_argument("--low-desks", nargs="+", default=["D-017","D-019"],
                   help="Desk IDs to keep low-util (few bookings).")
    p.add_argument("--ultra-room", default="MR-004",
                   help="Meeting room ID to cap at 3 bookings total.")
    p.add_argument("--output", default="synthetic_bookings.csv",
                   help="Output CSV file.")
    return p.parse_args()

def load_ids(mapping_path):
    df = pd.read_csv(mapping_path)
    desks = df.loc[df.Type=="desk","ID"].tolist()
    rooms= df.loc[df.Type=="meeting-room","ID"].tolist()
    return desks, rooms

def random_timestamp(start_dt, end_dt):
    """Uniform random datetime between start_dt and end_dt."""
    span = (end_dt - start_dt).total_seconds()
    offset = random.random() * span
    return start_dt + timedelta(seconds=offset)

def main():
    args = parse_args()
    desks, rooms = load_ids(args.mapping)
    start_dt = datetime.fromisoformat(args.start)
    end_dt   = datetime.fromisoformat(args.end)

    # Build weighted choices
    # normal desks get weight=1, low-desks weight=0.1
    desk_weights = {d: (0.1 if d in args.low_desks else 1.0) for d in desks}
    # normal rooms weight=1, ultra-room weight=0
    room_weights= {r: (0.0 if r==args.ultra_room else 1.0) for r in rooms}

    # normalize into lists for random.choices
    desk_ids = list(desk_weights.keys())
    desk_wts = np.array([desk_weights[d] for d in desk_ids], dtype=float)
    desk_wts /= desk_wts.sum()

    room_ids = list(room_weights.keys())
    room_wts = np.array([room_weights[r] for r in room_ids], dtype=float)
    room_wts /= room_wts.sum()

    # to enforce exactly 3 bookings for ultra-room, we’ll generate them manually later
    ultra_bookings = []
    for i in range(3):
        ts = random_timestamp(start_dt, end_dt)
        dur = random.choice([15,30,60,90])
        ultra_bookings.append({
            "Desk_ID": "",
            "Meeting_Room_ID": args.ultra_room,
            "Employee_Name": fake.name(),
            "Booking_Timestamp": ts.isoformat(sep=" "),
            "Duration": dur
        })

    records = []
    # generate approx args.rows bookings
    for _ in range(args.rows - len(ultra_bookings)):
        # decide desk vs room 80/20 split
        if random.random() < 0.8:
            # desk booking
            desk = random.choices(desk_ids, weights=desk_wts, k=1)[0]
            room = ""
        else:
            room = random.choices(room_ids, weights=room_wts, k=1)[0]
            desk = ""
        ts  = random_timestamp(start_dt, end_dt)
        # duration in minutes: peaked around 30–120
        dur = int(np.clip(np.random.normal(60,30), 15, 240) // 15 * 15)
        records.append({
            "Desk_ID": desk,
            "Meeting_Room_ID": room,
            "Employee_Name": fake.name(),
            "Booking_Timestamp": ts.isoformat(sep=" "),
            "Duration": dur
        })

    # combine and shuffle
    all_records = records + ultra_bookings
    random.shuffle(all_records)

    # write out
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "Desk_ID","Meeting_Room_ID","Employee_Name",
            "Booking_Timestamp","Duration"
        ])
        writer.writeheader()
        for r in all_records:
            writer.writerow(r)

    print(f"✨ Generated {len(all_records)} bookings → {args.output}")

if __name__=="__main__":
    main()
