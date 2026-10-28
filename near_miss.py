
import csv
import os
from pathlib import Path

import traci
from sumolib import checkBinary

PROJECT_DIR = Path(__file__).resolve().parent
SUMO_CONFIG = PROJECT_DIR / "configuration.sumocfg"
OUTPUT_DIR = PROJECT_DIR / "safety_results"

RAW_FILE = OUTPUT_DIR / "near_miss_observations.csv"
EVENT_FILE = OUTPUT_DIR / "near_miss_events.csv"

SIMULATION_STEPS = 200
TTC_THRESHOLD = 3.0
MAX_GAP = 15.0
MIN_CLOSING_SPEED = 0.1
HARD_DECELERATION = -8.0


def get_candidate(vehicle_a, vehicle_b):
    """Return TTC and estimated front-to-rear gap for same-lane vehicles."""
    lane_a = traci.vehicle.getLaneID(vehicle_a)
    lane_b = traci.vehicle.getLaneID(vehicle_b)

    if lane_a != lane_b:
        return None

    pos_a = traci.vehicle.getLanePosition(vehicle_a)
    pos_b = traci.vehicle.getLanePosition(vehicle_b)

    # Identify the leading and following vehicles.
    if pos_a > pos_b:
        leader, follower = vehicle_a, vehicle_b
        leader_pos, follower_pos = pos_a, pos_b
    else:
        leader, follower = vehicle_b, vehicle_a
        leader_pos, follower_pos = pos_b, pos_a

    leader_length = traci.vehicle.getLength(leader)
    gap = leader_pos - follower_pos - leader_length

    if gap < 0 or gap > MAX_GAP:
        return None

    leader_speed = traci.vehicle.getSpeed(leader)
    follower_speed = traci.vehicle.getSpeed(follower)
    closing_speed = follower_speed - leader_speed

    if closing_speed <= MIN_CLOSING_SPEED:
        return None

    ttc = gap / closing_speed

    if 0 <= ttc <= TTC_THRESHOLD:
        return {
            "vehicle_a": follower,
            "vehicle_b": leader,
            "gap_m": gap,
            "ttc_s": ttc,
        }

    return None


def main():
    if "SUMO_HOME" not in os.environ:
        raise EnvironmentError("Set SUMO_HOME before running this script.")

    if not SUMO_CONFIG.exists():
        raise FileNotFoundError(f"Missing configuration: {SUMO_CONFIG}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    raw_rows = []
    active_events = {}
    completed_events = []
    hard_braking_rows = []

    try:
        traci.start([
            checkBinary("sumo"),
            "-c", str(SUMO_CONFIG),
            "--no-step-log", "true",
            "--duration-log.disable", "true",
        ])

        for _ in range(SIMULATION_STEPS):
            traci.simulationStep()
            now = traci.simulation.getTime()
            vehicle_ids = list(traci.vehicle.getIDList())
            seen_pairs = set()

            # Record hard deceleration separately.
            for vehicle_id in vehicle_ids:
                acceleration = traci.vehicle.getAcceleration(vehicle_id)

                if acceleration <= HARD_DECELERATION:
                    hard_braking_rows.append({
                        "time_s": round(now, 2),
                        "vehicle_id": vehicle_id,
                        "acceleration_mps2": round(acceleration, 3),
                        "event_type": "hard_deceleration_observation",
                    })

            # Check same-lane vehicle pairs.
            for i, first in enumerate(vehicle_ids):
                for second in vehicle_ids[i + 1:]:
                    candidate = get_candidate(first, second)

                    if candidate is None:
                        continue

                    a = candidate["vehicle_a"]
                    b = candidate["vehicle_b"]
                    pair = (a, b)
                    seen_pairs.add(pair)

                    raw_rows.append({
                        "time_s": round(now, 2),
                        "vehicle_a": a,
                        "vehicle_b": b,
                        "gap_m": round(candidate["gap_m"], 3),
                        "ttc_s": round(candidate["ttc_s"], 3),
                    })

                    if pair not in active_events:
                        active_events[pair] = {
                            "vehicle_a": a,
                            "vehicle_b": b,
                            "start_s": now,
                            "end_s": now,
                            "min_ttc_s": candidate["ttc_s"],
                            "min_gap_m": candidate["gap_m"],
                            "observations": 1,
                        }
                    else:
                        event = active_events[pair]
                        event["end_s"] = now
                        event["min_ttc_s"] = min(
                            event["min_ttc_s"], candidate["ttc_s"]
                        )
                        event["min_gap_m"] = min(
                            event["min_gap_m"], candidate["gap_m"]
                        )
                        event["observations"] += 1

            # Close encounters that were not detected this step.
            for pair in list(active_events):
                if pair not in seen_pairs:
                    event = active_events.pop(pair)
                    completed_events.append(event)

        completed_events.extend(active_events.values())
        traci.close()

    except Exception:
        if traci.isLoaded():
            traci.close()
        raise

    with RAW_FILE.open("w", newline="", encoding="utf-8") as file:
        fields = ["time_s", "vehicle_a", "vehicle_b", "gap_m", "ttc_s"]
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(raw_rows)

    with EVENT_FILE.open("w", newline="", encoding="utf-8") as file:
        fields = [
            "vehicle_a", "vehicle_b", "start_s", "end_s",
            "duration_s", "min_ttc_s", "min_gap_m", "observations",
            "event_type",
        ]
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()

        for event in completed_events:
            writer.writerow({
                "vehicle_a": event["vehicle_a"],
                "vehicle_b": event["vehicle_b"],
                "start_s": round(event["start_s"], 2),
                "end_s": round(event["end_s"], 2),
                "duration_s": round(
                    event["end_s"] - event["start_s"], 2
                ),
                "min_ttc_s": round(event["min_ttc_s"], 3),
                "min_gap_m": round(event["min_gap_m"], 3),
                "observations": event["observations"],
                "event_type": "candidate_rear_end_encounter",
            })

    with (OUTPUT_DIR / "hard_deceleration.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        fields = [
            "time_s", "vehicle_id", "acceleration_mps2", "event_type"
        ]
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(hard_braking_rows)

    print("\nSafety screening completed.")
    print(f"Raw TTC observations: {len(raw_rows)}")
    print(f"Grouped candidate encounters: {len(completed_events)}")
    print(f"Hard-deceleration observations: {len(hard_braking_rows)}")
    print(f"Raw data: {RAW_FILE}")
    print(f"Grouped events: {EVENT_FILE}")
    print(f"Deceleration data: {OUTPUT_DIR / 'hard_deceleration.csv'}")


if __name__ == "__main__":
    main()
