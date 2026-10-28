import os
import time
import csv
import statistics
from pathlib import Path

import torch
import traci
from sumolib import checkBinary

from train import Agent, get_waiting_time, get_vehicle_numbers
from safety_metrics import get_ttc_candidates

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

PROJECT_DIR = Path(__file__).resolve().parent
SUMO_CONFIG = PROJECT_DIR / "configuration.sumocfg"
OUTPUT_DIR = PROJECT_DIR / "safety_results"
MODELS_DIR = PROJECT_DIR / "models"

SIMULATION_STEPS = 600
NUMBER_OF_TRIALS = 1
BASE_SEED = 101
HARD_DECELERATION = -8.0
MODEL_NAMES = ["safety_v2"]

SELECT_LANE = [
    ["yyyrrrrrrrrr", "GGGrrrrrrrrr"],
    ["rrryyyrrrrrr", "rrrGGGrrrrrr"],
    ["rrrrrryyyrrr", "rrrrrrGGGrrr"],
    ["rrrrrrrrryyy", "rrrrrrrrrGGG"],
]


def phaseDuration(junction, phase_time, phase_state):
    traci.trafficlight.setRedYellowGreenState(junction, phase_state)
    traci.trafficlight.setPhaseDuration(junction, phase_time)


def write_csv(path, rows, fields):
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def evaluate_model(model_name, trial_number, seed):
    model_path = MODELS_DIR / f"{model_name}.bin"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not SUMO_CONFIG.exists():
        raise FileNotFoundError(f"SUMO configuration not found: {SUMO_CONFIG}")

    raw_events = []
    hard_braking = []
    total_waiting_time = 0.0
    unique_lane_waiting_time = 0.0
    active_events = {}
    completed_events = []

    try:
        traci.start([
            checkBinary("sumo-gui"),
            "-c", str(SUMO_CONFIG),
            "--seed", str(seed),
            "--no-step-log", "true",
            "--duration-log.disable", "true",
        ])

        junctions = list(traci.trafficlight.getIDList())
        if not junctions:
            raise RuntimeError("No traffic-light junctions found.")

        brain = Agent(
            gamma=0.99,
            epsilon=0.0,
            lr=0.1,
            input_dims=4,
            fc1_dims=256,
            fc2_dims=256,
            batch_size=1024,
            n_actions=4,
            junctions=list(range(len(junctions))),
        )
        brain.Q_eval.load_state_dict(
            torch.load(model_path, map_location=brain.Q_eval.device)
        )
        brain.Q_eval.eval()

        traffic_lights_time = {junction: 0 for junction in junctions}

        for _ in range(SIMULATION_STEPS + 1):
            print(f"Simulation time: {traci.simulation.getTime():.0f}s")
            traci.simulationStep()
            time.sleep(0.1)
            now = traci.simulation.getTime()
            vehicle_ids = list(traci.vehicle.getIDList())
            seen_pairs = set()

            for junction in junctions:
                lanes = traci.trafficlight.getControlledLanes(junction)
                total_waiting_time += get_waiting_time(lanes)
                unique_lanes = list(dict.fromkeys(lanes))
                unique_lane_waiting_time += get_waiting_time(unique_lanes)

            for vehicle_id in vehicle_ids:
                acceleration = traci.vehicle.getAcceleration(vehicle_id)
                if acceleration <= HARD_DECELERATION:
                    hard_braking.append({
                        "time_s": round(now, 2),
                        "vehicle_id": vehicle_id,
                        "acceleration_mps2": round(acceleration, 3),
                    })

            for candidate in get_ttc_candidates(traci):
                pair = (candidate["follower"], candidate["leader"])
                seen_pairs.add(pair)
                raw_events.append({
                    "time_s": round(now, 2),
                    "vehicle_a": candidate["follower"],
                    "vehicle_b": candidate["leader"],
                    "gap_m": round(candidate["gap_m"], 3),
                    "ttc_s": round(candidate["ttc_s"], 3),
                })

                if pair not in active_events:
                    active_events[pair] = {
                        "vehicle_a": pair[0],
                        "vehicle_b": pair[1],
                        "start_s": now,
                        "end_s": now,
                        "min_ttc_s": candidate["ttc_s"],
                        "min_gap_m": candidate["gap_m"],
                        "observations": 1,
                    }
                else:
                    event = active_events[pair]
                    event["end_s"] = now
                    event["min_ttc_s"] = min(event["min_ttc_s"], candidate["ttc_s"])
                    event["min_gap_m"] = min(event["min_gap_m"], candidate["gap_m"])
                    event["observations"] += 1

            for pair in list(active_events):
                if pair not in seen_pairs:
                    completed_events.append(active_events.pop(pair))

            for junction in junctions:
                if traffic_lights_time[junction] == 0:
                    lanes = traci.trafficlight.getControlledLanes(junction)
                    vehicles_per_lane = get_vehicle_numbers(lanes)
                    state = list(vehicles_per_lane.values())[:4]
                    state += [0] * (4 - len(state))
                    with torch.no_grad():
                        action = brain.choose_action(state)
                    phaseDuration(junction, 6, SELECT_LANE[action][0])
                    phaseDuration(junction, 15, SELECT_LANE[action][1])
                    traffic_lights_time[junction] = 15
                else:
                    traffic_lights_time[junction] -= 1

        completed_events.extend(active_events.values())

    finally:
        if traci.isLoaded():
            traci.close()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    prefix = OUTPUT_DIR / f"{model_name}_trial_{trial_number:02d}"

    write_csv(
        Path(f"{prefix}_ttc_observations.csv"), raw_events,
        ["time_s", "vehicle_a", "vehicle_b", "gap_m", "ttc_s"],
    )

    grouped_rows = []
    for event in completed_events:
        row = dict(event)
        row["duration_s"] = round(event["end_s"] - event["start_s"], 2)
        grouped_rows.append(row)
    write_csv(
        Path(f"{prefix}_candidate_events.csv"), grouped_rows,
        ["vehicle_a", "vehicle_b", "start_s", "end_s", "duration_s",
         "min_ttc_s", "min_gap_m", "observations"],
    )
    write_csv(
        Path(f"{prefix}_hard_braking.csv"), hard_braking,
        ["time_s", "vehicle_id", "acceleration_mps2"],
    )

    result = {
        "model": model_name,
        "trial": trial_number,
        "seed": seed,
        "simulation_steps": SIMULATION_STEPS,
        "total_waiting_time": round(total_waiting_time, 3),
        "unique_lane_waiting_time": round(unique_lane_waiting_time, 3),
        "raw_ttc_observations": len(raw_events),
        "grouped_candidate_encounters": len(completed_events),
        "hard_deceleration_observations": len(hard_braking),
    }
    print(
        f"{model_name} | trial {trial_number}/{NUMBER_OF_TRIALS} | seed {seed} | "
        f"unique-lane waiting={result['unique_lane_waiting_time']:.1f} | "
        f"candidate encounters={result['grouped_candidate_encounters']} | "
        f"hard braking={result['hard_deceleration_observations']}"
    )
    return result


def mean_sd(values):
    average = statistics.mean(values)
    sd = statistics.stdev(values) if len(values) > 1 else 0.0
    return round(average, 3), round(sd, 3)


def create_summary_and_charts(results):
    trial_csv = OUTPUT_DIR / "multi_trial_results.csv"
    fields = list(results[0].keys())
    write_csv(trial_csv, results, fields)

    metrics = [
        "total_waiting_time",
        "unique_lane_waiting_time",
        "raw_ttc_observations",
        "grouped_candidate_encounters",
        "hard_deceleration_observations",
    ]
    summary = []
    for model in MODEL_NAMES:
        model_rows = [row for row in results if row["model"] == model]
        row = {"model": model, "trials": len(model_rows)}
        for metric in metrics:
            avg, sd = mean_sd([float(item[metric]) for item in model_rows])
            row[f"{metric}_mean"] = avg
            row[f"{metric}_std_dev"] = sd
        summary.append(row)

    summary_csv = OUTPUT_DIR / "multi_trial_summary.csv"
    write_csv(summary_csv, summary, list(summary[0].keys()))

    print("\n=== MULTI-TRIAL SUMMARY (mean ± standard deviation) ===")
    for row in summary:
        print(f"\nModel: {row['model']} ({row['trials']} trials)")
        for metric in metrics:
            print(f"  {metric}: {row[f'{metric}_mean']:.3f} ± {row[f'{metric}_std_dev']:.3f}")
    print(f"\nTrial-level CSV: {trial_csv}")
    print(f"Summary CSV: {summary_csv}")

    if plt is None:
        print("matplotlib is not installed; skipped graphs. Install with: python -m pip install matplotlib")
        return

    chart_metrics = [
        ("unique_lane_waiting_time", "Unique-lane waiting-time metric", "Mean waiting-time metric"),
        ("grouped_candidate_encounters", "Grouped candidate encounters", "Mean candidate encounters"),
        ("hard_deceleration_observations", "Hard-deceleration observations", "Mean hard-deceleration observations"),
    ]
    for metric, title, ylabel in chart_metrics:
        means = [row[f"{metric}_mean"] for row in summary]
        errors = [row[f"{metric}_std_dev"] for row in summary]
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.bar([row["model"] for row in summary], means, yerr=errors, capsize=6)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Model")
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        chart_path = OUTPUT_DIR / f"{metric}_comparison.png"
        fig.savefig(chart_path, dpi=200)
        plt.close(fig)
        print(f"Chart saved: {chart_path}")


def main():
    if "SUMO_HOME" not in os.environ:
        raise EnvironmentError("Set SUMO_HOME before running this script.")

    all_results = []
    for trial in range(1, NUMBER_OF_TRIALS + 1):
        seed = BASE_SEED + trial - 1
        for model_name in MODEL_NAMES:
            all_results.append(evaluate_model(model_name, trial, seed))
    create_summary_and_charts(all_results)


if __name__ == "__main__":
    main()
