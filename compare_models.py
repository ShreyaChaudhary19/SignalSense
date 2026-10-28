
import os
import csv
import statistics
from pathlib import Path

import torch
import traci
from sumolib import checkBinary

from train import Agent, get_vehicle_numbers
from safety_metrics import get_ttc_candidates

PROJECT_DIR = Path(__file__).resolve().parent
SUMO_CONFIG = PROJECT_DIR / "configuration.sumocfg"
MODELS_DIR = PROJECT_DIR / "models"
OUTPUT_DIR = PROJECT_DIR / "safety_results"

MODEL_NAMES = ["test_model", "safety_v2"]
SEEDS = [101, 102, 103, 104, 105]
SIMULATION_STEPS = 600

SELECT_LANE = [
    ["yyyrrrrrrrrr", "GGGrrrrrrrrr"],
    ["rrryyyrrrrrr", "rrrGGGrrrrrr"],
    ["rrrrrryyyrrr", "rrrrrrGGGrrr"],
    ["rrrrrrrrryyy", "rrrrrrrrrGGG"],
]


def set_signal(junction, duration, state):
    traci.trafficlight.setRedYellowGreenState(junction, state)
    traci.trafficlight.setPhaseDuration(junction, duration)


def run_trial(model_name, seed):
    model_path = MODELS_DIR / f"{model_name}.bin"

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    if not SUMO_CONFIG.exists():
        raise FileNotFoundError(f"SUMO config not found: {SUMO_CONFIG}")

    waiting_time = 0.0
    vehicle_count_sum = 0
    halting_count_sum = 0
    candidate_pairs = set()
    raw_ttc_observations = 0

    command = [
        checkBinary("sumo"),
        "-c", str(SUMO_CONFIG),
        "--seed", str(seed),
        "--no-step-log", "true",
        "--duration-log.disable", "true",
    ]

    try:
        traci.start(command)

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

        signal_timers = {junction: 0 for junction in junctions}

        for _ in range(SIMULATION_STEPS):
            traci.simulationStep()

            vehicles = traci.vehicle.getIDList()
            vehicle_count_sum += len(vehicles)

            # Count halting vehicles on non-internal lanes.
            halting = sum(
                traci.lane.getLastStepHaltingNumber(lane)
                for lane in traci.lane.getIDList()
                if not lane.startswith(":")
            )
            halting_count_sum += halting

            for junction in junctions:
                lanes = list(dict.fromkeys(
                    traci.trafficlight.getControlledLanes(junction)
                ))
                waiting_time += sum(
                    traci.lane.getWaitingTime(lane)
                    for lane in lanes
                    if not lane.startswith(":")
                )

            for candidate in get_ttc_candidates(traci):
                raw_ttc_observations += 1
                pair = tuple(sorted((
                    candidate["follower"],
                    candidate["leader"],
                )))
                candidate_pairs.add(pair)

            for junction in junctions:
                if signal_timers[junction] == 0:
                    lanes = traci.trafficlight.getControlledLanes(junction)
                    counts = get_vehicle_numbers(lanes)
                    state = list(counts.values())[:4]
                    state += [0] * (4 - len(state))

                    with torch.no_grad():
                        action = brain.choose_action(state)

                    set_signal(junction, 6, SELECT_LANE[action][0])
                    set_signal(junction, 15, SELECT_LANE[action][1])
                    signal_timers[junction] = 15
                else:
                    signal_timers[junction] -= 1

        result = {
            "model": model_name,
            "seed": seed,
            "avg_vehicles": vehicle_count_sum / SIMULATION_STEPS,
            "avg_halting": halting_count_sum / SIMULATION_STEPS,
            "waiting_time": waiting_time,
            "unique_candidate_pairs": len(candidate_pairs),
            "raw_ttc_observations": raw_ttc_observations,
        }

        print(
            f"{model_name:12} seed={seed} | "
            f"avg vehicles={result['avg_vehicles']:.2f} | "
            f"avg halting={result['avg_halting']:.2f} | "
            f"waiting={waiting_time:.1f} | "
            f"candidate pairs={len(candidate_pairs)} | "
            f"TTC observations={raw_ttc_observations}"
        )

        return result

    finally:
        if traci.isLoaded():
            traci.close()


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = []

    for seed in SEEDS:
        for model_name in MODEL_NAMES:
            results.append(run_trial(model_name, seed))

    csv_path = OUTPUT_DIR / "model_congestion_comparison.csv"

    fields = [
        "model", "seed", "avg_vehicles", "avg_halting",
        "waiting_time", "unique_candidate_pairs",
        "raw_ttc_observations",
    ]

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)

    print("\n=== MEAN RESULTS ACROSS SEEDS ===")

    for model_name in MODEL_NAMES:
        subset = [r for r in results if r["model"] == model_name]
        print(f"\n{model_name}")

        for metric in fields[2:]:
            values = [r[metric] for r in subset]
            mean = statistics.mean(values)
            sd = statistics.stdev(values) if len(values) > 1 else 0.0
            print(f"  {metric}: {mean:.2f} (SD {sd:.2f})")

    print(f"\nResults saved to: {csv_path}")


if __name__ == "__main__":
    main()
