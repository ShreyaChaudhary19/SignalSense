
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

    total_waiting = 0.0
    vehicle_count_sum = 0
    halting_count_sum = 0

    total_departed = 0
    total_arrived = 0
    total_teleport_starts = 0
    total_teleport_ends = 0

    max_vehicles = 0
    max_halting = 0

    raw_ttc_observations = 0
    candidate_pairs = set()

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

        signal_timers = {j: 0 for j in junctions}

        # Deduplicate controlled lanes across all junctions.
        controlled_lanes = sorted({
            lane
            for junction in junctions
            for lane in traci.trafficlight.getControlledLanes(junction)
            if not lane.startswith(":")
        })

        for _ in range(SIMULATION_STEPS):
            traci.simulationStep()

            vehicle_ids = traci.vehicle.getIDList()
            current_count = len(vehicle_ids)

            halting = sum(
                traci.lane.getLastStepHaltingNumber(lane)
                for lane in traci.lane.getIDList()
                if not lane.startswith(":")
            )

            vehicle_count_sum += current_count
            halting_count_sum += halting
            max_vehicles = max(max_vehicles, current_count)
            max_halting = max(max_halting, halting)

            total_waiting += sum(
                traci.lane.getWaitingTime(lane)
                for lane in controlled_lanes
            )

            # Count trip activity each simulation step.
            total_departed += traci.simulation.getDepartedNumber()
            total_arrived += traci.simulation.getArrivedNumber()

            # Count vehicles beginning and ending teleportation.
            total_teleport_starts += len(
                traci.simulation.getStartingTeleportIDList()
            )
            total_teleport_ends += len(
                traci.simulation.getEndingTeleportIDList()
            )

            # Count TTC observations and distinct candidate pairs.
            for candidate in get_ttc_candidates(traci):
                raw_ttc_observations += 1
                pair = tuple(sorted((
                    candidate["follower"],
                    candidate["leader"],
                )))
                candidate_pairs.add(pair)

            # Apply the trained model's traffic signal decisions.
            for junction in junctions:
                if signal_timers[junction] == 0:
                    lanes = traci.trafficlight.getControlledLanes(
                        junction
                    )
                    counts = get_vehicle_numbers(lanes)
                    state = list(counts.values())[:4]
                    state += [0] * (4 - len(state))

                    with torch.no_grad():
                        action = brain.choose_action(state)

                    set_signal(
                        junction, 6, SELECT_LANE[action][0]
                    )
                    set_signal(
                        junction, 15, SELECT_LANE[action][1]
                    )
                    signal_timers[junction] = 15
                else:
                    signal_timers[junction] -= 1

        result = {
            "model": model_name,
            "seed": seed,
            "avg_vehicles": vehicle_count_sum / SIMULATION_STEPS,
            "max_vehicles": max_vehicles,
            "avg_halting": halting_count_sum / SIMULATION_STEPS,
            "max_halting": max_halting,
            "waiting_time": total_waiting,
            "departed": total_departed,
            "arrived": total_arrived,
            "teleport_starts": total_teleport_starts,
            "teleport_ends": total_teleport_ends,
            "unique_candidate_pairs": len(candidate_pairs),
            "raw_ttc_observations": raw_ttc_observations,
        }

        print(
            f"{model_name} | seed={seed} | "
            f"avg vehicles={result['avg_vehicles']:.2f} | "
            f"avg halting={result['avg_halting']:.2f} | "
            f"max halting={max_halting} | "
            f"arrived={total_arrived} | "
            f"teleports started={total_teleport_starts} | "
            f"waiting={total_waiting:.1f} | "
            f"candidate pairs={len(candidate_pairs)}"
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

    csv_path = OUTPUT_DIR / "model_congestion_comparison_v2.csv"

    fields = list(results[0].keys())

    with csv_path.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)

    print("\n=== MEAN RESULTS ACROSS SEEDS ===")

    metrics = [
        "avg_vehicles",
        "max_vehicles",
        "avg_halting",
        "max_halting",
        "waiting_time",
        "departed",
        "arrived",
        "teleport_starts",
        "teleport_ends",
        "unique_candidate_pairs",
        "raw_ttc_observations",
    ]

    for model_name in MODEL_NAMES:
        subset = [
            row for row in results
            if row["model"] == model_name
        ]

        print(f"\n{model_name}")

        for metric in metrics:
            values = [row[metric] for row in subset]
            mean = statistics.mean(values)
            sd = statistics.stdev(values) if len(values) > 1 else 0.0
            print(f"  {metric}: {mean:.2f} (SD {sd:.2f})")

    print(f"\nDetailed results saved to:\n{csv_path}")


if __name__ == "__main__":
    main()
