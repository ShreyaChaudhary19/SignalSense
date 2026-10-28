
import os
import sys
import csv
from pathlib import Path

if "SUMO_HOME" not in os.environ:
    raise EnvironmentError("SUMO_HOME is not set.")

sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))

import traci

ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / "configuration.sumocfg"
OUTPUT = ROOT / "safety_results" / "congestion_log.csv"
STEPS = 600
SEED = 101

sumo_binary = os.path.join(
    os.environ["SUMO_HOME"], "bin", "sumo"
)

command = [
    sumo_binary,
    "-c", str(CONFIG),
    "--seed", str(SEED),
    "--no-step-log", "true",
    "--duration-log.disable", "true",
]

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
rows = []

try:
    traci.start(command)

    for step in range(STEPS):
        traci.simulationStep()

        vehicle_ids = traci.vehicle.getIDList()
        vehicle_count = len(vehicle_ids)

        halting_count = sum(
            traci.lane.getLastStepHaltingNumber(lane_id)
            for lane_id in traci.lane.getIDList()
            if not lane_id.startswith(":")
        )

        arrived_count = traci.simulation.getArrivedNumber()
        departed_count = traci.simulation.getDepartedNumber()

        rows.append({
            "time_s": traci.simulation.getTime(),
            "vehicles_on_network": vehicle_count,
            "halting_vehicles": halting_count,
            "departed_this_step": departed_count,
            "arrived_this_step": arrived_count,
        })

        if step % 50 == 0:
            print(
                f"Time={traci.simulation.getTime():.0f}s | "
                f"Vehicles={vehicle_count} | "
                f"Halting={halting_count}"
            )

finally:
    if traci.isLoaded():
        traci.close()

with OUTPUT.open("w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print(f"\nSaved congestion log: {OUTPUT}")
print(f"Final vehicles on network: {rows[-1]['vehicles_on_network']}")
print(f"Final halting vehicles: {rows[-1]['halting_vehicles']}")
