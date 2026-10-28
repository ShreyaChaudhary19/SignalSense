
import os
from pathlib import Path

import traci
from sumolib import checkBinary

from safety_metrics import get_ttc_candidates

PROJECT_DIR = Path(__file__).resolve().parent
CONFIG_FILE = PROJECT_DIR / "configuration.sumocfg"


def main():
    if "SUMO_HOME" not in os.environ:
        raise EnvironmentError("SUMO_HOME is not set.")

    if not CONFIG_FILE.exists():
        raise FileNotFoundError(CONFIG_FILE)

    traci.start([
        checkBinary("sumo"),
        "-c", str(CONFIG_FILE),
        "--no-step-log", "true",
        "--duration-log.disable", "true",
    ])

    total_candidates = 0
    minimum_ttc = None

    try:
        for _ in range(200):
            traci.simulationStep()

            candidates = get_ttc_candidates(traci)
            total_candidates += len(candidates)

            for candidate in candidates:
                ttc = candidate["ttc_s"]
                if minimum_ttc is None or ttc < minimum_ttc:
                    minimum_ttc = ttc

        print("Safety module simulation test completed.")
        print("Candidate observations:", total_candidates)

        if minimum_ttc is not None:
            print("Minimum TTC (seconds):", round(minimum_ttc, 3))
        else:
            print("No candidate conflicts detected in this run.")

    finally:
        if traci.isLoaded():
            traci.close()


if __name__ == "__main__":
    main()
