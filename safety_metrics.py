
"""Online, same-lane rear-end conflict screening for SUMO."""

TTC_THRESHOLD = 3.0
MAX_GAP = 15.0
MIN_CLOSING_SPEED = 0.1


def get_ttc_candidates(traci):
    """
    Return candidate same-lane rear-end conflicts in the current
    SUMO state.

    Each result contains the follower, leader, gap and TTC.
    These are screening indicators, not confirmed near-misses.
    """
    vehicle_ids = list(traci.vehicle.getIDList())
    lane_groups = {}

    # Group vehicles by their current lane.
    for vehicle_id in vehicle_ids:
        lane_id = traci.vehicle.getLaneID(vehicle_id)

        if not lane_id or lane_id.startswith(":"):
            continue

        lane_groups.setdefault(lane_id, []).append(
            (
                traci.vehicle.getLanePosition(vehicle_id),
                vehicle_id,
            )
        )

    candidates = []

    for lane_id, vehicles in lane_groups.items():
        vehicles.sort(key=lambda item: item[0])

        for index in range(len(vehicles) - 1):
            follower_pos, follower = vehicles[index]
            leader_pos, leader = vehicles[index + 1]

            leader_length = traci.vehicle.getLength(leader)
            gap = leader_pos - follower_pos - leader_length

            if gap < 0 or gap > MAX_GAP:
                continue

            follower_speed = traci.vehicle.getSpeed(follower)
            leader_speed = traci.vehicle.getSpeed(leader)
            closing_speed = follower_speed - leader_speed

            if closing_speed <= MIN_CLOSING_SPEED:
                continue

            ttc = gap / closing_speed

            if 0 <= ttc <= TTC_THRESHOLD:
                candidates.append(
                    {
                        "lane": lane_id,
                        "follower": follower,
                        "leader": leader,
                        "gap_m": gap,
                        "ttc_s": ttc,
                    }
                )

    return candidates


def safety_penalty_for_lanes(traci, controlled_lanes):
    """
    Return a nonnegative risk penalty for candidate conflicts
    on the specified lanes.

    TTC <= 1 second: penalty 2.0
    TTC > 1 and <= 2 seconds: penalty 1.0
    TTC > 2 and <= 3 seconds: penalty 0.25

    These are initial experimental weights, not validated
    safety thresholds.
    """
    controlled_lanes = set(controlled_lanes)
    penalty = 0.0
    risky_events = 0

    for candidate in get_ttc_candidates(traci):
        if candidate["lane"] not in controlled_lanes:
            continue

        ttc = candidate["ttc_s"]

        if ttc <= 1.0:
            penalty += 2.0
            risky_events += 1
        elif ttc <= 2.0:
            penalty += 1.0
            risky_events += 1
        else:
            penalty += 0.25
            risky_events += 1

    return penalty, risky_events
