DEFAULT_SPEED_LIMIT_KMH = 30.0


def set_npc_target_speed_kmh(tm, npc, target_speed_kmh):
    
    speed_limit = npc.get_speed_limit()  # km/h on the road the actor is on
    if speed_limit is None or speed_limit < 1.0:
        speed_limit = DEFAULT_SPEED_LIMIT_KMH

    diff_pct = (speed_limit - float(target_speed_kmh)) / speed_limit * 100.0

    if diff_pct > 100.0:
        diff_pct = 100.0
    elif diff_pct < -100.0:
        diff_pct = -100.0

    tm.vehicle_percentage_speed_difference(npc, diff_pct)


def set_npc_target_speed_mps(tm, npc, target_speed_mps):
    
    set_npc_target_speed_kmh(tm, npc, target_speed_mps * 3.6)
