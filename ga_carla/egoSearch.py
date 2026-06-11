import os
import sys
import time
import math
import random
import pickle
import copy
from datetime import datetime
from subprocess import Popen, PIPE

_PORT_DIR = os.path.dirname(os.path.abspath(__file__))
_DRIVEFUZZ_SRC = os.path.abspath(os.path.join(_PORT_DIR, os.pardir, "src"))
if _DRIVEFUZZ_SRC not in sys.path:
    sys.path.insert(0, _DRIVEFUZZ_SRC)
import config as _drivefuzz_config
_drivefuzz_config.set_carla_api_path()
import carla

import util
import autoware_runner
from ga_carla.simulation import _ensure_carla_running


DEFAULT_TOWN = "Town01"
NUM_CANDIDATES = 20 
TIME_SLICES_PER_TRIAL = 30 
TICKS_PER_TIME_SLICE = 20 
FIXED_DELTA_SECONDS = 0.05
EGO_READY_TIMEOUT = 60.0 
AUTOWARE_READY_TIMEOUT = 120.0
EGO_MOTION_TIMEOUT = 60.0 
EGO_MOTION_THRESHOLD_MPS = 0.5 
MIN_DEST_DISTANCE = 50.0 


def setup_world(town):
    ok, ready_client = _ensure_carla_running()
    if not ok:
        raise RuntimeError("CARLA could not be started")
    if ready_client is not None:
        client = ready_client
        client.set_timeout(20.0)
    else:
        client = carla.Client('localhost', 2000)
        client.set_timeout(20.0)
    current = client.get_world()
    if not current.get_map().name.endswith(town):
        client.load_world(town)
        time.sleep(2.0)
    world = client.get_world()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = FIXED_DELTA_SECONDS
    world.apply_settings(settings)
    return client, world


def cleanup_world(world):
    actors = world.get_actors()
    for s in actors.filter('sensor.*'):
        try: s.destroy()
        except Exception: pass
    for v in actors.filter('vehicle.*'):
        try: v.destroy()
        except Exception: pass


def evaluate_candidate(world, town, spawn_ego_tf, dest_ego_tf,
                       container_index=0):
    """
    Launch Autoware with the given spawn/dest, drive for
    TIME_SLICES_PER_TRIAL seconds, return (junction_ids_seen, junction_points).
    On any failure returns (set(), []).
    """
    cleanup_world(world)
    os.system("docker rm -f autoware-{} 2>/dev/null".format(
        os.getenv("USER") or "fuzz"))

    sp_loc = (spawn_ego_tf.location.x, spawn_ego_tf.location.y, spawn_ego_tf.location.z)
    sp_rpy = (spawn_ego_tf.rotation.roll, spawn_ego_tf.rotation.pitch,
              spawn_ego_tf.rotation.yaw)

    container = None
    junction_ids = set()
    junction_points = []
    try:
        container, _ = autoware_runner.launch_autoware_container(town, sp_loc, sp_rpy)

        ego = None
        deadline = time.time() + EGO_READY_TIMEOUT
        while time.time() < deadline:
            world.tick()
            for v in world.get_actors().filter("*vehicle.*"):
                if v.attributes.get("role_name") == "ego_vehicle":
                    ego = v
                    break
            if ego is not None:
                break
            time.sleep(0.2)
        if ego is None:
            print("[egoSearch] candidate {}: ego never spawned".format(container_index))
            return set(), []

        ego.set_transform(spawn_ego_tf)
        for _ in range(50):
            world.tick()
            if ego.get_location().distance(spawn_ego_tf.location) < 1.0:
                break

        proc = Popen(["rostopic echo /decision_maker/state"],
                     shell=True, stdout=PIPE, stderr=PIPE)
        ready = False
        deadline = time.time() + AUTOWARE_READY_TIMEOUT
        try:
            while time.time() < deadline:
                for _ in range(10):
                    world.tick()
                line = proc.stdout.readline()
                if line and b"VehicleReady" in line:
                    ready = True
                    break
                time.sleep(0.1)
        finally:
            proc.terminate()
            try: proc.wait(timeout=2)
            except Exception: proc.kill()
        if not ready:
            print("[egoSearch] candidate {}: Autoware never ready".format(container_index))
            return set(), []

        autoware_runner.publish_goal(
            (dest_ego_tf.location.x, dest_ego_tf.location.y, dest_ego_tf.location.z),
            dest_ego_tf.rotation.yaw)
        time.sleep(1.0)
        world.tick()
        motion_deadline = time.time() + EGO_MOTION_TIMEOUT
        moved = False
        wait_ticks = 0
        while time.time() < motion_deadline:
            world.tick()
            wait_ticks += 1
            if wait_ticks % 10 == 0:
                v = ego.get_velocity()
                speed = math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z)
                if speed > EGO_MOTION_THRESHOLD_MPS:
                    moved = True
                    print("[egoSearch] candidate {}: ego moved after "
                          "{:.1f}s plan lag".format(
                              container_index, wait_ticks * FIXED_DELTA_SECONDS))
                    break
        if not moved:
            print("[egoSearch] candidate {}: ego never moved (plan timeout)"
                  .format(container_index))
            return set(), []
        
        carla_map = world.get_map()
        for t in range(TIME_SLICES_PER_TRIAL):
            for _ in range(TICKS_PER_TIME_SLICE):
                world.tick()
            ego_loc = ego.get_location()
            wp = carla_map.get_waypoint(ego_loc)
            if wp is not None and wp.is_junction:
                jid = wp.get_junction().id
                if jid not in junction_ids:
                    junction_ids.add(jid)
                junction_points.append([ego_loc.x, ego_loc.y, ego_loc.z, t])
            if ego_loc.distance(dest_ego_tf.location) < 5.0:
                print("[egoSearch] candidate {}: reached goal at slice {}".format(
                    container_index, t))
                break

    except Exception as e:
        print("[egoSearch] candidate {}: exception: {}".format(container_index, e))
    finally:
        if container is not None:
            autoware_runner.kill_autoware_container(container)

    return junction_ids, junction_points


def main():
    town = sys.argv[1] if len(sys.argv) >= 2 else DEFAULT_TOWN
    n_candidates = int(sys.argv[2]) if len(sys.argv) >= 3 else NUM_CANDIDATES
    out_path = sys.argv[3] if len(sys.argv) >= 4 else 'ego_path.obj'

    client, world = setup_world(town)
    spawn_points = world.get_map().get_spawn_points()
    if len(spawn_points) < 2:
        print("[egoSearch] town has too few spawn points")
        sys.exit(1)

    TOP_N = 3
    top = []

    for i in range(n_candidates):
        for _ in range(20):
            sp = random.choice(spawn_points)
            dp = random.choice(spawn_points)
            if sp.location.distance(dp.location) >= MIN_DEST_DISTANCE:
                break

        print("\n[egoSearch] candidate {}/{}: sp=({:.1f},{:.1f}) dp=({:.1f},{:.1f})"
              .format(i + 1, n_candidates,
                      sp.location.x, sp.location.y, dp.location.x, dp.location.y))
        jids, jpoints = evaluate_candidate(world, town, sp, dp, container_index=i)
        score = len(jids)
        print("[egoSearch]   junction count = {}".format(score))

        candidate = {
            'score': score,
            'ego_pos': [
                [sp.location.x, sp.location.y, sp.location.z,
                 sp.rotation.pitch, sp.rotation.yaw, sp.rotation.roll],
                [dp.location.x, dp.location.y, dp.location.z,
                 dp.rotation.yaw],   # 4-tuple so we can publish goal yaw
            ],
            'junction_point': copy.deepcopy(jpoints),
        }

        if len(top) < TOP_N:
            top.append(candidate)
            top.sort(key=lambda c: c['score'], reverse=True)
            print("[egoSearch]   added to top-{} (now have {} candidates, "
                  "scores={})".format(TOP_N, len(top),
                                      [c['score'] for c in top]))
        elif score > top[-1]['score']:
            top[-1] = candidate
            top.sort(key=lambda c: c['score'], reverse=True)
            print("[egoSearch]   *** displaced worst of top-{}; scores={} ***"
                  .format(TOP_N, [c['score'] for c in top]))

    if not top or top[0]['score'] == 0:
        print("[egoSearch] no candidate produced any junctions; falling back "
              "to first random pair")
        sp = spawn_points[0]; dp = spawn_points[1]
        chosen = {
            'score': 0,
            'ego_pos': [
                [sp.location.x, sp.location.y, sp.location.z,
                 sp.rotation.pitch, sp.rotation.yaw, sp.rotation.roll],
                [dp.location.x, dp.location.y, dp.location.z, dp.rotation.yaw],
            ],
            'junction_point': [],
        }
    else:
        chosen = random.choice(top)
        print("\n[egoSearch] top-{} scores: {}".format(
            TOP_N, [c['score'] for c in top]))
        print("[egoSearch] randomly chose seed with score {}".format(chosen['score']))

    if os.path.isfile(out_path):
        os.remove(out_path)
    with open(out_path, 'wb') as f:
        pickle.dump(chosen, f)
    print("\n[egoSearch] saved seed (score={}) to {}".format(chosen['score'], out_path))

    settings = world.get_settings()
    settings.synchronous_mode = False
    world.apply_settings(settings)


if __name__ == "__main__":
    main()
