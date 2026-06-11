import os
import sys
import time
import math
import random
import pickle
import copy
from datetime import datetime

_PORT_DIR = os.path.dirname(os.path.abspath(__file__))
_DRIVEFUZZ_SRC = os.path.abspath(os.path.join(_PORT_DIR, os.pardir, "src"))
if _DRIVEFUZZ_SRC not in sys.path:
    sys.path.insert(0, _DRIVEFUZZ_SRC)
import config as _drivefuzz_config
_drivefuzz_config.set_carla_api_path()
import carla

import util
import autoware_runner
from carla_speed_helper import set_npc_target_speed_kmh


import socket
import subprocess

_CARLA_RPC_PORT = 2000
_CARLA_BOOT_TIMEOUT_S = 90.0


def _ensure_carla_running():

    name = "carla-{}".format(os.environ.get("USER", "fuzz"))

    def port_open():
        try:
            with socket.create_connection(("localhost", _CARLA_RPC_PORT),
                                          timeout=2.0):
                return True
        except (socket.timeout, OSError):
            return False

    def container_running():
        try:
            r = subprocess.run(
                ["docker", "inspect", "-f", "{{.State.Running}}", name],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            return r.returncode == 0 and r.stdout.strip() == b"true"
        except Exception:
            return False

    def container_logs(tail=40):
        try:
            r = subprocess.run(
                ["docker", "logs", "--tail", str(tail), name],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                timeout=5)
            return r.stdout.decode("utf-8", errors="replace")
        except Exception:
            return "(could not fetch logs)"

    def carla_rpc_works():
        """Real RPC, not just TCP probe. Returns the client on success."""
        try:
            c = carla.Client("localhost", _CARLA_RPC_PORT)
            c.set_timeout(60.0)
            w = c.get_world()
            _ = w.get_map().name
            return c
        except Exception:
            return None

    if container_running() and port_open():
        return True, None

    print("[sim] CARLA not healthy; restarting via run_carla.sh")

    subprocess.run(["docker", "rm", "-f", name],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(5.0)

    here = os.path.dirname(os.path.abspath(__file__))
    run_script = os.path.abspath(os.path.join(here, os.pardir, "run_carla.sh"))
    if not os.path.isfile(run_script):
        print("[sim] FATAL: run_carla.sh not found at {}".format(run_script))
        return False, None

    env = os.environ.copy()
    if not env.get("USER"):
        env["USER"] = "fuzz"

    proc = subprocess.run(["bash", run_script], env=env,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        print("[sim] run_carla.sh failed: rc={}, stderr={}".format(
            proc.returncode, proc.stderr.decode("utf-8", errors="replace")))
        return False, None

    time.sleep(3.0)
    if not container_running():
        print("[sim] container exited shortly after start; CARLA logs:\n{}"
              .format(container_logs()))
        return False, None

    deadline = time.time() + _CARLA_BOOT_TIMEOUT_S
    last_state = "no port"
    while time.time() < deadline:
        if not container_running():
            print("[sim] container died during boot; CARLA logs:\n{}"
                  .format(container_logs()))
            return False, None
        if not port_open():
            time.sleep(1.0)
            continue
        if last_state != "port_open":
            print("[sim] port open; testing RPC...")
            last_state = "port_open"
        client = carla_rpc_works()
        if client is not None:
            print("[sim] CARLA is ready")
            return True, client
        time.sleep(2.0)

    print("[sim] CARLA did not become RPC-responsive within {} s; logs:\n{}"
          .format(_CARLA_BOOT_TIMEOUT_S, container_logs()))
    return False, None

AUTOWARE_BRINGUP_MAX_ATTEMPTS = 3


DEFAULT_TOWN = "Town03"
FIXED_DELTA_SECONDS = 0.05      # 20 FPS in sync mode -- matches DriveFuzz
TICKS_PER_TIME_SLICE = 20       # 1 second per time slice (20 * 0.05 = 1.0)
EGO_BLUEPRINT = "vehicle.toyota.prius"  # what carla-autoware expects
AUTOWARE_PLAN_TIMEOUT_S = 60.0  # max wall-clock seconds to wait after goal
                                # publication for ego to start moving. If
                                # exceeded, the scenario is treated as a
                                # planning failure (fault=None, error flag).

NPC_DEADLOCK_GRACE_S = 3.0

NPC_SPAWN_Z_LIFT_M = 0.5

JUNCTION_POOL_PROB = 0.7


PP_LOOKAHEAD_M = 5.0
PP_THROTTLE_KP = 0.5
PP_BRAKE_KP = 0.5
PP_MAX_THROTTLE = 1.0
PP_MAX_BRAKE = 0.6 
PP_STEER_GAIN = 1.0
PP_QUEUE_TARGET = 8
JUNCTION_TURN_THRESHOLD_DEG = 30.0

LANE_CHANGE_FORWARD_M = 6.0

NPC_PERCEPTION_WARMUP_TICKS = 40


def _yaw_to_unit_vector(yaw_deg):

    rad = math.radians(yaw_deg)
    return (math.cos(rad), math.sin(rad))


def _signed_angle_deg(v_from, v_to):

    fx, fy = v_from
    tx, ty = v_to
    dot = fx * tx + fy * ty
    cross_z = fx * ty - fy * tx 
    return math.degrees(math.atan2(cross_z, dot))


def _classify_action(angle_deg):
    
    if abs(angle_deg) <= JUNCTION_TURN_THRESHOLD_DEG:
        return "S"
    return "R" if angle_deg > 0 else "L"


class LaneKeepController:
    
    def __init__(self, actor, carla_map, action_list):
        self.actor = actor
        self.map = carla_map
        self.actions = list(action_list)
        self.queue = []
        self.target_speed_mps = 0.0
        self._was_in_junction = False
        self._tick = 0
        self._seed_queue()

    def set_target_speed_kmh(self, speed_kmh):
        self.target_speed_mps = speed_kmh / 3.6

    def request_lane_change(self, direction):
        
        if direction not in (1, 2):
            return
        loc = self.actor.get_location()
        wp = self.map.get_waypoint(loc, project_to_road=True)
        if wp is None:
            return
        target_lane_wp = wp.get_left_lane() if direction == 1 else wp.get_right_lane()
        if target_lane_wp is None:
            return
        if target_lane_wp.lane_type != carla.LaneType.Driving:
            return
        if (wp.lane_id * target_lane_wp.lane_id) < 0:
            return
        try:
            forward_candidates = target_lane_wp.next(LANE_CHANGE_FORWARD_M)
        except Exception:
            forward_candidates = None
        if forward_candidates:
            seed_wp = forward_candidates[0]
        else:
            seed_wp = target_lane_wp
        self.queue = []
        self._extend_queue(start_wp=seed_wp)

    def _seed_queue(self):
        loc = self.actor.get_location()
        wp = self.map.get_waypoint(loc, project_to_road=True)
        if wp is None:
            return
        self._extend_queue(start_wp=wp)

    def _extend_queue(self, start_wp=None, target_size=PP_QUEUE_TARGET):
        
        if start_wp is None:
            if not self.queue:
                return
            current = self.queue[-1]
        else:
            current = start_wp
            self.queue.append(current)

        in_chosen_junction = bool(current.is_junction)

        while len(self.queue) < target_size:
            try:
                nexts = current.next(2.0)
            except Exception:
                break
            if not nexts:
                break
            if len(nexts) > 1:
                if in_chosen_junction:
                    chosen = nexts[0]
                else:
                    chosen = self._pick_junction_exit(current, nexts)
                    in_chosen_junction = True
            else:
                chosen = nexts[0]
                if in_chosen_junction and not chosen.is_junction:
                    in_chosen_junction = False
            self.queue.append(chosen)
            current = chosen

    def _pick_junction_exit(self, current, candidates):
        
        if self.actions:
            desired = self.actions.pop(0)
        else:
            desired = "S"

        cur_yaw = current.transform.rotation.yaw
        cur_vec = _yaw_to_unit_vector(cur_yaw)

        classified = []
        for c in candidates:
            cand_vec = _yaw_to_unit_vector(c.transform.rotation.yaw)
            angle = _signed_angle_deg(cur_vec, cand_vec)
            classified.append((c, angle, _classify_action(angle)))

        matched = [(c, a) for (c, a, b) in classified if b == desired]
        if matched:
            if desired == "S":
                matched.sort(key=lambda ca: abs(ca[1]))
            else:
                matched.sort(key=lambda ca: -abs(ca[1]))
            return matched[0][0]

        classified.sort(key=lambda x: abs(x[1]))
        return classified[0][0]

    def tick(self):
        
        self._tick += 1
        loc = self.actor.get_location()

        wp = self.map.get_waypoint(loc, project_to_road=True)
        if wp is not None:
            in_junction = wp.is_junction
            self._was_in_junction = in_junction

        actor_yaw_now = self.actor.get_transform().rotation.yaw
        actor_fwd = _yaw_to_unit_vector(actor_yaw_now)
        while self.queue:
            head = self.queue[0].transform.location
            dx = head.x - loc.x
            dy = head.y - loc.y
            close = (dx * dx + dy * dy) < (1.5 * 1.5)
            behind = (dx * actor_fwd[0] + dy * actor_fwd[1]) < 0.0
            if close or behind:
                self.queue.pop(0)
            else:
                break
        if len(self.queue) < PP_QUEUE_TARGET // 2:
            self._extend_queue()
        if not self.queue:
            self._seed_queue()

        if not self.queue:
            self.actor.apply_control(carla.VehicleControl(
                throttle=0.0, brake=PP_MAX_BRAKE, steer=0.0))
            return

        accumulated = 0.0
        target_wp = self.queue[0]
        prev = loc
        for wp in self.queue:
            accumulated += prev.distance(wp.transform.location)
            target_wp = wp
            if accumulated >= PP_LOOKAHEAD_M:
                break
            prev = wp.transform.location

        actor_yaw = self.actor.get_transform().rotation.yaw
        actor_vec = _yaw_to_unit_vector(actor_yaw)
        to_target = (target_wp.transform.location.x - loc.x,
                     target_wp.transform.location.y - loc.y)
        
        norm = math.sqrt(to_target[0] ** 2 + to_target[1] ** 2)
        if norm < 1e-3:
            steer = 0.0
            angle = 0.0
        else:
            to_target = (to_target[0] / norm, to_target[1] / norm)
            angle = _signed_angle_deg(actor_vec, to_target)
            steer = max(-1.0, min(1.0, (angle / 90.0) * PP_STEER_GAIN))

        v = self.actor.get_velocity()
        cur_speed = math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2)
        err = self.target_speed_mps - cur_speed
        if err > 0:
            throttle = max(0.0, min(PP_MAX_THROTTLE, err * PP_THROTTLE_KP))
            brake = 0.0
        else:
            throttle = 0.0
            brake = max(0.0, min(PP_MAX_BRAKE, -err * PP_BRAKE_KP))

        self.actor.apply_control(carla.VehicleControl(
            throttle=throttle, brake=brake, steer=steer))


class DriveFuzzAutowareSimulation:

    def __init__(self, num_time_slice, num_npc, ego_info, npc_spawn,
                 routes, is_init, town=DEFAULT_TOWN):
        self.world = None
        self.client = None
        self.tm = None
        self.ego = None
        self.autoware_container = None

        self.is_hit = False
        self.is_ego_fault = False
        self.collision_kind = None
        self.npc_list = []
        self.npc_indices = []
        self.npc_controllers = []
        self._npcs_activated = False
        self.detector_collision = None
        self.spectator = None
        self.scenario_obj = None

        self.num_time_slice = num_time_slice
        self.num_npc = num_npc
        self.is_init = is_init
        self.town = town

        self.ego_pos = ego_info['ego_pos']
        self.junction_point = ego_info['junction_point']
        self.npc_spawn_list = npc_spawn['pos']
        self.other_npc = npc_spawn['other']

        self.routes = routes

        self.scenario_pos = [[[] for _ in range(num_time_slice)]
                             for _ in range(num_npc + 1)]
        self.result_dic = {}

        self._init_simulator()
        self._init_ego_via_autoware()


    def _init_simulator(self):

        ok, ready_client = _ensure_carla_running()
        if not ok:
            raise RuntimeError("CARLA is unavailable; cannot proceed")

        if ready_client is not None:
            self.client = ready_client
            self.client.set_timeout(20.0)
        else:
            self.client = carla.Client('localhost', 2000)
            self.client.set_timeout(20.0)

        current_world = self.client.get_world()
        current_map = current_world.get_map().name
        if not current_map.endswith(self.town):
            util.print_debug("[sim] loading {} (was {})".format(self.town, current_map))
            self.client.load_world(self.town)
            time.sleep(2.0)

        self.world = self.client.get_world()
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = FIXED_DELTA_SECONDS
        self.world.apply_settings(settings)

        self.tm = self.client.get_trafficmanager()
        self.tm.set_synchronous_mode(True)

        self._cleanup_existing_actors()

    def _cleanup_existing_actors(self):
        actors = self.world.get_actors()
        for sensor in actors.filter('sensor.*'):
            try:
                sensor.destroy()
            except Exception:
                pass
        for vehicle in actors.filter('vehicle.*'):
            try:
                vehicle.destroy()
            except Exception:
                pass
        os.system("docker rm -f autoware-{} 2>/dev/null".format(
            os.getenv("USER") or "fuzz"))

    def _init_ego_via_autoware(self):

        last_err = None
        for attempt in range(1, AUTOWARE_BRINGUP_MAX_ATTEMPTS + 1):
            try:
                self._init_ego_via_autoware_once()
                return
            except RuntimeError as e:
                last_err = e
                util.print_debug("[sim] Autoware bring-up attempt {}/{} failed: {}"
                                 .format(attempt, AUTOWARE_BRINGUP_MAX_ATTEMPTS, e))
                self._kill_autoware()
                time.sleep(8.0)
        raise RuntimeError("Autoware bring-up exhausted retries: {}".format(last_err))

    def _init_ego_via_autoware_once(self):
        """
        One Autoware bring-up attempt. Raises RuntimeError on failure
        (no ego, VehicleReady never reached). Replaces simulation_a.initEV().
        """
        # ego_pos[0] = [x, y, z, pitch, yaw, roll]
        # ego_pos[1] = [x, y, z]  (goal)
        ego_sp = self.ego_pos[0]
        ego_dp = self.ego_pos[1]

        spawn_loc = (ego_sp[0], ego_sp[1], ego_sp[2])
        spawn_rpy = (ego_sp[5], ego_sp[3], ego_sp[4])  # roll, pitch, yaw

        self.autoware_container, _ = autoware_runner.launch_autoware_container(
            self.town, spawn_loc, spawn_rpy)

        ego = None
        deadline = time.time() + 90.0
        while time.time() < deadline:
            self.world.tick()
            vehicles = self.world.get_actors().filter("*vehicle.*")
            for v in vehicles:
                if v.attributes.get("role_name") == "ego_vehicle":
                    ego = v
                    break
            if ego is not None:
                break
            time.sleep(0.2)
        if ego is None:
            self._kill_autoware()
            raise RuntimeError("ego_vehicle never spawned by carla-ros-bridge")

        sp_transform = carla.Transform(
            carla.Location(x=ego_sp[0], y=ego_sp[1], z=ego_sp[2]),
            carla.Rotation(pitch=ego_sp[3], yaw=ego_sp[4], roll=ego_sp[5]))
        ego.set_transform(sp_transform)
        for _ in range(50):
            self.world.tick()
            if ego.get_location().distance(sp_transform.location) < 1.0:
                break
        self.ego = ego

        from subprocess import Popen, PIPE
        proc = Popen(["rostopic echo /decision_maker/state"],
                     shell=True, stdout=PIPE, stderr=PIPE)
        ready = False
        deadline = time.time() + 180.0
        try:
            while time.time() < deadline:
                for _ in range(10):
                    self.world.tick()
                line = proc.stdout.readline()
                if line and b"VehicleReady" in line:
                    ready = True
                    break
                time.sleep(0.1)
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=2)
            except Exception:
                proc.kill()
        if not ready:
            self._kill_autoware()
            raise RuntimeError("Autoware never reached VehicleReady")

        self.world.tick()

        loc = sp_transform.location
        rot = sp_transform.rotation
        self.scenario_pos[0][0] = [loc.x, loc.y, loc.z, rot.pitch, rot.yaw, rot.roll]

        bp_lib = self.world.get_blueprint_library()
        coll_bp = bp_lib.find('sensor.other.collision')
        self.detector_collision = self.world.spawn_actor(
            coll_bp, carla.Transform(), attach_to=self.ego)
        self.detector_collision.listen(self._on_collision)

        self.spectator = self.world.get_spectator()
        self.spectator.set_transform(carla.Transform(
            sp_transform.location + carla.Location(x=0, y=0, z=80),
            carla.Rotation(pitch=270)))


    def _on_collision(self, event):
        if self.is_hit:
            return  # only register the first contact
        other = event.other_actor

        if other is None or other.type_id == "static.road":
            return

        self.is_hit = True

        tid = other.type_id or ""
        parts = tid.split(".")
        if tid.startswith("vehicle."):
            kind = "vehicle"
        elif tid.startswith("walker."):
            kind = "walker"
        elif tid.startswith("static.") and len(parts) >= 2:
            kind = parts[1]
        elif parts:
            kind = parts[0]
        else:
            kind = "unknown"
        self.collision_kind = kind

        util.print_debug("***** Collision detected (against {}) *****"
                         .format(other.type_id))

        ego_velocity = self.ego.get_velocity()
        other_velocity = other.get_velocity()

        ev = ego_velocity
        ego_speed = math.sqrt(ev.x * ev.x + ev.y * ev.y + ev.z * ev.z)
        if other_velocity is not None:
            ov = other_velocity
            other_speed = math.sqrt(ov.x * ov.x + ov.y * ov.y + ov.z * ov.z)
        else:
            other_speed = 0.0
        relative_velocity = ego_speed - other_speed

        self.is_ego_fault = (relative_velocity > 0)


    def _try_spawn_at(self, transform):
        
        bp_lib = self.world.get_blueprint_library()
        vehicle_bp = random.choice(bp_lib.filter('vehicle.*.*'))
        lifted = carla.Transform(
            carla.Location(x=transform.location.x, y=transform.location.y,
                           z=transform.location.z + NPC_SPAWN_Z_LIFT_M),
            transform.rotation)
        npc = self.world.try_spawn_actor(vehicle_bp, lifted)
        if npc is None:
            return None, None
        return npc, lifted

    def _resample_from_pool(self, pool, used_keys, max_tries=20):
        
        candidates = [s for s in pool
                      if (s[0], s[1], s[2]) not in used_keys]
        random.shuffle(candidates)
        for s in candidates[:max_tries]:
            tf = carla.Transform(
                carla.Location(x=s[0], y=s[1], z=s[2]),
                carla.Rotation(pitch=s[3], yaw=s[4], roll=s[5]))
            npc, lifted = self._try_spawn_at(tf)
            if npc is not None:
                return npc, lifted, (s[0], s[1], s[2])
        return None, None, None

    def _add_npc_junction(self, scenario_npc, npc_spawn, first_flag, num,
                          used_spawn_keys):
        
        if first_flag:
            target = carla.Transform(
                carla.Location(x=npc_spawn[0], y=npc_spawn[1], z=npc_spawn[2]),
                carla.Rotation(pitch=npc_spawn[3], yaw=npc_spawn[4], roll=npc_spawn[5]))
            target_key = (npc_spawn[0], npc_spawn[1], npc_spawn[2])
        else:
            target = carla.Transform(
                carla.Location(x=scenario_npc[0][2], y=scenario_npc[0][3], z=scenario_npc[0][4]),
                carla.Rotation(pitch=scenario_npc[0][5], yaw=scenario_npc[0][6], roll=scenario_npc[0][7]))
            target_key = None

        npc, npc_pos = self._try_spawn_at(target)

        if npc is None and first_flag:
            npc, npc_pos, replacement_key = self._resample_from_pool(
                self.npc_spawn_list, used_spawn_keys)
            if npc is None:
                util.print_debug("[sim] junction NPC slot {}: junction "
                                 "set exhausted, falling back to other set"
                                 .format(num))
                npc, npc_pos, replacement_key = self._resample_from_pool(
                    self.other_npc, used_spawn_keys)
            if replacement_key is not None:
                used_spawn_keys.add(replacement_key)
        else:
            if first_flag and target_key is not None:
                used_spawn_keys.add(target_key)

        if npc is None:
            util.print_debug("[sim] WARN: junction NPC slot {} could not "
                             "spawn after resampling".format(num))
            return

        npc.set_simulate_physics(False)

        loc, rot = npc_pos.location, npc_pos.rotation
        self.scenario_pos[num + 1][0] = [loc.x, loc.y, loc.z, rot.pitch, rot.yaw, rot.roll]

        self.npc_list.append(npc)
        self.npc_indices.append(num)

    def _add_npc_other(self, scenario_npc, npc_spawn, first_flag, num,
                       used_spawn_keys):
        """
        Spawn an NPC intended for an 'other' slot. On collision, resample
        from the same 'other' pool only. Routing is handled later by the
        LaneKeepController.
        """
        if first_flag:
            target = carla.Transform(
                carla.Location(x=npc_spawn[0], y=npc_spawn[1], z=npc_spawn[2]),
                carla.Rotation(pitch=npc_spawn[3], yaw=npc_spawn[4], roll=npc_spawn[5]))
            target_key = (npc_spawn[0], npc_spawn[1], npc_spawn[2])
        else:
            target = carla.Transform(
                carla.Location(x=scenario_npc[0][2], y=scenario_npc[0][3], z=scenario_npc[0][4]),
                carla.Rotation(pitch=scenario_npc[0][5], yaw=scenario_npc[0][6], roll=scenario_npc[0][7]))
            target_key = None

        npc, npc_pos = self._try_spawn_at(target)

        if npc is None and first_flag:
            npc, npc_pos, replacement_key = self._resample_from_pool(
                self.other_npc, used_spawn_keys)
            if replacement_key is not None:
                used_spawn_keys.add(replacement_key)
        else:
            if first_flag and target_key is not None:
                used_spawn_keys.add(target_key)

        if npc is None:
            util.print_debug("[sim] WARN: 'other' NPC slot {} could not "
                             "spawn after resampling".format(num))
            return

        npc.set_simulate_physics(False)

        loc, rot = npc_pos.location, npc_pos.rotation
        self.scenario_pos[num + 1][0] = [loc.x, loc.y, loc.z, rot.pitch, rot.yaw, rot.roll]

        self.npc_list.append(npc)
        self.npc_indices.append(num)


    def run_simulation(self, scenario_obj):
        now = datetime.now()
        date_time = now.strftime("%m-%d-%Y-%H-%M-%S")
        util.print_debug("\n=== Run simulation [{}] ===".format(date_time))

        self.scenario_obj = scenario_obj

        self.record_date_time = date_time
        self.container_record_path = "/home/carla/recordings/recording_{}.log".format(date_time)
        _port_dir = os.path.dirname(os.path.abspath(__file__))
        self.host_record_path = os.path.join(
            _port_dir, "recordings", "recording_{}.log".format(date_time))
        try:
            self.client.start_recorder(self.container_record_path)
        except Exception as e:
            util.print_debug("[sim] start_recorder warn: {}".format(e))

        if len(self.other_npc) < self.num_npc:
            npc_first_spawn = random.sample(self.other_npc, len(self.other_npc))
        else:
            npc_first_spawn = random.sample(self.other_npc, self.num_npc)

        if len(self.npc_spawn_list) < self.num_npc:
            npc_junction = random.sample(self.npc_spawn_list, len(self.npc_spawn_list))
        else:
            npc_junction = random.sample(self.npc_spawn_list, self.num_npc)

        used_spawn_keys = set()

        if self.is_init:
            used = 0
            for n in range(self.num_npc):
                p = random.random()
                if p <= JUNCTION_POOL_PROB and used < len(npc_junction):
                    self._add_npc_junction(scenario_obj[n], npc_junction[used], True, n,
                                           used_spawn_keys)
                    used += 1
                else:
                    if n < len(npc_first_spawn):
                        self._add_npc_other(scenario_obj[n], npc_first_spawn[n], True, n,
                                            used_spawn_keys)
                    else:
                        util.print_debug("[sim] WARN: ran out of spawn points, skipping npc {}".format(n))
        else:
            for n in range(self.num_npc):
                if n < len(npc_first_spawn):
                    self._add_npc_other(scenario_obj[n], npc_first_spawn[n], False, n,
                                        used_spawn_keys)

        self.is_hit = False
        self.is_ego_fault = False
        self.collision_kind = None

        for _ in range(NPC_PERCEPTION_WARMUP_TICKS):
            self.world.tick()

        ego_dp = self.ego_pos[1]
        ego_sp = self.ego_pos[0]
        goal_yaw = ego_dp[3] if len(ego_dp) >= 4 else ego_sp[4]
        autoware_runner.publish_goal((ego_dp[0], ego_dp[1], ego_dp[2]), goal_yaw)
        time.sleep(1.0)
        self.world.tick()

        ok = self._wait_for_ego_motion(threshold_mps=0.5,
                                       timeout_s=AUTOWARE_PLAN_TIMEOUT_S)
        if not ok:
            util.print_debug("[sim] ego never moved within {}s of goal -- "
                             "treating as planning failure".format(
                                 AUTOWARE_PLAN_TIMEOUT_S))
            self.result_dic['pos'] = self.scenario_pos
            self.result_dic['time'] = date_time
            self.result_dic['fault'] = None
            self.result_dic['error'] = 'autoware_plan_timeout'
            self._teardown()
            return self.result_dic


        self._activate_npcs()
        for t in range(int(self.num_time_slice)):
            if t != 0:
                ego_loc = self.ego.get_location()
                self.scenario_pos[0][t] = [ego_loc.x, ego_loc.y, ego_loc.z]

            for k, ctrl in enumerate(self.npc_controllers):
                row = self.npc_indices[k]
                ctrl.set_target_speed_kmh(scenario_obj[row][t][0])
                turn_command = scenario_obj[row][t][1]
                # 0 = none, 1 = left, 2 = right
                if turn_command in (1, 2):
                    ctrl.request_lane_change(turn_command)

                if t != 0:
                    npc_loc = self.npc_list[k].get_location()
                    self.scenario_pos[row + 1][t] = [npc_loc.x, npc_loc.y, npc_loc.z]

            for _ in range(TICKS_PER_TIME_SLICE):
                for ctrl in self.npc_controllers:
                    try:
                        ctrl.tick()
                    except Exception as e:
                        util.print_debug("[sim] controller tick warn: {}".format(e))
                ego_position = self.ego.get_location()
                self.spectator.set_transform(carla.Transform(
                    ego_position + carla.Location(x=0, y=0, z=80),
                    carla.Rotation(pitch=270)))
                self.world.tick()
                if self.is_hit:
                    break
            if self.is_hit:
                break

        self.result_dic['pos'] = self.scenario_pos
        self.result_dic['time'] = date_time
        if self.is_hit:
            self.result_dic['fault'] = 'ego' if self.is_ego_fault else 'npc'
        else:
            self.result_dic['fault'] = None
        self.result_dic['collision_kind'] = self.collision_kind

        self._teardown()
        return self.result_dic


    def _wait_for_ego_motion(self, threshold_mps=0.5, timeout_s=60.0,
                             npc_grace_s=NPC_DEADLOCK_GRACE_S):
        
        deadline = time.time() + timeout_s
        npc_activation_deadline = time.time() + npc_grace_s
        ticks = 0
        check_every = 10
        ego_started_at_tick = None
        while time.time() < deadline:
            for ctrl in self.npc_controllers:
                try:
                    ctrl.tick()
                except Exception as e:
                    util.print_debug("[sim] wait-loop ctrl tick warn: {}"
                                     .format(e))
            self.world.tick()
            ticks += 1
            try:
                ego_loc = self.ego.get_location()
                self.spectator.set_transform(carla.Transform(
                    ego_loc + carla.Location(x=0, y=0, z=80),
                    carla.Rotation(pitch=270)))
            except Exception:
                pass
            if (not self._npcs_activated
                    and time.time() > npc_activation_deadline):
                util.print_debug(
                    "[sim] {:.1f}s grace elapsed; activating NPCs".format(
                        npc_grace_s))
                self._activate_npcs()
            if ticks % check_every == 0:
                v = self.ego.get_velocity()
                speed = math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z)
                if speed > threshold_mps:
                    if ego_started_at_tick is None:
                        ego_started_at_tick = ticks
                    if self._npcs_activated:
                        util.print_debug(
                            "[sim] ego started moving after {} ticks "
                            "({:.1f}s sim time); NPCs already active".format(
                                ego_started_at_tick,
                                ego_started_at_tick * FIXED_DELTA_SECONDS))
                        return True
        return False

    def _activate_npcs(self):
        
        if self._npcs_activated:
            return
        self._npcs_activated = True
        carla_map = self.world.get_map()
        for k, npc in enumerate(self.npc_list):
            try:
                npc.set_simulate_physics(True)
                row = self.npc_indices[k]
                action_list = self.routes[row] if row < len(self.routes) else []
                ctrl = LaneKeepController(npc, carla_map, action_list)
                try:
                    if (self.scenario_obj is not None
                            and row < len(self.scenario_obj)
                            and len(self.scenario_obj[row]) > 0):
                        ctrl.set_target_speed_kmh(self.scenario_obj[row][0][0])
                except Exception:
                    pass
                self.npc_controllers.append(ctrl)
            except Exception as e:
                util.print_debug("[sim] _activate_npcs warn: {}".format(e))


    def _kill_autoware(self):
        if self.autoware_container is not None:
            autoware_runner.kill_autoware_container(self.autoware_container)
            self.autoware_container = None

    def _teardown(self):
        util.print_debug("[sim] tearing down: {} NPCs, ego, sensor, container"
                         .format(len(self.npc_list)))
        try:
            if self.client is not None:
                self.client.stop_recorder()
        except Exception:
            pass
        error_set = isinstance(self.result_dic, dict) and self.result_dic.get('error') is not None
        host_path = getattr(self, "host_record_path", None)
        if error_set and host_path is not None:
            try:
                if os.path.isfile(host_path):
                    os.remove(host_path)
                    util.print_debug("[sim] deleted recording for failed "
                                     "scenario: {}".format(
                                         os.path.basename(host_path)))
            except Exception as e:
                util.print_debug("[sim] could not delete failed recording: {}"
                                 .format(e))
        if (not error_set and self.collision_kind is not None
                and host_path is not None):
            try:
                if os.path.isfile(host_path):
                    dirpath = os.path.dirname(host_path)
                    new_name = "recording_{}_{}.log".format(
                        self.collision_kind, self.record_date_time)
                    new_path = os.path.join(dirpath, new_name)
                    os.rename(host_path, new_path)
                    util.print_debug("[sim] renamed recording: {} -> {}"
                                     .format(os.path.basename(host_path),
                                             new_name))
                    self.host_record_path = new_path
            except Exception as e:
                util.print_debug("[sim] could not rename recording: {}"
                                 .format(e))
        for npc in self.npc_list:
            try:
                if npc.is_alive:
                    npc.destroy()
            except Exception:
                pass
        try:
            if self.detector_collision is not None and self.detector_collision.is_alive:
                self.detector_collision.destroy()
        except Exception:
            pass
        try:
            if self.ego is not None and self.ego.is_alive:
                self.ego.destroy()
        except Exception:
            pass
        self._kill_autoware()


def main():
    if len(sys.argv) < 9:
        print("usage: simulation_drivefuzz.py SCENARIO RES EGO NPCSPAWN ROUTES NUMNPC NUMSLICE ISINIT [TOWN]")
        sys.exit(1)

    scenario_path = sys.argv[1]
    result_path   = sys.argv[2]
    ego_path_obj  = sys.argv[3]
    npc_path_obj  = sys.argv[4]
    routes_path   = sys.argv[5]
    num_npc       = int(sys.argv[6])
    num_time_slice = int(sys.argv[7])
    is_init       = (sys.argv[8] == "True")
    town          = sys.argv[9] if len(sys.argv) >= 10 else DEFAULT_TOWN

    with open(scenario_path, 'rb') as f:
        scenario_obj = pickle.load(f)
    with open(ego_path_obj, 'rb') as f:
        ego_info = pickle.load(f)
    with open(npc_path_obj, 'rb') as f:
        npc_spawn = pickle.load(f)
    with open(routes_path, 'rb') as f:
        routes_obj = pickle.load(f)

    sim = None
    result_dic = {'pos': None, 'fault': None, 'time': ''}
    try:
        sim = DriveFuzzAutowareSimulation(
            num_time_slice=num_time_slice, num_npc=num_npc,
            ego_info=ego_info, npc_spawn=npc_spawn,
            routes=routes_obj,
            is_init=is_init, town=town)
        result_dic = sim.run_simulation(scenario_obj)
    except Exception as e:
        util.print_debug("[sim] FATAL: {}".format(e))
        if sim is not None:
            try:
                if isinstance(sim.result_dic, dict):
                    sim.result_dic['error'] = str(e)
                    if 'pos' not in sim.result_dic:
                        sim.result_dic['pos'] = None
            except Exception:
                pass
            try:
                sim._teardown()
            except Exception:
                pass
        result_dic = {'pos': None, 'fault': None, 'time': '',
                      'error': str(e)}

    if os.path.isfile(result_path):
        os.remove(result_path)
    with open(result_path, 'wb') as f:
        pickle.dump(result_dic, f)


if __name__ == "__main__":
    main()
