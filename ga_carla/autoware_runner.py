import os
import sys
import time
import math
import subprocess
from subprocess import Popen, PIPE

import numpy as np
import docker

_PORT_DIR = os.path.dirname(os.path.abspath(__file__))
_DRIVEFUZZ_SRC = os.path.abspath(os.path.join(_PORT_DIR, os.pardir, "src"))
if _DRIVEFUZZ_SRC not in sys.path:
    sys.path.insert(0, _DRIVEFUZZ_SRC)
import config as _drivefuzz_config
_drivefuzz_config.set_carla_api_path()
import carla


def quaternion_from_euler(ai, aj, ak, axes='sxyz'):
    
    _AXES2TUPLE = {
        'sxyz': (0, 0, 0, 0), 'sxyx': (0, 0, 1, 0), 'sxzy': (0, 1, 0, 0),
        'sxzx': (0, 1, 1, 0), 'syzx': (1, 0, 0, 0), 'syzy': (1, 0, 1, 0),
        'syxz': (1, 1, 0, 0), 'syxy': (1, 1, 1, 0), 'szxy': (2, 0, 0, 0),
        'szxz': (2, 0, 1, 0), 'szyx': (2, 1, 0, 0), 'szyz': (2, 1, 1, 0),
        'rzyx': (0, 0, 0, 1), 'rxyx': (0, 0, 1, 1), 'ryzx': (0, 1, 0, 1),
        'rxzx': (0, 1, 1, 1), 'rxzy': (1, 0, 0, 1), 'ryzy': (1, 0, 1, 1),
        'rzxy': (1, 1, 0, 1), 'ryxy': (1, 1, 1, 1), 'ryxz': (2, 0, 0, 1),
        'rzxz': (2, 0, 1, 1), 'rxyz': (2, 1, 0, 1), 'rzyz': (2, 1, 1, 1)}
    _NEXT_AXIS = [1, 2, 0, 1]
    firstaxis, parity, repetition, frame = _AXES2TUPLE[axes.lower()]
    i = firstaxis
    j = _NEXT_AXIS[i+parity]
    k = _NEXT_AXIS[i-parity+1]
    if frame:
        ai, ak = ak, ai
    if parity:
        aj = -aj
    ai /= 2.0; aj /= 2.0; ak /= 2.0
    ci = math.cos(ai); si = math.sin(ai)
    cj = math.cos(aj); sj = math.sin(aj)
    ck = math.cos(ak); sk = math.sin(ak)
    cc = ci*ck; cs = ci*sk; sc = si*ck; ss = si*sk
    q = np.empty((4,), dtype=np.float64)
    if repetition:
        q[i] = cj*(cs + sc); q[j] = sj*(cc + ss)
        q[k] = sj*(cs - sc); q[3] = cj*(cc - ss)
    else:
        q[i] = cj*sc - sj*cs; q[j] = cj*ss + sj*cc
        q[k] = cj*cs - sj*sc; q[3] = cj*cc + sj*ss
    if parity:
        q[j] *= -1
    return q 


DEFAULT_PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))


def make_sp_str(spawn_loc_xyz, spawn_rpy_deg):
    
    x, y, z = spawn_loc_xyz
    roll, pitch, yaw = spawn_rpy_deg
    return "{},{},{},{},{},{}".format(x, y, z, roll, pitch, -yaw)


def launch_autoware_container(town_name, spawn_loc_xyz, spawn_rpy_deg,
                              proj_root=None, container_name=None):
    
    if proj_root is None:
        proj_root = DEFAULT_PROJ_ROOT
    if container_name is None:
        container_name = "autoware-{}".format(os.getenv("USER") or "fuzz")

    sp_str = make_sp_str(spawn_loc_xyz, spawn_rpy_deg)
    autoware_cla = "{} '{}'".format(town_name, sp_str)

    docker_client = docker.from_env()
    xauth = os.path.join(os.getenv("HOME"), ".Xauthority")
    username = os.getenv("USER") or "fuzz"
    vol_dict = {
        "{}/carla-autoware/autoware-contents".format(proj_root): {
            "bind": "/home/autoware/autoware-contents",
            "mode": "ro"
        },
        "/tmp/.X11-unix": {"bind": "/tmp/.X11-unix", "mode": "rw"},
        "/home/{}/.Xauthority".format(username): {"bind": xauth, "mode": "rw"},
        "/tmp/fuzzerdata": {"bind": "/tmp/fuzzerdata", "mode": "rw"},
    }
    env_dict = {
        "DISPLAY": os.getenv("DISPLAY", ":0"),
        "XAUTHORITY": xauth,
    }

    autoware_container = None
    while autoware_container is None:
        try:
            autoware_container = docker_client.containers.run(
                "carla-autoware:improved",
                command=autoware_cla,
                detach=True,
                auto_remove=True,
                name=container_name,
                volumes=vol_dict,
                privileged=True,
                network_mode="host",
                runtime="nvidia",
                environment=env_dict,
            )
        except docker.errors.APIError as e:
            print("[autoware_runner] docker error:", e)
            if "Conflict" in str(e):
                os.system("docker rm -f {}".format(container_name))
            time.sleep(1)
        except Exception as e:
            print("[autoware_runner] fatal:", e)
            raise

    while True:
        running = docker_client.containers.list()
        if autoware_container in running:
            break
        print("[autoware_runner] waiting for container to start...")
        time.sleep(1)

    return autoware_container, autoware_cla


def wait_for_ego_vehicle(world, timeout_s=60.0):
    
    start = time.time()
    i = 0
    while time.time() - start < timeout_s:
        vehicles = world.get_actors().filter("*vehicle.*")
        for v in vehicles:
            if v.attributes.get("role_name") == "ego_vehicle":
                print("[autoware_runner] ego found id={} at {}".format(
                    v.id, v.get_location()))
                return v
        i += 1
        if i % 10 == 0:
            print("[autoware_runner] still waiting for ego_vehicle...")
        time.sleep(0.5)
    return None


def wait_for_vehicle_ready(timeout_s=120.0):
    
    proc = Popen(["rostopic echo /decision_maker/state"],
                 shell=True, stdout=PIPE, stderr=PIPE)
    start = time.time()
    try:
        while time.time() - start < timeout_s:
            line = proc.stdout.readline()
            if not line:
                time.sleep(0.1)
                continue
            if b"VehicleReady" in line:
                return True
        return False
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            proc.kill()


def publish_goal(goal_loc_xyz, goal_yaw_deg):
    
    gx, gy, gz = goal_loc_xyz
    yaw_rad = math.radians(goal_yaw_deg)
    qx, qy, qz, qw = quaternion_from_euler(0.0, 0.0, yaw_rad)

    goal_hdr = "header: {stamp: now, frame_id: 'map'}"
    goal_pose = ("pose: {{position: {{x: {:.6f}, y: {:.6f}, z: 0}}, "
                 "orientation: {{x: {:.6f}, y: {:.6f}, z: {:.6f}, w: {:.6f}}}}}"
                 ).format(gx, -gy, qx, qy, qz, qw)
    goal_msg = "'{" + goal_hdr + ", " + goal_pose + "}'"
    pub_cmd = ("rostopic pub --once /move_base_simple/goal "
               "geometry_msgs/PoseStamped {} > /dev/null".format(goal_msg))
    os.system(pub_cmd)
    print("[autoware_runner] goal published: ({:.2f}, {:.2f}) yaw={:.1f} deg"
          .format(gx, gy, goal_yaw_deg))


def kill_autoware_container(autoware_container, container_name=None):
    
    if container_name is None:
        container_name = "autoware-{}".format(os.getenv("USER") or "fuzz")
    try:
        autoware_container.kill()
    except docker.errors.APIError as e:
        print("[autoware_runner] couldn't kill container:", e)
    except Exception:
        print("[autoware_runner] container teardown anomaly; trying force-rm")
        os.system("docker rm -f {}".format(container_name))

    os.system("rosnode kill /recorder_video_front 2>/dev/null")
    os.system("rosnode kill /recorder_video_rear 2>/dev/null")
    os.system("rosnode kill /recorder_bag 2>/dev/null")


def bring_up_autoware_ego(world, town_name, ego_pos, goal_pos,
                          ego_ready_timeout=60.0, autoware_ready_timeout=120.0,
                          proj_root=None):
    
    spawn_loc = (ego_pos[0], ego_pos[1], ego_pos[2])
    spawn_rpy = (ego_pos[5], ego_pos[3], ego_pos[4])  # roll, pitch, yaw

    autoware_container, autoware_cla = launch_autoware_container(
        town_name, spawn_loc, spawn_rpy, proj_root=proj_root)

    ego = wait_for_ego_vehicle(world, timeout_s=ego_ready_timeout)
    if ego is None:
        kill_autoware_container(autoware_container)
        raise RuntimeError("Ego vehicle never spawned by carla-ros-bridge")

    sp_transform = carla.Transform(
        carla.Location(x=ego_pos[0], y=ego_pos[1], z=ego_pos[2]),
        carla.Rotation(pitch=ego_pos[3], yaw=ego_pos[4], roll=ego_pos[5]))
    world.tick()
    ego.set_transform(sp_transform)
    for _ in range(50):
        world.tick()
        if ego.get_location().distance(sp_transform.location) < 1.0:
            break

    if not wait_for_vehicle_ready(timeout_s=autoware_ready_timeout):
        kill_autoware_container(autoware_container)
        raise RuntimeError("Autoware /decision_maker/state never reached VehicleReady")

    goal_yaw = 0.0
    if len(goal_pos) >= 4:
        goal_yaw = goal_pos[3]
    publish_goal((goal_pos[0], goal_pos[1], goal_pos[2]), goal_yaw)
    time.sleep(1.0)
    world.tick()

    return autoware_container, ego, autoware_cla
