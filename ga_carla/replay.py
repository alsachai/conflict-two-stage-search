import os
import sys
import glob
import time
import math
import re
import argparse

_PORT_DIR = os.path.dirname(os.path.abspath(__file__))
_DRIVEFUZZ_SRC = os.path.abspath(os.path.join(_PORT_DIR, os.pardir, "src"))
if _DRIVEFUZZ_SRC not in sys.path:
    sys.path.insert(0, _DRIVEFUZZ_SRC)
import config as _drivefuzz_config
_drivefuzz_config.set_carla_api_path()
import carla


HOST_RECORDINGS_DIR = os.path.join(_PORT_DIR, "recordings")
CONTAINER_RECORDINGS_DIR = "/home/carla/recordings"
HOST_FRAMES_DIR = os.path.join(_PORT_DIR, "replay_frames")
HOST_ACCIDENT_DIR = os.path.join(_PORT_DIR, "AccidentScenario")

TIMESTAMP_RE = re.compile(r"(\d{2}-\d{2}-\d{4}-\d{2}-\d{2}-\d{2})")

CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CAMERA_FOV = 90
CAMERA_FPS = 20

CHASE_BEHIND_M = 6.0
CHASE_ABOVE_M = 3.0
CHASE_PITCH_DEG = -15.0


def host_to_container_path(host_path):
    
    abs_host = os.path.abspath(host_path)
    abs_recordings = os.path.abspath(HOST_RECORDINGS_DIR)
    if abs_host.startswith(abs_recordings):
        rel = abs_host[len(abs_recordings):].lstrip(os.sep)
        return os.path.join(CONTAINER_RECORDINGS_DIR, rel)
    return os.path.join(CONTAINER_RECORDINGS_DIR, os.path.basename(host_path))


def latest_recording():
    files = sorted(glob.glob(os.path.join(HOST_RECORDINGS_DIR, "recording_*.log")))
    return files[-1] if files else None


def extract_timestamp(s):
    
    m = TIMESTAMP_RE.search(s)
    return m.group(1) if m else None


def resolve_recording_path(arg):
    
    if arg is None:
        return latest_recording()

    if os.path.isfile(arg):
        base = os.path.basename(arg)
        if base.startswith("recording_") and base.endswith(".log"):
            return os.path.abspath(arg)
        if base.startswith("accident-"):
            ts = extract_timestamp(base)
            if ts is None:
                print("[replay] couldn't parse timestamp out of accident "
                      "scenario name: {}".format(base))
                return None
            return _find_recording_for_timestamp(ts)
        return os.path.abspath(arg)

    cand = os.path.join(HOST_RECORDINGS_DIR, arg)
    if os.path.isfile(cand):
        return cand
    cand = os.path.join(HOST_ACCIDENT_DIR, arg)
    if os.path.isfile(cand):
        ts = extract_timestamp(arg)
        return _find_recording_for_timestamp(ts) if ts else None
    ts = extract_timestamp(arg)
    if ts is not None:
        return _find_recording_for_timestamp(ts)
    return None


def _find_recording_for_timestamp(ts):
    
    candidates = glob.glob(os.path.join(HOST_RECORDINGS_DIR,
                                        "recording_*{}*.log".format(ts)))
    if not candidates:
        print("[replay] no recording in {} matches timestamp {}".format(
            HOST_RECORDINGS_DIR, ts))
        return None
    if len(candidates) > 1:
        print("[replay] multiple recordings match timestamp {}; using {}".format(
            ts, os.path.basename(candidates[0])))
    return os.path.abspath(candidates[0])


def recording_duration_seconds(client, container_path):
    
    info = client.show_recorder_file_info(container_path, False)
    for line in info.splitlines():
        s = line.strip().lower()
        if s.startswith("duration:"):
            try:
                return float(s.split(":", 1)[1].strip().split()[0])
            except Exception:
                pass
    return None


def find_ego(world):
    
    for v in world.get_actors().filter("*vehicle.*"):
        if v.attributes.get("role_name") == "ego_vehicle":
            return v
    vehicles = list(world.get_actors().filter("*vehicle.*"))
    return vehicles[0] if vehicles else None


def chase_transform(target):
    
    loc = target.get_transform().location
    yaw = target.get_transform().rotation.yaw
    rad = math.radians(yaw)
    return carla.Transform(
        carla.Location(
            x=loc.x - CHASE_BEHIND_M * math.cos(rad),
            y=loc.y - CHASE_BEHIND_M * math.sin(rad),
            z=loc.z + CHASE_ABOVE_M),
        carla.Rotation(pitch=CHASE_PITCH_DEG, yaw=yaw, roll=0.0))


def replay_spectator(client, container_path, duration_s):
    
    world = client.get_world()
    client.replay_file(container_path, 0.0, 0.0, 0)
    time.sleep(1.0)

    spectator = world.get_spectator()
    ego = find_ego(world)
    if ego is None:
        print("[replay] no ego in replayed world; just letting it run")
        time.sleep(duration_s or 30.0)
        return

    print("[replay] tracking ego with spectator for ~{:.1f}s".format(
        duration_s or 30.0))
    deadline = time.time() + (duration_s or 30.0) + 2.0
    while time.time() < deadline:
        try:
            spectator.set_transform(chase_transform(ego))
        except Exception:
            break
        time.sleep(1.0 / CAMERA_FPS)


def replay_capture(client, container_path, duration_s, recording_basename):
    
    out_dir = os.path.join(HOST_FRAMES_DIR, recording_basename)
    os.makedirs(out_dir, exist_ok=True)

    world = client.get_world()
    client.replay_file(container_path, 0.0, 0.0, 0)
    time.sleep(1.0)

    ego = find_ego(world)
    if ego is None:
        print("[replay] no ego in replayed world; cannot attach camera")
        return

    bp_lib = world.get_blueprint_library()
    cam_bp = bp_lib.find("sensor.camera.rgb")
    cam_bp.set_attribute("image_size_x", str(CAMERA_WIDTH))
    cam_bp.set_attribute("image_size_y", str(CAMERA_HEIGHT))
    cam_bp.set_attribute("fov", str(CAMERA_FOV))
    cam_bp.set_attribute("sensor_tick", str(1.0 / CAMERA_FPS))
    cam_tf = carla.Transform(
        carla.Location(x=-CHASE_BEHIND_M, z=CHASE_ABOVE_M),
        carla.Rotation(pitch=CHASE_PITCH_DEG))
    cam = world.spawn_actor(cam_bp, cam_tf, attach_to=ego)
    print("[replay] attached camera to ego; frames -> {}".format(out_dir))

    counter = [0]
    def _on_image(img):
        path = os.path.join(out_dir, "frame_{:06d}.png".format(counter[0]))
        try:
            img.save_to_disk(path)
            counter[0] += 1
        except Exception:
            pass

    cam.listen(_on_image)

    wait_s = (duration_s or 30.0) + 2.0
    print("[replay] capturing for ~{:.1f}s...".format(wait_s))
    end_at = time.time() + wait_s
    while time.time() < end_at:
        time.sleep(0.5)

    try:
        cam.stop()
        cam.destroy()
    except Exception:
        pass
    print("[replay] captured {} frames".format(counter[0]))

    mp4_path = os.path.join(HOST_FRAMES_DIR, "{}.mp4".format(recording_basename))
    rc = os.system(
        "ffmpeg -y -framerate {fps} -i {dir}/frame_%06d.png "
        "-c:v libx264 -pix_fmt yuv420p {out} >/dev/null 2>&1".format(
            fps=CAMERA_FPS, dir=out_dir, out=mp4_path))
    if rc == 0:
        print("[replay] encoded -> {}".format(mp4_path))
    else:
        print("[replay] ffmpeg not available or failed; PNG frames are in {}"
              .format(out_dir))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("path", nargs="?", default=None,
                   help="Recording file (host path or filename), OR an "
                        "AccidentScenario file (path or basename) -- in which "
                        "case we extract the timestamp and locate the matching "
                        "recording automatically. Default: most recent recording.")
    p.add_argument("--mode", choices=["spectator", "capture"], default="spectator",
                   help="spectator: chase camera via CARLA spectator (windowed CARLA only). "
                        "capture: record frames from an attached camera to disk (headless OK).")
    args = p.parse_args()

    path = resolve_recording_path(args.path)
    if path is None:
        if args.path is None:
            print("No recordings found in {}".format(HOST_RECORDINGS_DIR))
        else:
            print("Could not resolve {} to a recording file".format(args.path))
        sys.exit(1)
    print("Replaying: {}".format(os.path.basename(path)))

    container_path = host_to_container_path(path)
    print("Server-side path: {}".format(container_path))

    client = carla.Client('localhost', 2000)
    client.set_timeout(20.0)
    client.reload_world()
    time.sleep(1.0)

    duration_s = recording_duration_seconds(client, container_path)
    if duration_s is not None:
        print("Recording duration: {:.2f} s".format(duration_s))

    recording_basename = os.path.splitext(os.path.basename(path))[0]
    if args.mode == "capture":
        replay_capture(client, container_path, duration_s, recording_basename)
    else:
        replay_spectator(client, container_path, duration_s)


if __name__ == "__main__":
    main()
