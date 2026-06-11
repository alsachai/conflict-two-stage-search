#!/bin/bash

# Resolve script dir so the recordings folder is anchored to the project
# regardless of cwd. Recordings are written by the CARLA simulator inside
# the container, but the volume mount makes them appear on the host at
# <repo>/port/recordings/. CARLA's start_recorder RPC takes a path that
# the SERVER (the simulator) writes to -- so the path passed from Python
# must be the container path "/home/carla/recordings/...", not the host
# path. Files in container land in host via the bind mount.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RECORDINGS_DIR="$SCRIPT_DIR/port/recordings"

mkdir -p "$RECORDINGS_DIR"
# carlasim/carla:0.9.10.1 runs as a non-root user; world-writable avoids
# permission errors when the container writes the file.
chmod 777 "$RECORDINGS_DIR" 2>/dev/null || true

docker run --name="carla-$USER" \
  -d --rm \
  -p 2000-2002:2000-2002 \
  -v "$RECORDINGS_DIR":/home/carla/recordings \
  --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=0 --gpus 'device=0' \
  carlasim/carla:0.9.10.1 \
  /bin/bash -c  \
  'SDL_VIDEODRIVER=offscreen CUDA_DEVICE_ORDER=PCI_BUS_ID \
  CUDA_VISIBLE_DEVICES=1 ./CarlaUE4.sh -ResX=640 -ResY=360 \
  -nosound -windowed -opengl \
  -carla-rpc-port=2000 \
  -quality-level=Epic'
