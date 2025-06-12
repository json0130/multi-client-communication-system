#!/bin/bash

# Remove any previous container
docker rm -f docker-chatbot-socket 2>/dev/null

# Run the Docker container with necessary permissions
docker run -it \
  --network host \
  --env-file .env \
  --device=$(grep ARDUINO_PORT .env | cut -d '=' -f2) \
  --device=/dev/video0 \
  --device=/dev/video1 \
  --device=/dev/video2 \
  --device=/dev/video3 \
  --device=/dev/video4 \
  --volume=/dev:/dev \
  --device-cgroup-rule='c 189:* rmw' \
  --device=/dev/bus/usb \
  --privileged \
  -v /dev/bus/usb:/dev/bus/usb \
  -v "$(pwd)":/app \
  -v ${XDG_RUNTIME_DIR}/pulse/native:${XDG_RUNTIME_DIR}/pulse/native \
  -v ~/.config/pulse:/root/.config/pulse \
  -e PULSE_SERVER=unix:${XDG_RUNTIME_DIR}/pulse/native \
  --group-add audio \
  --runtime nvidia \
  -p 8080:8080 \
  --name docker-chatbot-socket \
  docker-chatbot-socket
