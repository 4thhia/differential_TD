#!/bin/bash

USER="4thhia"

# Specify UID & GID
UID=${UID:-$(id -u)}
GID=${GID:-$(id -g)}

# Build Docker Image
docker build -f docker/Dockerfile \
  --build-arg USER=${USER} \
  --build-arg UID=${UID} \
  --build-arg GID=${GID} \
  -t "${USER}_dtd" .
