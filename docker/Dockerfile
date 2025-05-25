ARG CUDA="12.8.0"
ARG TAG="devel"
ARG OS="ubuntu22.04"

FROM nvidia/cuda:${CUDA}-${TAG}-${OS}

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
    sudo \
    curl \
    wget \
    tmux \
    tree \
    git \
    nano \
    swig\
    xclip \
    ffmpeg \
    zip \
    unzip \
    build-essential \
    libssl-dev \
    libffi-dev \
    python3 \
    python3-pip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/* /usr/local/src/* /tmp/*

# Allow user to run with matched UID and GID
ARG USER
ARG UID
ARG GID
RUN groupadd --gid ${GID} ${USER} && \
    useradd \
    --no-log-init \
    --create-home \
    --uid ${UID} \
    --gid ${GID} \
    -s /bin/sh ${USER} && \
    usermod -aG sudo ${USER} && \
    echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
USER ${USER}

# Copy the requirements.txt to the working directory
COPY docker/requirements.txt /

# Set PYTHONPATH for the container
ENV PYTHONPATH=/home/${USER}/workdir
ENV PATH=${PATH}:/home/${USER}/.local/bin

# Upgrade pip and install requirements in a single RUN command
RUN python3 -m pip install --upgrade pip && \
    pip install -r /requirements.txt

# Set working directory before copying requirements.txt
WORKDIR /home/${USER}/workdir

CMD ["bash"]