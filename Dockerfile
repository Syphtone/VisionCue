# ---------------------------
# Base Image
# ---------------------------
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# ---------------------------
# Dependencies
# ---------------------------
RUN apt-get update && apt-get install -y \
    build-essential cmake git wget unzip pkg-config \
    libopenblas-dev liblapack-dev libx11-dev libgtk2.0-dev \
    libboost-all-dev libtbb-dev libeigen3-dev \
    python3 python3-pip python3-setuptools python3-wheel \
    libopencv-dev libsm6 libxext6 libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------
# Ensure pip tooling available and install 'packaging'
# ---------------------------
RUN python3 -m pip install --upgrade pip setuptools wheel packaging

# ---------------------------
# Install dlib C++ library
# ---------------------------
RUN git clone https://github.com/davisking/dlib.git /dlib && \
    cd /dlib && mkdir build && cd build && \
    cmake .. -DUSE_AVX_INSTRUCTIONS=1 -DDLIB_USE_CUDA=0 && \
    cmake --build . --config Release && \
    cmake --install . && \
    python3 -m pip install packaging && \
    cd /dlib && python3 setup.py install

# ---------------------------
# Clone OpenFace Source
# ---------------------------
RUN git clone https://github.com/TadasBaltrusaitis/OpenFace.git /openface

# ---------------------------
# Build OpenFace
# ---------------------------
RUN cd /openface && mkdir build && cd build && \
    cmake -DDLIB_INCLUDE_DIR=/usr/local/include \
          -DDLIB_LIB_DIR=/usr/local/lib \
          .. && \
    make -j$(nproc)

# ---------------------------
# Copy pre-downloaded OpenFace models
# ---------------------------
COPY models /openface/build/bin/model

# --------------------------
# Add Haarcascade
# --------------------------
RUN wget -O /openface/build/bin/haarcascade_frontalface_alt.xml \
    https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_alt.xml

# ---------------------------
# Working folder for input/output
# ---------------------------
WORKDIR /data

# ---------------------------
# Default tool: FeatureExtraction
# ---------------------------
ENTRYPOINT ["/openface/build/bin/FeatureExtraction"]

