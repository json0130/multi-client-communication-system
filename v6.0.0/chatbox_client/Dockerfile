# ChatBox client — v6.0.0 multi-robot server
#
# Build context must be the v6.0.0/ directory (one level up), so that
# both chatbox_client/ and client/ are accessible during the build:
#
#   cd v6.0.0/
#   docker build -f chatbox_client/Dockerfile -t chatbox-client .
#
# Run (with host networking for mDNS/ALSA/serial access):
#   docker run --rm -it \
#     --network host \
#     --device /dev/ttyUSB0 \
#     --device /dev/snd \
#     chatbox-client

FROM ultralytics/ultralytics:latest-jetson-jetpack4

# System dependencies
RUN apt-get update && \
    apt-get install -y \
        curl libssl-dev libusb-1.0-0-dev libudev-dev \
        pkg-config libgtk-3-dev libglfw3-dev libgl1-mesa-dev \
        espeak espeak-data libespeak-dev libpulse-dev alsa-utils \
        libportaudio2 libportaudiocpp0 portaudio19-dev \
        ffmpeg python3-dev python3-pip \
        libnss-mdns avahi-utils \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && sed -i 's/^hosts:.*/hosts: files mdns4_minimal [NOTFOUND=return] dns/' /etc/nsswitch.conf

RUN pip3 install --upgrade pip

RUN pip3 install \
    websockets \
    requests \
    gtts \
    numpy \
    pyserial

# PyAudio must be built from source on Jetson
RUN pip3 install --no-binary :all: pyaudio

# Piper TTS (pre-built aarch64 binary)
RUN PIPER_VERSION="2023.11.14-2" && \
    curl -L "https://github.com/rhasspy/piper/releases/download/${PIPER_VERSION}/piper_linux_aarch64.tar.gz" \
        -o /tmp/piper.tar.gz && \
    mkdir -p /opt/piper && \
    tar -xzf /tmp/piper.tar.gz -C /opt/piper --strip-components=1 && \
    ln -sf /opt/piper/piper /usr/local/bin/piper && \
    echo "/opt/piper" > /etc/ld.so.conf.d/piper.conf && ldconfig && \
    rm /tmp/piper.tar.gz

ENV LD_LIBRARY_PATH=/opt/piper:$LD_LIBRARY_PATH

# Fix GLIBCXX_3.4.26 (required by piper's bundled libpiper_phonemize.so.1)
RUN curl -fsSL "https://keyserver.ubuntu.com/pks/lookup?op=get&search=0x1E9377A2BA9EF27F" \
        | apt-key add - && \
    echo "deb http://ppa.launchpad.net/ubuntu-toolchain-r/test/ubuntu bionic main" \
        > /etc/apt/sources.list.d/ubuntu-toolchain-r.list && \
    apt-get update && \
    apt-get install -y --no-install-recommends libstdc++6 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Fix GLIBC_2.29 — compile espeak-ng from source
RUN apt-get update && apt-get install -y --no-install-recommends \
        autoconf automake libtool make g++ git \
    && git clone --depth 1 https://github.com/rhasspy/espeak-ng /tmp/espeak-ng-src \
    && cd /tmp/espeak-ng-src \
    && ./autogen.sh \
    && ./configure --without-sonic --without-mbrola \
    && make -j$(nproc) \
    && ESNG_LIB=$(find src/.libs -name 'libespeak-ng.so.1.*' ! -type l | head -1) \
    && [ -n "$ESNG_LIB" ] || { echo "libespeak-ng.so not found"; ls src/.libs/; exit 1; } \
    && rm -f /opt/piper/libespeak-ng.so* \
    && cp "$ESNG_LIB" /opt/piper/ \
    && ESNG_BASENAME=$(basename "$ESNG_LIB") \
    && ln -sf "$ESNG_BASENAME" /opt/piper/libespeak-ng.so.1 \
    && ln -sf "$ESNG_BASENAME" /opt/piper/libespeak-ng.so \
    && ldconfig \
    && apt-get remove -y autoconf automake libtool \
    && apt-get autoremove -y && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/espeak-ng-src

# Copy v6.0.0 base client (needed for 'from client import BasicClient')
COPY client /app/client

# Copy chatbox client
COPY chatbox_client /app/chatbox_client

WORKDIR /app/chatbox_client

CMD ["python3", "chatbox.py"]
