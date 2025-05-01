sudo apt update
sudo apt install -y \
    python3-pip \
    libopenblas-dev \
    libatlas-base-dev \
    libportaudio2 \
    portaudio19-dev \
    libprotobuf-dev \
    protobuf-compiler \
    sox \
    libsox-fmt-all

wget -O onnxruntime-1.11.1-cp39-cp39-linux_armv7l.whl https://github.com/nknytk/built-onnxruntime-for-raspberrypi-linux/raw/master/wheels/bullseye/onnxruntime-1.11.1-cp39-cp39-linux_armv7l.whl
wget -O silero_vad.onnx https://raw.githubusercontent.com/snakers4/silero-vad/master/src/silero_vad/data/silero_vad.onnx
pip3 install --upgrade pip
pip3 install -r requirements.txt
