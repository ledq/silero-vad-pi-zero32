# Silero VAD on Raspberry Pi

Real-time Voice Activity Detection (VAD) using the Silero VAD ONNX model, optimized for Raspberry Pi 32-bit OS.

---

## Setup Instructions

1. **Clone the repository:**

```bash
git clone https://github.com/ledq/silero-vad-pi-zero32.git
cd ilero-vad-pi-zero32
```

2. **Run the setup script:**

```bash
chmod +x setup.sh
./setup.sh
```


## Requirements

- Raspberry Pi Zero 2 W 
- Raspberry Pi OS 32-bit (Raspbian GNU/Linux 11 Bullseye)  
- Python 3.9+
- onnxruntime wheel from community: wget https://github.com/nknytk/built-onnxruntime-for-raspberrypi-linux/raw/master/wheels/bullseye/onnxruntime-1.11.1-cp39-cp39-linux_armv7l.whl

---


## Notes

- Input audio must be mono and sampled at 16kHz (or 8kHz).
- Silero VAD model (`silero_vad.onnx`) is included in the `models/` folder.
- The ONNX Runtime wheel is pre-built for Raspberry Pi ARMv7 architecture (community built, no official support).

---

## Run inference

```bash
python3 silero_live.py
```

