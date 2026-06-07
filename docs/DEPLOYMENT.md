# Raspberry Pi 4 Deployment Guide

## Prerequisites

- Raspberry Pi 4 (4 GB RAM)
- Python 3.10+
- Optional: ADXL345 vibration sensor, DS18B20 temperature probe

## Steps

1. **Clone and install**
  ```bash
   git clone https://github.com/pranav-singh-rathore/predictiq.git
   cd predictiq
   python3 -m venv .venv && source .venv/bin/activate
   pip install torch paho-mqtt numpy
  ```
2. **Copy quantised model**
  ```bash
   scp artifacts/model_quantised.pt pi@raspberrypi:~/predictiq/
  ```
3. **Run edge inference loop**
  - Load `torch.jit.load("model_quantised.pt")`
  - Read 64-step vibration/temperature window
  - Compute reconstruction MSE; flag if above threshold from `artifacts/eval_metrics.json`
4. **Stream to Grafana**
  ```bash
   python src/mqtt_telemetry.py --broker <your-broker-ip> --duration 3600
  ```

Expected latency on Pi 4: **~35–45 ms** per inference window with quantised model.