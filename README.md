# PredictIQ

Edge predictive maintenance with **LSTM autoencoder anomaly detection**, MQTT telemetry, and Grafana dashboards. Extends the [Maintenance-Strategy](https://github.com/pranav-singh-rathore/Maintenance-strategy) research with time-series fault detection and Raspberry Pi deployment.

## Highlights

| Metric | Result |
|--------|--------|
| Anomaly detection accuracy | ~93%+ on held-out fault patterns |
| Edge inference latency | ~38 ms (quantised) vs ~114 ms FP32 on ARM-class CPU |
| Telemetry throughput | 1,000+ sensor readings/min via MQTT |
| Fault patterns simulated | Bearing wear, imbalance, thermal overload, cavitation, misalignment |

## Architecture

```
Sensor Simulator → LSTM Autoencoder → Anomaly Score → MQTT Broker → Grafana
       ↑                    ↑
  5 fault patterns    Trained on normal ops only
```

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cd src
python train.py
python evaluate.py --model ../artifacts/model.pt
python quantize.py --model ../artifacts/model.pt
```

Or run the full pipeline:

```bash
chmod +x scripts/run_pipeline.sh && ./scripts/run_pipeline.sh
```

### MQTT telemetry (optional)

```bash
# Terminal 1: start a local broker (e.g. mosquitto)
brew services start mosquitto

# Terminal 2: publish simulated edge stream
cd src && python mqtt_telemetry.py --duration 120
```

Import `grafana/dashboard.json` after connecting InfluxDB/MQTT as your datasource.

## Project structure

```
predictiq/
├── src/
│   ├── data_generator.py   # 5 industrial fault simulators
│   ├── model.py            # LSTM autoencoder
│   ├── train.py            # Train on normal data
│   ├── evaluate.py         # Threshold + accuracy metrics
│   ├── quantize.py         # INT8 dynamic quantisation + latency bench
│   └── mqtt_telemetry.py   # Edge → cloud streaming
├── grafana/dashboard.json
├── scripts/run_pipeline.sh
└── artifacts/              # Generated after training
```

## Raspberry Pi deployment

1. Copy `artifacts/model_quantised.pt` to the Pi
2. Install `torch` (ARM build) and `paho-mqtt`
3. Run inference loop reading I²C vibration + temperature sensors
4. Publish scores to your MQTT broker

See `docs/DEPLOYMENT.md` for step-by-step Pi 4 setup.

## Related work

- Published paper: *Change of Maintenance Strategy for Effective Maintenance* (Grenze IJET, 2025)
- Prior repo: [Maintenance-strategy](https://github.com/pranav-singh-rathore/Maintenance-strategy)

## License

MIT
