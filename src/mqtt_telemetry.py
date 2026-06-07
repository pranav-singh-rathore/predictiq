"""MQTT telemetry publisher simulating edge sensor stream to Grafana."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone

import numpy as np
import paho.mqtt.client as mqtt

from data_generator import FaultPattern, GeneratorConfig, generate_dataset


def publish(
    broker: str = "localhost",
    port: int = 1883,
    topic: str = "predictiq/sensors",
    rate_hz: float = 16.0,
    duration_sec: int = 60,
) -> None:
    x, y = generate_dataset(GeneratorConfig())
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    client.connect(broker, port, keepalive=60)
    client.loop_start()

    interval = 1.0 / rate_hz
    sent = 0
    deadline = time.time() + duration_sec
    idx = 0

    print(f"Publishing to {broker}:{port}/{topic} at ~{rate_hz * 60:.0f} readings/min")

    while time.time() < deadline:
        seq = x[idx % len(x)]
        idx += 1
        for step, (vibration, temperature) in enumerate(seq):
            payload = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "vibration": float(vibration),
                "temperature": float(temperature),
                "fault": FaultPattern(list(FaultPattern)[y[(idx - 1) % len(y)]]).value,
                "step": step,
            }
            client.publish(topic, json.dumps(payload))
            sent += 1
            time.sleep(interval / len(seq))

    client.loop_stop()
    client.disconnect()
    print(f"Sent {sent} messages (~{sent / duration_sec * 60:.0f}/min)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--broker", default="localhost")
    parser.add_argument("--port", type=int, default=1883)
    parser.add_argument("--topic", default="predictiq/sensors")
    parser.add_argument("--duration", type=int, default=60)
    args = parser.parse_args()
    publish(broker=args.broker, port=args.port, topic=args.topic, duration_sec=args.duration)
