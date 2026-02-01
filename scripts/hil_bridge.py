"""
Hardware-in-the-loop bridge between Gymnasium simulation and an ESP32 over UART.

Protocol (line-oriented, ASCII):
PC -> ESP32:  "T,<ax>,<ay>,<az>,<gx>,<gy>,<gz>\n"  (floats in SI units)
ESP32 -> PC:  "A,<vibrate>\n" where <vibrate> is float 0.0-1.0 (haptic intensity)

Timing: The PC blocks each step until the ESP32 replies or a timeout fires.
RTT is measured per step and averaged; target <15 ms end-to-end.

Run:
    python scripts/hil_bridge.py --port /dev/ttyUSB0 --baud 115200 --steps 200

Notes:
- Ensure the ESP32 firmware uses the same protocol and baud rate.
- Adjust sensor mapping in `observation_to_sensors` to match your env signals.
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Tuple

import numpy as np
import serial

# Local imports (env lives in simulation/)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from simulation.motorcycle_env import MotorcycleEnv  # noqa: E402


def observation_to_sensors(obs: np.ndarray) -> Tuple[float, float, float, float, float, float]:
    """Map env observation to virtual IMU readings (ax, ay, az, gx, gy, gz).

    ax/ay use lateral g and speed; az ~ gravity. Gyro uses roll rate estimate.
    """
    velocity = float(obs[0])  # m/s
    lateral_g = float(obs[2])  # already in G
    roll_deg = float(obs[1])

    ax = 0.0  # assume no long accel in this simplified mapping
    ay = lateral_g * 9.81
    az = 9.81  # gravity

    roll_rate_deg_s = (roll_deg - getattr(observation_to_sensors, "_prev_roll", roll_deg)) / 0.02
    observation_to_sensors._prev_roll = roll_deg
    gx = 0.0
    gy = np.deg2rad(roll_rate_deg_s)
    gz = 0.0
    return ax, ay, az, gx, gy, gz


def send_sensor_packet(ser: serial.Serial, sensors: Tuple[float, float, float, float, float, float]) -> None:
    payload = "T,{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f}\n".format(*sensors)
    ser.write(payload.encode("ascii"))
    ser.flush()


def read_action_packet(ser: serial.Serial, timeout_s: float) -> Tuple[bool, float, float]:
    start = time.perf_counter_ns()
    ser.timeout = timeout_s
    line = ser.readline().decode("ascii", errors="ignore").strip()
    end = time.perf_counter_ns()
    if not line or not line.startswith("A,"):
        return False, 0.0, 0.0
    try:
        value = float(line.split(",", 1)[1])
    except ValueError:
        return False, 0.0, 0.0
    rtt_ms = (end - start) / 1e6
    return True, value, rtt_ms


def main() -> None:
    parser = argparse.ArgumentParser(description="HIL bridge for Moto-Edge-RL")
    parser.add_argument("--port", required=True, help="Serial port (e.g., /dev/ttyUSB0)")
    parser.add_argument("--baud", type=int, default=115200, help="Baud rate")
    parser.add_argument("--steps", type=int, default=200, help="Number of timesteps")
    parser.add_argument("--timeout", type=float, default=0.02, help="Serial timeout per step (s)")
    args = parser.parse_args()

    ser = serial.Serial(args.port, baudrate=args.baud, timeout=args.timeout)

    env = MotorcycleEnv(render_mode=None)
    obs, info = env.reset()

    rtts = []
    for t in range(args.steps):
        sensors = observation_to_sensors(obs)
        send_sensor_packet(ser, sensors)
        ok, vibrate, rtt_ms = read_action_packet(ser, args.timeout)
        if ok:
            rtts.append(rtt_ms)
        else:
            vibrate = 0.0

        action = np.array([vibrate, vibrate, 200.0], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            obs, info = env.reset()

    env.close()
    ser.close()

    if rtts:
        avg = sum(rtts) / len(rtts)
        worst = max(rtts)
        print(f"RTT avg: {avg:.3f} ms, worst: {worst:.3f} ms over {len(rtts)} replies")
    else:
        print("No valid RTT samples captured.")


if __name__ == "__main__":
    main()
