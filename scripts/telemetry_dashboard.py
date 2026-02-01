"""
Ghost Rider telemetry animation generator.

This script reads motorcycle telemetry logs (CSV or HDF5) containing both a
"pro" baseline and a trained agent, then renders a top-down animation plus
live plots for throttle/brake, lean angle, and a G-G diagram. The animation is
saved to .mp4 (preferred) or .gif.

Expected columns (case-insensitive):
- time: seconds
- x_pro, y_pro: pro rider global XY
- x_agent, y_agent: agent global XY
- throttle, brake: 0-1
- lean_angle_deg: degrees (+right, -left)
- ax, ay: longitudinal and lateral acceleration in m/s^2 (will be converted to G)

Usage:
    python scripts/telemetry_dashboard.py --data path/to/log.csv --output outputs/ghost.mp4 --fps 30

If only a subset of columns exist, the script will raise a helpful error.
"""

import argparse
import logging
import math
from pathlib import Path
from typing import Tuple

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Matplotlib defaults for crisper paper figures
plt.rcParams.update({
    "figure.figsize": (10, 8),
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
})

REQUIRED_COLUMNS = {
    "time",
    "x_pro",
    "y_pro",
    "x_agent",
    "y_agent",
    "throttle",
    "brake",
    "lean_angle_deg",
    "ax",
    "ay",
}


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Lowercase and trim column names for robust parsing."""
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    return df


def load_telemetry(path: Path) -> pd.DataFrame:
    """Load telemetry from CSV or HDF5 into a DataFrame sorted by time."""
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    if path.suffix.lower() in {".csv", ".txt"}:
        df = pd.read_csv(path)
    elif path.suffix.lower() in {".h5", ".hdf5"}:
        try:
            df = pd.read_hdf(path, key=None)
        except KeyError:
            # If no key was provided, load first available key
            with pd.HDFStore(path, "r") as store:
                first_key = store.keys()[0]
            df = pd.read_hdf(path, key=first_key)
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")

    df = _normalize_columns(df)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"Telemetry file missing columns: {sorted(missing)}.\n"
            "Expected columns: time, x_pro, y_pro, x_agent, y_agent, throttle, brake, "
            "lean_angle_deg, ax, ay"
        )

    df = df.sort_values("time").reset_index(drop=True)
    return df


def _compute_limits(df: pd.DataFrame) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """Compute XY limits with margin for the top-down view."""
    x_all = np.concatenate([df["x_pro"].values, df["x_agent"].values])
    y_all = np.concatenate([df["y_pro"].values, df["y_agent"].values])
    x_margin = 0.05 * (x_all.max() - x_all.min() + 1e-6)
    y_margin = 0.05 * (y_all.max() - y_all.min() + 1e-6)
    return (x_all.min() - x_margin, x_all.max() + x_margin), (y_all.min() - y_margin, y_all.max() + y_margin)


def animate(df: pd.DataFrame, output: Path, fps: int = 30) -> None:
    """Create and save the Ghost Rider animation."""
    frames = len(df)
    interval_ms = 1000 / fps

    x_lim, y_lim = _compute_limits(df)

    fig = plt.figure(constrained_layout=True)
    gs = fig.add_gridspec(3, 2, height_ratios=[2, 1, 1])

    ax_top = fig.add_subplot(gs[0, :])
    ax_tb = fig.add_subplot(gs[1, 0])  # throttle / brake
    ax_lean = fig.add_subplot(gs[1, 1])
    ax_gg = fig.add_subplot(gs[2, :])

    # Top-down track with trails
    ax_top.set_title("Ghost Rider: Pro (blue) vs Agent (red)")
    ax_top.set_xlim(*x_lim)
    ax_top.set_ylim(*y_lim)
    ax_top.set_aspect("equal", adjustable="box")
    ax_top.set_xlabel("X [m]")
    ax_top.set_ylabel("Y [m]")

    pro_point, = ax_top.plot([], [], "o", color="#1f77b4", markersize=8, label="Pro")
    agent_point, = ax_top.plot([], [], "o", color="#d62728", markersize=8, label="Agent")
    pro_trail, = ax_top.plot([], [], color="#1f77b4", alpha=0.4, linewidth=2)
    agent_trail, = ax_top.plot([], [], color="#d62728", alpha=0.4, linewidth=2)
    ax_top.legend(loc="upper right")

    # Throttle vs Brake
    ax_tb.set_title("Throttle vs Brake")
    ax_tb.set_xlim(df["time"].min(), df["time"].max())
    ax_tb.set_ylim(-0.05, 1.05)
    ax_tb.set_ylabel("Input")
    ax_tb.set_xlabel("Time [s]")
    throttle_line, = ax_tb.plot([], [], color="#1f77b4", label="Throttle")
    brake_line, = ax_tb.plot([], [], color="#d62728", label="Brake")
    tb_cursor = ax_tb.axvline(df["time"].min(), color="0.2", linestyle="--", linewidth=1)
    ax_tb.legend(loc="upper right")

    # Lean angle
    ax_lean.set_title("Lean Angle")
    ax_lean.set_xlim(df["time"].min(), df["time"].max())
    lean_limit = max(45, math.ceil(abs(df["lean_angle_deg"]).max() / 5) * 5)
    ax_lean.set_ylim(-lean_limit, lean_limit)
    ax_lean.set_ylabel("Degrees")
    ax_lean.set_xlabel("Time [s]")
    lean_line, = ax_lean.plot([], [], color="#ff7f0e", label="Lean")
    lean_cursor = ax_lean.axvline(df["time"].min(), color="0.2", linestyle="--", linewidth=1)
    ax_lean.legend(loc="upper right")

    # G-G diagram
    ax_gg.set_title("G-G Diagram (Long vs Lat)")
    ax_gg.set_xlabel("Longitudinal G")
    ax_gg.set_ylabel("Lateral G")
    ax_gg.set_xlim(df["ax"].min() / 9.81 - 0.2, df["ax"].max() / 9.81 + 0.2)
    ax_gg.set_ylim(df["ay"].min() / 9.81 - 0.2, df["ay"].max() / 9.81 + 0.2)
    gg_scatter = ax_gg.scatter([], [], s=12, alpha=0.7, c=np.linspace(0, 1, frames), cmap="viridis")

    def init():
        pro_point.set_data([], [])
        agent_point.set_data([], [])
        pro_trail.set_data([], [])
        agent_trail.set_data([], [])
        throttle_line.set_data([], [])
        brake_line.set_data([], [])
        lean_line.set_data([], [])
        gg_scatter.set_offsets(np.empty((0, 2)))
        return (
            pro_point,
            agent_point,
            pro_trail,
            agent_trail,
            throttle_line,
            brake_line,
            tb_cursor,
            lean_line,
            lean_cursor,
            gg_scatter,
        )

    def update(frame: int):
        t = df.loc[frame, "time"]

        pro_point.set_data(df.loc[frame, "x_pro"], df.loc[frame, "y_pro"])
        agent_point.set_data(df.loc[frame, "x_agent"], df.loc[frame, "y_agent"])

        pro_trail.set_data(df.loc[:frame, "x_pro"], df.loc[:frame, "y_pro"])
        agent_trail.set_data(df.loc[:frame, "x_agent"], df.loc[:frame, "y_agent"])

        throttle_line.set_data(df.loc[:frame, "time"], df.loc[:frame, "throttle"])
        brake_line.set_data(df.loc[:frame, "time"], df.loc[:frame, "brake"])
        tb_cursor.set_xdata(t)

        lean_line.set_data(df.loc[:frame, "time"], df.loc[:frame, "lean_angle_deg"])
        lean_cursor.set_xdata(t)

        gg_points = np.column_stack((df.loc[:frame, "ax"] / 9.81, df.loc[:frame, "ay"] / 9.81))
        gg_scatter.set_offsets(gg_points)

        return (
            pro_point,
            agent_point,
            pro_trail,
            agent_trail,
            throttle_line,
            brake_line,
            tb_cursor,
            lean_line,
            lean_cursor,
            gg_scatter,
        )

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=frames,
        init_func=init,
        blit=True,
        interval=interval_ms,
        repeat=False,
    )

    output.parent.mkdir(parents=True, exist_ok=True)

    try:
        writer = animation.FFMpegWriter(fps=fps, bitrate=1800)
        ani.save(output, writer=writer)
        logging.info("Saved animation to %s via ffmpeg", output)
        return
    except Exception as exc:  # noqa: BLE001
        logging.warning("FFmpeg unavailable or failed (%s). Falling back to Pillow/GIF.", exc)

    gif_path = output.with_suffix(".gif")
    ani.save(gif_path, writer="pillow", fps=fps)
    logging.info("Saved animation to %s", gif_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ghost Rider telemetry animation")
    parser.add_argument("--data", required=True, type=Path, help="Path to telemetry CSV or HDF5")
    parser.add_argument("--output", required=True, type=Path, help="Output file (.mp4 or .gif)")
    parser.add_argument("--fps", default=30, type=int, help="Frames per second for the animation")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    args = parse_args()

    df = load_telemetry(args.data)
    logging.info("Loaded telemetry with %d frames", len(df))

    animate(df, args.output, fps=args.fps)
    logging.info("Done")


if __name__ == "__main__":
    main()
