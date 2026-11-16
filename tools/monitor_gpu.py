#!/usr/bin/env python3
"""Show per-GPU utilization, VRAM, board power, and temperature until Ctrl+C."""

import argparse
import csv
import datetime as dt
import io
import shutil
import subprocess
import sys
import time


FIELDS = (
    "index",
    "name",
    "utilization.gpu",
    "utilization.memory",
    "memory.used",
    "memory.total",
    "power.draw",
    "power.limit",
    "temperature.gpu",
)


def number(value):
    value = value.strip()
    if value.upper() in {"N/A", "[NOT SUPPORTED]", ""}:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def sample():
    result = subprocess.run(
        [
            "nvidia-smi",
            f"--query-gpu={','.join(FIELDS)}",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    rows = []
    for row in csv.reader(io.StringIO(result.stdout)):
        if len(row) != len(FIELDS):
            continue
        rows.append(dict(zip(FIELDS, row)))
    if not rows:
        raise RuntimeError("nvidia-smi returned no GPU readings")
    return rows


def fmt(value, suffix=""):
    return "N/A" if value is None else f"{value:.0f}{suffix}"


def main():
    parser = argparse.ArgumentParser(
        description="Monitor GPU utilization, VRAM, board power, and temperature."
    )
    parser.add_argument(
        "-i",
        "--interval",
        type=float,
        default=5.0,
        metavar="SECONDS",
        help="sample interval (default: 5 seconds)",
    )
    args = parser.parse_args()
    if args.interval < 1:
        parser.error("--interval must be at least 1 second")
    if shutil.which("nvidia-smi") is None:
        parser.error("nvidia-smi was not found")

    print("GPU board power only; this does not include CPU or wall-plug power. Press Ctrl+C to stop.")
    try:
        while True:
            rows = sample()
            parts = []
            total_power = 0.0
            total_limit = 0.0
            for row in rows:
                gpu = number(row["index"])
                util = number(row["utilization.gpu"])
                mem_util = number(row["utilization.memory"])
                used = number(row["memory.used"])
                capacity = number(row["memory.total"])
                power = number(row["power.draw"])
                limit = number(row["power.limit"])
                temp = number(row["temperature.gpu"])
                if power is not None:
                    total_power += power
                if limit is not None:
                    total_limit += limit
                memory = (
                    "VRAM N/A"
                    if used is None or capacity is None
                    else f"VRAM {used / 1024:.1f}/{capacity / 1024:.1f} GiB"
                )
                power_text = (
                    "power N/A"
                    if power is None
                    else f"power {power:.0f}/{fmt(limit, ' W')}"
                )
                parts.append(
                    f"GPU{fmt(gpu)} {fmt(util, '%')} util, "
                    f"{fmt(mem_util, '%')} mem-util, {memory}, "
                    f"{power_text}, {fmt(temp, ' C')}"
                )
            stamp = dt.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
            print(
                f"{stamp} | {' | '.join(parts)} | "
                f"GPU total {total_power:.0f}/{total_limit:.0f} W",
                flush=True,
            )
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nGPU monitoring stopped.")
    except (OSError, subprocess.CalledProcessError, RuntimeError) as exc:
        print(f"monitor_gpu: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
