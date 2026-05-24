#!/usr/bin/env python3
"""Run a hardware-in-the-loop benchmark on the PYNQ/ZCU102 board."""

import argparse
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import numpy as np


def load_driver(driver_path: str):
    """Load a generated PYNQ driver module from a file path.

    Args:
        driver_path: Path to ``axi_stream_driver.py``.

    Returns:
        Imported Python module object.

    Raises:
        RuntimeError: If Python cannot create a module spec for the path.
    """
    spec = importlib.util.spec_from_file_location("axi_stream_driver", driver_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load driver spec from {driver_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def ensure_hwh_for_bit(bit_path: str, hwh_path: str | None) -> None:
    """Place the HWH file next to its bitstream when PYNQ needs that layout.

    Args:
        bit_path: Path to the bitstream loaded by PYNQ.
        hwh_path: Optional source HWH path.
    """
    if not hwh_path:
        return
    bit = Path(bit_path).resolve()
    source_hwh = Path(hwh_path).resolve()
    target_hwh = bit.with_suffix(".hwh")
    if source_hwh == target_hwh:
        return
    if target_hwh.exists():
        return
    shutil.copyfile(source_hwh, target_hwh)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for a board-side benchmark run."""
    parser = argparse.ArgumentParser(description="Run HIL benchmark on PYNQ/ZCU102.")
    parser.add_argument("--bit", required=True, help="Path to .bit file")
    parser.add_argument("--hwh", default=None, help="Path to .hwh file (optional)")
    parser.add_argument("--driver", required=True, help="Path to axi_stream_driver.py")
    parser.add_argument("--inputs", required=True, help="Path to input .npy")
    parser.add_argument("--expected", required=True, help="Path to expected output .npy")
    return parser.parse_args()


def main() -> int:
    """Run inference on the FPGA overlay and print JSON benchmark metrics."""
    args = parse_args()

    ensure_hwh_for_bit(args.bit, args.hwh)

    driver = load_driver(args.driver)

    x_test = np.load(args.inputs)
    y_expected = np.load(args.expected)

    y_shape = y_expected.shape

    nn = driver.NeuralNetworkOverlay(args.bit, x_test.shape, y_shape)

    y_hw, _, throughput = nn.predict(x_test, y_shape, profile=True)

    y_hw = np.asarray(y_hw)
    y_expected = np.asarray(y_expected)

    diff = y_hw - y_expected
    error_mae = float(np.mean(np.abs(diff)))
    error_mse = float(np.mean(diff * diff))

    _, latency_s, _ = nn.predict(x_test[:1], y_expected[:1].shape, profile=True)
    sample_throughput = float(throughput)
    pixels_per_sample = int(np.prod(x_test.shape[1:])) if x_test.ndim > 1 else int(np.prod(x_test.shape))
    pixel_throughput = sample_throughput * pixels_per_sample

    result = {
        "hil_latency": latency_s,
        "hil_throughput": sample_throughput,
        "hil_sample_throughput": sample_throughput,
        "hil_pixel_throughput": pixel_throughput,
        "hil_pixels_per_sample": pixels_per_sample,
        "hil_error_mae": error_mae,
        "hil_error_mse": error_mse,
        "latency_unit": "s",
        "throughput_unit": "inferences/s",
        "pixel_throughput_unit": "pixels/s",
    }

    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(json.dumps({"error": str(exc)}))
        sys.exit(1)
