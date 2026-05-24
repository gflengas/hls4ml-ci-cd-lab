#!/usr/bin/env python3
"""Generate model comparison JSON from local hls4ml build results."""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path


def extract_number(value: str) -> str:
    """Return the first integer or decimal number in a string.

    Args:
        value: Text containing a report table cell.

    Returns:
        First number found, or ``"N/A"`` when the string has no number.
    """
    match = re.search(r"-?\d+(?:\.\d+)?", value)
    return match.group(0) if match else "N/A"


def parse_hls_compile_report(hls_rpt: Path) -> dict[str, str]:
    """Parse latency metrics from a text hls_compile report."""
    lines = hls_rpt.read_text(encoding="utf-8", errors="ignore").splitlines()
    best: tuple[int, float, str] | None = None

    for line in lines:
        if not line.strip().startswith("|"):
            continue
        parts = [part.strip() for part in line.strip().strip("|").split("|")]
        if len(parts) < 7:
            continue
        if not parts[3].isdigit():
            continue
        try:
            latency_cycles = int(parts[3])
            latency_ns = float(parts[4])
        except ValueError:
            continue
        if best is None or latency_cycles > best[0]:
            interval = extract_number(parts[6])
            best = (latency_cycles, latency_ns, interval)

    if best is None:
        raise ValueError(f"Failed to parse HLS latency in {hls_rpt}")

    latency_cycles, latency_ns, interval = best
    latency_us = latency_ns / 1000.0
    return {
        "hls_ii_min": interval,
        "hls_ii_max": interval,
        "hls_latency_min_us": f"{latency_us:.3f} us",
        "hls_latency_avg_us": f"{latency_us:.3f} us",
        "hls_latency_max_us": f"{latency_us:.3f} us",
        "hls_latency_cycles": str(latency_cycles),
    }


def parse_rtl_cosim(cosim_rpt: Path) -> dict[str, str]:
    """Parse RTL co-simulation latency and interval metrics."""
    lines = cosim_rpt.read_text(encoding="utf-8", errors="ignore").splitlines()

    def parse_line(target: str) -> dict[str, str] | None:
        """Parse the first cosimulation table row matching a language target."""
        for line in lines:
            if target not in line or "|" not in line:
                continue
            numbers = re.findall(r"-?\d+(?:\.\d+)?", line)
            if len(numbers) < 6:
                continue
            metrics = {
                "rtl_latency_min_cycles": numbers[0],
                "rtl_latency_avg_cycles": numbers[1],
                "rtl_latency_max_cycles": numbers[2],
                "rtl_interval_min_cycles": numbers[3],
                "rtl_interval_avg_cycles": numbers[4],
                "rtl_interval_max_cycles": numbers[5],
            }
            if len(numbers) >= 7:
                metrics["rtl_total_exec_cycles"] = numbers[6]
            return metrics
        return None

    return parse_line("Verilog") or parse_line("VHDL") or {}


def parse_utilization(util_path: Path) -> dict[str, str]:
    """Parse implementation utilization metrics from a Vivado report."""
    lines = util_path.read_text(encoding="utf-8", errors="ignore").splitlines()

    for line in lines:
        if "(top)" not in line or not line.strip().startswith("|"):
            continue
        parts = [part.strip() for part in line.strip().strip("|").split("|")]
        if len(parts) < 11:
            break
        return {
            "impl_lut": extract_number(parts[2]),
            "impl_ff": extract_number(parts[6]),
            "impl_bram36": extract_number(parts[7]),
            "impl_bram18": extract_number(parts[8]),
            "impl_uram": extract_number(parts[9]),
            "impl_dsp": extract_number(parts[10]),
        }

    summary = {}

    patterns = {
        r"^\|\s*CLB LUTs\s*\|": "impl_lut",
        r"^\|\s*CLB Registers\s*\|": "impl_ff",
        r"^\|\s*RAMB36/FIFO": "impl_bram36",
        r"^\|\s*RAMB18\s*\|": "impl_bram18",
        r"^\|\s*DSPs\s*\|": "impl_dsp",
        r"^\|\s*URAM\s*\|": "impl_uram",
        r"^\|\s*URAMs\s*\|": "impl_uram",
    }

    for line in lines:
        for pattern, field in patterns.items():
            if re.search(pattern, line):
                parts = [part.strip() for part in line.strip().strip("|").split("|")]
                if len(parts) >= 2:
                    value = extract_number(parts[1])
                    if value != "N/A":
                        summary[field] = value
                break

    if summary:
        return summary
    raise ValueError(f"Failed to parse utilization in {util_path}")


def parse_timing_summary(timing_path: Path) -> dict[str, str]:
    """Parse routed timing slack and clock metrics from a Vivado report."""
    lines = timing_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    clk_period = "N/A"
    clk_freq = "N/A"

    in_clock_summary = False
    for line in lines:
        if "Clock Summary" in line:
            in_clock_summary = True
            continue
        if not in_clock_summary:
            continue
        if line.strip().startswith("Clock"):
            continue
        if not line.strip() or set(line.strip()) <= {"-"}:
            continue
        numbers = re.findall(r"-?\d+(?:\.\d+)?", line)
        if len(numbers) >= 2:
            clk_period = numbers[-2]
            clk_freq = numbers[-1]
            break

    in_summary = False
    for idx, line in enumerate(lines):
        if "Design Timing Summary" in line:
            in_summary = True
            continue
        if not in_summary:
            continue
        if not line.strip().startswith("WNS("):
            continue

        for j in range(idx + 1, min(idx + 6, len(lines))):
            candidate = lines[j].strip()
            if not candidate or set(candidate) <= {"-", " "}:
                continue
            values = re.findall(r"-?\d+(?:\.\d+)?", candidate)
            if len(values) < 12:
                continue
            return {
                "wns_ns": values[0],
                "clk_period_ns": clk_period,
                "clk_freq_mhz": clk_freq,
            }

    raise ValueError(f"Failed to parse timing summary in {timing_path}")


def parse_power_summary(power_path: Path) -> dict[str, str]:
    """Parse total, dynamic, and static power from a Vivado power report."""
    text = power_path.read_text(encoding="utf-8", errors="ignore")

    def find(pattern: str) -> str:
        """Return the first regex capture group or ``"N/A"``."""
        match = re.search(pattern, text)
        return match.group(1) if match else "N/A"

    power_total = find(r"Total On-Chip Power \(W\)\s*\|\s*([0-9.]+)")
    power_dynamic = find(r"Dynamic \(W\)\s*\|\s*([0-9.]+)")
    power_static = find(r"Device Static \(W\)\s*\|\s*([0-9.]+)")
    if "N/A" in (power_total, power_dynamic, power_static):
        raise ValueError(f"Failed to parse power in {power_path}")
    return {
        "power_total_w": power_total,
        "power_dynamic_w": power_dynamic,
        "power_static_w": power_static,
    }


def parse_impl_elapsed(runme_log: Path) -> dict[str, str]:
    """Parse total implementation elapsed seconds from Vivado run logs."""
    text = runme_log.read_text(encoding="utf-8", errors="ignore")
    matches = re.findall(r"elapsed = (\d+):(\d+):(\d+)", text)
    if not matches:
        raise ValueError(f"No elapsed time found in {runme_log}")

    total = 0
    for hours, minutes, seconds in matches:
        total += int(hours) * 3600 + int(minutes) * 60 + int(seconds)
    return {"impl_elapsed_sec": str(total)}


def update_if_present(metrics: dict[str, str], path: Path | None, parser) -> None:
    """Merge parsed metrics when an optional report exists.

    Args:
        metrics: Mutable metrics dictionary for one model row.
        path: Optional report path.
        parser: Callable that converts the report into a metrics dictionary.
    """
    if not path or not path.exists():
        return
    try:
        metrics.update(parser(path))
    except Exception as exc:
        metrics[f"parse_warning_{path.name}"] = str(exc)


def parse_model_info(model_info: Path) -> dict:
    """Read model metadata generated during a build job."""
    return json.loads(model_info.read_text(encoding="utf-8"))


def vitis_project_dir(model_dir: Path) -> Path:
    """Return the expected Vitis Unified hls4ml project directory."""
    return model_dir / f"hls4ml_project_{model_dir.name}_unified"


def first_glob(directory: Path, pattern: str) -> Path | None:
    """Return the first path matching a non-recursive glob pattern."""
    matches = sorted(directory.glob(pattern)) if directory.exists() else []
    return matches[0] if matches else None


def report_paths(model_dir: Path) -> dict[str, Path | None]:
    """Build explicit Vitis Unified report and artifact paths for one model."""
    project_dir = vitis_project_dir(model_dir)
    final_reports = project_dir / "final_reports"
    export_dir = project_dir / "export"
    impl_run_dir = (
        project_dir
        / "vitis_workspace/system_link/_x/link/vivado/vpl/prj/prj.runs/impl_1"
    )

    return {
        "util_rpt": final_reports / "impl_1_full_util_routed.rpt",
        "timing_rpt": first_glob(
            final_reports, "impl_1_*timing_summary_routed.rpt"
        ),
        "power_rpt": final_reports / "impl_1_power_routed.rpt",
        "cosim_rpt": final_reports / "hls_cosim.rpt",
        "impl_runme_log": impl_run_dir / "runme.log",
        "hls_compile_rpt": final_reports / "hls_compile.rpt",
        "model_info": model_dir / "model_info.json",
        "pynq_driver": export_dir / "axi_stream_driver.py",
        "bit_file": export_dir / "system.bit",
        "hwh_file": export_dir / "system.hwh",
    }


def main() -> None:
    """Collect completed model runs and write ``model_comparison.json``."""
    base = Path("results")
    model_dirs = sorted(
        entry
        for entry in base.iterdir()
        if entry.is_dir() and not entry.name.startswith(".")
    ) if base.exists() else []

    rows: list[dict[str, str]] = []
    for model_dir in model_dirs:
        model_name = model_dir.name
        metrics: dict[str, str] = {"model": model_name}
        paths = report_paths(model_dir)

        util_rpt = paths["util_rpt"]
        timing_rpt = paths["timing_rpt"]
        power_rpt = paths["power_rpt"]
        cosim_rpt = paths["cosim_rpt"]
        impl_runme_log = paths["impl_runme_log"]
        hls_compile_rpt = paths["hls_compile_rpt"]
        model_info = paths["model_info"]

        if not (
            util_rpt
            and util_rpt.exists()
            and timing_rpt
            and timing_rpt.exists()
        ):
            continue

        update_if_present(metrics, util_rpt, parse_utilization)
        update_if_present(metrics, timing_rpt, parse_timing_summary)
        update_if_present(metrics, power_rpt, parse_power_summary)
        update_if_present(metrics, impl_runme_log, parse_impl_elapsed)
        update_if_present(metrics, model_info, parse_model_info)
        update_if_present(metrics, hls_compile_rpt, parse_hls_compile_report)
        update_if_present(metrics, cosim_rpt, parse_rtl_cosim)

        artifacts_root = Path("ci_artifacts") / model_name
        artifacts_root.mkdir(parents=True, exist_ok=True)

        def stage_report(path: Path | None) -> str:
            """Copy a report into the CI artifacts directory when it exists."""
            if not path or not path.exists():
                return "N/A"
            destination = artifacts_root / path.name
            if destination != path:
                shutil.copy2(path, destination)
            return str(path)

        def stage_many(paths: list[Path | None]) -> list[str]:
            """Copy many artifacts into the CI artifacts directory."""
            staged = []
            seen = set()
            for path in paths:
                if not path or not path.exists():
                    continue
                destination = artifacts_root / path.name
                if destination != path:
                    shutil.copy2(path, destination)
                src_str = str(path)
                if src_str not in seen:
                    staged.append(src_str)
                    seen.add(src_str)
            return staged

        sources = {
            "util_rpt": stage_report(util_rpt),
            "timing_rpt": stage_report(timing_rpt),
            "power_rpt": stage_report(power_rpt),
            "cosim_rpt": stage_report(cosim_rpt),
            "impl_runme_log": stage_report(impl_runme_log),
            "hls_compile_rpt": stage_report(hls_compile_rpt),
            "model_info": stage_report(model_info),
            "pynq_drivers": stage_many([paths["pynq_driver"]]),
            "bit_files": stage_many([paths["bit_file"]]),
            "hwh_files": stage_many([paths["hwh_file"]]),
        }

        rows.append({**metrics, **sources})

    json_path = Path("ci_artifacts/model_comparison.json")
    json_path.parent.mkdir(parents=True, exist_ok=True)

    with json_path.open("w", encoding="utf-8") as file:
        json.dump(rows, file, indent=2)

    if not model_dirs:
        print("No model directories found under results/. Wrote empty comparison JSON.")
    elif not rows:
        print("No completed model runs found. Wrote empty comparison JSON.")

    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
