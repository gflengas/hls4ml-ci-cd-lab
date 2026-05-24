#!/usr/bin/env python3
"""Copy artifacts to a ZCU102, run HIL benchmarks, and merge the results."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import stat
import subprocess
import sys
import tempfile
from pathlib import Path


def run_cmd(cmd: list[str], *, check: bool = True) -> subprocess.CompletedProcess:
    """Run a subprocess and capture its output.

    Args:
        cmd: Command and arguments.
        check: Whether to raise on a non-zero exit code.

    Returns:
        Completed process with captured stdout and stderr.
    """
    return subprocess.run(cmd, check=check, text=True, capture_output=True)


def ensure_private_key_permissions(key_path: Path) -> None:
    """Set private-key permissions required by OpenSSH.

    Args:
        key_path: Path to the key file.
    """
    current_mode = stat.S_IMODE(key_path.stat().st_mode)
    required_mode = 0o600
    if current_mode != required_mode:
        key_path.chmod(required_mode)


def normalize_private_key_bytes(raw_key: bytes) -> bytes:
    """Normalize SSH private key bytes from GitLab file variables.

    Args:
        raw_key: Raw key bytes as provided by the CI variable.

    Returns:
        Key bytes with Unix newlines and a trailing newline.
    """
    normalized_key = raw_key.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    if b"\n" not in normalized_key and b"\\n" in normalized_key:
        normalized_key = normalized_key.replace(b"\\n", b"\n")
    if not normalized_key.endswith(b"\n"):
        normalized_key += b"\n"
    return normalized_key


def validate_private_key(key_path: Path) -> None:
    """Validate that a private key can be parsed by OpenSSH.

    Args:
        key_path: Path to the key file.

    Raises:
        ValueError: If ``ssh-keygen`` cannot parse the key.
    """
    result = run_cmd(["ssh-keygen", "-y", "-f", str(key_path)], check=False)
    if result.returncode != 0:
        raise ValueError(
            "ZCU102_SSH_KEY is not a valid OpenSSH private key after normalization. "
            "Check whether the GitLab file variable contains the raw private key text, "
            "not escaped text, not base64, and not a truncated copy."
        )


def prepare_private_key(key_path: Path) -> Path:
    """Create a normalized temporary copy of a CI private key.

    Args:
        key_path: Original key path from ``ZCU102_SSH_KEY``.

    Returns:
        Path to a sanitized temporary key file.
    """
    normalized_key = normalize_private_key_bytes(key_path.read_bytes())

    fd, sanitized_key_path = tempfile.mkstemp(prefix="zcu102_ssh_key_", text=False)
    os.close(fd)
    sanitized_key = Path(sanitized_key_path)
    sanitized_key.write_bytes(normalized_key)
    ensure_private_key_permissions(sanitized_key)
    validate_private_key(sanitized_key)
    return sanitized_key


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    return argparse.ArgumentParser().parse_args()


def main() -> int:
    """Run HIL benchmarks for all reportable models in model_comparison.json."""
    args = parse_args()
    pipeline_id = os.environ.get("CI_PIPELINE_ID", "local")
    remote_root = f"/home/xilinx/jupyter_notebooks/gflengas/hil_runs/{pipeline_id}"
    host = os.environ["ZCU102_HOST"]
    user = os.environ["ZCU102_USER"]
    original_key_path = Path(os.environ["ZCU102_SSH_KEY"])
    key_path = prepare_private_key(original_key_path)
    ssh_base = [
        "ssh",
        "-i",
        str(key_path),
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "UserKnownHostsFile=/dev/null",
    ]
    scp_base = [
        "scp",
        "-i",
        str(key_path),
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "UserKnownHostsFile=/dev/null",
    ]

    ssh_cmd = ssh_base[:]
    scp_cmd = scp_base[:]
    target = f"{user}@{host}"

    json_path = Path("ci_artifacts/model_comparison.json")
    rows = json.loads(json_path.read_text(encoding="utf-8"))

    models_dir = Path("models")

    result = run_cmd(
        ssh_cmd
        + [
            target,
            f"sudo -n mkdir -p {shlex.quote(remote_root)} && "
            f"sudo -n chown -R {shlex.quote(user)}:{shlex.quote(user)} {shlex.quote(remote_root)}",
        ],
        check=False,
    )
    if result.returncode != 0:
        sys.stderr.write(result.stdout)
        sys.stderr.write(result.stderr)
        raise SystemExit(result.returncode)

    try:
        for row in rows:
            model = row["model"]
            print(f"Model: {model}")
            artifacts_root = Path("ci_artifacts") / model
            bit_files = row.get("bit_files") or []
            hwh_files = row.get("hwh_files") or []
            pynq_drivers = row.get("pynq_drivers") or []
            if not bit_files or not hwh_files or not pynq_drivers:
                row["hil_status"] = "skipped_missing_board_artifacts"
                json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
                print(f"Skipping {model}: missing bitstream, HWH, or PYNQ driver artifact.")
                continue

            bit_file = artifacts_root / Path(bit_files[0]).name
            hwh_file = artifacts_root / Path(hwh_files[0]).name
            driver_file = artifacts_root / Path(pynq_drivers[0]).name
            inputs_file = models_dir / f"X_test_{model}.npy"
            expected_file = models_dir / f"y_test_{model}.npy"
            required_files = [bit_file, hwh_file, driver_file, inputs_file, expected_file]
            missing_files = [str(path) for path in required_files if not path.exists()]
            if missing_files:
                row["hil_status"] = "skipped_missing_files"
                row["hil_missing_files"] = missing_files
                json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
                print(f"Skipping {model}: missing files: {', '.join(missing_files)}")
                continue

            remote_dir = f"{remote_root}/{model}"
            run_cmd(
                ssh_cmd
                + [
                    target,
                    f"mkdir -p {shlex.quote(remote_dir)}",
                ]
            )

            run_cmd(scp_cmd + [str(bit_file), f"{target}:{remote_dir}/"])
            run_cmd(scp_cmd + [str(hwh_file), f"{target}:{remote_dir}/"])
            run_cmd(scp_cmd + [str(driver_file), f"{target}:{remote_dir}/"])
            run_cmd(scp_cmd + [str(inputs_file), f"{target}:{remote_dir}/"])
            run_cmd(scp_cmd + [str(expected_file), f"{target}:{remote_dir}/"])
            run_cmd(scp_cmd + ["ci/hil_bench.py", f"{target}:{remote_dir}/"])

            remote_py = (
                f"PYTHONPATH=/home/xilinx python3 {shlex.quote(remote_dir)}/hil_bench.py "
                f"--bit {shlex.quote(remote_dir)}/{shlex.quote(bit_file.name)} "
                f"--hwh {shlex.quote(remote_dir)}/{shlex.quote(hwh_file.name)} "
                f"--driver {shlex.quote(remote_dir)}/{shlex.quote(driver_file.name)} "
                f"--inputs {shlex.quote(remote_dir)}/{shlex.quote(inputs_file.name)} "
                f"--expected {shlex.quote(remote_dir)}/{shlex.quote(expected_file.name)}"
            )
            print(remote_py)
            remote_cmd = f"sudo -n bash -lc {shlex.quote('source /etc/profile && ' + remote_py)}"
            result = run_cmd(ssh_cmd + [target, remote_cmd], check=False)
            if result.returncode != 0:
                sys.stderr.write(result.stdout)
                sys.stderr.write(result.stderr)
                raise SystemExit(result.returncode)
            last_line = [line for line in result.stdout.splitlines() if line.strip()][-1]
            hil_metrics = json.loads(last_line)

            row.update(hil_metrics)
            json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    finally:
        run_cmd(
            ssh_cmd + [target, f"sudo -n rm -rf {shlex.quote(remote_root)}"], check=False
        )
        key_path.unlink(missing_ok=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
