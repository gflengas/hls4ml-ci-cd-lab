#!/usr/bin/env python3
"""
Generate a child GitLab CI pipeline that builds each model in parallel
through the vitis_unified backend, then compares, runs HIL, and renders
a dashboard.
"""

from pathlib import Path
import re

VITIS_IMAGE = "$VITIS_IMAGE"
BUILD_CMD = "python3 -u hls4ml_build.py --synth --cosim --export --bitfile"


def slug(value: str) -> str:
    """Return a GitLab-safe job-name fragment.

    Args:
        value: Arbitrary model name.

    Returns:
        Name containing only ASCII letters, digits, and underscores.
    """
    return re.sub(r"[^a-zA-Z0-9_]", "_", value)


def main() -> None:
    """Generate the child GitLab pipeline for selected models."""
    models_file = Path("ci_artifacts/models_to_test.txt")
    out_file = Path("ci_artifacts/model-child-pipeline.yml")

    models = [m.strip() for m in models_file.read_text(encoding="utf-8").splitlines() if m.strip()]
    if not models:
        raise SystemExit("No models selected.")

    lines: list[str] = []
    lines.extend(["stages:", "  - build", "  - compare", "  - hil", "  - results", ""])

    all_build_jobs: list[str] = []

    for model_path in models:
        model_stub = Path(model_path).stem
        model_slug = slug(model_stub)
        job = f"build_{model_slug}"
        all_build_jobs.append(job)
        model_filename = Path(model_path).name

        lines.append(f"{job}:")
        lines.append("  stage: build")
        lines.append(f"  image: {VITIS_IMAGE}")
        lines.append("  tags:")
        lines.append("    - ${HLS4ML_RUNNER_TAG}")
        lines.append("  allow_failure: true")
        lines.append("  needs:")
        lines.append("    - pipeline: $PARENT_PIPELINE_ID")
        lines.append("      job: check")
        lines.append("      artifacts: true")
        lines.append("  script:")
        lines.append("    - source /opt/conda/etc/profile.d/conda.sh")
        lines.append("    - mkdir -p ci_artifacts")
        lines.append("    - tar -xzf ci_artifacts/base_env.tar.gz -C /opt/conda/envs/")
        lines.append(f"    - run_dir=\"results/{model_stub}\"")
        lines.append(f"    - mkdir -p \"$run_dir\"")
        lines.append(f"    - cp \"{model_path}\" \"$run_dir/{model_filename}\"")
        lines.append("    - cp hls4ml_build.py \"$run_dir/\"")
        lines.append("    - cd \"$run_dir\"")
        lines.append("    - conda activate hls4ml-base")
        lines.append(
            f"    - python3 ../../ci/model_metadata.py --model {model_filename} --output model_info.json "
            f"--inputs ../../models/X_test_{model_stub}.npy --labels ../../models/labels_test_{model_stub}.npy"
        )
        lines.append(f"    - {BUILD_CMD} --model {model_filename} 2>&1")
        lines.append("  artifacts:")
        lines.append("    when: always")
        lines.append("    paths:")
        lines.append(f"      - results/{model_stub}/")
        lines.append("")

    lines.extend(
        [
            "compare_models:",
            "  stage: compare",
            "  image: $VITIS_IMAGE",
            "  tags:",
            "    - ${HLS4ML_RUNNER_TAG}",
            "  needs:",
        ]
    )
    for job in all_build_jobs:
        lines.append("    - job: " + job)
        lines.append("      optional: true")
    lines.extend(
        [
            "  script:",
            "    - mkdir -p ci_artifacts",
            "    - python3 ci/compare_models.py",
            "  artifacts:",
            "    when: always",
            "    paths:",
            "      - ci_artifacts/model_comparison.json",
            "      - ci_artifacts/*/",
            "",
        ]
    )

    lines.extend(
        [
            "hil_run:",
            "  stage: hil",
            "  image: $VITIS_IMAGE",
            "  tags:",
            "    - ${HLS4ML_RUNNER_TAG}",
            "  needs:",
            "    - job: compare_models",
            "      artifacts: true",
            "  script:",
            "    - python3 ci/hil_run.py",
            "  artifacts:",
            "    when: always",
            "    paths:",
            "      - ci_artifacts/model_comparison.json",
            "",
        ]
    )

    lines.extend(
        [
            "render_results:",
            "  stage: results",
            "  image: $VITIS_IMAGE",
            "  tags:",
            "    - ${HLS4ML_RUNNER_TAG}",
            "  needs:",
            "    - job: hil_run",
            "      artifacts: true",
            "  script:",
            "    - python3 ci/render_dashboard.py --input ci_artifacts/model_comparison.json --template ci/dashboard_template.html --output ci_artifacts/dashboard.html",
            "    - dash_date=\"$(date -u +%d.%m.%Y)\"",
            "    - dash_hash=\"$(printf '%s' \"${CI_COMMIT_SHORT_SHA:-$CI_COMMIT_SHA}\" | cut -c1-8)\"",
            "    - dash_name=\"${dash_date}_${dash_hash}.html\"",
            "    - mv ci_artifacts/dashboard.html \"ci_artifacts/${dash_name}\"",
            "    - report_url=\"\"",
            "    - |",
            "      if [ -n \"${DASHBOARD_PUBLIC_BASE_URL:-}\" ]; then",
            "        report_url=\"${DASHBOARD_PUBLIC_BASE_URL%/}/${dash_name}\"",
            "      fi",
            "    - printf \"REPORT_FILE=%s\\nREPORT_URL=%s\\nREPORT_UPLOAD_STATUS=pending\\n\" \"${dash_name}\" \"${report_url}\" > ci_artifacts/report.env",
            "    - |",
            "      if [ -n \"${DASHBOARD_UPLOAD_BASE_URL:-}\" ]; then",
            "        upload_url=\"${DASHBOARD_UPLOAD_BASE_URL%/}/${dash_name}\"",
            "        if curl -f -T \"ci_artifacts/${dash_name}\" -X PUT \"${upload_url}\"; then",
            "          printf \"REPORT_FILE=%s\\nREPORT_URL=%s\\nREPORT_UPLOAD_STATUS=uploaded\\n\" \"${dash_name}\" \"${report_url}\" > ci_artifacts/report.env",
            "        else",
            "          echo \"WARNING: dashboard upload failed; keeping HTML as a job artifact.\"",
            "          printf \"REPORT_FILE=%s\\nREPORT_URL=%s\\nREPORT_UPLOAD_STATUS=failed\\n\" \"${dash_name}\" \"${report_url}\" > ci_artifacts/report.env",
            "        fi",
            "      else",
            "        echo \"Dashboard upload disabled; keeping HTML as a job artifact.\"",
            "        printf \"REPORT_FILE=%s\\nREPORT_URL=%s\\nREPORT_UPLOAD_STATUS=skipped\\n\" \"${dash_name}\" \"${report_url}\" > ci_artifacts/report.env",
            "      fi",
            "  artifacts:",
            "    when: always",
            "    reports:",
            "      dotenv: ci_artifacts/report.env",
            "    paths:",
            "      - ci_artifacts/model_comparison.json",
            "      - ci_artifacts/*.html",
            "",
        ]
    )

    out_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Generated {out_file}")


if __name__ == "__main__":
    main()
