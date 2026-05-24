# hls4ml Model Comparison Bench

Reusable GitLab CI/CD workflow for comparing Keras/QKeras models on FPGA
hardware with the hls4ml Vitis Unified backend.

The pipeline builds selected models in parallel, runs HLS synthesis, RTL
co-simulation, implementation, bitstream export, optional hardware-in-the-loop
benchmarks on a ZCU102/PYNQ board, and renders an HTML comparison dashboard.

This repository is a workflow template. It intentionally ships with placeholder
runner, container, board, and dashboard-publishing settings so it can be adapted
to another GitLab instance, FPGA runner, board network, and artifact host. The
release package does not include trained model files or NumPy test vectors.

## Quick Start

1. Copy or clone this repository into a GitLab project with a runner that can
   run Vitis/Vivado.
2. Set `HLS4ML_RUNNER_TAG` and `VITIS_IMAGE` for your runner and Vitis image.
3. Add `.h5` or `.keras` models to `models/`.
4. Add optional HIL test vectors as `X_test_<model>.npy`,
   `y_test_<model>.npy`, and `labels_test_<model>.npy`.
5. Configure `ZCU102_HOST`, `ZCU102_USER`, and `ZCU102_SSH_KEY` if you want
   hardware-in-the-loop benchmarks.
6. Optionally configure `DASHBOARD_UPLOAD_BASE_URL` and
   `DASHBOARD_PUBLIC_BASE_URL` to publish dashboards outside GitLab artifacts.
7. Run the GitLab pipeline. Use `TEST_MODELS=model_a.h5,model_b.h5` to limit a
   run to selected models.

## Requirements

- GitLab runner with Docker or an equivalent executor, enough resources for
  Vitis builds, and tools such as `ssh`, `scp`, `ssh-keygen`, and `curl`.
- Vitis/Vivado 2023.2 container image with Conda under `/opt/conda`.
- Python 3.10 environment with TensorFlow 2.13, QKeras, hls4ml, numpy, h5py,
  pydigitalwavetools, pyyaml, and pyparsing.
- Optional ZCU102 target reachable over SSH from the CI runner, with PYNQ under
  `/home/xilinx` and passwordless `sudo -n` for the commands used by
  `ci/hil_run.py`.

The reference setup used for the poster targeted a Xilinx ZCU102 board with
PYNQ. Other boards can be used by adapting the hls4ml board, part, platform,
and board-side assumptions in `hls4ml_build.py` and `ci/hil_run.py`.

For the author's CERN lab setup, `HLS4ML_RUNNER_TAG` was `fpga-mid`,
`VITIS_IMAGE` was `registry.cern.ch/ci4fpga/vivado:2023.2`, and dashboards were
published to a CERNBox/WebDAV-backed public URL. These values are examples, not
requirements.

## CI Variables

| Variable | Default | Purpose |
| :--- | :--- | :--- |
| `HLS4ML_RUNNER_TAG` | `fpga-runner` | Runner tag used by CI jobs |
| `VITIS_IMAGE` | `your-vitis-2023.2-image` | Container image used by CI jobs |
| `MODELS_DIR` | `models` | Directory scanned for `.h5` and `.keras` models |
| `TEST_MODELS` | empty | Optional comma-separated model filenames to build |
| `DASHBOARD_PUBLIC_BASE_URL` | empty | Public base URL for rendered dashboards |
| `DASHBOARD_UPLOAD_BASE_URL` | empty | Upload base URL that accepts `curl -T ... -X PUT` |
| `ZCU102_HOST` | unset | ZCU102 hostname or IP address |
| `ZCU102_USER` | unset | SSH user for the ZCU102 |
| `ZCU102_SSH_KEY` | unset | GitLab file variable containing the private SSH key |

If `DASHBOARD_UPLOAD_BASE_URL` is not set, the rendered dashboard stays as a
GitLab job artifact. If both dashboard URL variables are set, the pipeline
uploads the dashboard and exposes the public report URL through `report.env`.

## Repository Layout

- `.gitlab-ci.yml`: parent GitLab pipeline.
- `ci/generate_child_pipeline.py`: writes the generated per-model child
  pipeline.
- `hls4ml_build.py`: hls4ml conversion and Vitis Unified build wrapper.
- `ci/compare_models.py`: parses Vitis reports and writes
  `ci_artifacts/model_comparison.json`.
- `ci/model_metadata.py`: extracts model-card metadata from Keras/QKeras files.
- `ci/hil_run.py`: CI-side ZCU102 SSH orchestration.
- `ci/hil_bench.py`: board-side benchmark runner.
- `ci/render_dashboard.py`: renders `model_comparison.json` into HTML.
- `ci/dashboard_template.html`: self-contained dashboard template.
- `envs/environment-base.yml`: base Python environment used by build jobs.
- `models/`: user-provided models and HIL test vectors.
- `fashion_mnist_models.py`: demo model and test-vector generator.

Generated directories are intentionally ignored:

- `hls4ml_project*/`
- `results/`
- `ci_artifacts/`
- `*.log`

## Model Inputs

Models live under `models/` and must be saved as `.h5` or `.keras`.

The HIL stage expects matching NumPy files for each model:

- `models/X_test_<model_name>.npy`
- `models/y_test_<model_name>.npy`
- `models/labels_test_<model_name>.npy`

Models without matching vectors can still be built and included in report
parsing. The HIL stage marks them as skipped in `model_comparison.json`.

## Build Configuration

The default build uses:

| Setting | Value |
| :--- | :--- |
| Backend | `vitis_unified` |
| Tool image | Vitis 2023.2 |
| Board | `zcu102` |
| Part | `xczu9eg-ffvb1156-2-e` |
| Clock period | `5ns` |
| IO type | `io_stream` |
| AXI mode | `axi_stream` |
| Build command | `--synth --cosim --export --bitfile` |

The base environment installs hls4ml from
`gflengas/hls4ml.git@fix-link-system-error-handling`. Precision and reuse
settings are applied in `hls4ml_build.py`.

## CI/CD Flow

The pipeline has a parent pipeline and a generated child pipeline:

1. Parent `check` selects model files, applies `TEST_MODELS`, creates or
   restores the cached `hls4ml-base` conda environment, and publishes setup
   artifacts.
2. Parent `generate_model_pipeline` writes
   `ci_artifacts/model-child-pipeline.yml`.
3. Parent `run_model_pipeline` triggers the generated child pipeline.
4. Child `build_<model>` jobs run in parallel and publish `results/<model>/`
   artifacts. Build jobs are `allow_failure: true` so one failed model does not
   block the full comparison.
5. Child `compare_models` parses completed Vitis Unified reports, stages board
   artifacts, and writes `model_comparison.json`.
6. Child `hil_run` optionally copies artifacts and test vectors to the ZCU102,
   runs the board benchmark, and merges HIL metrics.
7. Child `render_results` renders the HTML dashboard and optionally uploads it.

## Trust Model

This workflow is intended for self-hosted experimentation on infrastructure you
control. It loads model files, runs generated build scripts, imports generated
PYNQ drivers, copies artifacts over SSH, and runs board-side Python with
`sudo -n`.

Only run the full pipeline on trusted model files and trusted merge requests, or
isolate public contributions on runners without board access, protected
variables, or other sensitive credentials. Keep `ZCU102_SSH_KEY` and dashboard
upload URLs protected and masked where your GitLab instance supports that.

## Generating Demo Models

To generate Fashion-MNIST demo models and test vectors:

```bash
python3 fashion_mnist_models.py
```

This creates four 8-bit models in `models/`. The poster run also used
`fashion_dense_16_bit.h5` as a legacy 16-bit baseline; that file is not created
by the current demo script.

| Model | Conv Filters | Approx Params | Description |
| :--- | :--- | :--- | :--- |
| `fashion_dense_8_bit.h5` | 8 | 15.8k | Dense-heavy baseline |
| `fashion_cnn_8_bit_a.h5` | 5 | 9.9k | Single-convolution tiny CNN |
| `fashion_cnn_8_bit_b.h5` | 4, 16 | 8.5k | Two-convolution tiny CNN |
| `fashion_skip_8_bit.h5` | 5, 5 | 10.1k | Tiny CNN with a residual skip connection |

Each demo model comes with matching `X_test`, `y_test`, and `labels_test` NumPy
files for HIL validation and dashboard metadata.

Example generated dashboard:
<https://gflengas.web.cern.ch/results/13.05.2026_a8a88c69_pufvd.html>
