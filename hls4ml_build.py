"""Build one Keras/QKeras model with hls4ml's Vitis Unified backend."""

import argparse
import os
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import hls4ml
from hls4ml.utils.config import config_from_keras_model
from qkeras.utils import load_qmodel


def configure_common(config: dict, model) -> None:
    """Apply hls4ml settings shared by all benchmark models.

    Args:
        config: hls4ml configuration generated from the loaded Keras model.
        model: Loaded Keras/QKeras model. The model name determines whether
            8-bit or 16-bit precision defaults are used.
    """
    config["Model"]["ReuseFactor"] = 1
    config["Model"]["Strategy"] = "Latency"
    result_precision = (
        "fixed<8,4,RND,SAT>"
        if model.name.endswith("_8_bit")
        else "fixed<16,6,RND,SAT>"
    )
    accum_precision = "fixed<8,4>" if model.name.endswith("_8_bit") else "fixed<20,8>"
    output_layer_name = model.layers[-1].name
    if (
        output_layer_name in config["LayerName"]
        and "Precision" in config["LayerName"][output_layer_name]
    ):
        config["LayerName"][output_layer_name]["Precision"]["result"] = result_precision
    for layer_config in config["LayerName"].values():
        if "Precision" in layer_config:
            layer_config["Precision"]["accum"] = accum_precision


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for a single hls4ml build."""
    parser = argparse.ArgumentParser(description="HLS4ML build (Vitis Unified)")
    parser.add_argument("--model", required=True, help="Path to Keras model file")
    parser.add_argument("--csim", action="store_true", help="Run C simulation")
    parser.add_argument("--cosim", action="store_true", help="Run RTL co-simulation")
    parser.add_argument("--synth", action="store_true", help="Run synthesis")
    parser.add_argument(
        "--bitfile", action="store_true", help="Run implementation to generate bitfile"
    )
    parser.add_argument("--export", action="store_true", help="Export IP (ignored for Vitis Unified)")
    parser.add_argument(
        "--xpfm-path",
        default="/opt/Xilinx/Vitis/2023.2/base_platforms/xilinx_zcu102_base_202320_1/xilinx_zcu102_base_202320_1.xpfm",
        help="Platform path",
    )
    parser.add_argument(
        "--axi-mode",
        choices=["axi_stream", "axi_master"],
        default="axi_stream",
        help="AXI interface mode",
    )
    return parser.parse_args()


def main() -> int:
    """Convert and optionally build one model with the Vitis Unified backend."""
    args = parse_args()

    model_stem = Path(args.model).stem
    max_stem_len = 18 - len("_project")
    proj_stem = model_stem[:max_stem_len] if len(model_stem) > max_stem_len else model_stem
    project_name = f"{proj_stem}_project"
    output_dir = Path(f"hls4ml_project_{model_stem}_unified").resolve()

    print("\n" + "=" * 60)
    print("HLS4ML Build Script (vitis_unified)")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Output directory: {output_dir}")
    print(f"C Simulation: {args.csim}")
    print(f"RTL Co-Simulation: {args.cosim}")
    print(f"HLS Synthesis: {args.synth}")
    print(f"Bitfile: {args.bitfile}")
    print(f"Export IP: {args.export}")
    print(f"AXI mode: {args.axi_mode}")
    print(f"XPFM path: {args.xpfm_path}")
    print("=" * 60 + "\n")

    try:
        print("[1/4] Loading model...")
        loaded_model = load_qmodel(args.model)
        print("Model loaded\n")

        print("[2/4] Generating HLS4ML config...")
        config = config_from_keras_model(loaded_model, granularity="name")
        configure_common(config, loaded_model)

        fifo_flow = "vitisunified:fifo_depth_optimization"
        config["Flows"] = [fifo_flow]
        hls4ml.model.optimizer.get_optimizer(fifo_flow).configure(profiling_fifo_depth=100_000)
        print("Config generated\n")

        print("[3/4] Converting to HLS4ML project files...")
        hls_model = hls4ml.converters.convert_from_keras_model(
            loaded_model,
            hls_config=config,
            output_dir=str(output_dir),
            project_name=project_name,
            backend="VitisUnified",
            board="zcu102",
            part="xczu9eg-ffvb1156-2-e",
            clock_period="5ns",
            io_type="io_stream",
            input_type="float",
            output_type="float",
            xpfmPath=args.xpfm_path,
            axi_mode=args.axi_mode,
        )
        hls_model.compile()
        print("Project files generated\n")

        if args.csim or args.cosim or args.synth or args.bitfile or args.export:
            print("[4/4] Running HLS build...")
            hls_model.build(
                csim=args.csim,
                cosim=args.cosim,
                synth=args.synth,
                bitfile=args.bitfile,
            )
            print("HLS build complete\n")

        print("=" * 60)
        print("HLS4ML flow completed successfully")
        print(f"Output files in: {output_dir}")
        print("=" * 60 + "\n")

    except Exception as exc:
        print(f"\nError: {exc}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
