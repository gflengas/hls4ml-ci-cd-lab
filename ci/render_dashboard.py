#!/usr/bin/env python3
"""Render the model comparison dashboard from JSON and an HTML template."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Render dashboard HTML from model comparison JSON.")
    parser.add_argument(
        "--input",
        default="ci_artifacts/model_comparison.json",
        help="Path to model_comparison.json",
    )
    parser.add_argument(
        "--template",
        default="ci/dashboard_template.html",
        help="Path to HTML template with __MODEL_JSON__ placeholder",
    )
    parser.add_argument(
        "--output",
        default="ci_artifacts/dashboard.html",
        help="Path to rendered HTML output",
    )
    return parser.parse_args()


def main() -> int:
    """Inject model comparison JSON into the dashboard template."""
    args = parse_args()

    input_path = Path(args.input)
    template_path = Path(args.template)
    output_path = Path(args.output)

    rows = json.loads(input_path.read_text(encoding="utf-8"))
    template = template_path.read_text(encoding="utf-8")

    payload = json.dumps(rows, indent=2)
    rendered = template.replace("__MODEL_JSON__", payload)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(rendered, encoding="utf-8")

    print(f"Rendered dashboard: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
